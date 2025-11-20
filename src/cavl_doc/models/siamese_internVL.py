# src/models/siamese_internVL.py
from typing import Any, Callable, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------
# AttentionPooling + ProjectionHead (self-contained)
# ----------------------
class AttentionPooling(nn.Module):
    """
    Attention-based pooling that learns a global query vector and attends over token sequence.
    Input: tokens (B, seq_len, hidden_dim)
    Output: pooled (B, hidden_dim)
    Uses nn.MultiheadAttention (batch_first=False requires seq-first).
    """
    def __init__(self, hidden_dim: int = 1536, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        # use batch_first=False API (seq_len, batch, embed)
        # we'll transpose tokens accordingly
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=False)
        # learnable query token (1, hidden_dim)
        self.query = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)  # (1, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        tokens: (B, seq_len, hidden_dim)
        mask: optional bool tensor (B, seq_len) with True for valid tokens OR None
        returns: (B, hidden_dim)
        """
        if tokens is None:
            raise ValueError("tokens must be provided to AttentionPooling")

        b, seq_len, d = tokens.shape

        # --- Ensure dtype consistency between tokens and pool params (query / mha weights) ---
        # Use the dtype of the learnable query as canonical dtype for MHA ops
        target_dtype = self.query.dtype
        if tokens.dtype != target_dtype:
            tokens = tokens.to(dtype=target_dtype)

        tokens_t = tokens.transpose(0, 1)  # (seq_len, B, d)

        q = self.query.unsqueeze(1).expand(1, b, d)  # (1, B, d)
        # q already has the correct dtype (self.query.dtype)

        if mask is not None:
            # expected key_padding_mask: (B, seq_len) with True in positions that should be masked
            if mask.dtype == torch.bool:
                key_padding_mask = ~mask  # True = masked
            else:
                key_padding_mask = ~(mask.bool())
        else:
            key_padding_mask = None

        attn_out, _ = self.mha(q, tokens_t, tokens_t, key_padding_mask=key_padding_mask)
        pooled = attn_out.squeeze(0)  # (B, d)
        pooled = self.ln(pooled)
        return pooled


class ProjectionHead(nn.Module):
    """
    Projection MLP head (bigger): input_dim -> proj_hidden -> proj_out
    Includes LayerNorm on input and final L2 normalization.
    Default: proj_hidden=4096, proj_out=512
    """
    def __init__(self, input_dim: int = 1536, proj_hidden: int = 4096, proj_out: int = 512, use_norm: bool = True):
        super().__init__()
        self.ln = nn.LayerNorm(input_dim, eps=1e-6)
        self.fc1 = nn.Linear(input_dim, proj_hidden, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(proj_hidden, proj_out, bias=True)
        self.use_norm = use_norm

        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor):
        """
        x: (B, input_dim)
        returns: (B, proj_out) L2-normalized (if use_norm)
        """
        x = self.ln(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        if self.use_norm:
            x = F.normalize(x, p=2, dim=-1)
        return x

# ----------------------
# Siamese wrapper
# ----------------------
class SiameseInternVL(nn.Module):
    """
    Wrapper for InternVLChatModel to produce siamese embeddings using hidden states at `cut_layer`.
    - Replaces lm_head and final norm by nn.Identity() when possible
    - encode_fn(backbone, images, cut_layer) is recommended: returns (tokens, mask)
    - set_default_trainable() unfreezes typical layer-N parameters (layer == cut_layer)
    """
    def __init__(self,
                 backbone: Any,
                 cut_layer: int = 27,
                 hidden_dim: int = 1536,
                 proj_hidden: int = 4096,
                 proj_out: int = 512,
                 num_pool_heads: int = 8,
                 encode_fn: Optional[Callable] = None):
        super().__init__()
        self.backbone = backbone
        self.cut_layer = cut_layer
        self.encode_fn = encode_fn
        self.hidden_dim = hidden_dim

        # neutralize NTP-specific modules if present (safe guard)
        try:
            # many ChatModel wrappers have language_model.lm_head
            if hasattr(self.backbone.language_model, "lm_head"):
                self.backbone.language_model.lm_head = nn.Identity()
        except Exception:
            pass
        try:
            # some models have language_model.model.norm
            if hasattr(self.backbone.language_model.model, "norm"):
                self.backbone.language_model.model.norm = nn.Identity()
        except Exception:
            pass

        # new pooling + head
        self.pool = AttentionPooling(hidden_dim, num_heads=num_pool_heads)
        self.head = ProjectionHead(input_dim=hidden_dim, proj_hidden=proj_hidden, proj_out=proj_out)

        # freeze backbone by default
        self.freeze_all_backbone()

    # --- param control ---
    def freeze_all_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_params_by_substrings(self, substrings: List[str]):
        """
        Unfreeze any parameter whose name contains one of the substrings.
        Skip non-floating-point tensors (and log them).
        """
        skipped = []
        unfrozen = []
        for name, p in self.backbone.named_parameters():
            for s in substrings:
                if s in name:
                    # only allow floating-point tensors to require gradients
                    if not getattr(p, "dtype", None) or not p.dtype.is_floating_point:
                        skipped.append((name, str(p.dtype)))
                        break
                    p.requires_grad = True
                    unfrozen.append(name)
                    break

        if skipped:
            print("[WARN] Some parameters matched the requested substrings but were NOT unfrozen because they are not floating-point tensors:")
            for n, dt in skipped:
                print(f"  - {n} (dtype={dt})")
        if unfrozen:
            print(f"[INFO] Unfroze {len(unfrozen)} parameter tensors (examples): {unfrozen[:8]}")

    def set_default_trainable(self):
        """
        Default: freeze all backbone then unfreeze typical submodules in cut_layer
        (q_proj, k_proj, v_proj, o_proj, mlp projections, layernorms).
        """
        self.freeze_all_backbone()
        cut = self.cut_layer
        keys = [
            f"language_model.model.layers.{cut}.self_attn.q_proj",
            f"language_model.model.layers.{cut}.self_attn.k_proj",
            f"language_model.model.layers.{cut}.self_attn.v_proj",
            f"language_model.model.layers.{cut}.self_attn.o_proj",
            f"language_model.model.layers.{cut}.mlp.gate_proj",
            f"language_model.model.layers.{cut}.mlp.up_proj",
            f"language_model.model.layers.{cut}.mlp.down_proj",
            f"language_model.model.layers.{cut}.input_layernorm",
            f"language_model.model.layers.{cut}.post_attention_layernorm",
        ]
        self.unfreeze_params_by_substrings(keys)

    def list_trainable(self, prefix: str = ""):
        items = []
        for n, p in self.named_parameters():
            if p.requires_grad and (prefix == "" or prefix in n):
                items.append((n, tuple(p.shape)))
        return items

    def trainable_summary(self):
        total = 0
        trainable = 0
        print("Trainable parameter summary:")
        for n, p in self.named_parameters():
            nparams = p.numel()
            total += nparams
            if p.requires_grad:
                trainable += nparams
                print(f"  {n:90s} | TRAINABLE | shape={tuple(p.shape)}")
        pct = 100.0 * trainable / total if total > 0 else 0.0
        print(f"Total params: {total:,} | Trainable: {trainable:,} ({pct:.2f}%)")
        return total, trainable

    # --- token extraction ---
    def _extract_tokens_via_encode_fn(self, images: torch.Tensor, device: Optional[torch.device] = None,
                                      **encode_kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert callable(self.encode_fn), "encode_fn not provided."
        out = self.encode_fn(self.backbone, images, cut_layer=self.cut_layer, **(encode_kwargs or {}))
        if isinstance(out, tuple):
            tokens, mask = out
        else:
            tokens, mask = out, None
        return tokens, mask

    def _extract_tokens_via_hidden_states(self, input_ids: Optional[torch.Tensor] = None,
                                          attention_mask: Optional[torch.Tensor] = None,
                                          device: Optional[torch.device] = None,
                                          **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        lm = self.backbone.language_model.model
        call_args = dict(output_hidden_states=True, return_dict=True)
        if input_ids is not None:
            call_args['input_ids'] = input_ids.to(next(self.parameters()).device)
        if attention_mask is not None:
            call_args['attention_mask'] = attention_mask.to(next(self.parameters()).device)
        call_args.update(kwargs)
        out = lm(**call_args)
        hidden_states = out.hidden_states
        # hidden_states length may have +1 due to embeddings; adjust index accordingly
        if len(hidden_states) == (len(lm.layers) + 1):
            idx = self.cut_layer + 1
        else:
            idx = self.cut_layer
        tokens = hidden_states[idx]
        return tokens, None

    # --- forward ---
    def forward(self,
                images: Optional[torch.Tensor] = None,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                device: Optional[torch.device] = None,
                encode_kwargs: Optional[dict] = None) -> torch.Tensor:
        """
        Forward returns projected embedding z for given input.
        Preferred mode: provide `images` and `encode_fn` (the dataset uses per-document patch tensors).
        Alternative: provide input_ids + attention_mask and let the LM produce hidden states.
        """
        device = device or (next(self.parameters()).device)
        if self.encode_fn is not None and images is not None:
            tokens, mask = self._extract_tokens_via_encode_fn(images.to(device), device=device, **(encode_kwargs or {}))
        else:
            tokens, mask = self._extract_tokens_via_hidden_states(input_ids=input_ids,
                                                                  attention_mask=attention_mask,
                                                                  device=device,
                                                                  **(encode_kwargs or {}))
        # tokens: (batch, seq_len, hidden_dim) OR (1, seq_len, hidden_dim)
        pooled = self.pool(tokens, mask=mask)
        z = self.head(pooled)
        return z

    # convenience
    def save_head(self, path: str):
        sd = {'pool': self.pool.state_dict(), 'head': self.head.state_dict()}
        torch.save(sd, path)

    def load_head(self, path: str, map_location=None):
        sd = torch.load(path, map_location=map_location)
        self.pool.load_state_dict(sd['pool'])
        self.head.load_state_dict(sd['head'])


# ----------------------
# factory
# ----------------------
def build_siamese_internvl(
        backbone: Any,
        cut_layer: int = 27,
        encode_fn: Optional[Callable] = None,
        hidden_dim: int = 1536,
        proj_hidden: int = 4096,
        proj_out: int = 512,
        num_pool_heads: int = 8,
        pool_dim: Optional[int] = None,    # <-- compat alias
        set_trainable: bool = True
) -> SiameseInternVL:
    """
    Create SiameseInternVL configured with AttentionPooling + ProjectionHead.
    Backwards-compatible: accepts pool_dim (old name) as alias for hidden_dim.
    """
    if pool_dim is not None:
        hidden_dim = pool_dim

    siam = SiameseInternVL(
        backbone=backbone,
        cut_layer=cut_layer,
        hidden_dim=hidden_dim,
        proj_hidden=proj_hidden,
        proj_out=proj_out,
        num_pool_heads=num_pool_heads,
        encode_fn=encode_fn
    )
    if set_trainable:
        siam.set_default_trainable()
    return siam

