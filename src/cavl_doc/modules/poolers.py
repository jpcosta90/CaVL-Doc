# src/cavl_doc/modules/poolers.py
import torch
import torch.nn as nn
from typing import Optional

class AttentionPooling(nn.Module):
    """
    Attention-based pooling that learns a global query vector.
    
    Evolução (Retrocompatível):
    - Se num_queries=1 (padrão): Comporta-se EXATAMENTE como o antigo (mesmos nomes de pesos).
    - Se num_queries>1: Ativa o modo Multi-Query para capturar contextos distintos.
    """
    def __init__(self, hidden_dim: int = 1536, num_heads: int = 8, num_queries: int = 1, dropout: float = 0.0):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        
        # MHA padrão
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=False)
        
        # --- COMPATIBILIDADE DE PESOS ---
        # Mantemos o nome 'self.query' e 'self.ln' para carregar checkpoints antigos sem erro.
        if num_queries == 1:
            # Shape antigo: [1, hidden_dim]
            self.query = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        else:
            # Shape novo para multi-query: [num_queries, 1, hidden_dim]
            self.query = nn.Parameter(torch.randn(num_queries, 1, hidden_dim) * 0.02)
        
        # Mantemos o nome 'self.ln' (LayerNorm de entrada/saída única)
        self.ln = nn.LayerNorm(hidden_dim, eps=1e-6)
        
        # Camadas extras apenas para o modo Multi-Query
        if num_queries > 1:
            self.proj = nn.Linear(num_queries * hidden_dim, hidden_dim)
            self.ln_out = nn.LayerNorm(hidden_dim, eps=1e-6)
        else:
            self.proj = None

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if tokens is None:
            raise ValueError("tokens must be provided to AttentionPooling")
        
        b, seq_len, d = tokens.shape
        
        # Garante mesmo dtype (bfloat16/float32)
        target_dtype = self.query.dtype
        if tokens.dtype != target_dtype:
            tokens = tokens.to(dtype=target_dtype)
            
        # Transpose para [Seq_Len, Batch, Dim]
        tokens_t = tokens.transpose(0, 1)
        
        # --- Lógica de Expansão da Query ---
        if self.num_queries == 1:
            # Lógica Antiga: unsqueeze(1) manual
            # q shape: [1, B, Dim]
            q = self.query.unsqueeze(1).expand(1, b, d)
        else:
            # Lógica Nova: expand direto (pois já tem 3 dimensões)
            # q shape: [Num_Queries, B, Dim]
            q = self.query.expand(-1, b, -1)
        
        # Máscara
        if mask is not None:
            key_padding_mask = ~mask if mask.dtype == torch.bool else ~(mask.bool())
        else:
            key_padding_mask = None
            
        # Atenção
        attn_out, _ = self.mha(query=q, key=tokens_t, value=tokens_t, key_padding_mask=key_padding_mask)
        
        # --- Processamento de Saída ---
        if self.num_queries == 1:
            # Caminho Legado (Idêntico ao original)
            # Squeeze dim 0 -> [B, Dim] -> LayerNorm
            return self.ln(attn_out.squeeze(0))
        else:
            # Caminho Multi-Query
            # [Num, B, Dim] -> [B, Num, Dim]
            attn_out = attn_out.transpose(0, 1)
            # Flatten -> [B, Num*Dim]
            flat = attn_out.reshape(b, -1)
            # Projeção e Norm final
            return self.ln_out(self.proj(flat))

class PositionalAttentionPooling(nn.Module):
    """
    Attention-based pooling that adds learnable positional encodings to tokens before attention.
    Useful for tasks where the sequence order/position is critical and might be lost.
    """
    def __init__(self, hidden_dim: int = 1536, num_heads: int = 8, num_queries: int = 1, dropout: float = 0.0, max_len: int = 4096):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        
        # Learnable Positional Encoding
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, hidden_dim) * 0.02)
        
        # MHA
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=False)
        
        # Queries
        if num_queries == 1:
            self.query = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        else:
            self.query = nn.Parameter(torch.randn(num_queries, 1, hidden_dim) * 0.02)
        
        self.ln = nn.LayerNorm(hidden_dim, eps=1e-6)
        
        if num_queries > 1:
            self.proj = nn.Linear(num_queries * hidden_dim, hidden_dim)
            self.ln_out = nn.LayerNorm(hidden_dim, eps=1e-6)
        else:
            self.proj = None

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if tokens is None:
            raise ValueError("tokens must be provided")
        
        b, seq_len, d = tokens.shape
        
        # Garante mesmo dtype
        target_dtype = self.query.dtype
        if tokens.dtype != target_dtype:
            tokens = tokens.to(dtype=target_dtype)
            
        # --- Add Positional Encoding ---
        # Se a sequência for maior que o max_len, interpolamos o pos_embed
        if seq_len > self.pos_embed.shape[1]:
            # [1, Max, D] -> [1, D, Max] for interpolate
            pe = self.pos_embed.transpose(1, 2)
            pe = torch.nn.functional.interpolate(pe, size=seq_len, mode='linear', align_corners=False)
            pe = pe.transpose(1, 2) # [1, Seq, D]
        else:
            pe = self.pos_embed[:, :seq_len, :]
            
        tokens = tokens + pe.to(dtype=tokens.dtype, device=tokens.device)
        
        # Transpose para [Seq_Len, Batch, Dim] (MHA default é batch_first=False)
        tokens_t = tokens.transpose(0, 1)
        
        # --- Query Expansion ---
        if self.num_queries == 1:
            q = self.query.unsqueeze(1).expand(1, b, d)
        else:
            q = self.query.expand(-1, b, -1)
        
        # Mask
        if mask is not None:
            key_padding_mask = ~mask if mask.dtype == torch.bool else ~(mask.bool())
        else:
            key_padding_mask = None
            
        # Attention
        attn_out, _ = self.mha(query=q, key=tokens_t, value=tokens_t, key_padding_mask=key_padding_mask)
        
        # Output
        if self.num_queries == 1:
            return self.ln(attn_out.squeeze(0))
        else:
            attn_out = attn_out.transpose(0, 1)
            flat = attn_out.reshape(b, -1)
            return self.ln_out(self.proj(flat))

class MeanPooling(nn.Module):
    """Pooling simples (Média)."""
    def __init__(self, hidden_dim: int = 1536, **kwargs):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim, eps=1e-6)
        
    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None):
        target_dtype = self.ln.weight.dtype
        if tokens.dtype != target_dtype:
            tokens = tokens.to(dtype=target_dtype)
        if mask is None:
            pooled = tokens.mean(dim=1)
        else:
            mask = mask.unsqueeze(-1).float()
            tokens = tokens * mask
            sum_tokens = tokens.sum(dim=1)
            sum_mask = mask.sum(dim=1).clamp(min=1e-9)
            pooled = sum_tokens / sum_mask
        return self.ln(pooled)

class ModalPooler(nn.Module):
    """
    Asymmetric pooler: mean pool on visual, learned attention on text.

    Motivation
    ----------
    Visual tokens at cut_layer are many (~1792) and their activation distribution
    already encodes spatial semantics — mean pooling aggregates this well and
    behaves like soft attention over the activation map.

    Text tokens are few (5) and heterogeneous:
      - <s>, <img>  : precede the visual sequence, never attended to any patch
                      → hidden states carry almost no document information
      - </img>, ▁Analyze, ▁document : post-visual, rich cross-modal representations
                      → should dominate the text embedding

    A learned query attending over the text tokens naturally learns to weight
    the post-visual tokens heavily and suppress the pre-visual structural ones,
    without any explicit supervision.

    Interface
    ---------
    Same as other poolers + requires visual_mask to split modalities.
    Set via CaVLModel when pooler_type="modal".
    """

    requires_visual_mask = True

    def __init__(
        self,
        hidden_dim: int = 1536,
        num_heads: int = 8,
        dropout: float = 0.0,
        fixed_alpha: Optional[float] = None,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.text_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.text_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.ln_v = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.ln_t = nn.LayerNorm(hidden_dim, eps=1e-6)

        if fixed_alpha is None:
            self._alpha_logit = nn.Parameter(torch.zeros(1))
            self._fixed_alpha = None
        else:
            self._alpha_logit = None
            self._fixed_alpha = fixed_alpha

    @property
    def alpha(self) -> torch.Tensor:
        if self._fixed_alpha is not None:
            return torch.tensor(self._fixed_alpha)
        return torch.sigmoid(self._alpha_logit)

    def forward(
        self,
        tokens: torch.Tensor,                        # [B, seq, D]
        mask: Optional[torch.Tensor] = None,         # [B, seq] 1=valid 0=pad
        visual_mask: Optional[torch.Tensor] = None,  # [B, seq] True=visual token
    ) -> torch.Tensor:                               # [B, D]
        target_dtype = self.ln_v.weight.dtype
        if tokens.dtype != target_dtype:
            tokens = tokens.to(dtype=target_dtype)

        if visual_mask is None:
            pooled = tokens.mean(1) if mask is None else (
                (tokens * mask.unsqueeze(-1).float()).sum(1)
                / mask.float().sum(1, keepdim=True).clamp(min=1e-9)
            )
            return self.ln_v(pooled)

        visual_mask = visual_mask.bool()
        B = tokens.shape[0]

        # --- visual: mean pool ---
        vis_w = visual_mask.unsqueeze(-1).float()
        v_pool = (tokens * vis_w).sum(1) / vis_w.sum(1).clamp(min=1e-9)  # [B, D]

        # --- text: learned attention ---
        text_valid = ~visual_mask
        if mask is not None:
            text_valid = text_valid & mask.bool()

        q = self.text_query.expand(B, -1, -1)              # [B, 1, D]
        t_attended, _ = self.text_attn(
            query=q,
            key=tokens,
            value=tokens,
            key_padding_mask=~text_valid,                  # attend only to text tokens
        )
        t_attended = t_attended.squeeze(1)                 # [B, D]

        a = self.alpha.to(tokens.device)
        return a * self.ln_v(v_pool) + (1 - a) * self.ln_t(t_attended)


class PromptGuidedPooler(nn.Module):
    """
    Cross-modal pooler that uses text tokens as queries over visual tokens.

    Motivation
    ----------
    In causal VLMs (InternVL3, Qwen2-VL), visual tokens appear BEFORE text in
    the sequence. Because of the causal mask, visual hidden states at Layer -1
    have NEVER attended to the prompt — they are prompt-blind.
    Text tokens, however, have attended to all visual patches AND the full
    prompt, making them naturally cross-modal.

    This pooler exploits that asymmetry:
      1. text_pool  = mean(text hidden states)   ← already vision+prompt conditioned
      2. v_attended = CrossAttn(Q=text_pool, K=visual, V=visual)
                      ← visual summary *guided by what the prompt asked*
      3. output = α * v_attended + (1-α) * text_pool

    α is a learnable scalar (sigmoid-gated) that the model adjusts based on
    how much visual spatial detail vs. prompt semantics matters for the task.

    Args
    ----
    hidden_dim   : token dimension (1536 for InternVL3-2B)
    num_heads    : heads in the cross-attention MHA
    dropout      : dropout in cross-attention
    fixed_alpha  : if not None, fix the visual/text blend instead of learning it
    """

    requires_visual_mask = True

    # sigmoid(1.386) ≈ 0.80 — visual side gets 80% by default.
    _ALPHA_INIT = 1.386

    def __init__(
        self,
        hidden_dim: int = 1536,
        num_heads: int = 8,
        dropout: float = 0.0,
        fixed_alpha: Optional[float] = None,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_v = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.ln_t = nn.LayerNorm(hidden_dim, eps=1e-6)

        if fixed_alpha is None:
            # Learnable blend. Initialised at ~0.80 (visual-heavy) because:
            #   - v_attended is the truly prompt-guided output
            #   - t_pool is the query reused as a direct signal — double-counting
            #     at equal weight would over-represent the text side
            self._alpha_logit = nn.Parameter(torch.full((1,), self._ALPHA_INIT))
            self._fixed_alpha = None
        else:
            self._alpha_logit = None
            self._fixed_alpha = fixed_alpha

    @property
    def alpha(self) -> torch.Tensor:
        if self._fixed_alpha is not None:
            return torch.tensor(self._fixed_alpha)
        return torch.sigmoid(self._alpha_logit)

    def forward(
        self,
        tokens: torch.Tensor,                        # [B, seq, D]
        mask: Optional[torch.Tensor] = None,         # [B, seq] 1=valid 0=pad  (attention mask)
        visual_mask: Optional[torch.Tensor] = None,  # [B, seq] True=visual token (<IMG_CONTEXT>)
    ) -> torch.Tensor:                               # [B, D]
        """
        Unified interface identical to the other poolers, with one extra argument.

        visual_mask is mandatory for meaningful cross-modal behaviour.
        When absent (e.g. text-only forward), the pooler degrades gracefully
        to a plain mean pool so it never crashes existing code paths.
        """
        target_dtype = self.ln_v.weight.dtype
        if tokens.dtype != target_dtype:
            tokens = tokens.to(dtype=target_dtype)

        if visual_mask is None:
            # Graceful fallback: no modality information available
            pooled = tokens.mean(dim=1) if mask is None else (
                (tokens * mask.unsqueeze(-1).float()).sum(1)
                / mask.float().sum(1, keepdim=True).clamp(min=1e-9)
            )
            return self.ln_t(pooled)

        visual_mask = visual_mask.bool()

        # --- text mean: valid text positions (not visual, not padding) ---
        text_valid = ~visual_mask
        if mask is not None:
            text_valid = text_valid & mask.bool()
        txt_w = text_valid.unsqueeze(-1).float()
        t_pool = (tokens * txt_w).sum(1) / txt_w.sum(1).clamp(min=1e-9)  # [B, D]

        # --- cross-attention: t_pool as query, visual tokens as K/V ---
        # key_padding_mask: True = ignore → pass ~visual_mask (ignore non-visual positions)
        q = t_pool.unsqueeze(1)                                # [B, 1, D]
        v_attended, _ = self.cross_attn(
            query=q,
            key=tokens,
            value=tokens,
            key_padding_mask=~visual_mask,                     # ignore text+pad as K/V
        )
        v_attended = v_attended.squeeze(1)                     # [B, D]

        a = self.alpha.to(v_attended.device)
        return a * self.ln_v(v_attended) + (1 - a) * self.ln_t(t_pool)


class CrossModalPooler(nn.Module):
    """
    Bidirectional cross-modal attention pooler with learned queries.

    Two learned query vectors — one per direction — attend over opposite modalities:

      query_for_visual  (learned)  →  attends visual tokens  →  from_visual [B, D]
      query_for_text    (learned)  →  attends text tokens    →  from_text   [B, D]

      output = α · ln_v(from_visual) + (1-α) · ln_t(from_text)

    No mean pooling is used as a query anywhere. Each query is shaped end-to-end
    by the metric learning loss (CosFace/SubCenterCosFace) to extract the most
    discriminative features from each modality for document comparison.

    Why learned queries instead of modality-derived means:
      - mean(visual): uniform over ~3328 patches — doesn't isolate discriminative regions
      - mean(text): 5 tokens of which 2 (<s>, <img>) precede the image and carry
        no document information — corrupts the query before attention even runs

    The two directions are complementary:
      - query_for_visual learns "which patches matter to compare documents"
      - query_for_text   learns "which text tokens encode the most document identity"
        (converges to weight </img>, ▁Analyze, ▁document heavily — post-visual tokens)
    """

    requires_visual_mask = True

    def __init__(
        self,
        hidden_dim: int = 1536,
        num_heads: int = 8,
        dropout: float = 0.0,
        fixed_alpha: Optional[float] = None,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.attn_t2v = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.attn_v2t = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )

        # Learned queries — no dependence on input means
        self.query_for_visual = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.query_for_text   = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        self.ln_v = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.ln_t = nn.LayerNorm(hidden_dim, eps=1e-6)

        if fixed_alpha is None:
            self._alpha_logit = nn.Parameter(torch.zeros(1))
            self._fixed_alpha = None
        else:
            self._alpha_logit = None
            self._fixed_alpha = fixed_alpha

    @property
    def alpha(self) -> torch.Tensor:
        if self._fixed_alpha is not None:
            return torch.tensor(self._fixed_alpha)
        return torch.sigmoid(self._alpha_logit)

    def forward(
        self,
        tokens: torch.Tensor,                        # [B, seq, D]
        mask: Optional[torch.Tensor] = None,         # [B, seq] 1=valid 0=pad
        visual_mask: Optional[torch.Tensor] = None,  # [B, seq] True=visual token
    ) -> torch.Tensor:                               # [B, D]
        target_dtype = self.ln_v.weight.dtype
        if tokens.dtype != target_dtype:
            tokens = tokens.to(dtype=target_dtype)

        if visual_mask is None:
            pooled = tokens.mean(1) if mask is None else (
                (tokens * mask.unsqueeze(-1).float()).sum(1)
                / mask.float().sum(1, keepdim=True).clamp(min=1e-9)
            )
            return self.ln_v(pooled)

        visual_mask = visual_mask.bool()
        B = tokens.shape[0]

        text_valid = ~visual_mask
        if mask is not None:
            text_valid = text_valid & mask.bool()

        D = tokens.shape[-1]
        head_dim = D // self.attn_t2v.num_heads

        # --- learned query → visual tokens ---
        q_v = self.query_for_visual.expand(B, -1, -1)    # [B, 1, D]
        from_visual, _ = self.attn_t2v(
            query=q_v,
            key=tokens, value=tokens,
            key_padding_mask=~visual_mask,
        )
        from_visual = from_visual.squeeze(1)             # [B, D]

        # --- learned query → text tokens ---
        q_t = self.query_for_text.expand(B, -1, -1)      # [B, 1, D]
        from_text, _ = self.attn_v2t(
            query=q_t,
            key=tokens, value=tokens,
            key_padding_mask=~text_valid,
        )
        from_text = from_text.squeeze(1)                 # [B, D]

        # --- compute attention entropy manually (independent of MHA fast path) ---
        # Uses only W_Q projection from each MHA to reconstruct logits
        with torch.no_grad():
            # visual side: q_v projected through W_Q, compared with visual K tokens
            wq_v = self.attn_t2v.in_proj_weight[:D]          # W_Q: [D, D]
            q_v_proj = (q_v @ wq_v.T)                         # [B, 1, D]
            wk_v = self.attn_t2v.in_proj_weight[D:2*D]        # W_K: [D, D]
            k_v = tokens @ wk_v.T                              # [B, seq, D]
            scores_v = (q_v_proj @ k_v.transpose(1, 2)) / (head_dim ** 0.5)  # [B, 1, seq]
            scores_v = scores_v.squeeze(1)                     # [B, seq]
            scores_v = scores_v.masked_fill(~visual_mask, float('-inf'))
            p_v = torch.softmax(scores_v.float(), dim=-1)
            H_v = -(p_v * torch.log(p_v + 1e-9)).sum(-1).mean()

            # text side
            wq_t = self.attn_v2t.in_proj_weight[:D]
            q_t_proj = (q_t @ wq_t.T)                         # [B, 1, D]
            wk_t = self.attn_v2t.in_proj_weight[D:2*D]
            k_t = tokens @ wk_t.T                              # [B, seq, D]
            scores_t = (q_t_proj @ k_t.transpose(1, 2)) / (head_dim ** 0.5)  # [B, 1, seq]
            scores_t = scores_t.squeeze(1)
            scores_t = scores_t.masked_fill(~text_valid, float('-inf'))
            p_t = torch.softmax(scores_t.float(), dim=-1)
            H_t = -(p_t * torch.log(p_t + 1e-9)).sum(-1).mean()

        self._last_entropy_visual = H_v.item()
        self._last_entropy_text   = H_t.item()

        a = self.alpha.to(tokens.device)
        return a * self.ln_v(from_visual) + (1 - a) * self.ln_t(from_text)


# --- REGISTRO ---
POOLER_REGISTRY = {
    "attention": AttentionPooling,
    "positional_attention": PositionalAttentionPooling,
    "mean": MeanPooling,
    "modal": ModalPooler,
    "prompt_guided": PromptGuidedPooler,
    "cross_modal": CrossModalPooler,
}

def build_pooler(pooler_type: str, **kwargs):
    if pooler_type not in POOLER_REGISTRY:
        raise ValueError(f"Pooler '{pooler_type}' não encontrado.")

    import inspect
    cls = POOLER_REGISTRY[pooler_type]
    accepted = inspect.signature(cls.__init__).parameters
    valid_args = {k: v for k, v in kwargs.items() if k in accepted}

    if pooler_type == "attention":
        valid_args.setdefault('num_queries', 1)

    return cls(**valid_args)