# src/cavl_doc/modules/poolers.py
import torch
import torch.nn as nn
from typing import Optional

class AttentionPooling(nn.Module):
    """
    Attention-based pooling that learns a global query vector and attends over token sequence.
    Input: tokens (B, seq_len, hidden_dim)
    Output: pooled (B, hidden_dim)
    """
    def __init__(self, hidden_dim: int = 1536, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=False)
        self.query = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        self.ln = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if tokens is None:
            raise ValueError("tokens must be provided to AttentionPooling")
        
        b, seq_len, d = tokens.shape
        target_dtype = self.query.dtype
        if tokens.dtype != target_dtype:
            tokens = tokens.to(dtype=target_dtype)
            
        tokens_t = tokens.transpose(0, 1)
        q = self.query.unsqueeze(1).expand(1, b, d)
        
        if mask is not None:
            key_padding_mask = ~mask if mask.dtype == torch.bool else ~(mask.bool())
        else:
            key_padding_mask = None
            
        attn_out, _ = self.mha(q, tokens_t, tokens_t, key_padding_mask=key_padding_mask)
        return self.ln(attn_out.squeeze(0))

class MeanPooling(nn.Module):
    """
    Pooling simples que calcula a média dos tokens.
    Útil como baseline leve e rápido.
    """
    def __init__(self, hidden_dim: int = 1536, **kwargs):
        super().__init__()
        # hidden_dim é recebido para compatibilidade de argumentos, mas não usado na média
        self.ln = nn.LayerNorm(hidden_dim, eps=1e-6)
        
    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # tokens: [B, Seq, Dim]
        if mask is None:
            pooled = tokens.mean(dim=1)
        else:
            # Média ponderada pela máscara (ignorando padding)
            mask = mask.unsqueeze(-1).float() # [B, Seq, 1]
            tokens = tokens * mask
            sum_tokens = tokens.sum(dim=1)
            sum_mask = mask.sum(dim=1).clamp(min=1e-9)
            pooled = sum_tokens / sum_mask
            
        return self.ln(pooled)

# --- REGISTRO E BUILDER ---
POOLER_REGISTRY = {
    "attention": AttentionPooling,
    "mean": MeanPooling,
}

def build_pooler(pooler_type: str, **kwargs):
    if pooler_type not in POOLER_REGISTRY:
        raise ValueError(f"Pooler '{pooler_type}' não encontrado. Opções: {list(POOLER_REGISTRY.keys())}")
    return POOLER_REGISTRY[pooler_type](**kwargs)