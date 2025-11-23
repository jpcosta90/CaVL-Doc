import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    """
    Cabeça padrão do CaVL (SimCLR Style).
    Estrutura: LayerNorm -> Linear -> GELU -> Linear -> L2 Norm
    """
    def __init__(self, input_dim: int = 1536, proj_hidden: int = 4096, proj_out: int = 512, use_norm: bool = True, **kwargs):
        super().__init__()
        self.ln = nn.LayerNorm(input_dim, eps=1e-6)
        self.fc1 = nn.Linear(input_dim, proj_hidden, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(proj_hidden, proj_out, bias=True)
        self.use_norm = use_norm

        # Inicialização específica
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)
        if self.fc1.bias is not None: nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None: nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor):
        x = self.ln(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        if self.use_norm:
            x = F.normalize(x, p=2, dim=-1)
        return x

class MPProjectionHead(nn.Module):
    """
    Cabeça 'Legacy' usada nos experimentos de Mean Pooling.
    Estrutura: Linear -> ReLU -> Linear (Sem normalização)
    """
    def __init__(self, input_dim: int, proj_out: int = 512, proj_hidden: int = 2048, **kwargs):
        # Nota: Renomeei output_dim -> proj_out e hidden_dim -> proj_hidden 
        # para ser compatível com a chamada automática do CaVLModel.
        super().__init__()
        self.fc1 = nn.Linear(input_dim, proj_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(proj_hidden, proj_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))

# --- REGISTRO DE CABEÇAS ---
# Aqui definimos os nomes que você vai usar na config (ex: head_type="mlp" ou "simple")
HEAD_REGISTRY = {
    "mlp": ProjectionHead,         # A padrão robusta
    "simple_mlp": MPProjectionHead # A antiga simples
}

def build_head(head_type: str, **kwargs):
    if head_type not in HEAD_REGISTRY:
        raise ValueError(f"Head '{head_type}' não encontrado. Opções: {list(HEAD_REGISTRY.keys())}")
    
    # Passamos kwargs para que input_dim, proj_out, etc sejam injetados automaticamente
    return HEAD_REGISTRY[head_type](**kwargs)