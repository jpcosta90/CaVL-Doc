import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Implementa a função de perda contrastiva (Supervised Contrastive Loss para pares).
    Baseada na distância Euclidiana com margem.
    """
    def __init__(self, margin=1.0, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        """
        Calcula a perda contrastiva MÉDIA para o batch.
        Retorna: um único tensor escalar.
        """
        individual_losses = self._get_individual_losses(output1, output2, label)
        return torch.mean(individual_losses)

    def forward_individual(self, output1, output2, label):
        """
        Calcula e retorna a perda contrastiva INDIVIDUAL para cada par.
        Usado pelo 'Professor' de RL para obter o 'Estado' (State).
        Retorna: um tensor de shape [BatchSize]
        """
        return self._get_individual_losses(output1, output2, label)

    def _get_individual_losses(self, output1, output2, label):
        """
        Função helper interna para calcular as perdas individuais.
        """
        diff = output1 - output2
        # Adicionar epsilon para estabilidade numérica
        euclidean_distance = torch.sqrt(torch.sum(torch.pow(diff, 2), dim=1) + 1e-6)

        label = label.view_as(euclidean_distance)

        # Cálculo da perda individual (o que estava dentro do torch.mean)
        # Label 1 = Pares iguais (atrair)
        loss_positive = (label) * torch.pow(euclidean_distance, 2)
        
        # Label 0 = Pares diferentes (repelir até a margem)
        loss_negative = (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        
        individual_losses = loss_positive + loss_negative
        
        return individual_losses

# --- MENU DE OPÇÕES (REGISTRO) ---
LOSS_REGISTRY = {
    "contrastive": ContrastiveLoss,
    # Futuramente:
    # "arcface": ArcFaceLoss,
    # "triplet": TripletLoss,
    # "cosface": CosFaceLoss
}

def build_loss(loss_type: str, **kwargs):
    """
    Fábrica de Losses.
    Args:
        loss_type (str): Nome da loss (ex: 'contrastive')
        **kwargs: Argumentos específicos da loss (ex: margin=1.0)
    """
    if loss_type not in LOSS_REGISTRY:
        raise ValueError(f"Loss '{loss_type}' não encontrada. Opções disponíveis: {list(LOSS_REGISTRY.keys())}")
    
    return LOSS_REGISTRY[loss_type](**kwargs)