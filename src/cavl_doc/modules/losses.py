# src/cavl_doc/modules/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. CONTRASTIVE LOSS (Pairwise)
# ==========================================
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        individual_losses = self._get_individual_losses(output1, output2, label)
        return torch.mean(individual_losses)

    def forward_individual(self, output1, output2, label):
        return self._get_individual_losses(output1, output2, label)

    def _get_individual_losses(self, output1, output2, label):
        diff = output1 - output2
        euclidean_distance = torch.sqrt(torch.sum(torch.pow(diff, 2), dim=1) + 1e-6)
        label = label.view_as(euclidean_distance)
        loss_positive = (label) * torch.pow(euclidean_distance, 2)
        loss_negative = (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        return loss_positive + loss_negative

# ==========================================
# 2. ARCFACE (Classification) - CORRIGIDO
# ==========================================
class ArcFaceLoss(nn.Module):
    def __init__(self, in_features=512, num_classes=16, s=64.0, m=0.50, **kwargs):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def _get_logits(self, embeddings, labels):
        """Calcula os logits com a margem angular aplicada."""
        embeddings = F.normalize(embeddings, dim=1)
        weight = F.normalize(self.weight, dim=1)
        
        cosine = F.linear(embeddings, weight)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

    def forward(self, embeddings, labels):
        """Retorna a média da loss (escalar) para backpropagation."""
        output = self._get_logits(embeddings, labels)
        return F.cross_entropy(output, labels.long())

    def forward_individual(self, embeddings, labels):
        """Retorna a loss por amostra (vetor) para o RL."""
        output = self._get_logits(embeddings, labels)
        # reduction='none' retorna um vetor [BatchSize]
        return F.cross_entropy(output, labels.long(), reduction='none')

# ==========================================
# 3. COSFACE (Classification) - CORRIGIDO
# ==========================================
class CosFaceLoss(nn.Module):
    def __init__(self, in_features=512, num_classes=16, s=64.0, m=0.35, **kwargs):
        super(CosFaceLoss, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def _get_logits(self, embeddings, labels):
        embeddings = F.normalize(embeddings, dim=1)
        weight = F.normalize(self.weight, dim=1)
        cosine = F.linear(embeddings, weight)
        phi = cosine - self.m
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

    def forward(self, embeddings, labels):
        output = self._get_logits(embeddings, labels)
        return F.cross_entropy(output, labels.long())

    def forward_individual(self, embeddings, labels):
        output = self._get_logits(embeddings, labels)
        return F.cross_entropy(output, labels.long(), reduction='none')

# ==========================================
# REGISTRY
# ==========================================
LOSS_REGISTRY = {
    "contrastive": ContrastiveLoss,
    "arcface": ArcFaceLoss,
    "cosface": CosFaceLoss
}

def build_loss(loss_type: str, **kwargs):
    if loss_type not in LOSS_REGISTRY:
        raise ValueError(f"Loss '{loss_type}' não encontrada.")
    return LOSS_REGISTRY[loss_type](**kwargs)