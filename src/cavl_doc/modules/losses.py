# src/cavl_doc/modules/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. CONTRASTIVE LOSS (Pairwise)
# ==========================================
class ContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss para pares (Siamese).
    Baseada na distância Euclidiana com margem.
    """
    def __init__(self, margin=1.0, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        individual_losses = self._get_individual_losses(output1, output2, label)
        return torch.mean(individual_losses)

    def forward_individual(self, output1, output2, label):
        """Retorna a loss por amostra (para o RL Professor)."""
        return self._get_individual_losses(output1, output2, label)

    def _get_individual_losses(self, output1, output2, label):
        diff = output1 - output2
        euclidean_distance = torch.sqrt(torch.sum(torch.pow(diff, 2), dim=1) + 1e-6)
        label = label.view_as(euclidean_distance)
        
        # Label 1 = Iguais (minimizar distância)
        loss_positive = (label) * torch.pow(euclidean_distance, 2)
        # Label 0 = Diferentes (maximizar distância até margem)
        loss_negative = (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        
        return loss_positive + loss_negative

# ==========================================
# 2. ARCFACE (Classification)
# ==========================================
class ArcFaceLoss(nn.Module):
    """Additive Angular Margin Loss."""
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
        output = self._get_logits(embeddings, labels)
        return F.cross_entropy(output, labels.long())

    def forward_individual(self, embeddings, labels):
        output = self._get_logits(embeddings, labels)
        return F.cross_entropy(output, labels.long(), reduction='none')

# ==========================================
# 2.1 ELASTIC ARCFACE (Robustness)
# ==========================================
class ElasticArcFaceLoss(nn.Module):
    """
    ElasticFace: Elastic Angular Margin Loss.
    Relaxa a restrição de margem fixa amostrando m de uma distribuição normal N(m, std).
    Melhora a generalização e robustez (útil para Zero-Shot).
    Ref: https://arxiv.org/abs/2109.09416
    """
    def __init__(self, in_features=512, num_classes=16, s=64.0, m=0.50, std=0.05, **kwargs):
        super(ElasticArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.std = std
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def _get_logits(self, embeddings, labels):
        embeddings = F.normalize(embeddings, dim=1)
        weight = F.normalize(self.weight, dim=1)
        cosine = F.linear(embeddings, weight)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        
        # Elastic Margin: Amostra m aleatório para cada exemplo no batch
        # N(m, std)
        m_random = torch.normal(mean=self.m, std=self.std, size=cosine.size(), device=embeddings.device)
        
        cos_m = torch.cos(m_random)
        sin_m = torch.sin(m_random)
        
        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi = cosine * cos_m - sine * sin_m
        
        # Estabilidade numérica (usando parâmetros do m médio para o threshold)
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
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
# 3. COSFACE (Classification)
# ==========================================
class CosFaceLoss(nn.Module):
    """Large Margin Cosine Loss."""
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
# 3.1 ELASTIC COSFACE
# ==========================================
class ElasticCosFaceLoss(nn.Module):
    """
    Elastic CosFace.
    Aplica a aleatoriedade na margem de cosseno m.
    m ~ N(mean_m, std)
    """
    def __init__(self, in_features=512, num_classes=16, s=64.0, m=0.35, std=0.05, **kwargs):
        super(ElasticCosFaceLoss, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.std = std
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def _get_logits(self, embeddings, labels):
        embeddings = F.normalize(embeddings, dim=1)
        weight = F.normalize(self.weight, dim=1)
        cosine = F.linear(embeddings, weight)
        
        # Elastic Margin: Amostra m aleatório
        m_random = torch.normal(mean=self.m, std=self.std, size=cosine.size(), device=embeddings.device)
        
        phi = cosine - m_random
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
# 4. EXP-FACE (Exponential Angular Margin)
# ==========================================
class ExpFaceLoss(nn.Module):
    """
    ExpFace: Exponential Angular Margin Loss.
    Implementa T(theta) = cos((theta/pi)^m * pi).
    Foca em exemplos limpos (centro) e ignora ruidosos.
    Hiperparâmetros típicos: s=64, m=0.7[cite: 373].
    """
    def __init__(self, in_features=512, num_classes=16, s=64.0, m=0.7, **kwargs):
        super(ExpFaceLoss, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def _get_logits(self, embeddings, labels):
        embeddings = F.normalize(embeddings, dim=1)
        weight = F.normalize(self.weight, dim=1)
        
        # 1. Calcula Cosseno e Theta
        cosine = F.linear(embeddings, weight)
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cosine)
        
        # 2. Aplica Margem Exponencial (Eq. 8 do paper)
        # Penaliza mais o centro (ângulos pequenos) e menos a borda
        theta_m = torch.pow(theta / math.pi, self.m) * math.pi
        cosine_m = torch.cos(theta_m)
        
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        output = (one_hot * cosine_m) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

    def forward(self, embeddings, labels):
        output = self._get_logits(embeddings, labels)
        return F.cross_entropy(output, labels.long())

    def forward_individual(self, embeddings, labels):
        output = self._get_logits(embeddings, labels)
        return F.cross_entropy(output, labels.long(), reduction='none')

# ==========================================
# 4.1 ELASTIC EXP-FACE
# ==========================================
class ElasticExpFaceLoss(nn.Module):
    """
    Elastic ExpFace.
    Aplica a aleatoriedade na margem exponencial m.
    m ~ N(mean_m, std)
    """
    def __init__(self, in_features=512, num_classes=16, s=64.0, m=0.7, std=0.05, **kwargs):
        super(ElasticExpFaceLoss, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.std = std
        
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def _get_logits(self, embeddings, labels):
        embeddings = F.normalize(embeddings, dim=1)
        weight = F.normalize(self.weight, dim=1)
        
        # 1. Calcula Cosseno e Theta
        cosine = F.linear(embeddings, weight)
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cosine)
        
        # 2. Elastic Margin: Amostra m aleatório
        m_random = torch.normal(mean=self.m, std=self.std, size=cosine.size(), device=embeddings.device)
        
        # 3. Aplica Margem Exponencial com m aleatório
        theta_m = torch.pow(theta / math.pi, m_random) * math.pi
        cosine_m = torch.cos(theta_m)
        
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        output = (one_hot * cosine_m) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

    def forward(self, embeddings, labels):
        output = self._get_logits(embeddings, labels)
        return F.cross_entropy(output, labels.long())

    def forward_individual(self, embeddings, labels):
        output = self._get_logits(embeddings, labels)
        return F.cross_entropy(output, labels.long(), reduction='none')

# ==========================================
# 5. SUB-CENTER ARCFACE (Handling Variance)
# ==========================================
class SubCenterArcFaceLoss(nn.Module):
    """
    Permite K sub-centros por classe para lidar com variância intra-classe
    (ex: layouts diferentes para o mesmo tipo de documento).
    """
    def __init__(self, in_features=512, num_classes=16, k=3, s=64.0, m=0.50, **kwargs):
        super(SubCenterArcFaceLoss, self).__init__()
        self.num_classes = num_classes
        self.k = k
        self.s = s
        self.m = m
        
        # Pesos expandidos: [num_classes * k, in_features]
        self.weight = nn.Parameter(torch.FloatTensor(num_classes * k, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def _get_logits(self, embeddings, labels):
        embeddings = F.normalize(embeddings, dim=1)
        weight = F.normalize(self.weight, dim=1)
        
        # Cosseno com todos os sub-centros: [Batch, Class*K]
        cosine_all = F.linear(embeddings, weight)
        # Reshape para [Batch, Class, K]
        cosine_all = cosine_all.view(-1, self.num_classes, self.k)
        # Max-pooling: seleciona o melhor sub-centro para cada classe
        cosine_best, _ = torch.max(cosine_all, dim=2)
        
        sine = torch.sqrt((1.0 - torch.pow(cosine_best, 2)).clamp(0, 1))
        phi = cosine_best * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine_best > self.th, phi, cosine_best - self.mm)
        
        one_hot = torch.zeros(cosine_best.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine_best)
        output *= self.s
        return output

    def forward(self, embeddings, labels):
        output = self._get_logits(embeddings, labels)
        return F.cross_entropy(output, labels.long())

    def forward_individual(self, embeddings, labels):
        output = self._get_logits(embeddings, labels)
        return F.cross_entropy(output, labels.long(), reduction='none')

# ==========================================
# 6. CIRCLE LOSS (Reciprocity)
# ==========================================
class CircleLoss(nn.Module):
    """
    Circle Loss: Otimiza a similaridade de pares com ponderação dinâmica.
    Foca em Hard Negatives e relaxa em pares fáceis.
    """
    def __init__(self, in_features=512, num_classes=16, m=0.25, gamma=256, **kwargs):
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, dim=1)
        weight = F.normalize(self.weight, dim=1)
        
        # Similaridade de Cosseno [Batch, Num_Classes]
        sim = F.linear(embeddings, weight)
        
        # Máscaras para Positivos (sp) e Negativos (sn)
        one_hot = torch.zeros_like(sim)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        sp = sim[one_hot == 1]
        sn = sim[one_hot == 0]

        # Pesos Dinâmicos (alpha)
        ap = torch.clamp(1 + self.m - sp.detach(), min=0.0)
        an = torch.clamp(sn.detach() + self.m, min=0.0)
        
        delta_p = 1 - self.m
        delta_n = self.m
        
        logit_p = - self.gamma * ap * (sp - delta_p)
        logit_n = self.gamma * an * (sn - delta_n)
        
        # Loss = softplus(logsumexp(neg) + logsumexp(pos))
        loss = F.softplus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
        return loss

    def forward_individual(self, embeddings, labels):
        # Retorna a loss individual aproximada (Cross Entropy do cosseno reponderado) para o RL
        # Nota: Circle Loss original reduz o batch inteiro, aqui aproximamos a dificuldade.
        embeddings = F.normalize(embeddings, dim=1)
        weight = F.normalize(self.weight, dim=1)
        sim = F.linear(embeddings, weight)
        one_hot = torch.zeros_like(sim)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        # Dificuldade = 1 - similaridade com a classe correta
        sp = (sim * one_hot).sum(dim=1)
        return 1.0 - sp

# ==========================================
# 6.1 ELASTIC CIRCLE LOSS
# ==========================================
class ElasticCircleLoss(nn.Module):
    """
    Elastic Circle Loss.
    Aplica a aleatoriedade na margem m da Circle Loss para melhorar a generalização.
    m ~ N(mean_m, std)
    """
    def __init__(self, in_features=512, num_classes=16, m=0.25, gamma=256, std=0.0125, **kwargs):
        super(ElasticCircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.std = std
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, dim=1)
        weight = F.normalize(self.weight, dim=1)
        
        # Similaridade de Cosseno [Batch, Num_Classes]
        sim = F.linear(embeddings, weight)
        
        # Máscaras
        one_hot = torch.zeros_like(sim)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        sp = sim[one_hot == 1]
        sn = sim[one_hot == 0]

        # Elastic Margin: Amostra m para cada elemento
        # Para positivos
        m_p = torch.normal(mean=self.m, std=self.std, size=sp.size(), device=embeddings.device)
        # Para negativos
        m_n = torch.normal(mean=self.m, std=self.std, size=sn.size(), device=embeddings.device)

        # Pesos Dinâmicos (alpha) com margem elástica
        # ap = clamp(1 + m - sp)
        ap = torch.clamp(1 + m_p - sp.detach(), min=0.0)
        # an = clamp(sn + m)
        an = torch.clamp(sn.detach() + m_n, min=0.0)
        
        delta_p = 1 - m_p
        delta_n = m_n
        
        logit_p = - self.gamma * ap * (sp - delta_p)
        logit_n = self.gamma * an * (sn - delta_n)
        
        loss = F.softplus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
        return loss

    def forward_individual(self, embeddings, labels):
        # Mesma aproximação da CircleLoss padrão
        embeddings = F.normalize(embeddings, dim=1)
        weight = F.normalize(self.weight, dim=1)
        sim = F.linear(embeddings, weight)
        one_hot = torch.zeros_like(sim)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        sp = (sim * one_hot).sum(dim=1)
        return 1.0 - sp

# ==========================================
# REGISTRY & BUILDER
# ==========================================
LOSS_REGISTRY = {
    "contrastive": ContrastiveLoss,
    "arcface": ArcFaceLoss,
    "elastic_arcface": ElasticArcFaceLoss,
    "cosface": CosFaceLoss,
    "elastic_cosface": ElasticCosFaceLoss,
    "expface": ExpFaceLoss,
    "elastic_expface": ElasticExpFaceLoss,
    "subcenter_arcface": SubCenterArcFaceLoss,
    "circle": CircleLoss,
    "elastic_circle": ElasticCircleLoss
}

def build_loss(loss_type: str, **kwargs):
    """
    Fábrica de Losses.
    """
    if loss_type not in LOSS_REGISTRY:
        raise ValueError(f"Loss '{loss_type}' não encontrada. Opções: {list(LOSS_REGISTRY.keys())}")
    
    return LOSS_REGISTRY[loss_type](**kwargs)