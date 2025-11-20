# src/models/heads.py

import torch
import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 512, hidden_dim: int = 2048):
        """
        Cria um Projection Head para mapear embeddings de entrada para um espaço de saída.

        Args:
            input_dim (int): Dimensão dos embeddings de entrada (e.g., model.config.hidden_size).
            output_dim (int): Dimensão desejada dos embeddings de saída para a ContrastiveLoss.
            hidden_dim (int): Dimensão da camada oculta no MLP.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica a projeção.

        Args:
            x (torch.Tensor): Tensor de embeddings de entrada.

        Returns:
            torch.Tensor: Tensor de embeddings projetados.
        """
        return self.fc2(self.relu(self.fc1(x)))