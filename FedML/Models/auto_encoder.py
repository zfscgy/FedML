import torch
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, in_dim: int,  hidden_dim: int):
        super(AutoEncoder, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        return torch.sigmoid(self.decoder(torch.sigmoid(self.encoder(x))))

    def encode(self, x):
        return torch.sigmoid(self.encoder(x))
