"""Compact convolutional EEG encoder used for the retrieval baseline (NICE-style:
temporal convolution -> spatial convolution -> pooling -> MLP projection).
Input: (batch, channels, samples). Output: L2-normalised (batch, projection_dim)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEEGBaseline(nn.Module):
    def __init__(self, num_channels=17, num_samples=100, projection_dim=512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), padding=(0, 12)),   # temporal filters
            nn.BatchNorm2d(40), nn.ELU(),
            nn.Conv2d(40, 40, (num_channels, 1)),          # spatial filters across all electrodes
            nn.BatchNorm2d(40), nn.ELU(),
            nn.AvgPool2d((1, 5)), nn.Dropout(0.5), nn.Flatten(),
        )
        self.projection = nn.Sequential(
            nn.Linear(40 * (num_samples // 5), projection_dim), nn.GELU(),
            nn.Dropout(0.5), nn.Linear(projection_dim, projection_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(2.659))  # learnable temperature, ln(1/0.07)

    def forward(self, x):
        return F.normalize(self.projection(self.features(x.unsqueeze(1))), dim=-1)
