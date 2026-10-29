"""Decoupled Mixture-of-Experts head (experts: z only; router: [z || age embedding]).

Copied from the training code (DecoupledMixtureOfExpertsHead / _Expert). The SAP-Loss
(Soft Age Prior) is a training-time term only, so the inference head is identical.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EMBED_DIM = 1152
N_CLASSES = 3
N_EXPERTS = 3
N_BANDS = 3
AGE_DIM = 64
# Logit order used in training: index 0 / 1 / 2
CLASS_NAMES = ("Benign", "Malignant", "Borderline")


class _Expert(nn.Module):
    def __init__(self, embed_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, n_classes)
        )

    def forward(self, z):
        return self.net(z)


class DecoupledMixtureOfExpertsHead(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM, n_classes: int = N_CLASSES, n_experts: int = N_EXPERTS):
        super().__init__()
        self.n_experts = n_experts
        self.age_encoder = nn.Sequential(nn.Linear(1 + N_BANDS, AGE_DIM), nn.GELU())
        self.experts = nn.ModuleList([_Expert(embed_dim, n_classes) for _ in range(n_experts)])
        self.router = nn.Linear(embed_dim + AGE_DIM, n_experts)

    def forward(self, z, age_norm, band_id):
        band_oh = F.one_hot(band_id.long().clamp(0, N_BANDS - 1), num_classes=N_BANDS).float()
        a = self.age_encoder(torch.cat([age_norm.view(z.shape[0], -1), band_oh], dim=-1))
        logits_k = torch.stack([e(z) for e in self.experts], dim=1)
        pi = F.softmax(self.router(torch.cat([z, a], dim=-1)), dim=-1)
        logits = (pi.unsqueeze(-1) * logits_k).sum(dim=1)
        return logits, F.softmax(logits, dim=-1)
