"""Direct/non-generative JEV scoring modules.

These modules map state/question and runtime candidate meanings to logits
without generating a label token.  Candidate scoring weights are shared and
the set context is symmetric, so dynamic K and permutation equivariance are
native properties of the interface.
"""
from __future__ import annotations
import torch
from torch import nn


class DirectJEV(nn.Module):
    def __init__(self, hidden_size: int, bottleneck: int = 512, dropout: float = .1):
        super().__init__()
        d=min(hidden_size,bottleneck)
        self.state = nn.Sequential(nn.LayerNorm(hidden_size),nn.Linear(hidden_size,d),nn.GELU())
        self.candidate = nn.Sequential(nn.LayerNorm(hidden_size),nn.Linear(hidden_size,d),nn.GELU())
        self.energy = nn.Sequential(nn.LayerNorm(3*d),nn.Linear(3*d,d),nn.GELU(),nn.Dropout(dropout),nn.Linear(d,1))

    def forward(self, state: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2 or candidates.ndim != 3 or state.shape[0] != candidates.shape[0]:
            raise ValueError('expected state [B,H], candidates [B,K,H]')
        s=self.state(state).unsqueeze(1).expand(-1,candidates.shape[1],-1)
        c=self.candidate(candidates); context=c.mean(1,keepdim=True).expand_as(c)
        return -self.energy(torch.cat([s,c,context],-1)).squeeze(-1)

    @staticmethod
    def probabilities(energies: torch.Tensor) -> torch.Tensor:
        return energies.softmax(-1)
