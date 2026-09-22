"""Permutation-equivariant shared-state candidate scorer.

The backbone may encode the clinical state/question once.  Candidate
representations are then passed through the same scorer; no candidate index,
absolute position, or fixed label head is used.  The set context is symmetric
and therefore permuting candidates can only permute the output rows.
"""
from __future__ import annotations

import torch
from torch import nn


class SharedStateCandidateScorer(nn.Module):
    """Score a runtime candidate set with a shared, index-free path.

    Parameters are deliberately small so this module can be attached to a
    frozen/LoRA backbone.  ``state`` is ``[B,H]`` and ``candidates`` is
    ``[B,K,H]``.  The same scorer is applied to every candidate.
    """

    def __init__(self, hidden_size: int, bottleneck: int | None = None):
        super().__init__()
        d = bottleneck or hidden_size
        self.state_proj = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, d), nn.GELU())
        self.candidate_proj = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, d), nn.GELU())
        self.score = nn.Sequential(
            nn.LayerNorm(3 * d), nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1)
        )

    def forward(self, state: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2 or candidates.ndim != 3:
            raise ValueError("expected state [B,H] and candidates [B,K,H]")
        if state.shape[0] != candidates.shape[0] or state.shape[-1] != candidates.shape[-1]:
            raise ValueError("state/candidate batch or hidden dimensions do not match")
        s = self.state_proj(state).unsqueeze(1)
        c = self.candidate_proj(candidates)
        # Symmetric context excludes no candidate: it is invariant to set order.
        set_context = c.mean(dim=1, keepdim=True).expand_as(c)
        features = torch.cat([s.expand_as(c), c, set_context], dim=-1)
        return self.score(features).squeeze(-1)


def permutation_error(module: nn.Module, state: torch.Tensor, candidates: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    """Return max equivariance error for a candidate permutation."""
    with torch.no_grad():
        base = module(state, candidates)
        permuted = module(state, candidates[:, permutation])
        return (permuted - base[:, permutation]).abs().max()
