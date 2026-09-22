import torch

from medjev.shared_state import SharedStateCandidateScorer, permutation_error


def test_shared_state_is_permutation_equivariant():
    torch.manual_seed(17)
    model = SharedStateCandidateScorer(16, bottleneck=8)
    state = torch.randn(3, 16)
    candidates = torch.randn(3, 5, 16)
    error = permutation_error(model, state, candidates, torch.tensor([3, 0, 4, 1, 2]))
    assert float(error) < 1e-6


def test_shared_state_supports_dynamic_k():
    torch.manual_seed(23)
    model = SharedStateCandidateScorer(12, bottleneck=6)
    state = torch.randn(2, 12)
    for k in (2, 3, 5, 10):
        scores = model(state, torch.randn(2, k, 12))
        assert scores.shape == (2, k)
