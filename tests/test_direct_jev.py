import torch
from medjev.direct_jev import DirectJEV

def test_direct_jev_dynamic_k_and_equivariance():
    torch.manual_seed(17); m=DirectJEV(20, bottleneck=8).eval()
    s=torch.randn(2,20); c=torch.randn(2,5,20); p=torch.tensor([3,0,4,1,2])
    a=m(s,c); b=m(s,c[:,p])
    assert a.shape==(2,5) and torch.allclose(b,a[:,p],atol=1e-6)
    assert torch.allclose(m.probabilities(a).sum(-1),torch.ones(2))

def test_direct_jev_supports_runtime_candidate_count():
    m=DirectJEV(12,bottleneck=6).eval(); s=torch.randn(1,12)
    for k in (2,3,10): assert m(s,torch.randn(1,k,12)).shape==(1,k)
