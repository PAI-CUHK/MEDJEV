import numpy as np
import pytest
import torch
from medjev.model import DecisionHead
from medjev.metrics import probabilities, metrics, fit_temperature
from medjev.data import dev_partition


def test_candidate_permutation_and_query_isolation():
    torch.manual_seed(2)
    model = DecisionHead(12, 8).eval()
    e = torch.randn(4, 7, 12)
    mask = torch.ones(4, 7, dtype=torch.bool)
    c = torch.randn(4, 3, 12)
    full = model(e, mask, c)
    assert torch.allclose(full[:1], model(e[:1], mask[:1], c[:1]), atol=1e-6)
    perm = [2, 0, 1]
    assert torch.allclose(full[:, perm], model(e, mask, c[:, perm]), atol=1e-6)


def test_padding_invariance():
    model = DecisionHead(12, 8).eval()
    e = torch.randn(1, 3, 12)
    c = torch.randn(1, 3, 12)
    a = model(e, torch.ones(1, 3, dtype=torch.bool), c)
    b = model(torch.cat([e, torch.randn(1, 4, 12)], 1),
              torch.tensor([[1,1,1,0,0,0,0]], dtype=torch.bool), c)
    assert torch.allclose(a,b,atol=1e-6)


def test_group_split():
    rows = [{"group": str(i//3)} for i in range(60)]
    a,b = dev_partition(rows)
    assert not ({r["group"] for r in a} & {r["group"] for r in b})
    assert len(a)+len(b)==60


def test_probability_metrics():
    y = np.array([0,1,2])
    perfect = np.eye(3)
    m = metrics(y, perfect)
    assert m["accuracy"]==1 and m["brier"]==0
    p = probabilities(np.ones((3,3)))
    assert np.allclose(p, 1/3)
    assert fit_temperature(np.ones((3,3)), y)>0
    with pytest.raises(ValueError):
        metrics(y, np.ones((3,3)))


def test_development_audit_never_opens_test(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from medjev.data import audit, LABELS
    for split in ('train','dev'):
        records = [dict(pairID=f'{split}{i}',sentence1=f'{split} evidence {i}',sentence2='claim',gold_label=LABELS[0]) for i in range(6)]
        (tmp_path/f'mli_{split}_v1.jsonl').write_text('\n'.join(map(json.dumps,records)),encoding='utf-8')
    (tmp_path/'mli_test_v1.jsonl').write_text('DO NOT OPEN',encoding='utf-8')
    text, binary = Path.read_text, Path.read_bytes
    def checked_text(path,*a,**kw):
        assert path.name != 'mli_test_v1.jsonl'
        return text(path,*a,**kw)
    def checked_binary(path,*a,**kw):
        assert path.name != 'mli_test_v1.jsonl'
        return binary(path,*a,**kw)
    monkeypatch.setattr(Path,'read_text',checked_text)
    monkeypatch.setattr(Path,'read_bytes',checked_binary)
    result = audit(tmp_path,include_test=False)
    assert set(result['splits']) == {'train','dev'}
    assert set(result['overlap']) == {'train:dev'}
