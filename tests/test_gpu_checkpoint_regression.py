"""CPU regression tests extracted from the GPU module without optional HF dependencies."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn

SOURCE = Path(__file__).resolve().parents[1] / 'src' / 'medjev' / 'gpu.py'

def namespace():
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    names = {'ClinicalDecisionModel', 'save_model', 'load_model'}
    selected = ast.Module(body=[n for n in tree.body if getattr(n, 'name', '') in names], type_ignores=[])
    ns = dict(torch=torch, nn=nn, Path=Path, json=json)
    ns['write_json'] = lambda p, d: Path(p).write_text(json.dumps(d))
    exec(compile(selected, str(SOURCE), 'exec'), ns)
    return ns

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = nn.Embedding(10, 8)
    def forward(self, input_ids, attention_mask, use_cache=False):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))
    def save_pretrained(self, path):
        Path(path).mkdir(exist_ok=True)
        torch.save(self.state_dict(), Path(path)/'dummy.pt')

class Tokenizer:
    def save_pretrained(self, path):
        Path(path).mkdir(exist_ok=True)

def test_cross_checkpoint_roundtrip(tmp_path):
    ns = namespace()
    model = ns['ClinicalDecisionModel'](Backbone(), 'dynamic', 5, 'cross_attention').eval()
    batch = {'input_ids': torch.arange(10).reshape(5,2), 'attention_mask': torch.ones(5,2,dtype=torch.long)}
    expected = model(batch, 5).detach()
    meta = dict(model='dummy', revision='dummy', mode='dynamic', n_labels=5, interaction='cross_attention')
    ns['save_model'](model, Tokenizer(), tmp_path, meta)
    ns['AutoModel'] = SimpleNamespace(from_pretrained=lambda *a, **kw: Backbone())
    def restore(base, path):
        base.load_state_dict(torch.load(Path(path)/'dummy.pt', weights_only=True))
        return base
    ns['PeftModel'] = SimpleNamespace(from_pretrained=restore)
    ns['AutoTokenizer'] = SimpleNamespace(from_pretrained=lambda p: Tokenizer())
    restored, _, _ = ns['load_model'](tmp_path, device='cpu')
    torch.testing.assert_close(restored(batch,5), expected)
    (tmp_path/'cross.pt').unlink()
    with pytest.raises(ValueError, match='missing cross.pt'):
        ns['load_model'](tmp_path, device='cpu')

def test_optimizer_covers_set_projection_and_updates_it():
    ns = namespace()
    model = ns['ClinicalDecisionModel'](Backbone(), 'dynamic', 5, 'set_attention')
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    train = next(n for n in tree.body if getattr(n,'name','') == 'train')
    assignments = [n for n in train.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('body_params','head_params') for t in n.targets)]
    scope = {'model':model}
    exec(compile(ast.Module(body=assignments,type_ignores=[]),str(SOURCE),'exec'),scope)
    params = scope['body_params'] + scope['head_params']
    assert {id(p) for p in params} == {id(p) for p in model.parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(params,lr=.01)
    before = model.set_proj[1].weight.detach().clone()
    batch = {'input_ids':torch.arange(10).reshape(5,2), 'attention_mask':torch.ones(5,2,dtype=torch.long)}
    nn.functional.cross_entropy(model(batch,5),torch.tensor([2])).backward()
    optimizer.step()
    assert not torch.equal(before,model.set_proj[1].weight)

def test_candidate_template_alignment_and_legacy_compatibility():
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    fn = next(n for n in tree.body if getattr(n,'name','') == 'candidate_text')
    ns = {}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(SOURCE),'exec'),ns)
    render = ns['candidate_text']
    for index, meaning in enumerate(['no disease', 'Candidate meaning: literal text', 'dose 5 mg']):
        training = f'Candidate ID: {index}\n{meaning}'
        assert render(meaning,index,'aligned_v2') == render(training,index,'aligned_v2')
        assert render(meaning,index,'legacy') == f'Candidate ID: {index}\nCandidate meaning: {meaning}'
        assert render(training,index,'legacy') == training
    # Identity travels with the description during permutations.
    assert render('Candidate ID: 2\ndose 5 mg',0,'aligned_v2') == 'Candidate ID: 2\ndose 5 mg'
