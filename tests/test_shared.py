from medjev.shared import common_prefix


def test_branch_prefix_keeps_a_nonempty_suffix():
    assert common_prefix([[1,2,3], [1,2,4]]) == 2
    assert common_prefix([[1,2], [1,2]]) == 1
    assert common_prefix([[1,2], [1,2,3]]) == 1
    assert common_prefix([[1], [2]]) == 0


def test_cached_padded_prefix_matches_full_float32():
    import pytest
    pytest.importorskip("peft")
    import torch
    from torch import nn
    from transformers import Qwen3Config, Qwen3Model
    from medjev.gpu import render
    from medjev.data import DESCRIPTIONS
    from medjev.shared import shared_infer

    class Tokenizer:
        pad_token_id = 0
        def __init__(self):
            self.vocab = {}
        def __call__(self, texts, **kwargs):
            ids = []
            for text in texts:
                seq = []
                for word in text.split():
                    if word not in self.vocab:
                        self.vocab[word] = len(self.vocab)+1
                    seq.append(self.vocab[word])
                ids.append(seq)
            return {"input_ids": ids}

    torch.manual_seed(9)
    model = nn.Module()
    model.mode = "dynamic"
    config = Qwen3Config(vocab_size=256, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=8, max_position_embeddings=256)
    config._attn_implementation = "sdpa"
    model.backbone = Qwen3Model(config).eval()
    model.head = nn.Linear(32, 1)
    tok = Tokenizer()
    rows = [{"premise": "short record", "hypothesis": "a claim"},
            {"premise": "a substantially longer separate record here", "hypothesis": "different claim"},
            {"premise": "short record", "hypothesis": "another claim"}]
    paths = tok([render(row,d) for row in rows for d in DESCRIPTIONS])["input_ids"]
    width = max(map(len, paths))
    tokens = torch.zeros(len(paths), width, dtype=torch.long)
    for i, path in enumerate(paths):
        tokens[i, :len(path)] = torch.tensor(path)
    with torch.inference_mode():
        hidden = model.backbone(input_ids=tokens, attention_mask=tokens.ne(0), use_cache=False).last_hidden_state
        fresh = model.head(hidden[torch.arange(len(paths)), torch.tensor(list(map(len,paths)))-1]).reshape(-1,3)
        shared, stats = shared_infer(model, tok, rows)
    assert torch.allclose(fresh, torch.tensor(shared).float(), atol=2e-5, rtol=2e-5)
    assert stats["computed_prefix_tokens"]+stats["computed_suffix_tokens"] < stats["full_tokens"]
    # Different K and question text must preserve path grouping and original row order.
    binary = ("supported", "not supported or unresolved")
    question = "Does this record support the assertion?"
    paths = tok([render(row,d,question) for row in rows for d in binary])["input_ids"]
    tokens = torch.zeros(len(paths), max(map(len,paths)), dtype=torch.long)
    for i, path in enumerate(paths):
        tokens[i,:len(path)] = torch.tensor(path)
    with torch.inference_mode():
        h = model.backbone(input_ids=tokens, attention_mask=tokens.ne(0), use_cache=False).last_hidden_state
        expected = model.head(h[torch.arange(len(paths)),torch.tensor(list(map(len,paths)))-1]).reshape(-1,2)
        actual,_ = shared_infer(model,tok,rows,descriptions=binary,question=question)
        swapped,_ = shared_infer(model,tok,rows,descriptions=binary[::-1],question=question)
    assert torch.allclose(expected,torch.tensor(actual).float(),atol=2e-5,rtol=2e-5)
    assert torch.allclose(torch.tensor(actual),torch.tensor(swapped[:,::-1].copy()),atol=2e-5,rtol=2e-5)
