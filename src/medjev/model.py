from __future__ import annotations
import torch
from torch import nn


class DecisionHead(nn.Module):
    """Jevlike-inspired option queries over shared evidence, independently per query."""
    def __init__(self, width, rank=128):
        super().__init__()
        self.width, self.rank = width, rank
        self.q = nn.Linear(width, rank)
        self.k = nn.Linear(width, rank)
        self.v = nn.Linear(width, rank)
        self.norm = nn.LayerNorm(width)
        self.score = nn.Sequential(nn.Linear(rank * 4, rank), nn.GELU(),
                                   nn.Linear(rank, 1))

    def forward(self, evidence, mask, candidates):
        # evidence: B,L,D; candidates: B,K,D; no cross-question attention.
        q = self.q(self.norm(candidates))
        k = self.k(self.norm(evidence))
        v = self.v(self.norm(evidence))
        attn = torch.einsum("bkr,blr->bkl", q, k) / self.rank ** .5
        attn = attn.masked_fill(~mask[:, None, :].bool(), torch.finfo(attn.dtype).min)
        read = torch.einsum("bkl,blr->bkr", attn.softmax(-1), v)
        return self.score(torch.cat([q, read, q * read, torch.abs(q - read)], -1)).squeeze(-1)


class PooledClassifier(nn.Module):
    """Same frozen encoder, fixed-label comparison; not a dynamic JEV model."""
    def __init__(self, width, rank=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width * 4, rank), nn.GELU(), nn.Linear(rank, 3))

    def forward(self, evidence, mask, hypothesis):
        p = (evidence * mask[:, :, None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return self.net(torch.cat([p, hypothesis, p * hypothesis, torch.abs(p - hypothesis)], -1))


class TextEncoder:
    def __init__(self, model_id, revision=None, device="cpu", max_length=256, threads=4):
        from transformers import AutoTokenizer, AutoModel
        torch.set_num_threads(threads)
        self.model_id, self.revision = model_id, revision
        self.max_length, self.device = max_length, device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = AutoModel.from_pretrained(model_id, revision=revision, torch_dtype=dtype).to(device).eval()
        self.model.requires_grad_(False)
        self.width = self.model.config.hidden_size
        self.truncated = 0

    @torch.inference_mode()
    def encode(self, texts, batch_size=32, tokens=False):
        output = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            lengths = [len(self.tokenizer.encode(s, add_special_tokens=True)) for s in chunk]
            self.truncated += sum(n > self.max_length for n in lengths)
            batch = self.tokenizer(chunk, padding=True, truncation=True,
                                   max_length=self.max_length, return_tensors="pt").to(self.device)
            hidden = self.model(**batch).last_hidden_state.float()
            mask = batch.attention_mask
            if tokens:
                output.extend([hidden[i, :int(mask[i].sum())].cpu().half() for i in range(len(chunk))])
            else:
                pooled = (hidden * mask[:, :, None]).sum(1) / mask.sum(1, keepdim=True)
                output.extend(pooled.cpu())
        return output if tokens else torch.stack(output)


def pad_evidence(items, device="cpu"):
    length, width = max(len(x) for x in items), items[0].shape[-1]
    x = torch.zeros(len(items), length, width)
    mask = torch.zeros(len(items), length, dtype=torch.bool)
    for i, item in enumerate(items):
        x[i, :len(item)] = item.float()
        mask[i, :len(item)] = True
    return x.to(device), mask.to(device)
