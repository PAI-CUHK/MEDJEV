"""Exact-token shared-prefix inference for the trained dynamic candidate scorer.

No weight changes. Prefixes for different evidence groups are batched; suffix
positions preserve their original unpadded offsets. This is an experimental
inference backend, not a claim of speedup for every input length.
"""
from collections import defaultdict
import numpy as np
import torch
from .data import DESCRIPTIONS, QUESTION


def common_prefix(paths):
    n = 0
    for parts in zip(*paths):
        if len(set(parts)) != 1:
            break
        n += 1
    return min(n, min(map(len, paths))-1)


@torch.inference_mode()
def shared_infer(model, tokenizer, rows, groups_per_batch=16, max_length=768,
                 descriptions=None, question=QUESTION):
    from .gpu import render
    if model.mode != "dynamic":
        raise ValueError("Only dynamic candidate-path checkpoints are supported")
    model.eval()
    descriptions = DESCRIPTIONS if descriptions is None else descriptions
    if len(descriptions) < 2 or len(set(descriptions)) != len(descriptions):
        raise ValueError("Candidates must have at least two distinct meanings")
    k_candidates = len(descriptions)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["premise"]].append((i, row))
    groups = list(groups.values())
    result = np.zeros((len(rows), k_candidates))
    stats = {"full_tokens": 0, "computed_prefix_tokens": 0, "computed_suffix_tokens": 0,
             "groups": len(groups), "calls": 0}
    device = next(model.parameters()).device
    for start in range(0, len(groups), groups_per_batch):
        units = []
        for group in groups[start:start+groups_per_batch]:
            texts = [render(row, desc, question) for _, row in group for desc in descriptions]
            paths = tokenizer(texts, add_special_tokens=True)["input_ids"]
            if max(map(len, paths)) > max_length:
                raise ValueError("Complete input exceeds max_length")
            n = common_prefix(paths)
            if n < 1:
                raise ValueError("No shared token prefix")
            units.append((group, paths, n))
        width = max(n for _, _, n in units)
        prefixes = torch.full((len(units), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
        prefix_mask = torch.zeros_like(prefixes)
        mapping, suffixes, positions, destinations = [], [], [], []
        for j, (group, paths, n) in enumerate(units):
            prefixes[j, :n] = torch.tensor(paths[0][:n], device=device)
            prefix_mask[j, :n] = 1
            stats["computed_prefix_tokens"] += n
            stats["full_tokens"] += sum(map(len, paths))
            for k, path in enumerate(paths):
                mapping.append(j)
                suffixes.append(path[n:])
                positions.append(n)
                destinations.append((group[k//k_candidates][0], k % k_candidates))
        prefix = model.backbone(input_ids=prefixes, attention_mask=prefix_mask, use_cache=True)
        cache = prefix.past_key_values
        index = torch.tensor(mapping, dtype=torch.long, device=device)
        cache.batch_select_indices(index)
        max_suffix = max(map(len, suffixes))
        tokens = torch.full((len(suffixes), max_suffix), tokenizer.pad_token_id, dtype=torch.long, device=device)
        suffix_mask = torch.zeros_like(tokens)
        pos = torch.zeros_like(tokens)
        for j, (suffix, offset) in enumerate(zip(suffixes, positions)):
            tokens[j, :len(suffix)] = torch.tensor(suffix, device=device)
            suffix_mask[j, :len(suffix)] = 1
            pos[j, :len(suffix)] = torch.arange(offset, offset+len(suffix), device=device)
        mask = torch.cat([prefix_mask[index], suffix_mask], dim=1)
        hidden = model.backbone(input_ids=tokens, attention_mask=mask, position_ids=pos,
                                past_key_values=cache, use_cache=True).last_hidden_state
        last = suffix_mask.sum(1)-1
        h = hidden[torch.arange(len(suffixes), device=device), last].float()
        # Keep scorer arithmetic aligned with the original GPU autocast path.
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=hidden.dtype == torch.bfloat16):
            scores = model.head(h).float().flatten().cpu().numpy()
        for (row_id, candidate), score in zip(destinations, scores):
            result[row_id, candidate] = score
        stats["computed_suffix_tokens"] += sum(map(len, suffixes))
        stats["calls"] += 2
    return result, stats
