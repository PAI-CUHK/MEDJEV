from __future__ import annotations
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

LABELS = os.environ.get("MEDJEV_LABELS", "entailment,contradiction,neutral").split(",")
DESCRIPTIONS = os.environ.get(
    "MEDJEV_DESCRIPTIONS",
    "The clinical evidence supports the statement.|The clinical evidence contradicts the statement.|The clinical evidence does not establish or contradict the statement."
).split("|")
QUESTION = os.environ.get("MEDJEV_QUESTION", "What is the relationship between the clinical evidence and the statement?")
if len(LABELS) != len(DESCRIPTIONS) or len(LABELS) < 2:
    raise ValueError("MEDJEV_LABELS and MEDJEV_DESCRIPTIONS must have the same length >= 2")


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def find_data(path: str | Path) -> Path:
    path = Path(path)
    if (path / "mli_train_v1.jsonl").exists():
        return path
    found = list(path.rglob("mli_train_v1.jsonl"))
    if len(found) != 1:
        raise ValueError(f"Expected exactly one MedNLI directory, found {len(found)}")
    return found[0].parent


def load_split(path, split):
    rows = []
    # Split only on JSONL record newlines; str.splitlines() also splits Unicode
    # line separators that may legitimately occur inside medical text strings.
    for line in (find_data(path) / f"mli_{split}_v1.jsonl").read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        item = json.loads(line)
        if item["gold_label"] not in LABELS:
            raise ValueError("Unknown gold label")
        rows.append({
            "id": item["pairID"], "premise": item["sentence1"],
            "hypothesis": item["sentence2"], "label": LABELS.index(item["gold_label"]),
            "group": item.get("source_group", digest(item["sentence1"].strip())),
            **({"candidate_meanings": item["candidate_meanings"]} if "candidate_meanings" in item else {}),
        })
    return rows


def dev_partition(rows, seed=17):
    groups = [r["group"] for r in rows]
    select, calibrate = next(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=seed)
                             .split(np.arange(len(rows)), groups=groups))
    return [rows[i] for i in select], [rows[i] for i in calibrate]


def audit(path, include_test=True):
    path = find_data(path)
    names = ["train", "dev", "test"] if include_test else ["train", "dev"]
    splits = {s: load_split(path, s) for s in names}
    report = {"labels": LABELS, "splits": {}, "overlap": {},
              "group_note": "Exact premise grouping; patient identifiers are not available here, so patient disjointness is not established."}
    for name, rows in splits.items():
        report["splits"][name] = {
            "count": len(rows), "unique_premises": len({r["group"] for r in rows}),
            "label_counts": dict(Counter(LABELS[r["label"]] for r in rows)),
            "sha256": hashlib.sha256((path / f"mli_{name}_v1.jsonl").read_bytes()).hexdigest(),
        }
    for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        if a not in splits or b not in splits:
            continue
        report["overlap"][a + ":" + b] = len({r["group"] for r in splits[a]} & {r["group"] for r in splits[b]})
    if any(report["overlap"].values()):
        raise ValueError("Premise leakage across official splits; resolve before training")
    a, b = dev_partition(splits["dev"])
    report["test_audit"] = "included" if include_test else "not read; requires separate custodian audit"
    report["development_partition"] = {"selection": len(a), "calibration": len(b),
                                       "seed": 17, "group_overlap": len({r["group"] for r in a} & {r["group"] for r in b})}
    return report


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
