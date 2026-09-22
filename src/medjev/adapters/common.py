from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

@dataclass
class DecisionRecord:
    id: str
    source: str
    state: str
    question: str
    candidates: list[dict[str, str]]
    target: str
    evidence_spans: list[str] = field(default_factory=list)
    phenomena: list[str] = field(default_factory=list)
    source_group: str = ""
    split: str = ""
    original_label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self): return asdict(self)

def write_jsonl(records, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict() if hasattr(r, "to_dict") else r, ensure_ascii=False) + "\n")
