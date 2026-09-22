"""Runtime semantic-query interface for the trained candidate-path engine."""
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
from .data import LABELS, DESCRIPTIONS, QUESTION
from .metrics import probabilities
from .schemas import schema_key


@dataclass(frozen=True)
class Candidate:
    id: str
    meaning: str


DEFAULT_CANDIDATES = tuple(Candidate(k, v) for k, v in zip(LABELS, DESCRIPTIONS))


@dataclass(frozen=True)
class Query:
    id: str
    statement: str
    question: str = QUESTION
    candidates: tuple[Candidate, ...] = DEFAULT_CANDIDATES

    @classmethod
    def from_schema(cls, id: str, statement: str, schema_id: str):
        from .schemas import BY_ID
        schema = BY_ID[schema_id]
        return cls(id, statement, schema.question,
                   tuple(Candidate(f"option_{i}", meaning) for i, meaning in enumerate(schema.candidates)))

    def __post_init__(self):
        if not self.id.strip() or not self.statement.strip() or not self.question.strip():
            raise ValueError("Query id, statement and question must be nonempty")
        if len(self.candidates) < 2:
            raise ValueError("At least two exclusive candidate meanings are required")
        ids = [c.id for c in self.candidates]
        meanings = [c.meaning for c in self.candidates]
        if len(set(ids)) != len(ids) or len(set(meanings)) != len(meanings):
            raise ValueError("Candidate ids and meanings must be unique")
        if any(not s.strip() for s in ids + meanings):
            raise ValueError("Empty candidates are not allowed")


class DecisionEngine:
    def __init__(self, checkpoint, batch_size=32, backend="fresh"):
        from .gpu import load_model
        if backend not in {"fresh", "shared"}:
            raise ValueError("backend must be fresh or shared")
        self.backend = backend
        self.model, self.tokenizer, self.metadata = load_model(checkpoint)
        if self.model.mode != "dynamic":
            raise ValueError("Runtime candidate queries require a dynamic checkpoint")
        self.args = SimpleNamespace(batch_size=batch_size, max_length=self.metadata["max_length"])

    def evaluate(self, evidence: str, queries: list[Query]):
        from .gpu import infer
        if not evidence.strip():
            raise ValueError("Evidence must be nonempty")
        if len({q.id for q in queries}) != len(queries):
            raise ValueError("Query ids must be unique")
        # Batch compatible schemas; one scalar scorer serves all candidate meanings.
        groups = {}
        for q in queries:
            groups.setdefault((q.question, q.candidates), []).append(q)
        results = {}
        for (question, candidates), items in groups.items():
            canonical = question == QUESTION and set(c.meaning for c in candidates) == set(DESCRIPTIONS)
            key = schema_key(question, [c.meaning for c in candidates])
            schema_calibrated = key in self.metadata.get("schema_temperatures", {})
            temperature = self.metadata.get("schema_temperatures", {}).get(key,
                self.metadata["temperature"] if canonical else 1.)
            rows = [{"premise": evidence, "hypothesis": q.statement} for q in items]
            shared = self.backend == "shared"
            if shared:
                from .shared import shared_infer
                scores, _ = shared_infer(self.model, self.tokenizer, rows,
                    max_length=self.args.max_length,
                    descriptions=[c.meaning for c in candidates], question=question)
            else:
                scores = infer(self.model, self.tokenizer, rows, self.args,
                               [c.meaning for c in candidates], question)
            probs = probabilities(scores, temperature)
            for q, p in zip(items, probs):
                results[q.id] = {"id": q.id, "statement": q.statement,
                    "probabilities": {c.id: float(v) for c, v in zip(candidates, p)},
                    "prediction": candidates[int(np.argmax(p))].id,
                    "calibration": ("fresh-checkpoint temperature transferred; shared execution calibration not revalidated" if shared and (canonical or schema_calibrated) else
                                    "MedNLI held-out development temperature for this exact schema" if canonical or schema_calibrated else "unvalidated schema; uncalibrated"),
                    "execution_backend": "experimental_shared" if shared else "fresh",
                    "scope": "evidence relationship, not disease probability"}
        return [results[q.id] for q in queries]
