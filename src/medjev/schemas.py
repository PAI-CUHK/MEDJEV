"""Label partitions with explicit semantics. They are views of NLI, not new tasks."""
from dataclasses import dataclass
import hashlib
import json
from .data import QUESTION, DESCRIPTIONS


@dataclass(frozen=True)
class Schema:
    id: str
    family: str
    question: str
    candidates: tuple[str, ...]
    mapping: tuple[int, int, int]
    split: str = "train"


SCHEMAS = [
    Schema("nli", "nli", QUESTION, tuple(DESCRIPTIONS), (0,1,2)),
    Schema("nli_reword_a", "nli", "How does the record relate to the assertion?",
           ("The assertion is supported by the record.", "The assertion conflicts with the record.",
            "The record leaves the assertion unresolved."), (0,1,2)),
    Schema("nli_reword_b", "nli", "Choose the evidence status of this claim.",
           ("The claim follows from the documented evidence.", "The documented evidence rules out the claim.",
            "The evidence neither establishes nor rules out the claim."), (0,1,2)),
    Schema("support", "support", "Does the provided record support this statement?",
           ("The statement is supported by the record.",
            "The statement is not supported: it is contradicted or information is insufficient."), (0,1,1)),
    Schema("support_reword", "support", "Is this assertion established by the evidence?",
           ("Yes: the evidence establishes the assertion.",
            "No: the evidence refutes the assertion or leaves it unknown."), (0,1,1)),
    Schema("refute", "refute", "Does the provided record contradict this statement?",
           ("The record contradicts the statement.",
            "The record does not contradict the statement: it supports it or leaves it unresolved."), (1,0,1)),
    Schema("refute_reword", "refute", "Is this assertion refuted by the evidence?",
           ("Yes: the evidence refutes the assertion.",
            "No: the assertion is supported or there is insufficient information to refute it."), (1,0,1)),
    Schema("determined", "determined", "Can the evidence determine whether this statement holds?",
           ("The relationship is determined: the evidence supports or contradicts the statement.",
            "The relationship is undetermined: the evidence neither supports nor contradicts it."), (0,0,1)),
    Schema("determined_reword", "determined", "Does the record settle this assertion?",
           ("The assertion can be affirmed or refuted from the record.",
            "The record leaves the assertion unresolved."), (0,0,1)),
    Schema("nli_unseen", "nli", "Which evidential assessment applies to the proposed conclusion?",
           ("The available information warrants the conclusion.",
            "The available information is incompatible with the conclusion.",
            "The available information cannot settle the conclusion either way."), (0,1,2), "heldout_wording"),
    Schema("support_unseen", "support", "Is there sufficient backing in this record for the proposed claim?",
           ("There is sufficient backing for the claim.",
            "There is no sufficient backing, whether due to opposing evidence or missing information."), (0,1,1), "heldout_wording"),
    Schema("refute_unseen", "refute", "Is the proposed claim incompatible with the available information?",
           ("It is incompatible with the available information.",
            "Incompatibility is not established; the claim may be supported or unresolved."), (1,0,1), "heldout_wording"),
    Schema("determined_unseen", "determined", "Is the evidential relationship of this claim resolved?",
           ("It is resolved, either in favor of the claim or against it.",
            "It remains unresolved in either direction."), (0,0,1), "heldout_wording"),
]
BY_ID = {s.id: s for s in SCHEMAS}
EVALUATION_IDS = ["nli", "support", "refute", "determined",
                  "nli_unseen", "support_unseen", "refute_unseen", "determined_unseen"]


def schema_key(question, descriptions):
    text = json.dumps([question, sorted(descriptions)], ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def draw_schema(rng, curriculum):
    if not curriculum:
        return BY_ID["nli"]
    p = rng.random()
    if p < .5:
        return BY_ID["nli"]
    if p < .7:
        return BY_ID[rng.choice(["nli_reword_a", "nli_reword_b"])]
    return BY_ID[rng.choice(["support", "support_reword", "refute", "refute_reword",
                            "determined", "determined_reword"])]
