"""Validate the model-independent MEDJEV query contract.

This example intentionally does not load a checkpoint or clinical dataset.
It is safe to run after a clean clone and demonstrates schema validation only.
"""

import sys
from pathlib import Path


# Keep this model-free example runnable directly from a clean source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medjev import Candidate, Query


def main() -> None:
    candidates = (
        Candidate("supported", "The evidence supports the statement."),
        Candidate("contradicted", "The evidence contradicts the statement."),
        Candidate("unresolved", "The evidence does not settle the statement."),
    )
    query = Query(
        id="demo_claim",
        statement="The patient has chest pain.",
        question="What is the relationship between the record and the statement?",
        candidates=candidates,
    )

    print("query_id:", query.id)
    print("question:", query.question)
    print("candidate_ids:", [candidate.id for candidate in query.candidates])
    print("status: query contract validated")


if __name__ == "__main__":
    main()
