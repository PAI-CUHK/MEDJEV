"""Local LLM draft -> sentence queries -> MedJEV review -> LLM revision demo.

Synthetic examples only. Verifier scores are not independent clinical ground truth.
"""
import argparse
import re
from pathlib import Path
from .api import DecisionEngine, Query
from .data import write_json


def split_claims(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="artifacts/demo.json")
    args = parser.parse_args()
    from .native import NativeEngine
    import torch
    torch.set_num_threads(8)
    generator = NativeEngine()
    verifier = DecisionEngine(args.checkpoint)
    cases = [
        {"id": "synthetic_negation", "evidence": "A 54-year-old man presents with cough. He denies chest pain and shortness of breath. His temperature is 37.0 degrees Celsius. Chest radiograph shows no focal infiltrate. No microbiology results are available."},
        {"id": "synthetic_uncertainty", "evidence": "A 70-year-old woman is admitted for evaluation of dizziness. Her hemoglobin is 8.2 g/dL. She has a history of hypertension. The cause of the anemia has not been determined. No endoscopy has been performed."},
        {"id": "controlled_injected_error", "evidence": "A 54-year-old man presents with cough. He denies chest pain and shortness of breath. No microbiology results are available.",
         "injected_draft": "The patient has chest pain. The patient presents with cough."}
    ]
    output = []
    for case in cases:
        evidence = case["evidence"]
        draft = case.get("injected_draft") or generator.generate("Summarize only documented clinical evidence. Do not invent diagnoses or tests.",
                                   evidence + "\nWrite a concise three-sentence clinical summary.", 180)
        claims = split_claims(draft)
        checks = verifier.evaluate(evidence, [Query(f"claim_{i}", s) for i, s in enumerate(claims)])
        flagged = [x for x in checks if x["prediction"] != "entailment"]
        revision = draft
        if flagged:
            import json
            revision = generator.generate("Revise the draft using only the evidence. The automated review may be wrong; check each flagged claim against the record. Preserve documented facts and remove unsupported assertions.",
                f"Evidence:\n{evidence}\nDraft:\n{draft}\nAutomated review:\n{json.dumps(flagged)}\nWrite the revised concise summary.", 200)
        after = verifier.evaluate(evidence, [Query(f"revised_{i}", s) for i, s in enumerate(split_claims(revision))])
        output.append({**case, "draft_source": "manually injected error for branch test" if "injected_draft" in case else "LLM generation",
                       "draft": draft, "draft_checks": checks, "revision": revision,
                       "revision_checks": after, "revision_triggered": bool(flagged)})
    write_json(args.output, {"generator": "Qwen/Qwen3-0.6B", "verifier": args.checkpoint,
        "claim_extraction": "deterministic sentence splitting; sentences may contain multiple claims",
        "limitation": "Workflow demonstration on synthetic cases; no independent evidence of improved factuality or clinical outcomes.",
        "cases": output})
    print(str(Path(args.output).resolve()))


if __name__ == "__main__":
    main()
