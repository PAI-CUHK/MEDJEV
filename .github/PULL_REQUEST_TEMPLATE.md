## Summary

Describe the change and the public or experimental contract it affects.

## Validation

- [ ] `pytest -q`
- [ ] `ruff check src tests` (when Python code changes)
- [ ] `python -m build` (when packaging changes)

## Research and safety review

- [ ] No patient data, credentials, checkpoints, or private artifacts are included.
- [ ] Benchmark claims remain separate from clinical claims.
- [ ] Dataset/model licenses and revisions are documented when relevant.
- [ ] Calibration, candidate semantics, and data leakage implications are described when relevant.
