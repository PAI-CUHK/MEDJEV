# Development Guide

## Repository workflow

The project uses a `src/` layout, editable installs, focused tests, and GitHub Actions. Keep local experiment outputs outside source control.

```bash
python -m pip install -e ".[test,dev]"
pytest -q
ruff check src tests
python -m build
```

## Test layers

- API and schema tests validate public query construction.
- Core model tests validate candidate permutation, query isolation, masking, and probabilities.
- Checkpoint regression tests isolate CPU-safe checkpoint behavior from optional GPU dependencies.
- SleepJEV tests validate temporal windows, event indexes, sparse/dense parity, and unlabeled events.

## Style

Use type annotations for public interfaces, keep imports explicit, and add comments only where the implementation is not self-evident. Avoid hidden global state, silent data coercion, and broad exception handling around model or data errors.

## Adding a model route

Document the route in `docs/architecture.md`, add a focused baseline comparison, define checkpoint metadata, and state whether calibration is supported. Do not add a new route only by copying a result folder into the repository.

## Adding a dataset

Add an adapter and a data card. Include the source URL, version, license, access requirements, expected schema, split policy, and whether patient-level disjointness is actually known. Raw data stays outside Git.

## Pull requests

Every pull request should state:

- what public or experimental contract changed;
- which tests were added or run;
- whether calibration, data leakage, or result claims are affected;
- whether new external dependencies or licenses are involved.
