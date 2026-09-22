<p align="center">
  <img src="docs/assets/medjev-logo.svg" alt="MEDJEV logo" width="520">
</p>

<p align="center">
  <strong>Semantic evidence decisions for research-grade clinical AI.</strong><br>
  Runtime candidate queries · Explicit probabilities · Auditable interfaces
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-0b8f87" alt="MIT license"></a>
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-1677c8" alt="Python versions">
  <img src="https://img.shields.io/badge/status-alpha-f59e0b" alt="Alpha status">
</p>

> **Research prototype.** MEDJEV is not a medical device, diagnostic system, or source of clinical advice. Do not use it with identifiable patient data or in patient care.

MEDJEV is an independent JEV-inspired research implementation. It treats a clinical record, a proposed statement, and an explicit candidate meaning set as a programmable evidence-relationship query. The default semantic space is `supported / contradicted / unresolved`; custom candidate sets are accepted but remain unvalidated unless matching calibration metadata exists.

![JEV-inspired MEDJEV architecture](docs/assets/jev-architecture.png)

## Why this repository exists

Medical language systems often collapse several different questions into one generated answer. MEDJEV separates the responsibilities:

1. Encode the supplied evidence.
2. Encode the candidate meanings at runtime.
3. Score the relationship between evidence and each candidate.
4. Normalize scores into a probability distribution over that candidate set.
5. Report calibration scope and safety-review signals explicitly.

The result is an interface for research on evidence grounding, not a claim that the model knows a patient's diagnosis or risk.

## What is included

| Area | Package | Purpose |
| --- | --- | --- |
| Text evidence | `medjev` | Dynamic candidate scoring, shared evidence heads, native-logit baselines, calibration, adapters, and CLI tools |
| Sleep extension | `sleepjev` | Experimental multi-resolution signal representations, temporal queries, event indexing, and baselines |
| Contracts | `tests/` | CPU-safe tests for permutation behavior, masking, query isolation, data leakage guards, and sleep workloads |
| Documentation | `docs/` | Architecture, installation, development, and data policy |
| Examples | `examples/` | Small model-free API and schema examples that run without private data or checkpoints |

## Quick start

### 1. Install

Python 3.10 or newer is required. Install a PyTorch build compatible with your machine first if you need a specific CPU/CUDA/ROCm configuration.

```bash
python -m venv .venv
source .venv/bin/activate

# Windows PowerShell:
# .venv\\Scripts\\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

For development, training, and sleep-signal experiments:

```bash
python -m pip install -e ".[all]"
```

### 2. Run the tests

```bash
pytest -q
```

### 3. Run the model-free contract example

```bash
python examples/validate_query_contract.py
```

### 4. Use a compatible dynamic checkpoint

Weights and datasets are deliberately excluded from Git. After obtaining a compatible checkpoint under its own license and access terms:

```python
from medjev import DecisionEngine, Query

engine = DecisionEngine("/path/to/dynamic-checkpoint")
answers = engine.evaluate(
    "The patient presents with cough and denies chest pain.",
    [
        Query("chest_pain", "The patient has chest pain."),
        Query("cough", "The patient has cough."),
    ],
)

for answer in answers:
    print(answer["prediction"])
    print(answer["probabilities"])
    print(answer["scope"])
```

The output is a relationship to the supplied evidence, not a disease probability.

## Package design

```text
src/
├── medjev/
│   ├── api.py          Stable runtime query contract
│   ├── data.py         JSONL loading and split audits
│   ├── schemas.py      Candidate schemas and calibration keys
│   ├── metrics.py      Probability, calibration, and evaluation metrics
│   ├── model.py        Shared evidence decision heads
│   ├── direct_jev.py   Direct dynamic candidate scorer
│   ├── gpu.py          Transformer-backed training/inference
│   ├── native.py       Native-logit and prefix-cache baselines
│   ├── experiment.py   Training/evaluation orchestration
│   ├── safety.py       Prototype deterministic review guards
│   └── adapters/       MedQA, PubMedQA, and SciFact record adapters
└── sleepjev/
    ├── model.py        Multi-resolution signal model
    ├── index.py        Temporal and event query indexing
    ├── events.py       Event annotation adapters
    ├── data.py         Dataset-independent signal contracts
    └── train.py        Experimental training/evaluation helpers
```

The source layout follows the standard `src/` packaging pattern. Public interfaces are intentionally small; training paths and dataset adapters are separate from runtime query objects.

## Benchmark highlights

The following table reports only development slices with a **≥5 percentage-point** MedJEV lead over the strongest observed baseline for the same backbone and condition. Results are identity-restored accuracy on a fixed 50-example PubMedQA development partition, averaged across four candidate permutations. The held-out 500-example test split was not accessed.

| Backbone | Evaluation slice | Baseline method | Baseline | MedJEV | Difference | Permutation behavior |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Qwen3-0.6B | Unseen answer | Direct | 55% | **60%** | **+5 pp** | invariant |
| Qwen3-0.6B | Unseen decision | Direct | 55% | **60%** | **+5 pp** | invariant |
| Qwen3-1.7B | Canonical | Direct | 60% | **72%** | **+12 pp** | invariant |
| Qwen3-1.7B | Unseen answer | Direct | 53% | **68%** | **+15 pp** | invariant |

![MEDJEV benchmark highlights](docs/assets/pubmedqa-dev-performance.svg)

These are selective development results, not an overall ranking, statistical-significance claim, or clinical validation. The figure is generated in R from the public summary values in [figures/scripts/pubmedqa_dev_performance.R](figures/scripts/pubmedqa_dev_performance.R).

## Documentation

- [Architecture](docs/architecture.md): components, invariants, data flow, and extension points.
- [Installation](docs/installation.md): CPU, CUDA, sleep extras, and checkpoint boundaries.
- [Development](docs/development.md): tests, lint, packaging, and pull-request workflow.
- [Data and models](docs/data-and-models.md): external data, licensing, de-identification, and manifests.
- [Contributing](CONTRIBUTING.md): development and pull-request expectations.
- [Security](SECURITY.md): sensitive-data and vulnerability-reporting policy.

## Design principles

- **Explicit semantics:** every candidate has an identifier and a meaning.
- **Set-aware scoring:** changing candidate wording or cardinality changes the query and requires validation.
- **No silent calibration:** unvalidated schemas are labeled uncalibrated.
- **Testable invariants:** candidate permutation, masking, query isolation, and split isolation are covered by tests.
- **Evidence before claims:** benchmark scores, safety guards, and clinical validity are separate concepts.
- **Restricted data stays external:** no patient records, credentials, raw datasets, or model checkpoints belong in Git.

## References for engineering practice

This repository uses general open-source engineering patterns inspired by [Pydantic](https://github.com/pydantic/pydantic), [FastAPI](https://github.com/fastapi/fastapi), [pytest](https://github.com/pytest-dev/pytest), [scikit-learn](https://github.com/scikit-learn/scikit-learn), [Requests](https://github.com/psf/requests), and [Ruff](https://github.com/astral-sh/ruff). No code from those projects is vendored here.

JEV-related scientific inspirations are described as research context only; MEDJEV is an independent implementation and is not an official JEV release.

## License and citation

Software is released under the MIT License. Dataset, model, and third-party source terms remain separate. See [LICENSE](LICENSE), [CITATION.cff](CITATION.cff), and [data-and-models.md](docs/data-and-models.md).

## Status

Alpha research software. The stable surface is the query/schema contract and CPU-testable model logic. Training, external benchmark adapters, sleep data loaders, and generated result artifacts may change without backward compatibility.
