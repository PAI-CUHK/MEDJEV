# Installation

## Requirements

- Python 3.10, 3.11, or 3.12.
- A PyTorch wheel appropriate for the target CPU, CUDA, or ROCm environment.
- Access to any external checkpoint and dataset used by a training or inference command.

## Development installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test,dev]"
```

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test,dev]"
```

## Optional extras

```bash
python -m pip install -e ".[train]"  # PEFT and Accelerate
python -m pip install -e ".[sleep]"  # EDF and tabular signal utilities
python -m pip install -e ".[all]"    # all project extras
```

## Verify installation

```bash
python examples/validate_query_contract.py
pytest -q
```

## Checkpoint boundary

The repository does not download or distribute model weights automatically. A checkpoint directory must contain the metadata expected by its loading route. Record the base model, immutable revision, tokenizer revision, schema, seed, calibration split, and software environment in a separate non-sensitive manifest.

## GPU use

The GPU route is hardware-dependent. Install a PyTorch build that matches the machine before installing the project. Do not assume the local development environment and a remote training host are interchangeable. Start with the CPU test suite, then run a small synthetic or development-only smoke test.
