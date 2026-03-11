# Setup

## Prerequisites

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- A CUDA-capable GPU (for training; CPU works for small tests)

## Install dependencies

```bash
# Core + dev tools (ruff, pytest, jupyterlab, pre-commit, etc.)
uv sync --group dev

# Also install docs tooling
uv sync --group dev --group docs
```

## Pre-commit hooks

```bash
uv run pre-commit install
```

This automatically runs formatting (ruff), linting, and notebook output
stripping before every `git commit`.

## Common commands

| Command | What it does |
|---|---|
| `make format` | Auto-format code with ruff |
| `make lint` | Lint with ruff (no auto-fix) |
| `make test` | Run tests in parallel, skip `@slow` tests |
| `make all` | format → lint → test |
| `make docs-serve` | Live-preview docs at `http://localhost:8000` |
| `make docs-deploy` | Build and publish docs to GitHub Pages |
| `make clean` | Remove `__pycache__`, `.pytest_cache`, `dist/`, `site/` |

## Running training

```bash
# Single-user personalized model (recommended for development)
uv run python -m emg2qwerty.train \
  user=single_user \
  trainer.accelerator=gpu \
  trainer.devices=1

# Generic (multi-user) model
uv run python -m emg2qwerty.train \
  user=generic \
  trainer.accelerator=gpu \
  trainer.devices=8 \
  --multirun
```

Config overrides are passed directly on the CLI via Hydra.
See `config/base.yaml` and the `config/` subdirectories for all options.

## Optional: kenlm (beam-search decoder)

The `CTCBeamDecoder` requires
[KenLM](https://github.com/kpu/kenlm) compiled from source. If you only
need greedy decoding (the default), you can skip this.

```bash
# Build kenlm from source, then:
pip install https://github.com/kpu/kenlm/archive/master.zip
```
