# Setup

## Prerequisites

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) package manager

## Install dependencies

```bash
# Core + dev dependencies
uv sync --group dev

# If you also want docs tooling
uv sync --group dev --group docs
```

## Pre-commit hooks

```bash
uv run pre-commit install
```

This will run formatting (ruff), linting, and notebook output stripping on
every commit.

## Common commands

| Command | What it does |
|---|---|
| `make format` | Auto-format code with ruff |
| `make lint` | Lint with ruff |
| `make test` | Run tests (parallel, skips slow tests) |
| `make all` | Format → lint → test |
| `make docs-serve` | Live-preview docs at `localhost:8000` |
| `make clean` | Remove caches and build artifacts |
