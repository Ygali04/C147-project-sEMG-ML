# Setup

## Prerequisites

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- A CUDA-capable GPU (for training; CPU works for evaluation and tests)
- [cmake](https://cmake.org/) + [Boost](https://www.boost.org/) (only if you want beam-search decoding via kenlm)

## Install Dependencies

```bash
# Core + dev tools (ruff, pytest, jupyterlab, pre-commit, etc.)
uv sync --group dev

# Also install docs tooling
uv sync --group dev --group docs
```

## Pre-commit Hooks

```bash
uv run pre-commit install
```

This automatically runs formatting (ruff), linting, and notebook output
stripping before every `git commit`.

## Backblaze B2 Credentials

The data pipeline needs Backblaze B2 credentials to download and store HDF5
session files.

```bash
cp .env.example .env
# Edit .env — fill in B2_KEY_ID and B2_APPLICATION_KEY
```

The `.env` file is git-ignored and never committed.

## Common Commands

| Command | What it does |
|---|---|
| `make format` | Auto-format code with ruff |
| `make lint` | Lint with ruff (no auto-fix) |
| `make test` | Run tests in parallel, skip `@slow` and `@integration` tests |
| `make all` | format → lint → test |
| `make docs-serve` | Live-preview docs at `http://localhost:8000` |
| `make docs-deploy` | Build and publish docs to GitHub Pages |
| `make rclone-config` | Configure rclone for Backblaze B2 |
| `make data-download` | Download baseline data |
| `make clean` | Remove `__pycache__`, `.pytest_cache`, `dist/`, `site/` |

## Training

### Single-User Personalized Model

```bash
# Train on baseline user (89335547) — 18 sessions, ~150 epochs
uv run python -m emg2qwerty.train \
  user=single_user \
  trainer.accelerator=gpu \
  trainer.devices=1
```

### Generic (Multi-User) Model

```bash
uv run python -m emg2qwerty.train \
  user=generic \
  trainer.accelerator=gpu \
  trainer.devices=8 \
  --multirun
```

### Personalized Fine-Tuned Evaluation (Multi-Run)

```bash
# Greedy decoding across all user configs
uv run python -m emg2qwerty.train \
  user="glob(user*)" \
  checkpoint="${HOME}/emg2qwerty/models/personalized-finetuned/\${user}.ckpt" \
  train=False trainer.accelerator=cpu \
  decoder=ctc_greedy \
  hydra.launcher.mem_gb=64 \
  --multirun

# Beam-search decoding with 6-gram character language model
uv run python -m emg2qwerty.train \
  user="glob(user*)" \
  checkpoint="${HOME}/emg2qwerty/models/personalized-finetuned/\${user}.ckpt" \
  train=False trainer.accelerator=cpu \
  decoder=ctc_beam \
  hydra.launcher.mem_gb=64 \
  --multirun
```

Config overrides are passed directly on the CLI via Hydra.
See `config/base.yaml` and the `config/` subdirectories for all options.

## KenLM (Beam-Search Decoder)

The `CTCBeamDecoder` requires [KenLM](https://github.com/kpu/kenlm) for
n-gram language model scoring during beam search. If you only need greedy
decoding (the default), you can skip this section entirely.

### Prerequisites

KenLM requires cmake and Boost to be installed:

```bash
# macOS
brew install cmake boost

# Ubuntu / Debian
sudo apt install cmake libboost-all-dev
```

### Automatic Install (via uv)

KenLM is vendored in `vendor/kenlm/` with Cython bindings regenerated for
Python 3.14+ compatibility. It's declared as a dependency in `pyproject.toml`
and installed automatically by `uv sync`:

```bash
uv sync --group dev
```

kenlm will persist across `uv sync` calls — no manual reinstallation needed.

!!! info "How the vendored kenlm works"
    The upstream kenlm ships a pre-generated `kenlm.cpp` from an older Cython
    that is incompatible with Python 3.14+. The vendored copy in `vendor/kenlm/`
    has this file regenerated with Cython 3.2.4. The `[tool.uv.sources]` section
    in `pyproject.toml` tells `uv` to build from this local copy.

### Manual Install (alternative)

If you prefer not to use the vendored version:

```bash
pip install https://github.com/kpu/kenlm/archive/master.zip
```

!!! tip "Python 3.14+ compatibility"
    If building fails on Python 3.14+ due to Cython compatibility errors, you
    can fix it by regenerating the Cython bindings:

    ```bash
    git clone https://github.com/kpu/kenlm.git /tmp/kenlm
    pip install cython
    cython /tmp/kenlm/python/kenlm.pyx --cplus -o /tmp/kenlm/python/kenlm.cpp
    pip install /tmp/kenlm
    ```

The `MAX_ORDER` environment variable controls the maximum n-gram order:

```bash
MAX_ORDER=20 pip install https://github.com/kpu/kenlm/archive/master.zip
```

### Build the 6-gram Character Language Model

The beam-search decoder uses a 6-gram character-level language model built from
the [WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext) raw
corpus. A pre-built binary is expected at `models/lm/wikitext-103-6gram-charlm.bin`.

To regenerate it:

1. Build kenlm from source: <https://github.com/kpu/kenlm#compiling>
2. Run the build script:

```bash
./scripts/lm/build_char_lm.sh 6
```

This downloads WikiText-103, preprocesses it to character-level tokens (lowercase
alphabets only, no cross-word n-grams), and builds the ARPA + binary LM files
under `models/lm/`.

### Beam-Search Decoder Config

The beam-search decoder is configured in `config/decoder/ctc_beam.yaml`:

```yaml
decoder:
  _target_: emg2qwerty.decoder.CTCBeamDecoder
  beam_size: 50
  max_labels_per_timestep: 10
  lm_path: ${hydra:runtime.cwd}/models/lm/wikitext-103-6gram-charlm.bin
  lm_weight: 2.0
  insertion_bonus: 2.0
  delete_key: Key.backspace
```

Switch between decoders at the CLI:

```bash
# Greedy (default, no kenlm needed)
uv run python -m emg2qwerty.train decoder=ctc_greedy ...

# Beam search (requires kenlm + built LM)
uv run python -m emg2qwerty.train decoder=ctc_beam ...
```
