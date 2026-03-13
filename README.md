# EMG2QWERTY — sEMG Keystroke Prediction

**Predicting QWERTY keystrokes from surface electromyography (sEMG) signals using deep learning.**

UCLA ECE C147/C247 Final Project — Winter 2026

---

## Overview

This project decodes typed keystrokes from sEMG signals recorded at both wrists.
We build on the [emg2qwerty dataset and baseline](https://arxiv.org/abs/2410.20081)
from Meta, extending it with new model architectures, training configurations, and
systematic experiments.

Each session is an HDF5 file containing:
- **Left + right wrist EMG** — 16 electrode channels per wrist, sampled at 2 kHz
- **Keystroke ground-truth** — key-press timestamps recorded by a keylogger

---

## End-to-End Pipeline

```
HDF5 session files (308 GB tar.gz from Meta S3)
  → EMGDownloader (stream-filter tar.gz → local + Backblaze B2)
  → WindowedEMGDataset (data.py)
  → LogSpectrogram + SpecAugment (transforms.py)
  → Model encoder (modules.py / lightning.py)
  → CTC Loss (nn.CTCLoss)
  → Greedy / Beam decoder (decoder.py)
  → CER metric (metrics.py)
```

---

## Architectures

| Architecture | Status | Description |
|---|---|---|
| **TDS-CNN** | ✅ Baseline | Time-Depth Separable CNN (Meta baseline) |
| **BiLSTM / CNN+BiLSTM** | ✅ Implemented | Bidirectional recurrent CTC encoders |
| **GRU** | 🔬 In progress | Recurrent sequential encoder |
| **Transformer** | 🔬 In progress | Self-attention over EMG frames |
| **Hybrid** | 🔬 Planned | CNN front-end + Transformer/RNN encoder |

---

## Quick Start

### 1. Install dependencies

```bash
uv sync --group dev
```

### 2. Configure B2 credentials (for data pipeline)

```bash
cp .env.example .env   # fill in B2_KEY_ID and B2_APPLICATION_KEY
```

### 3. Download baseline sessions

Streams the 308 GB tar.gz from Meta S3, extracts the 18 baseline sessions:

```bash
uv run python scripts/download_data.py --baseline
```

### 4. Train the baseline model

```bash
uv run python -m emg2qwerty.train user=single_user trainer.accelerator=gpu
```

### 5. Evaluate with greedy decoding

```bash
uv run python -m emg2qwerty.train \
  user=single_user checkpoint=path/to/best.ckpt \
  train=False trainer.accelerator=cpu decoder=ctc_greedy
```

### 6. Run unit tests

```bash
make test
```

### 7. Serve documentation locally

```bash
make docs-serve
```

---

## Source Layout

```
src/emg2qwerty/
  charset.py        # Key ↔ label ↔ unicode mappings
  data.py           # EMGSessionData + WindowedEMGDataset
  decoder.py        # CTCGreedyDecoder + CTCBeamDecoder (kenlm)
  lightning.py      # LightningDataModule + TDSConvCTCModule
  metrics.py        # CharacterErrorRates (CER / IER / DER / SER)
  modules.py        # TDSConvEncoder + RotationInvariantMLP + SpectrogramNorm
  train.py          # Hydra entry-point (train + eval loop)
  transforms.py     # LogSpectrogram, SpecAugment, RandomBandRotation, …
  utils.py          # Optimizer/scheduler instantiation, checkpoint helpers
  pipeline/
    config.py       # Pydantic models (B2Config, DownloadConfig, …)
    downloader.py   # tar.gz streaming from Meta S3 → local + B2
    registry.py     # FileRegistry (dedup JSON manifest in B2)
    trainer.py      # BatchTrainer (multi-profile orchestration)
config/             # Hydra YAML configs (model, optimizer, LR scheduler, …)
scripts/            # Dataset utilities, download/train CLIs, LM build
tests/              # pytest tests (charset, decoder, data, pipeline)
docs/               # MkDocs Material documentation
```

---

## Configuration

All training parameters are managed through [Hydra](https://hydra.cc/) YAML
configs under `config/`. Key override examples:

```bash
# Beam decoder with KenLM language model
uv run python -m emg2qwerty.train decoder=ctc_beam

# Multi-user (generic) training
uv run python -m emg2qwerty.train user=generic

# Different LR schedule
uv run python -m emg2qwerty.train lr_scheduler=cosine_annealing

# Larger batch
uv run python -m emg2qwerty.train batch_size=64 num_workers=8
```

---

## Evaluation Metric

**Character Error Rate (CER)** = edit distance between predicted and reference character sequence,
divided by reference length. Also tracked: insertion (IER), deletion (DER), and substitution (SER) rates.

---

## References

- [emg2qwerty: A Large Dataset and Baselines for EMG-to-Text](https://arxiv.org/abs/2410.20081) — Sivakumar et al., Meta, 2024
- [Sequence-to-Sequence Speech Recognition with Time-Depth Separable Convolutions](https://arxiv.org/abs/1904.02619) — Hannun et al., 2019
- [KenLM Language Model Toolkit](https://github.com/kpu/kenlm)

---

## License

See [LICENSE](LICENSE) for details.
