# EMG2QWERTY

**Predicting Keystrokes from Electromyography (EMG) Signals**

UCLA ECE C147/C247 Final Project — Winter 2026

---

## Overview

This project decodes QWERTY keystrokes from surface electromyography (sEMG)
signals recorded at the wrist. We build on the
[emg2qwerty dataset and baseline](https://arxiv.org/abs/2410.20081) from Meta,
extending it with new model architectures and experiments.

Each session is an HDF5 file containing:

- **Left + right wrist EMG** — 16 electrode channels per wrist, 2 kHz sampling rate
- **Keystroke ground-truth** — keylogger-recorded key-press timestamps per character

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

## Loss & Metric

| | |
|---|---|
| **Loss** | CTC (Connectionist Temporal Classification) — handles alignment without frame-level labels |
| **Metric** | CER (Character Error Rate) = edit distance / reference length |

## Architectures

| Architecture | Status | Description |
|---|---|---|
| **TDS-CNN** | ✅ Baseline | Time-Depth Separable CNN (Meta baseline) |
| **RNN / LSTM / GRU** | 🔬 In progress | Recurrent sequential decoder |
| **Transformer** | 🔬 In progress | Self-attention over EMG frames |
| **Hybrid** | 🔬 Planned | CNN front-end + Transformer/RNN encoder |

## Quick Start

```bash
# 1. Install dependencies
uv sync --group dev

# 2. Set up B2 credentials for data pipeline
cp .env.example .env   # edit with your B2_KEY_ID / B2_APPLICATION_KEY

# 3. Download baseline data (streams 308 GB tar.gz, extracts 18 sessions)
uv run python scripts/download_data.py --baseline

# 4. Train the baseline model
uv run python -m emg2qwerty.train user=single_user trainer.accelerator=gpu

# 5. Evaluate with greedy decoding
uv run python -m emg2qwerty.train \
  user=single_user checkpoint=path/to/best.ckpt \
  train=False trainer.accelerator=cpu decoder=ctc_greedy

# 6. Run tests
make test

# 7. Serve docs locally
make docs-serve
```

## Source Layout

```
src/emg2qwerty/
  __init__.py
  charset.py        # Key ↔ label ↔ unicode mappings
  data.py           # EMGSessionData + WindowedEMGDataset
  decoder.py        # CTCGreedyDecoder + CTCBeamDecoder (kenlm)
  lightning.py       # LightningDataModule + TDSConvCTCModule
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
config/              # Hydra YAML configs (model, optimizer, LR scheduler, …)
scripts/             # Dataset utilities, download/train CLIs, LM build
tests/               # pytest tests (charset, decoder, data, pipeline)
notebooks/           # EDA and result visualization
docs/                # MkDocs Material documentation (this site)
```
