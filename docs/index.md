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
  → CER / DER / IER / SER metrics (metrics.py)
```

## Loss & Metric

| | |
|---|---|
| **Loss** | CTC (Connectionist Temporal Classification) — handles alignment without frame-level labels |
| **Selection metric** | CER (Character Error Rate) = edit distance / reference length |
| **Diagnostic metrics** | DER, IER, SER to separate deletion, insertion, and substitution errors |

## Architectures

| Architecture | Status | Description |
|---|---|---|
| **TDS-CNN** | ✅ Baseline | Time-Depth Separable CNN (Meta baseline) |
| **BiLSTM / CNN+BiLSTM** | ✅ Implemented | Recurrent CTC encoders with bidirectional context |
| **Whisper-CTC** | ✅ Implemented | Transfer-learning variant using a pretrained Whisper encoder with a CTC head |
| **Transformer** | 🔬 In progress | Generic self-attention encoder over EMG frames |
| **Hybrid** | 🔬 Planned | CNN front-end + Transformer/RNN encoder |

## Current Best Documented Result

The strongest documented run so far is the CNN + BiLSTM model on the
single-user split for user 89335547 with greedy decoding.

| Split | CER (%) | DER (%) | IER (%) | SER (%) | Loss |
|---|---|---|---|---|---|
| Validation | 13.76 | 1.77 | 3.15 | 8.84 | 0.544 |
| Test | 14.89 | 1.36 | 2.64 | 10.89 | 0.556 |

Compared with the documented TDS-CNN baseline, this reduces CER on both
validation and test for the same baseline user split.

## Additional Evaluated Runs

| Model | Val CER (%) | Test CER (%) | Takeaway |
|---|---|---|---|
| BiLSTM | 15.37 | 22.07 | Better than the TDS-CNN baseline on validation, but behind CNN + BiLSTM on both splits |
| Whisper-CTC | 17.72 | 99.91 | Validation looked reasonable, but test performance collapsed due to insertion-heavy decoding |

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
