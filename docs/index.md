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

## Pipeline

```
HDF5 session files
  → WindowedEMGDataset (data.py)
  → LogSpectrogram transform (transforms.py)
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

## Quick start

```bash
# Install all dependencies
uv sync --group dev

# Run tests (charset + decoder)
make test

# Train the baseline (requires GPU + data)
uv run python -m emg2qwerty.train user=single_user trainer.accelerator=gpu

# Serve docs locally
make docs-serve
```

## Source layout

```
src/emg2qwerty/
  __init__.py
  charset.py     # Key ↔ label ↔ unicode mappings
  data.py        # EMGSessionData + WindowedEMGDataset
  decoder.py     # CTCGreedyDecoder + CTCBeamDecoder
  lightning.py   # LightningDataModule + TDSConvCTCModule
  metrics.py     # CharacterErrorRates (CER / IER / DER / SER)
  modules.py     # TDSConvEncoder + RotationInvariantMLP + SpectrogramNorm
  train.py       # Hydra entry-point (train + eval loop)
  transforms.py  # LogSpectrogram, SpecAugment, RandomBandRotation, ...
  utils.py       # Optimizer/scheduler instantiation, checkpoint helpers
config/          # Hydra YAML configs (model, optimizer, LR scheduler, ...)
scripts/         # Dataset utilities (generate_splits, print_stats, ...)
tests/           # pytest tests for charset, decoder, data
notebooks/       # EDA and result visualization
```
