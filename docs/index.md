# EMG2QWERTY

**Predicting Keystrokes from Electromyography (EMG) Signals**

UCLA ECE C147/C247 Final Project

---

## Overview

This project explores decoding QWERTY keystrokes from surface electromyography
(sEMG) signals recorded at the wrist. We use the
[emg2qwerty](https://github.com/facebookresearch/emg2qwerty) dataset, which
provides multi-electrode sEMG recordings paired with ground-truth keystroke
labels.

## Approach

- **Loss function:** CTC (Connectionist Temporal Classification) loss, which
  handles the alignment between variable-length EMG windows and keystroke
  sequences without requiring frame-level labels.
- **Metric:** Character Error Rate (CER) — the edit distance between predicted
  and ground-truth keystroke sequences, normalized by the reference length.

## Architectures

| Architecture | Description |
|---|---|
| **TDS-CNN** | Time-Depth Separable convolutions (baseline from the paper) |
| **RNN / LSTM / GRU** | Recurrent models for sequential EMG decoding |
| **Transformer** | Self-attention-based encoder for EMG sequences |
| **Hybrid** | Combinations (e.g., CNN feature extractor + Transformer encoder) |

## Quick start

```bash
uv sync --group dev
make test
make docs-serve
```
