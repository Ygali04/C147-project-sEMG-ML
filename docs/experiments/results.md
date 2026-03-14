# Experiment Results

## Final CER Comparison — All Architectures

All experiments trained on user 89335547 (single user, greedy CTC decoding, no language model),
on an 8×H200 Verda instance with one experiment per GPU.

### Full Results Table
| Architecture | Params | Val CER (%) | Test CER (%) | Val Loss | Test Loss | Best Epoch | Notes |
|---|---|---|---|---|---|---|---|
| **TDS-ConvNet (baseline)** | 5.3M | **19.45** | **22.48** | 1.017 | 1.184 | 124/150 | Branch 2 results used (lower CER on both splits) |
| **CNN + BiLSTM** | — | **13.76** | **14.89** | — | — | 132 | Best overall; greedy decoder |
| BiLSTM (bidir) | — | 15.37 | 22.07 | — | — | 132 | Plain recurrent encoder |
| Whisper-CTC | — | 17.72 | 99.91 | — | — | — | Whisper-tiny transfer; val improved, test generalization collapsed |
| **Large Transformer + CNN** | ~5M | **16.79** | 78.52 | 0.532 | 4.910 | 76/80 | Strong val CER but significant train/test gap |
| Small Transformer + CNN | ~1.3M | 28.71 | 67.54 | 0.862 | 3.030 | 77/80 | |
| Trans + CNN, LR=5e-4 | ~1.3M | 39.52 | 58.81 | 1.219 | 2.049 | 78/80 | Higher LR hurts val |
| Trans + CNN + blank penalty | ~1.3M | 47.76 | 64.82 | 49.57 | 50.89 | 78/80 | Blank penalty destabilizes loss |
| Pure Transformer (no CNN) | ~1.3M | 48.76 | 84.78 | 1.265 | 5.218 | 79/80 | |
| Trans + blank penalty (no CNN) | ~1.3M | 94.84 | 98.92 | 50.25 | 52.87 | 74/80 | Blank penalty + no CNN: collapsed |
| Tiny Transformer (d=64) | ~300K | 96.37 | 100.0 | 77.73 | 66.26 | 1/80 | Undertrained / too small |

## RTX PRO 6000 Campaign

The active troubleshooting campaign runs on an on-demand Verda instance with
`8x RTX PRO 6000` GPUs and treats the machine as an 8-lane experiment cluster.
The governing rule is `1 GPU = 1 independent run`; the campaign does not use
standard DDP for core CTC training because earlier 8-GPU DDP experiments caused
blank-collapse by reducing each GPU to too few warmup steps.

### Active Wave Ledger

Use this table to track live and recently completed runs for the
`yahvin/transformer-troubleshoot` branch.

| Wave | GPU slot | Model | Commit SHA | Inference mode | Train windows | Positional encoding | Frontend | Decoder | Status | Checkpoint | Train CER | Val CER | Test CER | Notes |
|---|---:|---|---|---|---|---|---|---|---|---|---:|---:|---:|---|
| wave-0-docs | 0 | docs foundation | pending | n/a | n/a | n/a | n/a | n/a | completed | — | — | — | — | initial ledger scaffold |
| wave-1-inference | 0 | Large CNN + Transformer | pending | `full_session` | `4s` | sinusoidal | spectrogram | greedy | planned | — | — | — | — | control |
| wave-1-inference | 1 | Large CNN + Transformer | pending | `windowed_chunk_decode` | `4s` | sinusoidal | spectrogram | greedy | planned | — | — | — | — | chunk decode |
| wave-1-inference | 2 | Large CNN + Transformer | pending | `windowed_logits_merge` | `4s` | sinusoidal | spectrogram | greedy | planned | — | — | — | — | main candidate |
| wave-1-inference | 3 | Small CNN + Transformer | pending | `windowed_logits_merge` | `4s` | sinusoidal | spectrogram | greedy | planned | — | — | — | — | small control |
| wave-1-inference | 4 | Whisper-CTC | pending | `windowed_logits_merge` | `4s` | n/a | spectrogram | greedy | planned | — | — | — | — | transfer control |
| wave-1-inference | 5 | TDS-ConvNet | pending | `windowed_logits_merge` | `4s` | n/a | spectrogram | greedy | planned | — | — | — | — | local control |
| wave-1-inference | 6 | Large CNN + Transformer | pending | `windowed_logits_merge` | `4s` | sinusoidal | spectrogram | greedy | planned | — | — | — | — | alt stride |
| wave-1-inference | 7 | Large CNN + Transformer | pending | `windowed_logits_merge` | `4s` | sinusoidal | spectrogram | greedy | planned | — | — | — | — | alt trim |

### Inference Policy Comparison

| Policy | Window length | Stride | Trim margin | Merge strategy | Model | Val CER | Test CER | IER | DER | SER | Notes |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---|
| `full_session` | — | — | — | none | Large CNN + Transformer | 16.79 | 78.52 | 61.16 | 0.13 | 17.22 | current failure baseline |
| `windowed_chunk_decode` | 8000 | 8000 | 0 | transcript concat/alignment | pending | — | — | — | — | — | to be filled |
| `windowed_logits_merge` | 8000 | 8000 | 0 | log-prob merge | pending | — | — | — | — | — | to be filled |

### Position Encoding Comparison

| Model | Train windows | Inference mode | Sinusoidal Val/Test | ALiBi Val/Test | RoPE Val/Test | Winner | Notes |
|---|---|---|---|---|---|---|---|
| Large CNN + Transformer | pending | pending | pending | pending | pending | pending | wave not yet run |

### Variable-Length Window Sweep

| Model | Train window lengths | Sampling weights | Inference mode | Val CER | Test CER | Gap | Status | Notes |
|---|---|---|---|---:|---:|---:|---|---|
| Large CNN + Transformer | `[8000]` | `[1.0]` | pending | — | — | — | planned | fixed-length control |
| Large CNN + Transformer | `[8000,16000]` | `[1.0,1.0]` | pending | — | — | — | planned | first multi-length regime |
| Large CNN + Transformer | `[8000,16000,24000]` | `[1.0,1.0,1.0]` | pending | — | — | — | planned | broader range |
| Large CNN + Transformer | `[8000,16000,24000,32000]` | `[1.0,1.0,1.0,1.0]` | pending | — | — | — | planned | max planned regime |

### Architecture Sweep Template

| Family | Frontend | Encoder | Downsample | Positional encoding | Decoder | Train CER | Val CER | Test CER | Status | Notes |
|---|---|---|---:|---|---|---:|---:|---:|---|---|
| transformer | spectrogram | transformer | 1x | sinusoidal | greedy | — | 16.79 | 78.52 | completed | current control |
| recurrent | spectrogram | CNN + BiLSTM | 1x | n/a | greedy | — | 13.76 | 14.89 | completed | strongest current control |

### Detailed Metrics (Val / Test)
Each training run now auto-saves a curve plot at:

`logs/<date>/<time>/training_progress.png`

This includes the recurrent and transfer models (`bilstm_ctc`, `cnn_bilstm_ctc`, `whisper_ctc`).

You can also regenerate plots for completed runs from their saved Lightning
metrics without retraining:

```bash
# Rebuild plots for every run under a given date directory
uv run python -m emg2qwerty.plot_training_curves logs/2026-03-12/*

# Or target a single historical run
uv run python -m emg2qwerty.plot_training_curves logs/2026-03-12/09-20-04
```

The utility looks for `lightning_logs/version_*/metrics.csv` inside each run
directory and rewrites `training_progress.png` in place.

The current best documented recurrent run is:

`logs/2026-03-12/03-23-36/training_progress.png`

## Baseline Results

| Architecture | CER | IER | DER | SER |
|-------------|-----|-----|-----|-----|
| **TDS-ConvNet baseline** | 19.45 / **22.48** | 5.74 / 5.81 | 1.97 / 2.36 | 11.74 / 14.31 |
| **Large Trans + CNN** | **16.79** / 78.52 | 4.32 / 61.16 | 1.62 / 0.13 | 10.86 / 17.22 |
| Small Trans + CNN | 28.71 / 67.54 | 8.88 / 35.34 | 2.17 / 3.70 | 17.66 / 28.51 |
| Trans + CNN, LR=5e-4 | 39.52 / 58.81 | 13.38 / 24.25 | 2.57 / 6.07 | 23.57 / 28.48 |
| Trans + CNN + penalty | 47.76 / 64.82 | 21.62 / 44.35 | 2.22 / 0.91 | 23.93 / 19.56 |
| Pure Transformer | 48.76 / 84.78 | 36.24 / 79.77 | 0.73 / 0.00 | 11.79 / 5.01 |
| Trans + penalty (no CNN) | 94.84 / 98.92 | 94.51 / 98.92 | 0.00 / 0.00 | 0.33 / 0.00 |
| Tiny Transformer | 96.37 / 100.0 | 95.33 / 99.98 | 0.02 / 0.00 | 1.02 / 0.00 |

## Critical Finding: Sequence-Length Generalization Gap

The most important result from this sweep is the **massive val-to-test generalization gap**
in transformer models:

| Model | Val CER | Test CER | Gap |
|-------|---------|----------|-----|
| TDS-ConvNet baseline | 19.45% | 22.48% | **+3.0%** |
| Large Trans + CNN | 16.79% | 78.52% | **+61.7%** |
| Small Trans + CNN | 28.71% | 67.54% | **+38.8%** |

**Why this happens:**

- **Training** uses 4-second windows (8,000 timesteps at 2kHz)
- **Validation** uses the same windowing scheme
- **Test** feeds the **entire session** at once (~140,000 timesteps) without windowing

The transformer's self-attention mechanism (which attends over all positions) cannot
extrapolate from 8K-length sequences to 140K-length sequences. The sinusoidal positional
encoding helps somewhat but is insufficient for a 17.5× length increase.

The baseline **TDS-ConvNet** uses local convolutions with fixed receptive fields, so it
naturally handles arbitrary-length sequences — hence its tiny 3% gap.

**Implications for future work:**
- Use **windowed inference** at test time (sliding window + merge predictions)
- Apply **ALiBi** or **RoPE** positional encodings which extrapolate better
- Train with **variable-length windows** to improve length generalization
- Consider **Conformer** architecture which combines local conv + global attention

## Convergence Curves

### Baseline TDS-ConvNet (GPU 0, 150 epochs)

```
Epoch   0: 1365.1%  (random)
Epoch   1:  100.0%  (blank collapse during LR warmup)
Epoch   9:   99.5%  (warmup finishing → learning starts)
Epoch  10:   93.5%
Epoch  14:   74.9%
Epoch  17:   56.0%
Epoch  19:   40.1%
Epoch  23:   30.2%
Epoch  28:   25.5%
Epoch  42:   21.5%
Epoch  81:   20.0%
Epoch 124:   19.5%  (converged)
```

### Large Transformer + CNN (GPU 5, 80 epochs)

```
Epoch   0: 3497.2%  (random — higher initial CER than baseline)
Epoch   1:  100.0%  (blank collapse)
Epoch  10:   88.3%  (learning starts, slower than baseline)
Epoch  20:   69.1%
Epoch  30:   48.2%
Epoch  38:   33.4%
Epoch  44:   28.3%
Epoch  47:   26.4%
Epoch  60:   19.9%
Epoch  76:   16.8%  (best val CER — beats baseline!)
```

## Architecture Ablation Summary

| Feature | Effect on Val CER | Evidence |
|---------|-------------------|----------|
| **Temporal CNN** | Essential for early learning | CNN variants escape collapse by epoch 10; non-CNN variants stuck until epoch 40+ |
| **Model size** | Larger is better | Large (16.8%) >> Small (28.7%) >> Tiny (96.4%) |
| **Anti-blank penalty** | Harmful | Penalty variants have ~2× worse CER and artificially high losses |
| **Learning rate** | 1e-3 > 5e-4 | Default LR (28.7%) beats 5e-4 (39.5%) |
| **Sequence-length generalization** | TDS >> Transformer | TDS gap: +3%; Transformer gap: +39-62% |

## CTC Blank Collapse with DDP

See [Transformer docs](../models/transformer.md#ddp-pitfall-cer-collapse) for full details
on the CTC blank collapse phenomenon when training with DDP across 8 GPUs. In summary:

- With 8 GPUs: 15 steps/epoch → 150 warmup steps → **permanent blank collapse**
- With 1 GPU: 120 steps/epoch → 1200 warmup steps → **escapes collapse at epoch 10**
- Affected **every architecture**, including the proven TDS-ConvNet baseline

## Training Infrastructure

| Spec | Value |
|------|-------|
| Instance | Verda FIN-03, 8×NVIDIA H200 |
| GPU VRAM | 141 GB each (1128 GB total) |
| CPU / RAM | 176 vCPU, 1450 GB |
| OS | Ubuntu 24.04, CUDA 12.8 |
| Cost | $11.42/hr |
| Baseline (150 ep) | ~25 min |
| Transformer sweep (80 ep × 7) | ~45 min |
| Eval (8 models) | ~5 min |
| **Total session cost** | **~$15** |
| Metric | Validation | Test |
|--------|-----------|------|
| CER (%) | 20.18 | 23.56 |
| DER (%) | 2.22 | 2.36 |
| IER (%) | 4.94 | 5.25 |
| SER (%) | 13.03 | 15.95 |
| Loss | 1.126 | 1.295 |

## CNN + BiLSTM Results

After 150 epochs of training on user 89335547 with `model=cnn_bilstm_ctc`
and greedy decoding, the best checkpoint was saved at epoch 132.

![CNN + BiLSTM Training Progress](../images/cnn_bilstm_training_progress.png)

This run improves validation CER from 20.18% to 13.76% and test CER from
23.56% to 14.89% relative to the TDS-CNN baseline on the same single-user split.

| Metric | Validation | Test |
|--------|-----------|------|
| CER (%) | 13.76 | 14.89 |
| DER (%) | 1.77 | 1.36 |
| IER (%) | 3.15 | 2.64 |
| SER (%) | 8.84 | 10.89 |
| Loss | 0.544 | 0.556 |

## Whisper-CTC Results

After 150 epochs of training on user 89335547 with `model=whisper_ctc`
and greedy decoding, the best checkpoint was saved at epoch 143.

![CNN + BiLSTM Training Progress](../images/cnn_bilstm_training_progress.png)

This transfer-learning run reached a respectable validation CER, but it does
not generalize to the test split. The failure mode is almost entirely
insertions, which pushes test CER to nearly 100% even though validation loss
and validation CER looked competitive.

| Metric | Validation | Test |
|--------|-----------|------|
| CER (%) | 17.72 | 99.91 |
| DER (%) | 2.55 | 0.00 |
| IER (%) | 4.10 | 99.91 |
| SER (%) | 11.08 | 0.00 |
| Loss | 0.615 | inf |
