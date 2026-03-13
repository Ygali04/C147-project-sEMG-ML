# Experiment Results

## Final CER Comparison — All Architectures

All experiments trained on user 89335547 (single user, greedy CTC decoding, no language model),
on an 8×H200 Verda instance with one experiment per GPU.

### Full Results Table

| # | Architecture | Params | Val CER | Test CER | Val Loss | Test Loss | Best Epoch |
|---|-------------|--------|---------|----------|----------|-----------|------------|
| 0 | **TDS-ConvNet (baseline)** | 5.3M | **19.45%** | **22.48%** | 1.017 | 1.184 | 124/150 |
| 5 | **Large Transformer + CNN** | ~5M | **16.79%** | 78.52% | 0.532 | 4.910 | 76/80 |
| 2 | Small Transformer + CNN | ~1.3M | 28.71% | 67.54% | 0.862 | 3.030 | 77/80 |
| 7 | Trans + CNN, LR=5e-4 | ~1.3M | 39.52% | 58.81% | 1.219 | 2.049 | 78/80 |
| 4 | Trans + CNN + blank penalty | ~1.3M | 47.76% | 64.82% | 49.57 | 50.89 | 78/80 |
| 1 | Pure Transformer (no CNN) | ~1.3M | 48.76% | 84.78% | 1.265 | 5.218 | 79/80 |
| 3 | Trans + blank penalty (no CNN) | ~1.3M | 94.84% | 98.92% | 50.25 | 52.87 | 74/80 |
| 6 | Tiny Transformer (d=64) | ~300K | 96.37% | 100.0% | 77.73 | 66.26 | 1/80 |

### Detailed Metrics (Val / Test)

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
