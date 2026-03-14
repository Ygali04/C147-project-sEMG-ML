# Transformer Encoder + CTC

> **Status:** Whisper-based transfer model implemented. General Transformer architecture sweep ran on 8×H200 (Verda).
> **Module:** `emg2qwerty.lightning.T5CTCModule`
> **Config:** `config/model/t5_ctc.yaml`

> The active remediation campaign now targets the `yahvin/transformer-troubleshoot`
> branch on an `8x RTX PRO 6000` Verda instance and tracks every transformer-family
> wave in [`docs/experiments/transformer_troubleshooting.md`](../experiments/transformer_troubleshooting.md).

## Architecture

The Transformer-CTC model reuses the same EMG front-end as the TDS-ConvNet baseline
(SpectrogramNorm → MultiBandRotationInvariantMLP → Flatten) and replaces the TDS
convolutional encoder with a standard PyTorch `nn.TransformerEncoder`.

```
Raw sEMG (2kHz, 2 bands × 16 channels)
  │
  ▼
SpectrogramNorm (BatchNorm2d per channel)
  │
  ▼
MultiBandRotationInvariantMLP → 768 features
  │
  ▼
Linear projection → d_model
  │
  ▼
[Optional] Temporal CNN Featurizer (3 × Conv1d, BN, GELU)
  │
  ▼
LayerNorm + Sinusoidal Positional Encoding + Dropout
  │
  ▼
nn.TransformerEncoder (Pre-LN, GELU, configurable depth)
  │
  ▼
Linear → log_softmax → CTC Loss
```

### Key design choices

| Choice | Detail |
|--------|--------|
| Pre-LayerNorm | `norm_first=True` for better gradient flow |
| Activation | GELU (matches modern transformer practice) |
| Positional encoding | Sinusoidal (no learned PE — avoids overfitting on small data) |
| Temporal CNN | Optional 3-layer Conv1d block before transformer (kernel sizes 5, 5, 3) |
| Anti-blank bias | Output layer initialized with `bias[blank] = -5.0` to discourage CTC blank collapse |
| CTC variant | `zero_infinity=True` to handle edge cases |
| Time format | Time-first `(T, N, D)` throughout, matching baseline convention |

### Available and planned models

| Model | Key idea |
|---|---|
| Whisper-CTC (`model=whisper_ctc`) | Project EMG features into a pretrained Whisper encoder and train a CTC head |
| Encoder-only Transformer | Self-attention over EMG spectrogram frames |
| CNN + Transformer | TDS/Conv front-end → Transformer encoder |
| Conformer-CTC (planned) | Convolution-augmented attention encoder for better local/global balance |
| Raw-CNN + Transformer (planned) | FairEMG-style raw sEMG frontend before attention |
| MyoText-style Refiner (planned) | Stage-A motor model plus transformer text correction |

## Documented Whisper-CTC result

The current transformer-family result comes from the `whisper_ctc` experiment,
which reuses the pretrained `openai/whisper-tiny` encoder with the top two
encoder layers unfrozen.

| Metric | Validation | Test |
|---|---|---|
| CER (%) | 17.72 | 99.91 |
| DER (%) | 2.55 | 0.00 |
| IER (%) | 4.10 | 99.91 |
| SER (%) | 11.08 | 0.00 |
| Loss | 0.615 | inf |

The validation split looked promising, but the test split collapsed due to
massive insertion errors. Right now that makes the Whisper transfer approach a
useful experiment, not a competitive model for this task.

### Minimal example skeleton

### Configurable parameters

All controllable via Hydra overrides:

```yaml
module:
  d_model: 128        # hidden dimension
  num_layers: 4        # transformer layers
  num_heads: 4         # attention heads
  d_ff: 512            # feedforward dimension
  use_cnn: true        # toggle temporal CNN featurizer
  blank_penalty_epochs: 40   # epochs of anti-blank penalty (0 = disable)
  blank_alpha_max: 50.0      # peak penalty weight
```

### Model size variants

| Variant | d_model | Layers | Heads | d_ff | ~Params |
|---------|---------|--------|-------|------|---------|
| Tiny | 64 | 2 | 2 | 256 | ~300K |
| Small | 128 | 4 | 4 | 512 | ~1.3M |
| Large | 256 | 6 | 8 | 1024 | ~5M |

## Architecture Sweep Results (March 2026)

### Setup

- **Hardware:** 8×NVIDIA H200 (141 vCPU, 1450GB RAM, 1128GB GPU VRAM)
- **Instance:** Verda FIN-03, Ubuntu 24.04, CUDA 12.8
- **Training:** Single GPU per experiment (no DDP — see [DDP Pitfall](#ddp-pitfall-cer-collapse) below)
- **Data:** User 89335547 (baseline profile), 3835 training windows, 120 steps/epoch
- **LR schedule:** LinearWarmupCosineAnnealing, warmup=10 epochs, max_epochs=80 (transformers) / 150 (baseline)

### CER Leaderboard (our experiments)

| GPU | Architecture | Params | Val CER | Test CER | Epoch |
|-----|-------------|--------|---------|----------|-------|
| 0 | **TDS-ConvNet (baseline)** | 5.3M | 19.45% | **22.48%** | 124/150 ✅ |
| 5 | **Large Transformer + CNN** (d=256, 6L, 8H) | ~5M | **16.79%** | 78.52% | 76/80 ✅ |
| 2 | Transformer + CNN (Small, d=128, 4L) | ~1.3M | 28.71% | 67.54% | 77/80 ✅ |
| 7 | Transformer + CNN, LR=5e-4 | ~1.3M | 39.52% | 58.81% | 78/80 ✅ |
| 4 | Transformer + CNN + blank penalty | ~1.3M | 47.76% | 64.82% | 78/80 ✅ |
| 1 | Pure Transformer (no CNN) | ~1.3M | 48.76% | 84.78% | 79/80 ✅ |
| 3 | Transformer + blank penalty (no CNN) | ~1.3M | 94.84% | 98.92% | 74/80 ✅ |
| 6 | Tiny Transformer (d=64, 2L) | ~300K | 96.37% | 100.0% | 1/80 ❌ |

> ⚠️ **Critical caveat:** Test CER is much higher than Val CER for all transformers
> because test feeds the entire session (~140K timesteps) without windowing, while
> training/val use 4-sec windows (8K timesteps). See
> [Sequence-Length Generalization Gap](../experiments/results.md#critical-finding-sequence-length-generalization-gap).

## RTX PRO 6000 Troubleshooting Campaign

The active campaign treats the new Verda machine as an 8-lane experiment cluster.
Each GPU runs one independent job rather than participating in standard DDP.

| Resource | Value |
|---|---|
| GPUs | 8x RTX PRO 6000 |
| Per-GPU VRAM | ~96 GB |
| CPU | 240 vCPU |
| RAM | 720 GB |
| Planning mode | one GPU per run, rolling 8-lane waves |

### Wave 1 Objective

The first RTX PRO 6000 wave focuses exclusively on inference policy:

| GPU | Planned run |
|---:|---|
| 0 | Large CNN + Transformer, `full_session` control |
| 1 | Large CNN + Transformer, `windowed_chunk_decode` |
| 2 | Large CNN + Transformer, `windowed_logits_merge` |
| 3 | Small CNN + Transformer, `windowed_logits_merge` |
| 4 | Whisper-CTC, `windowed_logits_merge` |
| 5 | TDS-ConvNet, `windowed_logits_merge` control |
| 6 | Large CNN + Transformer, alternate stride |
| 7 | Large CNN + Transformer, alternate trim |

This wave is designed to answer one question before any retraining starts:
how much of the transformer test collapse is caused by full-session inference
rather than by the encoder itself?

### Key findings

1. **Large Transformer + CNN beats baseline on validation.** Val CER 16.79% vs baseline's
   19.45% — a **14% relative improvement**. However, it catastrophically fails on test
   (78.5%) due to the sequence-length generalization gap (see below).

2. **CNN featurizer is essential.** All CNN-equipped variants escaped blank collapse by
   epoch 10. The pure transformer (no CNN) didn't start learning until epoch ~40 and
   plateaued at 48.8% val CER. Temporal convolutions provide local context that
   bootstraps early learning.

3. **Anti-blank penalty hurts more than it helps.** CNN + penalty (47.8%) vs CNN alone
   (28.7%). The worst combination is penalty + no CNN (94.8%). The penalty's additive
   loss term interferes with the CTC gradient. Anti-blank bias initialization alone
   (biasing output layer weights) is sufficient.

4. **Default LR (1e-3) beats higher LR (5e-4).** GPU 2 (28.7%) outperforms GPU 7 (39.5%)
   at the same epoch count with the same architecture.

5. **Model capacity matters.** Large (16.8%) >> Small (28.7%) >> Tiny (96.4%).
   This aligns with FairEMG which found scaling consistently improves CER.

6. **Sequence-length generalization is the #1 open problem.** Transformers trained on
   4-sec windows cannot generalize to 70-sec full sessions at test time. The baseline
   TDS-ConvNet has only a +3% gap (19.5% → 22.5%) while the Large Transformer has a
   +62% gap (16.8% → 78.5%). This is because self-attention patterns and positional
   encodings don't extrapolate to 17× longer sequences.

## Difficulties and Lessons Learned

### DDP Pitfall: CER Collapse

**The most significant challenge** was a complete CTC blank collapse when training with
Distributed Data Parallel (DDP) across 8 GPUs. The model would predict only blank tokens,
resulting in 100% CER that never improved.

#### What happened

```
Epoch 0: val/CER = 1365.1  (random predictions)
Epoch 1: val/CER = 100.0   (BLANK COLLAPSE — all blank tokens)
Epoch 2: val/CER = 100.0   (stuck)
Epoch 3: val/CER = 100.0   (stuck)
...
Epoch 53: val/CER = 99.15   (barely moved after 53 epochs)
```

This pattern repeated across **every architecture** we tried:
- HuggingFace T5EncoderModel + CTC
- Standard nn.TransformerEncoder + CTC
- nn.TransformerEncoder + CNN + CTC
- **Even the baseline TDS-ConvNet** collapsed with DDP

#### Root cause

With 8 GPUs and batch_size=32, each GPU saw only `3835 / 8 / 32 ≈ 15 steps per epoch`.
The LinearWarmupCosineAnnealing schedule warms up over 10 epochs, so the model had only
`15 × 10 = 150 gradient updates` at a very small learning rate (1e-8 → 1e-3) before
the warmup finished. This was insufficient to learn any character-level patterns before
the CTC loss landscape pushed the model toward the blank attractor.

On a **single GPU**, steps per epoch = `3835 / 32 = 120`, giving `120 × 10 = 1200`
gradient updates during warmup — 8× more signal, enough to escape the blank attractor.

#### Solution

**Train on a single GPU per experiment.** The H200 has enough memory (141GB VRAM) to
handle the full dataset comfortably. For larger datasets, increase batch size or use
gradient accumulation rather than DDP, unless the per-GPU step count is high enough
(recommendation: ≥50 steps/epoch per GPU).

#### Things we tried that did NOT fix the DDP collapse

| Attempt | Result |
|---------|--------|
| Anti-blank bias initialization (bias[blank] = -5.0) | ❌ Still collapsed |
| Blank penalty loss (alpha * mean_blank_prob) | ❌ Made it worse |
| Reduce num_layers from 6 → 1 | ❌ Still collapsed |
| Add dropout 0.1 everywhere | ❌ Still collapsed |
| Lower LR (3e-4 instead of 1e-3) | ❌ Still collapsed |
| Gradient clipping (1.0) | ❌ Still collapsed |
| Reduce warmup_epochs to 1 | ❌ Still collapsed |
| Replace T5 with nn.TransformerEncoder | ❌ Still collapsed |
| Add CNN featurizer | ❌ Still collapsed (with DDP) |
| Mixed precision (fp16) | ❌ Still collapsed |

**The only thing that worked was switching to single GPU.**

### Other issues encountered

| Issue | Fix |
|-------|-----|
| `pkill -f 'emg2qwerty'` kills the SSH session | The grep pattern matches the SSH command itself; use PID-based kill instead |
| `uv: command not found` on remote | Add `export PATH=$HOME/.local/bin:$PATH` in every SSH block |
| `Python.h: No such file` building kenlm | Install `python3-dev build-essential cmake zlib1g-dev libbz2-dev liblzma-dev` |
| `pl_bolts` broken with PyTorch Lightning 2.x | Patch `__init__.py` with sed to bypass eager imports |
| Hydra `+` prefix needed for new keys | Use `+optimizer.weight_decay=0.01` not `optimizer.weight_decay=0.01` |
| Zombie DDP processes after crashes | `killall -9 python3` before relaunching |
| Stale `__pycache__` bytecode | Always `find . -name __pycache__ -exec rm -rf {} +` after rsync |

## Verda 8×H200 Training Setup

### Instance specs

| Spec | Value |
|------|-------|
| GPUs | 8× NVIDIA H200 |
| GPU VRAM | 1128 GB total (141 GB each) |
| CPU | 176 vCPU |
| RAM | 1450 GB |
| OS | Ubuntu 24.04, CUDA 12.8 |
| Location | FIN-03 |
| Price | $11.42/hr |

### Deployment workflow

```bash
# 1. Sync code to remote
rsync -avz --delete -e "ssh -i ~/.ssh/id_ed25519" \
    --exclude='.git/' --exclude='data/' --exclude='logs/' \
    --exclude='.venv/' --exclude='.env' \
    ./ root@<verda-ip>:/root/emg2qwerty/

# 2. Remote setup (first time only)
ssh root@<verda-ip>
curl -LsSf https://astral.sh/uv/install.sh | sh
sudo apt-get install -y python3-dev build-essential cmake \
    zlib1g-dev libbz2-dev liblzma-dev
cd /root/emg2qwerty && uv sync

# 3. Run single experiment
CUDA_VISIBLE_DEVICES=0 uv run python -m emg2qwerty.train \
    user=single_user model=t5_ctc '~cluster' trainer.devices=1

# 4. Run architecture sweep (all 8 GPUs)
bash scripts/sweep_gpus.sh

# 5. Monitor
grep 'reached' /root/sweep_gpu*.log
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
```

### Key training parameters

```yaml
# Base config
batch_size: 32
max_epochs: 150 (baseline) / 80 (transformer sweep)
lr: 1e-3 (Adam)
lr_schedule: LinearWarmupCosineAnnealing
warmup_epochs: 10
warmup_start_lr: 1e-8
window_length: 8000  # 4 seconds at 2kHz
padding: [1800, 200] # 900ms past, 100ms future
```

## Literature Context

Our results align with the findings from recent emg2qwerty papers:

| Paper | Architecture | Zero-shot CER | Personalized CER |
|-------|-------------|---------------|-----------------|
| emg2qwerty Baseline (NeurIPS 2024) | TDS-ConvNet + CTC | 51.8% | 6.95% (+LM) |
| **Ours (baseline repro)** | **TDS-ConvNet + CTC** | 19.45% val / **22.48% test*** | — |
| SplashNet (NeurIPS 2025) | Split-Share TDS | 35.7% | 5.5% (+LM) |
| FairEMG (TMLR 2025) | CNN + Transformer | ~30-40% | — |
| **Ours (Large Transformer)** | **CNN + Transformer + CTC** | **16.79% val** / 78.52% test* | — |
| Typing Reinvented (2025) | Transformer/Conformer | 20.34%** | 10.10%** |
| MyoText (2026) | CNN-BiLSTM + T5 | — | 5.4% |

\* Single user (89335547), no LM. Not zero-shot — trained on that user's data.
\*\* Causal/online setting — not directly comparable.

### FairEMG comparison (most relevant)

The FairEMG paper (Paper 4) found that even a ~600K-param Tiny transformer substantially
outperforms the 5.3M TDS-ConvNet baseline. They use a **CNN featurizer on raw sEMG**
(not spectrograms), which gave ~8 CER improvement. Our results confirm that CNN features
before the transformer are essential — our CNN variants all learn while non-CNN variants
struggled to escape blank collapse during warmup.

## Next Steps

Based on the sweep results and literature analysis, prioritized by impact:

1. **🔴 Fix sequence-length generalization** — This is the #1 blocker. Options:
   - **Sliding-window inference**: chunk the test session into overlapping 4-sec windows,
     run inference on each, merge predictions. Simple and effective.
   - **ALiBi / RoPE** positional encodings: these extrapolate to unseen lengths much
     better than sinusoidal PE. RoPE is used by LLaMA, ALiBi by BLOOM.
   - **Train with variable-length windows**: randomly sample window lengths during
     training (e.g., 2-16 sec) to improve length robustness.
   - **Conformer architecture**: combines local convolution (length-invariant) with
     global attention (content-dependent). Used in Typing Reinvented (Paper 5).

2. **Raw sEMG CNN featurizer** (FairEMG approach) — bypass log-spectrogram, apply CNN
   directly to the raw 2kHz signal. FairEMG shows ~8 CER improvement.

3. **Language model decoding** — Add 6-gram KenLM beam search (already in
   `decoder/ctc_beam.yaml`). This alone drops CER from ~20% → ~7% in published results.

4. **SplashNet innovations** — Rolling Time Normalization and Aggressive Channel Masking
   are architecture-agnostic and gave 31% relative improvement.

5. **Scale up** — FairEMG shows gains up to 110M params. Our Large has ~5M;
   try d=512, 12L (~20M).

6. **Knowledge distillation** — Train a large teacher, distill to a 2M student
   (FairEMG: <1.5% CER loss with 50× fewer params).
