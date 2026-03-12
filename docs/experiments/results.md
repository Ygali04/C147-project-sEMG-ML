# Experiment Results

## CER comparison across architectures

| Architecture | Val CER (%) | Test CER (%) | Notes |
|---|---|---|---|
| TDS-CNN (baseline) | 20.18 | 23.56 | Reference model on user 89335547 |
| CNN + BiLSTM | 13.76 | 14.89 | Best documented run, greedy decoder, checkpoint at epoch 132 |
| BiLSTM (bidir) | 15.37 | 22.07 | Plain recurrent encoder, checkpoint at epoch 132 |
| Whisper-CTC | 17.72 | 99.91 | Whisper-tiny transfer run; validation improved but test generalization collapsed |
| Transformer | — | — | Generic transformer encoder still not benchmarked here |
| CNN + Transformer | — | — | Hybrid |

## Training curves

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

## BiLSTM Results

Evaluated with:

```bash
uv run python -m emg2qwerty.train user=single_user train=False model=bilstm_ctc "checkpoint=logs/2026-03-12/08-51-56/checkpoints/epoch\=132-step\=15960.ckpt"
```

| Metric | Validation | Test |
|--------|-----------|------|
| CER (%) | 15.37 | 22.07 |
| DER (%) | 1.51 | 4.91 |
| IER (%) | 3.35 | 1.73 |
| SER (%) | 10.52 | 15.43 |
| Loss | 0.537 | 0.814 |

## Baseline Results

After 150 epochs of training on user 89335547 (best checkpoint at epoch 111):

![Baseline Training Progress](../images/baseline_training_progress.png)

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