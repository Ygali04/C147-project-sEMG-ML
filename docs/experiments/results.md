# Experiment Results

## CER comparison across architectures

| Architecture | Val CER (%) | Test CER (%) | Notes |
|---|---|---|---|
| TDS-CNN (baseline) | 20.18 | 23.56 | Reference model on user 89335547 |
| CNN + BiLSTM | 13.76 | 14.89 | Best documented run, greedy decoder, checkpoint at epoch 132 |
| BiLSTM (bidir) | — | — | Implemented, but results not yet summarized here |
| Transformer | — | — | |
| CNN + Transformer | — | — | Hybrid |

## Training curves

Each training run now auto-saves a curve plot at:

`logs/<date>/<time>/training_progress.png`

This includes the recurrent models (`bilstm_ctc`, `cnn_bilstm_ctc`).

The current best documented recurrent run is:

`logs/2026-03-12/03-23-36/training_progress.png`

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