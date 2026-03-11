# Experiment Results

## CER comparison across architectures

*Results will be filled in as experiments are run.*

| Architecture | Val CER (%) | Test CER (%) | Notes |
|---|---|---|---|
| TDS-CNN (baseline) | 20.18 | 23.56 | Reference model |
| LSTM (bidir) | — | — | |
| GRU (bidir) | — | — | |
| Transformer | — | — | |
| CNN + Transformer | — | — | Hybrid |

## Training curves

*(Plots will be added under `docs/images/` and referenced here.)*
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