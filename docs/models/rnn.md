# RNN / LSTM

> **Status:** Implemented (BiLSTM + CNN-BiLSTM)

## Implemented approach

Recurrent CTC architectures are implemented as `pl.LightningModule`
subclasses in `src/emg2qwerty/lightning.py` and share the same training
loop, optimizer/scheduler wiring, and decoder metrics as the baseline model.

### Available models

| Model | Key idea |
|---|---|
| BiLSTM (`model=bilstm_ctc`) | Full-sequence bidirectional context after spectral front-end |
| CNN + BiLSTM (`model=cnn_bilstm_ctc`) | Temporal Conv1D stack before BiLSTM encoder |

Both variants keep the existing front-end:

1. `SpectrogramNorm`
2. `MultiBandRotationInvariantMLP`
3. Feature flattening across bands

and then replace the temporal encoder with recurrent layers.

## Documented results

The strongest documented recurrent result so far is the
`cnn_bilstm_ctc` run on the `single_user` split for user 89335547.
That run used greedy decoding, batch size 32, and trained for 150 epochs,
with the best checkpoint saved at epoch 132.

![CNN + BiLSTM Training Progress](../images/cnn_bilstm_training_progress.png)

| Metric | Validation | Test |
|---|---|---|
| CER (%) | 13.76 | 14.89 |
| DER (%) | 1.77 | 1.36 |
| IER (%) | 3.15 | 2.64 |
| SER (%) | 8.84 | 10.89 |
| Loss | 0.544 | 0.556 |

The remaining error is dominated by substitutions rather than insertions or
deletions, which suggests the recurrent encoder is aligning sequences well but
still confuses some characters at decode time.

For comparison, the plain `bilstm_ctc` checkpoint evaluated on the same split
reached weaker but still competitive numbers:

| Metric | Validation | Test |
|---|---|---|
| CER (%) | 15.37 | 22.07 |
| DER (%) | 1.51 | 4.91 |
| IER (%) | 3.35 | 1.73 |
| SER (%) | 10.52 | 15.43 |
| Loss | 0.537 | 0.814 |

That comparison suggests the convolutional front-end is doing meaningful work
before the recurrent encoder: it improves both validation and test CER, while
also keeping deletion and substitution errors lower than the pure BiLSTM.

## Usage

```bash
# Pure BiLSTM encoder
uv run python -m emg2qwerty.train model=bilstm_ctc user=single_user

# CNN + BiLSTM hybrid
uv run python -m emg2qwerty.train model=cnn_bilstm_ctc user=single_user
```

## Minimal architecture sketch

```python
class BiLSTMCTCModule(pl.LightningModule):
    def __init__(self, ...):
        self.frontend = nn.Sequential(...)
        self.encoder = nn.LSTM(
            input_size=..., hidden_size=..., num_layers=...,
            bidirectional=True, batch_first=False
        )
        self.classifier = nn.Linear(2 * hidden_size, num_classes)
```

## Expected trade-offs vs TDS-CNN

| | TDS-CNN | BiLSTM |
|---|---|---|
| Parallelism | ✅ Fully parallel | ❌ Sequential |
| Long-range context | ⚠️ Limited by kernel | ✅ Unbounded |
| Training speed | Fast | Slower |
| Overfitting risk | Low | Higher (more params) |

In practice, the CNN + BiLSTM hybrid has been a better fit than the plain
baseline CNN for the documented single-user experiment, likely because the
convolutional front-end reduces local noise before the recurrent stack models
longer-range temporal structure.
