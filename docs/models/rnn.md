# RNN / LSTM / GRU

> **Status:** In progress — this page will be updated as models are implemented.

## Planned approach

New recurrent architectures are added as new `pl.LightningModule` subclasses
in `src/emg2qwerty/lightning.py`, following the same interface as
`TDSConvCTCModule`.

### Planned models

| Model | Key idea |
|---|---|
| Bidirectional LSTM | Full-sequence context; larger receptive field than TDS |
| GRU | Lighter alternative to LSTM, fewer parameters |
| CNN + BiLSTM | TDS-style front-end → bidirectional LSTM encoder |

### Minimal example skeleton

```python
class BiLSTMCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(self, in_features, hidden_size, num_layers, ...):
        super().__init__()
        self.model = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(in_features=in_features, ...),
            nn.Flatten(start_dim=2),
            # Replace TDSConvEncoder with BiLSTM:
            nn.LSTM(input_size=..., hidden_size=hidden_size,
                    num_layers=num_layers, bidirectional=True, batch_first=False),
            # Extract output, project to classes
            ...
            nn.LogSoftmax(dim=-1),
        )
```

## Expected trade-offs vs TDS-CNN

| | TDS-CNN | BiLSTM |
|---|---|---|
| Parallelism | ✅ Fully parallel | ❌ Sequential |
| Long-range context | ⚠️ Limited by kernel | ✅ Unbounded |
| Training speed | Fast | Slower |
| Overfitting risk | Low | Higher (more params) |
