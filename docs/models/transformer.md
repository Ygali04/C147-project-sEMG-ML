# Transformer

> **Status:** In progress — this page will be updated as models are implemented.

## Planned approach

Transformer encoders are added as new `pl.LightningModule` subclasses in
`src/emg2qwerty/lightning.py`, sharing the same CTC training loop as the baseline.

### Planned models

| Model | Key idea |
|---|---|
| Encoder-only Transformer | Self-attention over EMG spectrogram frames |
| CNN + Transformer | TDS/Conv front-end → Transformer encoder |

### Minimal example skeleton

```python
class TransformerCTCModule(pl.LightningModule):
    def __init__(self, in_features, d_model, nhead, num_layers, ...):
        super().__init__()
        self.input_proj = nn.Linear(in_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, charset().num_classes)

    def forward(self, inputs):
        # inputs: (T, N, bands=2, C=16, freq) after SpectrogramNorm + MLP
        x = self.input_proj(inputs)          # (T, N, d_model)
        x = self.transformer(x)              # (T, N, d_model)
        return F.log_softmax(self.output_proj(x), dim=-1)
```

## Expected trade-offs vs TDS-CNN

| | TDS-CNN | Transformer |
|---|---|---|
| Parallelism | ✅ Parallel | ✅ Fully parallel |
| Long-range context | ⚠️ Limited | ✅ Global attention |
| Training speed | Fast | Moderate (O(T²) attention) |
| Data hunger | Low | Higher |
| Interpretability | Low | ✅ Attention maps |
