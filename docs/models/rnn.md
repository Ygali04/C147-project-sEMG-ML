# RNN / LSTM / GRU

## Recurrent architectures for EMG decoding

*This page will be updated as we implement and evaluate recurrent models.*

### Planned experiments

- Bidirectional LSTM with varying hidden sizes
- GRU as a lighter alternative
- CNN feature extractor → RNN encoder (hybrid)

### Expected trade-offs

- Better at capturing long-range temporal dependencies than TDS-CNN
- Slower training due to sequential nature
- Risk of vanishing gradients mitigated by LSTM/GRU gating
