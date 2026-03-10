# Transformer

## Self-attention for EMG sequences

*This page will be updated as we implement and evaluate Transformer models.*

### Planned experiments

- Encoder-only Transformer with positional encoding over EMG frames
- Comparison of learned vs. sinusoidal positional embeddings
- CNN front-end → Transformer encoder (hybrid)

### Expected trade-offs

- Parallelizable (faster training than RNNs)
- May need more data or regularization to avoid overfitting on small datasets
- Attention maps could provide interpretability into which time steps matter
