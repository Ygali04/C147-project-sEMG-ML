# TDS-CNN Baseline

## Time-Depth Separable Convolutions

The baseline architecture uses **Time-Depth Separable (TDS)** convolutional
blocks, following the approach from the emg2qwerty paper.

### Key ideas

- **Depthwise temporal convolution**: convolves along the time axis independently
  per channel, capturing local temporal patterns in the EMG signal.
- **Pointwise (1×1) convolution**: mixes information across channels after the
  temporal convolution.
- **Residual connections** and **layer normalization** stabilize training.

### Architecture summary

```
Input (C channels × T time steps)
  → [TDS Block] × N
  → Linear projection → num_classes (26 letters + blank)
  → CTC loss
```

### Notes

- This is the reference model we compare all other architectures against.
- Hyperparameters (kernel size, number of blocks, channel width) are configured
  via Hydra configs in `config/`.
