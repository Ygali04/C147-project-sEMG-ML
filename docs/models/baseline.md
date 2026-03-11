# TDS-CNN Baseline

> Reference: [Sequence-to-Sequence Speech Recognition with Time-Depth Separable
> Convolutions, Hannun et al. (2019)](https://arxiv.org/abs/1904.02619)

## Architecture

The baseline `TDSConvCTCModule` (defined in `src/emg2qwerty/lightning.py`) is a
stack of Time-Depth Separable convolutional blocks followed by a linear CTC head.

```
Input: (T, N, bands=2, C=16, freq)            # LogSpectrogram of left+right EMG
  ↓
SpectrogramNorm                                # BatchNorm2d per band×channel
  ↓
MultiBandRotationInvariantMLP                  # MLP with electrode-rotation pooling
  → (T, N, bands=2, mlp_features[-1])
  ↓
Flatten → (T, N, num_features)
  ↓
TDSConvEncoder                                 # Stack of TDSConv2dBlock + FC blocks
  ↓
Linear(num_features, num_classes)
  ↓
LogSoftmax → CTC Loss
```

## Key modules (`src/emg2qwerty/modules.py`)

### `SpectrogramNorm`
Applies `nn.BatchNorm2d` independently over each of the
`num_bands × electrode_channels` = 2 × 16 = 32 channels.

### `RotationInvariantMLP`
Shifts electrode channels by each offset in `(-1, 0, 1)` (band rotation
augmentation), runs an MLP on each shifted version, then mean-pools. This
makes the model robust to electrode placement variation.

### `TDSConv2dBlock`
```
Conv2d(channels, channels, kernel=(1, kernel_width))   # temporal conv per channel
  → ReLU
  → skip connection (last T_out frames of input)
  → LayerNorm
```

### `TDSFullyConnectedBlock`
```
Linear(num_features, num_features) → ReLU → Linear
  → skip connection
  → LayerNorm
```

### `TDSConvEncoder`
Stacks alternating `TDSConv2dBlock` + `TDSFullyConnectedBlock` for each
entry in `block_channels`.

## Config

The baseline hyperparameters live in `config/model/tds_conv_ctc.yaml`:

```yaml
module:
  _target_: emg2qwerty.lightning.TDSConvCTCModule
  in_features: 33        # n_fft // 2 + 1 frequency bins
  mlp_features: [384]
  block_channels: [24, 24, 24, 24, 24, 24, 24, 24]
  kernel_width: 32
```

## Decoding

Two decoders are available (`src/emg2qwerty/decoder.py`):

| Decoder | Config | Notes |
|---|---|---|
| `CTCGreedyDecoder` | `decoder=ctc_greedy` | Fast, no dependencies |
| `CTCBeamDecoder` | `decoder=ctc_beam` | Requires kenlm, uses 6-gram char LM |
