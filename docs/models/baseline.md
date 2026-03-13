# TDS-CNN Baseline

> Reference: [Sequence-to-Sequence Speech Recognition with Time-Depth Separable
> Convolutions, Hannun et al. (2019)](https://arxiv.org/abs/1904.02619)

## Architecture

The baseline `TDSConvCTCModule` (defined in `src/emg2qwerty/lightning.py`) is a
stack of Time-Depth Separable convolutional blocks followed by a linear CTC head.

```
Input: (T, N, bands=2, C=16, freq=33)       # LogSpectrogram of left+right EMG
  ↓
SpectrogramNorm                              # BatchNorm2d per band×channel (32 maps)
  ↓
MultiBandRotationInvariantMLP                # MLP with electrode-rotation pooling
  → (T, N, bands=2, mlp_features=384)
  ↓
Flatten → (T, N, 768)
  ↓
TDSConvEncoder                               # 4× (TDSConv2dBlock + FCBlock)
  ↓
Linear(768, num_classes=80)
  ↓
LogSoftmax → CTC Loss
```

## Key Modules (`src/emg2qwerty/modules.py`)

### `SpectrogramNorm`

Applies `nn.BatchNorm2d` independently over each of the
`num_bands × electrode_channels` = 2 × 16 = 32 channels.

### `RotationInvariantMLP`

Shifts electrode channels by each offset in `(-1, 0, 1)` (band rotation
augmentation), runs an MLP on each shifted version, then mean-pools. This
makes the model robust to electrode placement variation across sessions.

### `MultiBandRotationInvariantMLP`

Applies `RotationInvariantMLP` independently to the left-wrist and right-wrist
bands, then concatenates the outputs.

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
  in_features: 528          # freq × channels = (n_fft//2 + 1) × 16
  mlp_features: [384]
  block_channels: [24, 24, 24, 24]
  kernel_width: 32           # Total temporal receptive field = 125 samples

datamodule:
  _target_: emg2qwerty.lightning.WindowedEMGDataModule
  window_length: 8000        # 4 sec windows at 2 kHz
  padding: [1800, 200]       # 900 ms past + 100 ms future context
```

## Decoding

Two decoders are available (`src/emg2qwerty/decoder.py`):

| Decoder | Config | Dependencies | Notes |
|---|---|---|---|
| `CTCGreedyDecoder` | `decoder=ctc_greedy` | None | Fast, default |
| `CTCBeamDecoder` | `decoder=ctc_beam` | [KenLM](https://github.com/kpu/kenlm) | 6-gram char LM, beam_size=50 |

### Greedy Decoder

Performs argmax at each timestep, collapses consecutive duplicates, and removes
blank tokens. This is the default decoder and requires no additional dependencies.

### Beam-Search Decoder

Uses a 6-gram character-level language model built from WikiText-103 to rescore
beam hypotheses. The LM biases predictions toward likely character sequences,
reducing substitution and insertion errors.

| Parameter | Default | Description |
|---|---|---|
| `beam_size` | 50 | Number of beams |
| `max_labels_per_timestep` | 10 | Labels expanded per step |
| `lm_weight` | 2.0 | LM score weight |
| `insertion_bonus` | 2.0 | Bonus for character insertions |

To use the beam decoder:

```bash
# Install kenlm (see Getting Started → Setup → KenLM)
pip install https://github.com/kpu/kenlm/archive/master.zip

# Build the 6-gram character LM from WikiText-103
./scripts/lm/build_char_lm.sh 6

# Train / evaluate with beam decoding
uv run python -m emg2qwerty.train decoder=ctc_beam ...
```

## Training

```bash
# Single-user personalized model on baseline user
uv run python -m emg2qwerty.train \
  user=single_user \
  trainer.accelerator=gpu \
  trainer.devices=1

# With custom hyperparameters
uv run python -m emg2qwerty.train \
  user=single_user \
  batch_size=64 \
  trainer.max_epochs=200 \
  optimizer.lr=5e-4
```
