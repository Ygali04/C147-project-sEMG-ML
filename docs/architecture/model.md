# Model Architecture

---

## TDS-CNN Baseline (`modules.py`, `lightning.py`)

The default model is a Time-Depth Separable CNN, following
[Hannun et al. (2019)](https://arxiv.org/abs/1904.02619):

```
Input: (T, N, bands=2, C=16, freq=33)       # LogSpectrogram of L+R EMG
  ↓
SpectrogramNorm                              # BatchNorm2d per band×channel
  ↓
MultiBandRotationInvariantMLP                # MLP + electrode-rotation pooling
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

---

## Sub-Modules

| Module | Purpose |
|---|---|
| `SpectrogramNorm` | `BatchNorm2d` over `bands × channels` = 32 feature maps |
| `RotationInvariantMLP` | Shifts electrodes by each offset `(-1, 0, 1)`, runs MLP, mean-pools — robust to electrode placement |
| `MultiBandRotationInvariantMLP` | Applies `RotationInvariantMLP` independently to left and right wrist bands |
| `TDSConv2dBlock` | `Conv2d(C, C, (1, K))` → ReLU → skip connection → LayerNorm |
| `TDSFullyConnectedBlock` | `Linear → ReLU → Linear` → skip → LayerNorm |
| `TDSConvEncoder` | Alternating stack of conv + FC blocks |

---

## Hyperparameters (`config/model/tds_conv_ctc.yaml`)

```yaml
module:
  _target_: emg2qwerty.lightning.TDSConvCTCModule
  in_features: 528          # freq × channels
  mlp_features: [384]
  block_channels: [24, 24, 24, 24]
  kernel_width: 32           # Temporal receptive field

datamodule:
  _target_: emg2qwerty.lightning.WindowedEMGDataModule
  window_length: 8000        # 4 sec at 2 kHz
  padding: [1800, 200]       # 900 ms past + 100 ms future
```
