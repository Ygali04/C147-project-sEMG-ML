# Transform Pipeline

Transforms are defined in `config/transforms/log_spectrogram.yaml` and
implemented in `src/emg2qwerty/transforms.py`.

---

## Training Transforms

```
ToTensor                    # Convert numpy arrays to torch tensors
  ↓
RandomBandRotation          # Shift electrode channels by (-1, 0, +1)
  ↓
TemporalAlignmentJitter     # Random ±60 ms temporal offset
  ↓
LogSpectrogram              # STFT: n_fft=64, hop=16 → (T', 33) at 125 Hz
  ↓
SpecAugment                 # Mask 3 time bands + 2 freq bands
```

---

## Validation / Test Transforms

```
ToTensor → LogSpectrogram   # No augmentation
```

---

## Key Parameters

| Parameter | Value | Meaning |
|---|---|---|
| `n_fft` | 64 | STFT window size (32 ms at 2 kHz) |
| `hop_length` | 16 | STFT hop (8 ms) → output rate = 125 Hz |
| `n_time_masks` | 3 | SpecAugment: number of time masks |
| `time_mask_param` | 25 | Max width per time mask (200 ms at 125 Hz) |
| `n_freq_masks` | 2 | SpecAugment: number of frequency masks |
| `freq_mask_param` | 4 | Max width per frequency mask |
