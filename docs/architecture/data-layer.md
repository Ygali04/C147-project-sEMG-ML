# Data Layer

This page describes how raw HDF5 session files are loaded and served to the
training pipeline.

---

## HDF5 Session Files

Each recording session is a single HDF5 file (~150–400 MB) structured as:

```
emg2qwerty/
  timeseries       # Compound dataset with fields:
    emg_left       # (T, 16) — left wrist, 16 electrodes, 2 kHz
    emg_right      # (T, 16) — right wrist, 16 electrodes, 2 kHz
    time           # (T,) — timestamps in seconds
  attrs:
    session_name   # Unique session identifier
    user           # User ID (e.g. 89335547)
    condition      # "on_keyboard"
    duration_mins  # Session length in minutes
    keystrokes     # JSON array: [{key, start, end}, ...]
    prompts        # JSON array: [{payload: {text}, start, end}, ...]
```

---

## `EMGSessionData` (`data.py`)

Context-manager wrapper around a single HDF5 file. Provides:

- Lazy loading of EMG timeseries and keystrokes
- `ground_truth()` — returns keystroke labels as a `LabelData` named tuple
- Indexing `session[start:end]` returns a structured NumPy array

---

## `WindowedEMGDataset` (`data.py`)

A `torch.utils.data.Dataset` that:

1. Loads an HDF5 session via `EMGSessionData`
2. Segments the continuous EMG stream into fixed-length windows
   (`window_length=8000` samples = 4 seconds at 2 kHz)
3. Adds configurable padding for past/future context (`[1800, 200]` →
   900 ms past, 100 ms future)
4. Returns `(emg_window, labels, input_lengths, label_lengths)` tuples

---

## `WindowedEMGDataModule` (`lightning.py`)

PyTorch Lightning `LightningDataModule` that:

- Accepts train/val/test session lists (from Hydra config)
- Instantiates `WindowedEMGDataset` for each split
- Applies per-split transforms (augmentation for train, none for val/test)
- Creates `DataLoader` instances with collation
