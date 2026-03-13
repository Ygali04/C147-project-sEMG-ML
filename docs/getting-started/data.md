# Dataset

## emg2qwerty HDF5 Dataset

The project uses the **emg2qwerty** dataset — surface EMG recordings from wrist
electrodes captured while subjects typed on a QWERTY keyboard.

**Dataset stats:**

- 1,135 session files spanning **108 users** and **346 hours** of recording
- Format: HDF5 (`.hdf5`), one file per session
- EMG: 2 kHz, 16 channels per wrist (left + right)
- Labels: keylogger-recorded keystroke timestamps
- Total size: ~308 GB (compressed tar.gz archive)

Each HDF5 file is structured under the `emg2qwerty` group:

```
emg2qwerty/
  timeseries      # Compound dataset: emg_left (T,16), emg_right (T,16), time (T,)
  attrs:
    session_name  # Unique session ID
    user          # User ID (e.g. 89335547)
    condition     # "on_keyboard"
    duration_mins # Session length
    keystrokes    # JSON: [{key, start, end}, ...]
    prompts       # JSON: [{payload: {text}, start, end}, ...]
```

## Download

### Automated Pipeline (recommended)

Use the [data pipeline](../pipeline/index.md) to stream HDF5 files from Meta's
public S3 archive and store them in Backblaze B2:

```bash
# 1. Set up credentials
cp .env.example .env
# Edit .env with your B2 key ID and application key

# 2. Download baseline data (single user, 18 sessions)
uv run python scripts/download_data.py --baseline

# 3. Or download 10 random users for evaluation
uv run python scripts/download_data.py --test

# 4. Or use Makefile shortcuts
make download-baseline
```

See [Pipeline → Download](../pipeline/download.md) for full details and all
download modes.

!!! info "Streaming architecture"
    The Meta dataset is distributed as a single 308 GB gzip-compressed tar
    archive. Our downloader streams through it, extracting only the files
    for your selected users, and uploads them to Backblaze B2 for future access.

### Manual Download

Alternatively, download from UCLA Box:

> **[UCLA Box — emg2qwerty dataset](https://ucla.box.com/s/3xc4nwpfjfpo6ydjs94t0v2kuq37d5eg)**

Place the downloaded `.hdf5` files under `data/` at the project root:

```
C147-project-sEMG-ML/
  data/
    <user_id>/
      <session_id>.hdf5
      ...
    metadata.csv
```

The `data/` directory is git-ignored.

## Subject

We primarily work with **subject #89335547** (18 sessions across 2 days).

## Train / Val / Test Split

Splits are defined in `config/user/single_user.yaml`.
Each entry lists a `session` ID; the loader resolves these to
`<dataset.root>/<session>.hdf5` paths.

```bash
# Print dataset statistics
uv run python scripts/print_dataset_stats.py

# Re-generate splits
uv run python scripts/generate_splits.py
```

## Loading Data in Python

```python
from emg2qwerty.data import EMGSessionData

with EMGSessionData("data/89335547/session.hdf5") as session:
    print(session)           # EMGSessionData: (N samples, K keystrokes, T mins)
    window = session[0:2000] # Structured numpy array: emg_left, emg_right, time
    labels = session.ground_truth()
    print(labels.text)       # e.g. "the quick brown"
```

### Using the DataModule

```python
from hydra import compose, initialize
from emg2qwerty.lightning import WindowedEMGDataModule

with initialize(config_path="../config"):
    cfg = compose(config_name="base", overrides=["user=single_user"])

dm = WindowedEMGDataModule(cfg)
dm.setup("fit")

for batch in dm.train_dataloader():
    emg, labels, input_lengths, label_lengths = batch
    print(emg.shape)  # (batch_size, window_length, freq_bins)
    break
```
