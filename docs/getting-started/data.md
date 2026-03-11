# Dataset

## emg2qwerty HDF5 dataset

The project uses the **emg2qwerty** dataset — surface EMG recordings from wrist
electrodes captured while subjects typed on a QWERTY keyboard.

**Dataset stats:**

- 1,135 session files spanning **108 users** and **346 hours** of recording
- Format: HDF5 (`.hdf5`), one file per session
- EMG: 2 kHz, 16 channels per wrist (left + right)
- Labels: keylogger-recorded keystroke timestamps

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

Download the dataset from UCLA Box:

> **[UCLA Box — emg2qwerty dataset](https://ucla.box.com/s/3xc4nwpfjfpo6ydjs94t0v2kuq37d5eg)**

Place the downloaded `.hdf5` files under `data/` at the project root:

```
C147-project-sEMG-ML/
  data/
    <session_id>.hdf5
    ...
    metadata.csv
```

The `data/` directory is git-ignored.

## Subject

We primarily work with **subject #89335547**.

## Train / Val / Test split

Splits are defined in `config/user/single_user.yaml`.
Each entry lists a `session` ID; the loader resolves these to
`<dataset.root>/<session>.hdf5` paths.

```bash
# Print dataset statistics
uv run python scripts/print_dataset_stats.py

# Re-generate splits
uv run python scripts/generate_splits.py
```

## Loading data in Python

```python
from emg2qwerty.data import EMGSessionData

with EMGSessionData("data/89335547_session1.hdf5") as session:
    print(session)           # EMGSessionData: (N samples, K keystrokes, T mins)
    window = session[0:2000] # Structured numpy array: emg_left, emg_right, time
    labels = session.ground_truth()
    print(labels.text)       # e.g. "the quick brown"
```
