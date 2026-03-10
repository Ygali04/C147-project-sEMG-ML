# Dataset

## emg2qwerty HDF5 dataset

The project uses the **emg2qwerty** dataset — surface EMG recordings from wrist
electrodes captured while subjects typed on a QWERTY keyboard.

Each session is stored as an **HDF5** (`.hdf5`) file containing:

- `emg`: multi-channel sEMG signal array (electrodes × time steps)
- `labels`: ground-truth keystroke sequence

## Download

Download the dataset from UCLA Box:

> **[UCLA Box — emg2qwerty dataset](https://ucla.box.com/s/3xc4nwpfjfpo6ydjs94t0v2kuq37d5eg)**

Place the downloaded `.hdf5` files under a `data/` directory at the project root
(this path is git-ignored).

## Subject

We primarily work with **subject #89335547**. The dataset includes multiple
recording sessions for this subject.

## Train / Val / Test split

The train/validation/test split is defined in the YAML config files under
`config/`. See the Hydra experiment configs for the exact file lists per split.
