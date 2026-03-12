# Batched Training

The training script (`scripts/train_batched.py`) orchestrates model training
across multiple user profiles stored in Backblaze B2.

## Usage

```bash
# Train on baseline profile (user 89335547)
uv run python scripts/train_batched.py --baseline

# Train baseline with BiLSTM CTC
uv run python scripts/train_batched.py --baseline --model bilstm_ctc

# Train baseline with CNN + BiLSTM CTC
uv run python scripts/train_batched.py --baseline --model cnn_bilstm_ctc

# Train on 10 random profiles from the registry
uv run python scripts/train_batched.py --test

# Train on ALL profiles in the registry
uv run python scripts/train_batched.py --all

# Resume from a checkpoint
uv run python scripts/train_batched.py --baseline --checkpoint logs/2026-03-11/best.ckpt
```

Or use the Makefile shortcuts:

```bash
make train-baseline
make train-test
make train-all
```

## How It Works

### 1. Registry Scan

The `BatchTrainer` reads the B2 file registry (`emg2qwerty_registry.json`)
to discover which user profiles and sessions are available.

### 2. Profile Filtering

Based on the `--mode` flag:

| Mode | Strategy |
|---|---|
| `--baseline` | Only user `89335547` |
| `--test` | First `n_test_users` users from the registry |
| `--all` | Every user in the registry |

### 3. Data Sync

For each profile, HDF5 files are downloaded from B2 to the local `data/`
directory using the B2 S3-compatible API (`boto3`). Files already present
locally are skipped.

### 4. Training

The existing Hydra-based training entry-point (`emg2qwerty.train`) is invoked
with profile-specific command-line overrides.

## Configuration

Training hyperparameters are controlled by the existing Hydra config system:

| Config | Purpose |
|---|---|
| `config/base.yaml` | Batch size, epochs, seed, logging |
| `config/model/tds_conv_ctc.yaml` | TDS-CNN architecture |
| `config/model/bilstm_ctc.yaml` | Bidirectional LSTM encoder |
| `config/model/cnn_bilstm_ctc.yaml` | Temporal CNN front-end + BiLSTM encoder |
| `config/model/whisper_ctc.yaml` | Whisper encoder transfer model with CTC head |
| `config/optimizer/adam.yaml` | Adam optimizer (lr=1e-3) |
| `config/lr_scheduler/*.yaml` | LR schedule options |
| `config/decoder/*.yaml` | Greedy vs beam decoder |
| `config/user/*.yaml` | Single-user vs generic splits |

The batched trainer passes user-specific overrides to Hydra automatically:

```yaml
# These are injected per-profile:
user: <user_id>
model: <model_name>
checkpoint: <checkpoint_path>   # optional
```

You can select the model from CLI:

```bash
uv run python scripts/train_batched.py --baseline --model tds_conv_ctc
uv run python scripts/train_batched.py --baseline --model bilstm_ctc
uv run python scripts/train_batched.py --baseline --model cnn_bilstm_ctc
uv run python scripts/train_batched.py --baseline --model whisper_ctc
```

## Checkpoint Management

- Checkpoints are saved to `logs/<date>/<time>/checkpoints/`
- Use `--checkpoint` to resume training from a saved checkpoint
- The `ModelCheckpoint` callback saves:
    - Best model by validation CER
    - Last epoch checkpoint

## Output Structure

```
logs/
  2026-03-11/
    14-30-00/
      checkpoints/
        epoch=42-step=1234.ckpt
        last.ckpt
      hydra_configs/
        base.yaml
        overrides.yaml
      training_progress.png
```

`training_progress.png` is generated automatically at the end of training for
all model choices (`tds_conv_ctc`, `bilstm_ctc`, `cnn_bilstm_ctc`, `whisper_ctc`).

If you change the plotting code or want to backfill older runs, regenerate the
images directly from the saved metrics CSV files:

```bash
# Rebuild all plots for one training day
uv run python -m emg2qwerty.plot_training_curves logs/2026-03-12/*

# Rebuild a single run
uv run python -m emg2qwerty.plot_training_curves logs/2026-03-12/09-20-04
```

This command scans each run directory for `lightning_logs/version_*/metrics.csv`
and writes a fresh `training_progress.png` next to the run's checkpoints and
Hydra config outputs.

## Monitoring

Training progress is logged to the console with:

- Per-epoch train loss, validation CER
- Learning rate schedule
- Checkpoint save events

For experiment tracking, PyTorch Lightning supports multiple loggers
(TensorBoard, W&B, etc.) configured through Hydra.
