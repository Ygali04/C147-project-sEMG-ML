# Batched Training

The training script (`scripts/train_batched.py`) orchestrates model training
across multiple user profiles stored in Backblaze B2.

## Usage

```bash
# Train on baseline profile (user 89335547)
uv run python scripts/train_batched.py --baseline

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
directory using `rclone sync`. Files already present locally are skipped.

### 4. Training

The existing Hydra-based training entry-point (`emg2qwerty.train`) is invoked
with appropriate overrides for each user profile, using `OmegaConf.merge`
and `OmegaConf.from_dotlist` to compose the final config.

## Configuration

Training hyperparameters are controlled by the existing Hydra config system:

| Config | Purpose |
|---|---|
| `config/base.yaml` | Batch size, epochs, seed, logging |
| `config/model/tds_conv_ctc.yaml` | TDS-CNN architecture |
| `config/optimizer/adam.yaml` | Adam optimizer (lr=1e-3) |
| `config/lr_scheduler/*.yaml` | LR schedule options |
| `config/decoder/*.yaml` | Greedy vs beam decoder |
| `config/user/*.yaml` | Single-user vs generic splits |

The batched trainer passes user-specific overrides to Hydra automatically:

```yaml
# These are injected per-profile:
dataset.root: data/<user_id>
user: single_user       # or the appropriate user config
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
```

## Monitoring

Training progress is logged to the console with:

- Per-epoch train loss, validation CER
- Learning rate schedule
- Checkpoint save events

For experiment tracking, PyTorch Lightning supports multiple loggers
(TensorBoard, W&B, etc.) configured through Hydra.
