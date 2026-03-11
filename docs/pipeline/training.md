# Batched Training

The training script (`scripts/train_batched.py`) orchestrates model training across multiple user profiles stored in Backblaze B2.

## Usage

```bash
# Train on baseline profile (user 89335547)
uv run python scripts/train_batched.py --baseline

# Train on first 10 profiles from the registry
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

1. **Registry scan** — The trainer reads the B2 file registry to discover which user profiles and sessions are available.

2. **Profile filtering** — Based on the `--mode` flag:
    - `--baseline`: only user `89335547`
    - `--test`: first `n_test_users` users from the registry
    - `--all`: every user in the registry

3. **Data sync** — For each profile, HDF5 files are downloaded from B2 to the local `data/` directory. Files already present locally are skipped.

4. **Training** — The existing Hydra-based training entry-point (`emg2qwerty.train`) is invoked with appropriate overrides for each user profile.

## Configuration

Training hyperparameters are controlled by the existing Hydra config system:

- `config/base.yaml` — base configuration (batch size, epochs, etc.)
- `config/model/tds_conv_ctc.yaml` — model architecture
- `config/optimizer/adam.yaml` — optimizer settings
- `config/lr_scheduler/` — learning rate schedules

The batched trainer passes user-specific overrides to Hydra automatically.

## Checkpoint Management

- Checkpoints are saved to `logs/<date>/<time>/checkpoints/`
- Use `--checkpoint` to resume training from a saved checkpoint
- The `ModelCheckpoint` callback saves the best model by validation CER and the last epoch

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
