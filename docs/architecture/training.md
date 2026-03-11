# Training Loop

---

## Hydra Configuration System

All training parameters are managed through Hydra YAML configs under `config/`:

```
config/
  base.yaml                              # Top-level defaults + trainer settings
  model/tds_conv_ctc.yaml                # Model architecture
  optimizer/adam.yaml                     # Adam: lr=1e-3
  lr_scheduler/
    linear_warmup_cosine_annealing.yaml  # 10-epoch warmup, cosine decay
    cosine_annealing.yaml
    reduce_on_plateau.yaml
    step.yaml
  decoder/
    ctc_greedy.yaml                      # Default: no LM
    ctc_beam.yaml                        # KenLM beam search
  transforms/log_spectrogram.yaml        # STFT + augmentation
  user/
    single_user.yaml                     # Baseline user (89335547)
    generic.yaml                         # Multi-user training
  cluster/
    local.yaml                           # Single-machine training
    slurm.yaml                           # SLURM cluster
```

---

## Entry Point (`train.py`)

```bash
uv run python -m emg2qwerty.train [HYDRA_OVERRIDES...]
```

This:

1. Loads the composed Hydra config
2. Instantiates the `WindowedEMGDataModule` and `TDSConvCTCModule`
3. Creates a `pl.Trainer` with configured callbacks (LR monitor, checkpointing)
4. Calls `trainer.fit()` if `train=True`, then `trainer.test()`

---

## Key Training Parameters (`config/base.yaml`)

| Parameter | Default | Description |
|---|---|---|
| `seed` | 1501 | Random seed for reproducibility |
| `batch_size` | 32 | Mini-batch size |
| `num_workers` | 4 | DataLoader worker processes |
| `trainer.max_epochs` | 150 | Training epochs |
| `trainer.accelerator` | `gpu` | Device type |
| `monitor_metric` | `val/CER` | Checkpoint selection metric |
| `dataset.root` | `data` | HDF5 file root directory |

---

## Optimizer & LR Schedule

- **Optimizer**: Adam with `lr=1e-3`
- **LR Schedule**: Linear warmup (10 epochs, from `1e-8`) → cosine annealing to `1e-6`
