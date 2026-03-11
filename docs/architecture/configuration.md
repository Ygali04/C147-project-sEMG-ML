# Configuration Reference

---

## Hydra Config Groups

| Group | Options | Default |
|---|---|---|
| `user` | `single_user`, `generic` | `single_user` |
| `model` | `tds_conv_ctc` | `tds_conv_ctc` |
| `optimizer` | `adam` | `adam` |
| `lr_scheduler` | `linear_warmup_cosine_annealing`, `cosine_annealing`, `reduce_on_plateau`, `step` | `linear_warmup_cosine_annealing` |
| `decoder` | `ctc_greedy`, `ctc_beam` | `ctc_greedy` |
| `transforms` | `log_spectrogram` | `log_spectrogram` |
| `cluster` | `local`, `slurm` | `local` |

---

## Override Examples

```bash
# Change optimizer + LR schedule
uv run python -m emg2qwerty.train optimizer=adam lr_scheduler=cosine_annealing

# Use beam decoder
uv run python -m emg2qwerty.train decoder=ctc_beam

# Train on generic (multi-user) split
uv run python -m emg2qwerty.train user=generic

# Evaluation only, from checkpoint
uv run python -m emg2qwerty.train \
  train=False checkpoint=logs/best.ckpt decoder=ctc_greedy

# Adjust batch size + workers
uv run python -m emg2qwerty.train batch_size=64 num_workers=8
```
