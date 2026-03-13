#!/usr/bin/env bash
# eval_sweep.sh — Run test evaluation on all sweep checkpoints.
#
# Usage (on Verda):
#     bash scripts/eval_sweep.sh
#
set -euo pipefail
cd /root/emg2qwerty
export PATH="$HOME/.local/bin:$PATH"

COMMON="user=single_user ~cluster trainer.devices=1 train=False"

declare -A NAMES
declare -A CKPTS
declare -A OVERRIDES

# GPU 0: Baseline TDS-ConvNet
NAMES[0]="baseline_tds"
CKPTS[0]="/root/emg2qwerty/logs/2026-03-13/13-17-38/checkpoints/epoch=124-step=15000.ckpt"
OVERRIDES[0]="model=tds_conv_ctc"

# GPU 1: Pure Transformer (no CNN, no penalty)
NAMES[1]="transformer_pure"
CKPTS[1]="/root/emg2qwerty/logs/2026-03-13/13-39-08/checkpoints/epoch=79-step=9600.ckpt"
OVERRIDES[1]="model=t5_ctc module.use_cnn=false module.blank_penalty_epochs=0"

# GPU 2: Transformer + CNN (no penalty)
NAMES[2]="transformer_cnn"
CKPTS[2]="/root/emg2qwerty/logs/2026-03-13/13-36-28/checkpoints/epoch=77-step=9360.ckpt"
OVERRIDES[2]="model=t5_ctc module.use_cnn=true module.blank_penalty_epochs=0"

# GPU 3: Transformer + penalty (no CNN)
NAMES[3]="transformer_penalty"
CKPTS[3]="/root/emg2qwerty/logs/2026-03-13/13-39-08/checkpoints/epoch=74-step=9000.ckpt"
OVERRIDES[3]="model=t5_ctc module.use_cnn=false module.blank_penalty_epochs=40 module.blank_alpha_max=50.0"

# GPU 4: Transformer + CNN + penalty (full)
NAMES[4]="transformer_full"
CKPTS[4]="/root/emg2qwerty/logs/2026-03-13/13-36-28/checkpoints/epoch=78-step=9480.ckpt"
OVERRIDES[4]="model=t5_ctc module.use_cnn=true module.blank_penalty_epochs=40 module.blank_alpha_max=50.0"

# GPU 5: Large Transformer + CNN
NAMES[5]="transformer_large"
CKPTS[5]="/root/emg2qwerty/logs/2026-03-13/13-36-28/checkpoints/epoch=76-step=9240.ckpt"
OVERRIDES[5]="model=t5_ctc module.use_cnn=true module.blank_penalty_epochs=0 module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024"

# GPU 6: Tiny Transformer (did not converge)
NAMES[6]="transformer_tiny"
CKPTS[6]="/root/emg2qwerty/logs/2026-03-13/13-39-08/checkpoints/epoch=1-step=240-v1.ckpt"
OVERRIDES[6]="model=t5_ctc module.use_cnn=false module.blank_penalty_epochs=0 module.d_model=64 module.num_layers=2 module.num_heads=2 module.d_ff=256"

# GPU 7: Transformer + CNN, higher LR
NAMES[7]="transformer_cnn_highlr"
CKPTS[7]="/root/emg2qwerty/logs/2026-03-13/13-36-28/checkpoints/epoch=78-step=9480-v1.ckpt"
OVERRIDES[7]="model=t5_ctc module.use_cnn=true module.blank_penalty_epochs=0 optimizer.lr=5e-4"

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Test Evaluation — 8 models on GPUs 0-7                  ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

for gpu in 0 1 2 3 4 5 6 7; do
    name="${NAMES[$gpu]}"
    ckpt="${CKPTS[$gpu]}"
    ovr="${OVERRIDES[$gpu]}"
    logfile="/root/eval_${name}.log"

    echo "[GPU $gpu] Evaluating $name"
    echo "  Checkpoint: $ckpt"
    echo "  Log: $logfile"

    # Symlink the checkpoint to a simple name to avoid Hydra '=' parsing issues
    simple_ckpt="/root/eval_ckpt_${name}.ckpt"
    ln -sf "$ckpt" "$simple_ckpt"

    CUDA_VISIBLE_DEVICES=$gpu nohup uv run python -m emg2qwerty.train \
        $COMMON $ovr \
        checkpoint="$simple_ckpt" \
        > "$logfile" 2>&1 &
    echo "  PID=$!"
    echo ""
done

echo "All 8 evaluations launched. Monitor with:"
echo "  tail -f /root/eval_*.log"
echo "  # Results will show val_metrics and test_metrics at the end of each log."
