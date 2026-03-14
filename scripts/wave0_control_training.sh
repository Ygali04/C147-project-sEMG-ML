#!/usr/bin/env bash
# wave0_control_training.sh — Train the control checkpoints needed by wave 1.
#
# This script assumes the baseline profile has already been synced locally via:
#   uv run python scripts/sync_baseline_from_b2.py
#
set -euo pipefail

cd /root/emg2qwerty
export PATH="$HOME/.local/bin:$PATH"

COMMON="user=single_user ~cluster trainer.devices=1"

run_train() {
    local gpu="$1"
    local name="$2"
    shift 2
    local extra=("$@")

    echo "[GPU ${gpu}] ${name}"
    CUDA_VISIBLE_DEVICES="${gpu}" nohup uv run python -m emg2qwerty.train \
        ${COMMON} \
        "${extra[@]}" \
        > "/root/${name}.log" 2>&1 &
    echo "  PID=$!"
}

echo "=== Wave 0: Control Checkpoint Training ==="

run_train 0 wave0_gpu0_tds_control \
    model=tds_conv_ctc trainer.max_epochs=150

run_train 1 wave0_gpu1_transformer_large_control \
    model=t5_ctc trainer.max_epochs=80 \
    module.use_cnn=true module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024 \
    module.blank_penalty_epochs=0

run_train 2 wave0_gpu2_transformer_small_control \
    model=t5_ctc trainer.max_epochs=80 \
    module.use_cnn=true module.d_model=128 module.num_layers=4 module.num_heads=4 module.d_ff=512 \
    module.blank_penalty_epochs=0

run_train 3 wave0_gpu3_whisper_control \
    model=whisper_ctc trainer.max_epochs=150

echo
echo "Wave 0 launched. Monitor with:"
echo "  tail -f /root/wave0_gpu*.log"
echo "  grep 'reached' /root/wave0_gpu*.log"
