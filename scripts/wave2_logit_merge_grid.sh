#!/usr/bin/env bash
# wave2_logit_merge_grid.sh — Benchmark logit-merge hyperparameters on 8 GPUs.
#
# Required env vars:
#   LARGE_TRANSFORMER_CKPT
#   SMALL_TRANSFORMER_CKPT
#
set -euo pipefail

cd /root/emg2qwerty
export PATH="$HOME/.local/bin:$PATH"

: "${LARGE_TRANSFORMER_CKPT:?Set LARGE_TRANSFORMER_CKPT to the large transformer checkpoint path}"
: "${SMALL_TRANSFORMER_CKPT:?Set SMALL_TRANSFORMER_CKPT to the small transformer checkpoint path}"

COMMON="user=single_user train=False ~cluster trainer.devices=1 inference=windowed_logits_merge"

run_eval() {
    local gpu="$1"
    local name="$2"
    local checkpoint_path="$3"
    shift 3
    local extra=("$@")

    local checkpoint_link="/root/${name}.ckpt"
    ln -sf "$checkpoint_path" "$checkpoint_link"

    echo "[GPU ${gpu}] ${name}"
    CUDA_VISIBLE_DEVICES="${gpu}" nohup uv run python -m emg2qwerty.train \
        ${COMMON} \
        "hydra.run.dir=logs/${name}" \
        "checkpoint=${checkpoint_link}" \
        "${extra[@]}" \
        > "/root/${name}.log" 2>&1 &
    echo "  PID=$!"
}

echo "=== Wave 2: Logit Merge Grid ==="

run_eval 0 wave2_gpu0_large_flat_stride8k "$LARGE_TRANSFORMER_CKPT" \
    model=t5_ctc module.use_cnn=true module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024 \
    inference.window_length=8000 inference.stride=8000 inference.trim_margin=0 inference.merge_mode=flat

run_eval 1 wave2_gpu1_large_flat_stride4k "$LARGE_TRANSFORMER_CKPT" \
    model=t5_ctc module.use_cnn=true module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024 \
    inference.window_length=8000 inference.stride=4000 inference.trim_margin=0 inference.merge_mode=flat

run_eval 2 wave2_gpu2_large_flat_trim256 "$LARGE_TRANSFORMER_CKPT" \
    model=t5_ctc module.use_cnn=true module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024 \
    inference.window_length=8000 inference.stride=4000 inference.trim_margin=256 inference.merge_mode=flat

run_eval 3 wave2_gpu3_large_triangular_trim256 "$LARGE_TRANSFORMER_CKPT" \
    model=t5_ctc module.use_cnn=true module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024 \
    inference.window_length=8000 inference.stride=4000 inference.trim_margin=256 inference.merge_mode=triangular

run_eval 4 wave2_gpu4_large_triangular_trim512 "$LARGE_TRANSFORMER_CKPT" \
    model=t5_ctc module.use_cnn=true module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024 \
    inference.window_length=8000 inference.stride=4000 inference.trim_margin=512 inference.merge_mode=triangular

run_eval 5 wave2_gpu5_small_flat_stride8k "$SMALL_TRANSFORMER_CKPT" \
    model=t5_ctc module.use_cnn=true module.d_model=128 module.num_layers=4 module.num_heads=4 module.d_ff=512 \
    inference.window_length=8000 inference.stride=8000 inference.trim_margin=0 inference.merge_mode=flat

run_eval 6 wave2_gpu6_small_triangular_trim256 "$SMALL_TRANSFORMER_CKPT" \
    model=t5_ctc module.use_cnn=true module.d_model=128 module.num_layers=4 module.num_heads=4 module.d_ff=512 \
    inference.window_length=8000 inference.stride=4000 inference.trim_margin=256 inference.merge_mode=triangular

run_eval 7 wave2_gpu7_small_triangular_trim512 "$SMALL_TRANSFORMER_CKPT" \
    model=t5_ctc module.use_cnn=true module.d_model=128 module.num_layers=4 module.num_heads=4 module.d_ff=512 \
    inference.window_length=8000 inference.stride=4000 inference.trim_margin=512 inference.merge_mode=triangular

echo
echo "Wave 2 launched. Monitor with:"
echo "  tail -f /root/wave2_gpu*.log"
