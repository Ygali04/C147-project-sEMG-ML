#!/usr/bin/env bash
# wave1_inference_sweep.sh — Launch the first 8-GPU inference-policy sweep.
#
# Run this on the remote RTX PRO 6000 instance after syncing the repo and
# populating the required checkpoint environment variables.
#
# Required env vars:
#   TDS_CKPT
#   LARGE_TRANSFORMER_CKPT
#   SMALL_TRANSFORMER_CKPT
#   WHISPER_CKPT
#
set -euo pipefail

cd /root/emg2qwerty
export PATH="$HOME/.local/bin:$PATH"

: "${TDS_CKPT:?Set TDS_CKPT to the baseline checkpoint path}"
: "${LARGE_TRANSFORMER_CKPT:?Set LARGE_TRANSFORMER_CKPT to the large transformer checkpoint path}"
: "${SMALL_TRANSFORMER_CKPT:?Set SMALL_TRANSFORMER_CKPT to the small transformer checkpoint path}"
: "${WHISPER_CKPT:?Set WHISPER_CKPT to the Whisper checkpoint path}"

COMMON="user=single_user train=False ~cluster trainer.devices=1"

run_eval() {
    local gpu="$1"
    local name="$2"
    local model="$3"
    local inference_cfg="$4"
    local checkpoint_path="$5"
    shift 5
    local extra=("$@")

    local checkpoint_link="/root/${name}.ckpt"
    ln -sf "$checkpoint_path" "$checkpoint_link"

    echo "[GPU ${gpu}] ${name}"
    CUDA_VISIBLE_DEVICES="${gpu}" nohup uv run python -m emg2qwerty.train \
        ${COMMON} \
        model="${model}" \
        inference="${inference_cfg}" \
        "hydra.run.dir=logs/${name}" \
        "checkpoint=${checkpoint_link}" \
        "${extra[@]}" \
        > "/root/${name}.log" 2>&1 &
    echo "  PID=$!"
}

echo "=== Wave 1: Inference Policy Sweep ==="

run_eval 0 wave1_gpu0_large_full_session t5_ctc full_session \
    "$LARGE_TRANSFORMER_CKPT" \
    module.use_cnn=true module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024

run_eval 1 wave1_gpu1_large_chunk_decode t5_ctc windowed_chunk_decode \
    "$LARGE_TRANSFORMER_CKPT" \
    module.use_cnn=true module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024

run_eval 2 wave1_gpu2_large_logits_merge t5_ctc windowed_logits_merge \
    "$LARGE_TRANSFORMER_CKPT" \
    module.use_cnn=true module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024

run_eval 3 wave1_gpu3_small_logits_merge t5_ctc windowed_logits_merge \
    "$SMALL_TRANSFORMER_CKPT" \
    module.use_cnn=true module.d_model=128 module.num_layers=4 module.num_heads=4 module.d_ff=512

run_eval 4 wave1_gpu4_whisper_logits_merge whisper_ctc windowed_logits_merge \
    "$WHISPER_CKPT"

run_eval 5 wave1_gpu5_tds_logits_merge tds_conv_ctc windowed_logits_merge \
    "$TDS_CKPT"

run_eval 6 wave1_gpu6_large_logits_stride4k t5_ctc windowed_logits_merge \
    "$LARGE_TRANSFORMER_CKPT" \
    module.use_cnn=true module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024 \
    inference.stride=4000

run_eval 7 wave1_gpu7_large_logits_trim512 t5_ctc windowed_logits_merge \
    "$LARGE_TRANSFORMER_CKPT" \
    module.use_cnn=true module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024 \
    inference.trim_margin=512

echo
echo "Wave 1 launched. Monitor with:"
echo "  tail -f /root/wave1_gpu*.log"
