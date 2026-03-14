#!/usr/bin/env bash
# wave1b_transformer_eval.sh — Transformer-only single-GPU inference comparison.
#
# Each run uses exactly one visible GPU and Lightning is pinned to
# trainer.devices=1, so no DDP or multi-GPU strategy is involved.
set -euo pipefail

cd /root/emg2qwerty
export PATH="$HOME/.local/bin:$PATH"

bash scripts/link_wave0_checkpoints.sh

COMMON="user=single_user train=False ~cluster trainer.accelerator=gpu trainer.devices=1 trainer.strategy=auto"

run_eval() {
    local gpu="$1"
    local name="$2"
    shift 2
    local extra=("$@")

    echo "[GPU ${gpu}] ${name}"
    CUDA_VISIBLE_DEVICES="${gpu}" nohup /root/.local/bin/uv run python -m emg2qwerty.train \
        ${COMMON} \
        "hydra.run.dir=logs/${name}" \
        "${extra[@]}" \
        > "/root/${name}.log" 2>&1 &
    echo "  PID=$!"
}

echo "=== Wave 1B: Transformer Inference Comparison ==="

run_eval 4 wave1b_gpu4_large_full \
    model=t5_ctc inference=full_session checkpoint=/root/checkpoints/transformer_large_best.ckpt \
    module.use_cnn=true module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024

run_eval 5 wave1b_gpu5_large_logits_merge \
    model=t5_ctc inference=windowed_logits_merge checkpoint=/root/checkpoints/transformer_large_best.ckpt \
    module.use_cnn=true module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024

run_eval 6 wave1b_gpu6_small_full \
    model=t5_ctc inference=full_session checkpoint=/root/checkpoints/transformer_small_best.ckpt \
    module.use_cnn=true module.d_model=128 module.num_layers=4 module.num_heads=4 module.d_ff=512

run_eval 7 wave1b_gpu7_small_logits_merge \
    model=t5_ctc inference=windowed_logits_merge checkpoint=/root/checkpoints/transformer_small_best.ckpt \
    module.use_cnn=true module.d_model=128 module.num_layers=4 module.num_heads=4 module.d_ff=512

echo
echo "Wave 1B launched. Monitor with:"
echo "  tail -f /root/wave1b_gpu*.log"
