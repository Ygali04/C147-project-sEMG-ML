#!/usr/bin/env bash
# sweep_gpus.sh — Launch architecture sweep on Verda (8×H200).
#
# GPU 0: baseline TDS-ConvNet (already running — skip)
# GPU 1: Transformer (no CNN, no blank penalty) — pure transformer
# GPU 2: Transformer + CNN (no blank penalty)
# GPU 3: Transformer + blank penalty (no CNN)
# GPU 4: Transformer + CNN + blank penalty (full)
# GPU 5: Transformer large (d=256, 6 layers, 8 heads)
# GPU 6: Transformer tiny (d=64, 2 layers, 2 heads)
# GPU 7: Transformer + CNN, higher LR (1e-3)
#
# Usage (run on Verda after syncing code):
#     bash scripts/sweep_gpus.sh
#
# To monitor:
#     tail -f /root/sweep_gpu*.log
#
set -euo pipefail

cd /root/emg2qwerty
export PATH="$HOME/.local/bin:$PATH"

COMMON="user=single_user model=t5_ctc ~cluster trainer.devices=1 trainer.max_epochs=80"

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Architecture Sweep — 7 experiments on GPUs 1-7          ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# GPU 1: Pure Transformer (no CNN, no blank penalty)
echo "[GPU 1] Pure Transformer (no CNN, no blank penalty)"
CUDA_VISIBLE_DEVICES=1 nohup uv run python -m emg2qwerty.train \
    $COMMON \
    module.use_cnn=false \
    module.blank_penalty_epochs=0 \
    > /root/sweep_gpu1_transformer_pure.log 2>&1 &
echo "  PID=$!"

# GPU 2: Transformer + CNN (no blank penalty)
echo "[GPU 2] Transformer + CNN (no blank penalty)"
CUDA_VISIBLE_DEVICES=2 nohup uv run python -m emg2qwerty.train \
    $COMMON \
    module.use_cnn=true \
    module.blank_penalty_epochs=0 \
    > /root/sweep_gpu2_transformer_cnn.log 2>&1 &
echo "  PID=$!"

# GPU 3: Transformer + blank penalty (no CNN)
echo "[GPU 3] Transformer + blank penalty (no CNN)"
CUDA_VISIBLE_DEVICES=3 nohup uv run python -m emg2qwerty.train \
    $COMMON \
    module.use_cnn=false \
    module.blank_penalty_epochs=40 \
    module.blank_alpha_max=50.0 \
    > /root/sweep_gpu3_transformer_penalty.log 2>&1 &
echo "  PID=$!"

# GPU 4: Transformer + CNN + blank penalty (full config)
echo "[GPU 4] Transformer + CNN + blank penalty (full)"
CUDA_VISIBLE_DEVICES=4 nohup uv run python -m emg2qwerty.train \
    $COMMON \
    module.use_cnn=true \
    module.blank_penalty_epochs=40 \
    module.blank_alpha_max=50.0 \
    > /root/sweep_gpu4_transformer_full.log 2>&1 &
echo "  PID=$!"

# GPU 5: Large Transformer (d=256, 6 layers, 8 heads, d_ff=1024)
echo "[GPU 5] Large Transformer (d=256, 6L, 8H)"
CUDA_VISIBLE_DEVICES=5 nohup uv run python -m emg2qwerty.train \
    $COMMON \
    module.use_cnn=true \
    module.blank_penalty_epochs=0 \
    module.d_model=256 \
    module.num_layers=6 \
    module.num_heads=8 \
    module.d_ff=1024 \
    > /root/sweep_gpu5_transformer_large.log 2>&1 &
echo "  PID=$!"

# GPU 6: Tiny Transformer (d=64, 2 layers, 2 heads, d_ff=256)
echo "[GPU 6] Tiny Transformer (d=64, 2L, 2H)"
CUDA_VISIBLE_DEVICES=6 nohup uv run python -m emg2qwerty.train \
    $COMMON \
    module.use_cnn=false \
    module.blank_penalty_epochs=0 \
    module.d_model=64 \
    module.num_layers=2 \
    module.num_heads=2 \
    module.d_ff=256 \
    > /root/sweep_gpu6_transformer_tiny.log 2>&1 &
echo "  PID=$!"

# GPU 7: Transformer + CNN, higher LR (1e-3 instead of default 1e-3)
echo "[GPU 7] Transformer + CNN, LR=5e-4"
CUDA_VISIBLE_DEVICES=7 nohup uv run python -m emg2qwerty.train \
    $COMMON \
    module.use_cnn=true \
    module.blank_penalty_epochs=0 \
    optimizer.lr=5e-4 \
    > /root/sweep_gpu7_transformer_cnn_highlr.log 2>&1 &
echo "  PID=$!"

echo ""
echo "All 7 experiments launched. Monitor with:"
echo "  tail -f /root/sweep_gpu*.log"
echo "  # Or check CER progress:"
echo "  grep 'reached' /root/sweep_gpu*.log"
