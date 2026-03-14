#!/usr/bin/env bash
# wave10_eval_and_retrain.sh — 8-GPU eval sweep + retraining of failed lanes
#
# GPUs 0-5: Windowed inference eval on best checkpoints (no training)
# GPUs 6-7: Retrain the failed transformer runs (ALiBi + Large)
#
set -euo pipefail

cd /root/emg2qwerty
export PATH="$HOME/.local/bin:$PATH"

COMMON_EVAL="user=single_user train=False ~cluster trainer.devices=1 trainer.accelerator=gpu trainer.strategy=auto"
COMMON_TRAIN="user=single_user ~cluster trainer.devices=1 trainer.accelerator=gpu trainer.strategy=auto"

# Best checkpoints from wave-9
CNN_BILSTM_CKPT="/root/emg2qwerty/logs/wave9_gpu0_cnn_bilstm/checkpoints/epoch=130-step=15720.ckpt"
HYBRID_CKPT="/root/emg2qwerty/logs/wave9_gpu1_hybrid/checkpoints/epoch=148-step=17880.ckpt"
SMALL_TRANS_CKPT="/root/emg2qwerty/logs/wave9_gpu4_small_transformer/checkpoints/epoch=135-step=16320.ckpt"
WHISPER_CKPT="/root/emg2qwerty/logs/wave0_gpu3_whisper_control/checkpoints/epoch=139-step=16800.ckpt"
CONFORMER_SMALL_CKPT="/root/emg2qwerty/logs/wave9_gpu2_conformer_small/checkpoints/epoch=148-step=17880.ckpt"

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Wave 10: Eval Sweep + Retrain Failed Lanes              ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# GPU 0: CNN-BiLSTM with windowed_logits_merge (should be ~14.96 or better)
echo "[GPU 0] CNN-BiLSTM + windowed_logits_merge"
CUDA_VISIBLE_DEVICES=0 nohup uv run python -m emg2qwerty.train \
    $COMMON_EVAL model=cnn_bilstm_ctc \
    inference=windowed_logits_merge \
    "checkpoint=$CNN_BILSTM_CKPT" \
    "hydra.run.dir=logs/wave10_gpu0_cnn_bilstm_merge" \
    > /root/wave10_gpu0_cnn_bilstm_merge.log 2>&1 &
echo "  PID=$!"

# GPU 1: Hybrid with windowed_logits_merge (should close 38% -> ~15%)
echo "[GPU 1] Hybrid + windowed_logits_merge"
CUDA_VISIBLE_DEVICES=1 nohup uv run python -m emg2qwerty.train \
    $COMMON_EVAL model=cnn_bilstm_transformer_ctc \
    inference=windowed_logits_merge \
    "checkpoint=$HYBRID_CKPT" \
    "hydra.run.dir=logs/wave10_gpu1_hybrid_merge" \
    > /root/wave10_gpu1_hybrid_merge.log 2>&1 &
echo "  PID=$!"

# GPU 2: Small Transformer with windowed_logits_merge (should close 87% -> ~20%)
echo "[GPU 2] Small Transformer + windowed_logits_merge"
CUDA_VISIBLE_DEVICES=2 nohup uv run python -m emg2qwerty.train \
    $COMMON_EVAL model=t5_ctc \
    inference=windowed_logits_merge \
    module.use_cnn=true module.d_model=128 module.num_layers=4 module.num_heads=4 module.d_ff=512 \
    module.blank_penalty_epochs=0 module.temporal_stride=1 module.positional_encoding=sinusoidal \
    "checkpoint=$SMALL_TRANS_CKPT" \
    "hydra.run.dir=logs/wave10_gpu2_small_trans_merge" \
    > /root/wave10_gpu2_small_trans_merge.log 2>&1 &
echo "  PID=$!"

# GPU 3: Hybrid with windowed_chunk_decode (compare merge vs chunk decode)
echo "[GPU 3] Hybrid + windowed_chunk_decode"
CUDA_VISIBLE_DEVICES=3 nohup uv run python -m emg2qwerty.train \
    $COMMON_EVAL model=cnn_bilstm_transformer_ctc \
    inference=windowed_chunk_decode \
    "checkpoint=$HYBRID_CKPT" \
    "hydra.run.dir=logs/wave10_gpu3_hybrid_chunk" \
    > /root/wave10_gpu3_hybrid_chunk.log 2>&1 &
echo "  PID=$!"

# GPU 4: Small Transformer with windowed_chunk_decode
echo "[GPU 4] Small Transformer + windowed_chunk_decode"
CUDA_VISIBLE_DEVICES=4 nohup uv run python -m emg2qwerty.train \
    $COMMON_EVAL model=t5_ctc \
    inference=windowed_chunk_decode \
    module.use_cnn=true module.d_model=128 module.num_layers=4 module.num_heads=4 module.d_ff=512 \
    module.blank_penalty_epochs=0 module.temporal_stride=1 module.positional_encoding=sinusoidal \
    "checkpoint=$SMALL_TRANS_CKPT" \
    "hydra.run.dir=logs/wave10_gpu4_small_trans_chunk" \
    > /root/wave10_gpu4_small_trans_chunk.log 2>&1 &
echo "  PID=$!"

# GPU 5: Whisper with windowed_logits_merge (test if windowing fixes 100%)
echo "[GPU 5] Whisper + windowed_logits_merge"
CUDA_VISIBLE_DEVICES=5 nohup uv run python -m emg2qwerty.train \
    $COMMON_EVAL model=whisper_ctc \
    inference=windowed_logits_merge \
    "checkpoint=$WHISPER_CKPT" \
    "hydra.run.dir=logs/wave10_gpu5_whisper_merge" \
    > /root/wave10_gpu5_whisper_merge.log 2>&1 &
echo "  PID=$!"

# GPU 6: Retrain Small Transformer + ALiBi (was missing due to PATH issue)
echo "[GPU 6] RETRAIN: Small Transformer + ALiBi (150ep)"
CUDA_VISIBLE_DEVICES=6 nohup uv run python -m emg2qwerty.train \
    $COMMON_TRAIN model=t5_ctc trainer.max_epochs=150 \
    module.use_cnn=true module.d_model=128 module.num_layers=4 module.num_heads=4 module.d_ff=512 \
    module.blank_penalty_epochs=0 module.temporal_stride=1 module.positional_encoding=alibi \
    "hydra.run.dir=logs/wave10_gpu6_small_trans_alibi" \
    > /root/wave10_gpu6_small_trans_alibi.log 2>&1 &
echo "  PID=$!"

# GPU 7: Retrain Large Transformer (was missing due to PATH issue)
echo "[GPU 7] RETRAIN: Large Transformer (150ep)"
CUDA_VISIBLE_DEVICES=7 nohup uv run python -m emg2qwerty.train \
    $COMMON_TRAIN model=t5_ctc trainer.max_epochs=150 \
    module.use_cnn=true module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024 \
    module.blank_penalty_epochs=0 module.temporal_stride=1 module.positional_encoding=sinusoidal \
    "hydra.run.dir=logs/wave10_gpu7_large_trans" \
    > /root/wave10_gpu7_large_trans.log 2>&1 &
echo "  PID=$!"

echo ""
echo "Wave 10 launched. Monitor with:"
echo "  tail -f /root/wave10_gpu*.log"
