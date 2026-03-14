#!/usr/bin/env bash
# wave3_variable_window_sweep.sh — Launch the first variable-length window sweep.
#
# Required env vars:
#   LARGE_TRANSFORMER_CKPT  (optional resume checkpoint/control)
#   SMALL_TRANSFORMER_CKPT  (optional resume checkpoint/control)
#
set -euo pipefail

cd /root/emg2qwerty
export PATH="$HOME/.local/bin:$PATH"

COMMON="user=single_user model=t5_ctc ~cluster trainer.devices=1 trainer.max_epochs=80"

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

echo "=== Wave 3: Variable-Length Window Sweep ==="

run_train 0 wave3_gpu0_large_fixed4s \
    module.use_cnn=true module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024 \
    datamodule.train_window_lengths=[8000] datamodule.train_window_weights=[1.0]

run_train 1 wave3_gpu1_large_4s_8s \
    module.use_cnn=true module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024 \
    datamodule.train_window_lengths=[8000,16000] datamodule.train_window_weights=[1.0,1.0]

run_train 2 wave3_gpu2_large_4s_8s_12s \
    module.use_cnn=true module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024 \
    datamodule.train_window_lengths=[8000,16000,24000] datamodule.train_window_weights=[1.0,1.0,1.0]

run_train 3 wave3_gpu3_large_4s_8s_12s_16s \
    module.use_cnn=true module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024 \
    datamodule.train_window_lengths=[8000,16000,24000,32000] datamodule.train_window_weights=[1.0,1.0,1.0,1.0]

run_train 4 wave3_gpu4_small_4s_8s_12s \
    module.use_cnn=true module.d_model=128 module.num_layers=4 module.num_heads=4 module.d_ff=512 \
    datamodule.train_window_lengths=[8000,16000,24000] datamodule.train_window_weights=[1.0,1.0,1.0]

run_train 5 wave3_gpu5_pure_4s_8s_12s \
    module.use_cnn=false module.d_model=128 module.num_layers=4 module.num_heads=4 module.d_ff=512 \
    module.blank_penalty_epochs=0 \
    datamodule.train_window_lengths=[8000,16000,24000] datamodule.train_window_weights=[1.0,1.0,1.0]

run_train 6 wave3_gpu6_whisper_4s_8s \
    model=whisper_ctc \
    datamodule.train_window_lengths=[8000,16000] datamodule.train_window_weights=[1.0,1.0]

run_train 7 wave3_gpu7_cnn_bilstm_4s_8s_12s \
    model=cnn_bilstm_ctc \
    datamodule.train_window_lengths=[8000,16000,24000] datamodule.train_window_weights=[1.0,1.0,1.0]

echo
echo "Wave 3 launched. Monitor with:"
echo "  tail -f /root/wave3_gpu*.log"
