#!/usr/bin/env bash
# wave6_downsample_sweep.sh — Launch the temporal-downsampling sweep.
set -euo pipefail

cd /root/emg2qwerty
export PATH="$HOME/.local/bin:$PATH"

COMMON="user=single_user model=t5_ctc ~cluster trainer.accelerator=gpu trainer.devices=1 trainer.strategy=auto trainer.max_epochs=80"

run_train() {
    local gpu="$1"
    local name="$2"
    shift 2
    local extra=("$@")

    echo "[GPU ${gpu}] ${name}"
    CUDA_VISIBLE_DEVICES="${gpu}" nohup uv run python -m emg2qwerty.train \
        ${COMMON} \
        "hydra.run.dir=logs/${name}" \
        "${extra[@]}" \
        > "/root/${name}.log" 2>&1 &
    echo "  PID=$!"
}

echo "=== Wave 6: Temporal Downsampling Sweep ==="

run_train 0 wave6_gpu0_large_stride1 \
    module.use_cnn=true module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024 module.temporal_stride=1

run_train 1 wave6_gpu1_large_stride2 \
    module.use_cnn=true module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024 module.temporal_stride=2

run_train 2 wave6_gpu2_large_stride4 \
    module.use_cnn=true module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024 module.temporal_stride=4

run_train 3 wave6_gpu3_large_stride8 \
    module.use_cnn=true module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024 module.temporal_stride=8

run_train 4 wave6_gpu4_small_stride1 \
    module.use_cnn=true module.d_model=128 module.num_layers=4 module.num_heads=4 module.d_ff=512 module.temporal_stride=1

run_train 5 wave6_gpu5_small_stride2 \
    module.use_cnn=true module.d_model=128 module.num_layers=4 module.num_heads=4 module.d_ff=512 module.temporal_stride=2

run_train 6 wave6_gpu6_small_stride4 \
    module.use_cnn=true module.d_model=128 module.num_layers=4 module.num_heads=4 module.d_ff=512 module.temporal_stride=4

run_train 7 wave6_gpu7_large_stride4_varlen \
    module.use_cnn=true module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024 module.temporal_stride=4 \
    datamodule.train_window_lengths=[8000,16000,24000] datamodule.train_window_weights=[1.0,1.0,1.0]

echo
echo "Wave 6 launched. Monitor with:"
echo "  tail -f /root/wave6_gpu*.log"
