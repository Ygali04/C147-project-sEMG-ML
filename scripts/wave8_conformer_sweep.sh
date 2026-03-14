#!/usr/bin/env bash
# wave8_conformer_sweep.sh — First Conformer-family architecture sweep.
#
# One architecture per GPU, single visible device, no DDP.
set -euo pipefail

cd /root/emg2qwerty
export PATH="$HOME/.local/bin:$PATH"

COMMON="user=single_user model=conformer_ctc ~cluster trainer.accelerator=gpu trainer.devices=1 trainer.strategy=auto trainer.max_epochs=80"

run_train() {
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

echo "=== Wave 8: Conformer Sweep ==="

run_train 0 wave8_gpu0_small_conformer \
    module.d_model=128 module.num_layers=4 module.num_heads=4 module.d_ff=512 module.conv_kernel_size=15

run_train 1 wave8_gpu1_small_conformer_alibi \
    module.d_model=128 module.num_layers=4 module.num_heads=4 module.d_ff=512 module.conv_kernel_size=15 \
    module.positional_encoding=alibi

run_train 2 wave8_gpu2_small_conformer_stride2 \
    module.d_model=128 module.num_layers=4 module.num_heads=4 module.d_ff=512 module.conv_kernel_size=15 \
    ++module.temporal_stride=2

run_train 4 wave8_gpu4_large_conformer \
    module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024 module.conv_kernel_size=31

run_train 5 wave8_gpu5_large_conformer_alibi \
    module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024 module.conv_kernel_size=31 \
    module.positional_encoding=alibi

run_train 6 wave8_gpu6_large_conformer_stride2_varlen \
    module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024 module.conv_kernel_size=31 \
    ++module.temporal_stride=2 \
    datamodule.train_window_lengths=[8000,16000,24000] datamodule.train_window_weights=[1.0,1.0,1.0]

run_train 7 wave8_gpu7_small_conformer_stride2_varlen \
    module.d_model=128 module.num_layers=4 module.num_heads=4 module.d_ff=512 module.conv_kernel_size=15 \
    ++module.temporal_stride=2 \
    datamodule.train_window_lengths=[8000,16000,24000] datamodule.train_window_weights=[1.0,1.0,1.0]

echo
echo "Wave 8 launched. Monitor with:"
echo "  tail -f /root/wave8_gpu*.log"
