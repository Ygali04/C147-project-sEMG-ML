#!/usr/bin/env bash
# wave5_alibi_sweep.sh — Launch the ALiBi positional-encoding sweep.
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

echo "=== Wave 5: ALiBi Positional-Encoding Sweep ==="

run_train 0 wave5_gpu0_large_sinusoidal \
    module.use_cnn=true module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024 \
    module.positional_encoding=sinusoidal

run_train 1 wave5_gpu1_large_alibi \
    module.use_cnn=true module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024 \
    module.positional_encoding=alibi

run_train 2 wave5_gpu2_small_sinusoidal \
    module.use_cnn=true module.d_model=128 module.num_layers=4 module.num_heads=4 module.d_ff=512 \
    module.positional_encoding=sinusoidal

run_train 3 wave5_gpu3_small_alibi \
    module.use_cnn=true module.d_model=128 module.num_layers=4 module.num_heads=4 module.d_ff=512 \
    module.positional_encoding=alibi

run_train 4 wave5_gpu4_pure_sinusoidal \
    module.use_cnn=false module.blank_penalty_epochs=0 module.d_model=128 module.num_layers=4 module.num_heads=4 module.d_ff=512 \
    module.positional_encoding=sinusoidal

run_train 5 wave5_gpu5_pure_alibi \
    module.use_cnn=false module.blank_penalty_epochs=0 module.d_model=128 module.num_layers=4 module.num_heads=4 module.d_ff=512 \
    module.positional_encoding=alibi

run_train 6 wave5_gpu6_large_alibi_varlen \
    module.use_cnn=true module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024 \
    module.positional_encoding=alibi \
    datamodule.train_window_lengths=[8000,16000,24000] datamodule.train_window_weights=[1.0,1.0,1.0]

run_train 7 wave5_gpu7_large_sinusoidal_varlen \
    module.use_cnn=true module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024 \
    module.positional_encoding=sinusoidal \
    datamodule.train_window_lengths=[8000,16000,24000] datamodule.train_window_weights=[1.0,1.0,1.0]

echo
echo "Wave 5 launched. Monitor with:"
echo "  tail -f /root/wave5_gpu*.log"
