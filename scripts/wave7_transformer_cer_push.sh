#!/usr/bin/env bash
# wave7_transformer_cer_push.sh — Focused transformer CER-reduction runs.
#
# One architecture per GPU, single visible device, no DDP.
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
    CUDA_VISIBLE_DEVICES="${gpu}" nohup /root/.local/bin/uv run python -m emg2qwerty.train \
        ${COMMON} \
        "hydra.run.dir=logs/${name}" \
        "${extra[@]}" \
        > "/root/${name}.log" 2>&1 &
    echo "  PID=$!"
}

echo "=== Wave 7: Transformer CER Push ==="

run_train 0 wave7_gpu0_small_alibi_stride2_varlen \
    module.use_cnn=true module.d_model=128 module.num_layers=4 module.num_heads=4 module.d_ff=512 \
    module.positional_encoding=alibi ++module.temporal_stride=2 \
    datamodule.train_window_lengths=[8000,16000,24000] datamodule.train_window_weights=[1.0,1.0,1.0]

run_train 1 wave7_gpu1_large_alibi_stride2_varlen \
    module.use_cnn=true module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024 \
    module.positional_encoding=alibi ++module.temporal_stride=2 \
    datamodule.train_window_lengths=[8000,16000,24000] datamodule.train_window_weights=[1.0,1.0,1.0]

run_train 2 wave7_gpu2_small_alibi_stride4_varlen \
    module.use_cnn=true module.d_model=128 module.num_layers=4 module.num_heads=4 module.d_ff=512 \
    module.positional_encoding=alibi ++module.temporal_stride=4 \
    datamodule.train_window_lengths=[8000,16000,24000] datamodule.train_window_weights=[1.0,1.0,1.0]

run_train 4 wave7_gpu4_large_sinusoidal_stride2_varlen \
    module.use_cnn=true module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024 \
    module.positional_encoding=sinusoidal ++module.temporal_stride=2 \
    datamodule.train_window_lengths=[8000,16000,24000] datamodule.train_window_weights=[1.0,1.0,1.0]

run_train 5 wave7_gpu5_small_sinusoidal_stride2_varlen \
    module.use_cnn=true module.d_model=128 module.num_layers=4 module.num_heads=4 module.d_ff=512 \
    module.positional_encoding=sinusoidal ++module.temporal_stride=2 \
    datamodule.train_window_lengths=[8000,16000,24000] datamodule.train_window_weights=[1.0,1.0,1.0]

run_train 6 wave7_gpu6_large_alibi_stride4_varlen \
    module.use_cnn=true module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024 \
    module.positional_encoding=alibi ++module.temporal_stride=4 \
    datamodule.train_window_lengths=[8000,16000,24000] datamodule.train_window_weights=[1.0,1.0,1.0]

run_train 7 wave7_gpu7_small_alibi_stride2_varlen_longmix \
    module.use_cnn=true module.d_model=128 module.num_layers=4 module.num_heads=4 module.d_ff=512 \
    module.positional_encoding=alibi ++module.temporal_stride=2 \
    datamodule.train_window_lengths=[8000,16000,24000,32000] datamodule.train_window_weights=[1.0,1.0,1.0,1.0]

echo
echo "Wave 7 launched. Monitor with:"
echo "  tail -f /root/wave7_gpu*.log"
