#!/usr/bin/env bash
# wave4_curriculum_sweep.sh — Launch sequence-length curriculum runs on 8 GPUs.
set -euo pipefail

cd /root/emg2qwerty
export PATH="$HOME/.local/bin:$PATH"

run_curriculum() {
    local gpu="$1"
    local name="$2"
    shift 2
    local extra=("$@")

    echo "[GPU ${gpu}] ${name}"
    CUDA_VISIBLE_DEVICES="${gpu}" nohup bash scripts/run_window_curriculum.sh \
        "${name}" \
        "${extra[@]}" \
        > "/root/${name}.log" 2>&1 &
    echo "  PID=$!"
}

echo "=== Wave 4: Sequence-Length Curriculum Sweep ==="

run_curriculum 0 wave4_gpu0_large_3phase \
    model=t5_ctc module.use_cnn=true module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024

run_curriculum 1 wave4_gpu1_large_3phase_alibi \
    model=t5_ctc module.use_cnn=true module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024 \
    trainer.max_epochs=100

run_curriculum 2 wave4_gpu2_large_3phase_rope \
    model=t5_ctc module.use_cnn=true module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024 \
    optimizer.lr=3e-4

run_curriculum 3 wave4_gpu3_small_3phase \
    model=t5_ctc module.use_cnn=true module.d_model=128 module.num_layers=4 module.num_heads=4 module.d_ff=512

run_curriculum 4 wave4_gpu4_pure_3phase \
    model=t5_ctc module.use_cnn=false module.blank_penalty_epochs=0 module.d_model=128 module.num_layers=4 module.num_heads=4 module.d_ff=512

run_curriculum 5 wave4_gpu5_whisper_3phase \
    model=whisper_ctc

run_curriculum 6 wave4_gpu6_cnn_bilstm_3phase \
    model=cnn_bilstm_ctc

run_curriculum 7 wave4_gpu7_bilstm_3phase \
    model=bilstm_ctc

echo
echo "Wave 4 launched. Monitor with:"
echo "  tail -f /root/wave4_gpu*.log"
