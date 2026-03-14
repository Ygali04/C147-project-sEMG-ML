#!/usr/bin/env bash
# wave9_architecture_sweep.sh — Focused architecture sweep: 7 architectures, 7 GPUs.
#
# Each GPU trains exactly ONE architecture from scratch for 150 epochs.
# No DDP, no multi-GPU. CUDA_VISIBLE_DEVICES pins each run to a single GPU.
#
# GPU 0: CNN-BiLSTM           (strong recurrent baseline)
# GPU 1: CNN-BiLSTM-Transformer hybrid (BiLSTM feeds transformer)
# GPU 2: Conformer-small       (convolution + attention)
# GPU 3: [RESERVED — Whisper control still training]
# GPU 4: Small Transformer     (sinusoidal PE, no downsampling, 150ep control)
# GPU 5: Small Transformer + ALiBi (isolate ALiBi effect, no other changes)
# GPU 6: Large Transformer     (sinusoidal PE, no downsampling, 150ep control)
# GPU 7: Conformer-large       (larger conformer variant)
#
set -euo pipefail

cd /root/emg2qwerty
export PATH="$HOME/.local/bin:$PATH"

COMMON="user=single_user ~cluster trainer.devices=1 trainer.accelerator=gpu trainer.strategy=auto"

run_train() {
    local gpu="$1"
    local name="$2"
    local model="$3"
    local max_epochs="$4"
    shift 4
    local extra=("$@")

    echo "[GPU ${gpu}] ${name} — ${model} (${max_epochs} epochs)"
    CUDA_VISIBLE_DEVICES="${gpu}" nohup uv run python -m emg2qwerty.train \
        ${COMMON} \
        model="${model}" \
        "trainer.max_epochs=${max_epochs}" \
        "hydra.run.dir=logs/${name}" \
        "${extra[@]}" \
        > "/root/${name}.log" 2>&1 &
    echo "  PID=$!"
}

echo "╔═════════════════════════════════════════════════════════════╗"
echo "║  Wave 9: Architecture Sweep — 7 architectures, 7 GPUs      ║"
echo "╚═════════════════════════════════════════════════════════════╝"
echo ""

# GPU 0: CNN-BiLSTM (strong recurrent baseline — should get ~20-25% CER)
run_train 0 wave9_gpu0_cnn_bilstm cnn_bilstm_ctc 150

# GPU 1: CNN-BiLSTM-Transformer hybrid (the key new architecture)
run_train 1 wave9_gpu1_hybrid cnn_bilstm_transformer_ctc 150

# GPU 2: Conformer-small (d=128, 4 layers)
run_train 2 wave9_gpu2_conformer_small conformer_ctc 150

# GPU 3: RESERVED for Whisper control (already running)
echo "[GPU 3] RESERVED — Whisper control still training"

# GPU 4: Small Transformer control (sinusoidal, no downsampling, 150ep)
run_train 4 wave9_gpu4_small_transformer t5_ctc 150 \
    module.use_cnn=true module.d_model=128 module.num_layers=4 module.num_heads=4 module.d_ff=512 \
    module.blank_penalty_epochs=0 module.temporal_stride=1 module.positional_encoding=sinusoidal

# GPU 5: Small Transformer + ALiBi (isolate ALiBi, everything else same)
run_train 5 wave9_gpu5_small_transformer_alibi t5_ctc 150 \
    module.use_cnn=true module.d_model=128 module.num_layers=4 module.num_heads=4 module.d_ff=512 \
    module.blank_penalty_epochs=0 module.temporal_stride=1 module.positional_encoding=alibi

# GPU 6: Large Transformer control (sinusoidal, no downsampling, 150ep)
run_train 6 wave9_gpu6_large_transformer t5_ctc 150 \
    module.use_cnn=true module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024 \
    module.blank_penalty_epochs=0 module.temporal_stride=1 module.positional_encoding=sinusoidal

# GPU 7: Conformer-large (d=256, 6 layers)
run_train 7 wave9_gpu7_conformer_large conformer_ctc 150 \
    module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024 \
    module.conv_kernel_size=31

echo ""
echo "Wave 9 launched. Monitor with:"
echo "  tail -f /root/wave9_gpu*.log"
echo "  for f in /root/wave9_gpu*.log; do echo \"=== \$(basename \$f) ===\"; grep 'reached' \$f | tail -3; done"
