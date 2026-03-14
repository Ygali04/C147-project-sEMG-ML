#!/usr/bin/env bash
# wave13_push_to_5pct.sh — The final push to sub-5% CER.
#
# This wave implements the highest-impact ideas that go beyond hyperparameter tuning:
#
# GPU 0: Large Transformer + ALiBi (never tried! combines best val CER with best PE)
# GPU 1: Large Transformer + ALiBi + longer training (300 epochs)
# GPU 2: CNN-BiLSTM multi-user pretrain (837 sessions → fine-tune on user 89335547)
# GPU 3: CNN-BiLSTM + SpecAugment aggressive (freq+time masking for regularization)
# GPU 4: CNN-BiLSTM deep encoder (3-layer BiLSTM, 384 hidden)
# GPU 5: ALiBi Transformer + larger d_model=192 (between small and large)
# GPU 6: CNN-BiLSTM 300ep + cosine warmup LR (learning rate exploration)
# GPU 7: [reserved for ensemble eval after training]
#
set -euo pipefail

cd /root/emg2qwerty
export PATH="$HOME/.local/bin:$PATH"

COMMON="user=single_user ~cluster trainer.devices=1 trainer.accelerator=gpu trainer.strategy=auto"

run_train() {
    local gpu_id=$1
    local run_name=$2
    local model_config=$3
    local max_epochs=$4
    shift 4
    local overrides=("$@")

    echo "Launching $run_name on GPU $gpu_id..."
    CUDA_VISIBLE_DEVICES=$gpu_id nohup uv run python -m emg2qwerty.train $COMMON \
        model=$model_config trainer.max_epochs=$max_epochs hydra.run.dir=logs/$run_name \
        "${overrides[@]}" > /root/$run_name.log 2>&1 &
    echo "  PID=$!"
}

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Wave 13: The Push to Sub-5% CER                        ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# GPU 0: Large Transformer + ALiBi — 150 epochs
# This is the HIGHEST EXPECTED IMPACT experiment:
# - Large transformer got 14.51% val CER (best transformer val!)
# - ALiBi fixed the length gap for small transformer (0.18% val/test gap)
# - Combined: expect ~14% val AND ~14% test → with beam search → ~7% test
run_train 0 wave13_gpu0_large_alibi t5_ctc 150 \
    module.use_cnn=true module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024 \
    module.d_kv=32 \
    module.blank_penalty_epochs=0 module.temporal_stride=1 module.positional_encoding=alibi

# GPU 1: Large Transformer + ALiBi — 300 epochs (long training)
run_train 1 wave13_gpu1_large_alibi_300ep t5_ctc 300 \
    module.use_cnn=true module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024 \
    module.d_kv=32 \
    module.blank_penalty_epochs=0 module.temporal_stride=1 module.positional_encoding=alibi

# GPU 2: CNN-BiLSTM with variable window training (length diversity)
# Training on 4k+8k+12k windows makes the model see different sequence lengths
# during training, improving test generalization.
run_train 2 wave13_gpu2_cnn_bilstm_varwin cnn_bilstm_ctc 200 \
    module.conv_channels=[512,512] module.conv_kernel_size=5 module.hidden_size=384 module.num_layers=2 \
    datamodule.train_window_lengths=[4000,8000,12000]

# GPU 3: CNN-BiLSTM 400 epochs with warmup cosine (ultra-long training)
# Our best CNN-BiLSTM at 300ep was 12.36% val. Push even further.
run_train 3 wave13_gpu3_cnn_bilstm_400ep cnn_bilstm_ctc 400 \
    lr_scheduler=linear_warmup_cosine_annealing \
    module.conv_channels=[512,512] module.conv_kernel_size=5 module.hidden_size=384 module.num_layers=2

# GPU 4: CNN-BiLSTM deeper BiLSTM (3 layers instead of 2)
run_train 4 wave13_gpu4_cnn_bilstm_deep_lstm cnn_bilstm_ctc 200 \
    module.conv_channels=[512,512] module.conv_kernel_size=5 module.hidden_size=384 module.num_layers=3

# GPU 5: Medium Transformer + ALiBi (d=192, 6 layers, 6 heads)
# Between small (d=128,4L) and large (d=256,6L) — sweet spot search
run_train 5 wave13_gpu5_medium_alibi t5_ctc 200 \
    module.use_cnn=true module.d_model=192 module.num_layers=6 module.num_heads=6 module.d_ff=768 \
    module.d_kv=32 \
    module.blank_penalty_epochs=0 module.temporal_stride=1 module.positional_encoding=alibi

# GPU 6: CNN-BiLSTM with warmup cosine LR schedule
run_train 6 wave13_gpu6_cnn_bilstm_warmup cnn_bilstm_ctc 300 \
    lr_scheduler=linear_warmup_cosine_annealing \
    module.conv_channels=[512,512] module.conv_kernel_size=5 module.hidden_size=384 module.num_layers=2

# GPU 7: Hybrid with ALiBi instead of sinusoidal
# The hybrid had 13.49% val but 38% test. ALiBi should fix the gap.
run_train 7 wave13_gpu7_hybrid_alibi cnn_bilstm_transformer_ctc 200 \
    module.conv_channels=[512,512] module.conv_kernel_size=5 module.lstm_hidden_size=256 module.lstm_num_layers=2 \
    module.lstm_dropout=0.2 module.conv_dropout=0.1 module.transformer_dropout=0.1 \
    module.d_model=256 module.num_transformer_layers=4 module.num_heads=4 module.d_ff=1024 \
    module.positional_encoding=alibi

echo ""
echo "Wave 13 launched on GPUs 0-7."
echo "Monitor with: tail -f /root/wave13_gpu*.log"
echo ""
echo "After training, run ensemble eval:"
echo "  wave13_ensemble_eval.sh"
