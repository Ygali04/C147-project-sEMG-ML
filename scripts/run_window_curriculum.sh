#!/usr/bin/env bash
# run_window_curriculum.sh — Run a multi-phase window-length curriculum.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_window_curriculum.sh \
#       wave4_large_3phase \
#       model=t5_ctc \
#       trainer.max_epochs=80 \
#       module.use_cnn=true module.d_model=256 module.num_layers=6 module.num_heads=8 module.d_ff=1024
#
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/run_window_curriculum.sh <run_name> [hydra overrides...]"
    exit 1
fi

RUN_NAME="$1"
shift

ROOT_DIR="/root/emg2qwerty"
PHASE_DIR="/root/window_curricula/${RUN_NAME}"
mkdir -p "${PHASE_DIR}"

cd "${ROOT_DIR}"
export PATH="$HOME/.local/bin:$PATH"

COMMON=("$@" "user=single_user" "~cluster" "trainer.devices=1")

run_phase() {
    local phase_name="$1"
    local max_epochs="$2"
    local lengths="$3"
    local weights="$4"
    local checkpoint_arg=()

    if [[ -f "${PHASE_DIR}/latest.ckpt" ]]; then
        checkpoint_arg=("checkpoint=${PHASE_DIR}/latest.ckpt")
    fi

    echo "--- ${RUN_NAME} / ${phase_name} ---"
    uv run python -m emg2qwerty.train \
        "${COMMON[@]}" \
        "hydra.run.dir=logs/${RUN_NAME}/${phase_name}" \
        "trainer.max_epochs=${max_epochs}" \
        "datamodule.train_window_lengths=${lengths}" \
        "datamodule.train_window_weights=${weights}" \
        "${checkpoint_arg[@]}" \
        > "${PHASE_DIR}/${phase_name}.log" 2>&1

    latest_ckpt="$(find "${ROOT_DIR}/logs/${RUN_NAME}/${phase_name}" -path '*/checkpoints/last*.ckpt' -type f | sort | tail -1)"
    if [[ -z "${latest_ckpt}" ]]; then
        echo "No checkpoint produced for ${phase_name}"
        exit 1
    fi
    ln -sf "${latest_ckpt}" "${PHASE_DIR}/latest.ckpt"
}

run_phase phase1 20 "[8000]" "[1.0]"
run_phase phase2 40 "[8000,16000]" "[1.0,1.0]"
run_phase phase3 80 "[8000,16000,24000,32000]" "[1.0,1.0,1.0,1.0]"

echo "Curriculum run complete: ${RUN_NAME}"
echo "Latest checkpoint: ${PHASE_DIR}/latest.ckpt"
