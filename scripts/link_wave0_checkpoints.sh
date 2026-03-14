#!/usr/bin/env bash
# link_wave0_checkpoints.sh — Materialize stable symlinks for wave-0 control checkpoints.
set -euo pipefail

ROOT_DIR="/root/emg2qwerty"
LINK_DIR="/root/checkpoints"
mkdir -p "${LINK_DIR}"

link_best() {
    local run_name="$1"
    local link_name="$2"
    local ckpt

    ckpt="$(find "${ROOT_DIR}/logs/${run_name}/checkpoints" -name 'epoch=*.ckpt' -type f | sort | tail -1)"
    if [[ -z "${ckpt}" ]]; then
        echo "No checkpoint found for ${run_name}"
        return 1
    fi

    ln -sfn "${ckpt}" "${LINK_DIR}/${link_name}"
    echo "${link_name} -> ${ckpt}"
}

link_best wave0_gpu0_tds_control tds_best.ckpt
link_best wave0_gpu1_transformer_large_control transformer_large_best.ckpt
link_best wave0_gpu2_transformer_small_control transformer_small_best.ckpt
link_best wave0_gpu3_whisper_control whisper_best.ckpt
