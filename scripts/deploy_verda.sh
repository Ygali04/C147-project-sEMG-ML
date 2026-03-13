#!/usr/bin/env bash
# deploy_verda.sh — One-command deploy & train on a Verda compute instance.
#
# Usage:
#     ./scripts/deploy_verda.sh user@<verda-ip>
#
#     # With a custom remote directory:
#     REMOTE_DIR=/workspace/emg2qwerty ./scripts/deploy_verda.sh user@<verda-ip>
#
#     # Dry-run (sync only, don't train):
#     DRY_RUN=1 ./scripts/deploy_verda.sh user@<verda-ip>
#
# What it does:
#   1. rsync's your local project to the remote (excluding data/, logs/, .git, build artifacts)
#   2. Copies your .env (B2 credentials) to the remote
#   3. Installs uv + dependencies on the remote (first run only, cached after)
#   4. Kicks off: uv run python scripts/train_batched.py --baseline --model t5_ctc
#   5. Lightning auto-detects all available GPUs → DDP when >1

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REMOTE="${1:?Usage: ./scripts/deploy_verda.sh user@host}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR="${REMOTE_DIR:-/root/emg2qwerty}"
MODEL="${MODEL:-t5_ctc}"
MODE="${MODE:---baseline}"
DRY_RUN="${DRY_RUN:-0}"

# Verda SSH key (ygali@g.ucla.edu)
SSH_KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS="-i $SSH_KEY"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Verda Deploy — ${REMOTE}                               "
echo "║  Model: ${MODEL}   Mode: ${MODE}                        "
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ---------------------------------------------------------------------------
# Phase 1: Sync code
# ---------------------------------------------------------------------------
echo "=== Phase 1/4: Syncing code to ${REMOTE}:${REMOTE_DIR} ==="
rsync -avz --delete -e "ssh $SSH_OPTS" \
    --exclude='.git/' \
    --exclude='data/' \
    --exclude='logs/' \
    --exclude='.venv/' \
    --exclude='.env' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='*.so' \
    --exclude='*.dylib' \
    --exclude='*.egg-info/' \
    --exclude='vendor/kenlm/build/' \
    --exclude='site/' \
    --exclude='.cursor/' \
    --exclude='.ruff_cache/' \
    --exclude='uv.lock' \
    "$LOCAL_DIR/" "${REMOTE}:${REMOTE_DIR}/"
echo ""

# ---------------------------------------------------------------------------
# Phase 2: Push .env credentials
# ---------------------------------------------------------------------------
echo "=== Phase 2/4: Pushing .env credentials ==="
if [[ -f "$LOCAL_DIR/.env" ]]; then
    scp $SSH_OPTS "$LOCAL_DIR/.env" "${REMOTE}:${REMOTE_DIR}/.env"
    echo "  .env copied."
else
    echo "  WARNING: No .env found at $LOCAL_DIR/.env"
    echo "  Make sure B2_KEY_ID and B2_APPLICATION_KEY are set on the remote."
fi
echo ""

# ---------------------------------------------------------------------------
# Phase 3: Remote setup (uv + dependencies)
# ---------------------------------------------------------------------------
echo "=== Phase 3/4: Installing dependencies on remote ==="
ssh -t $SSH_OPTS "$REMOTE" << SETUP_EOF
    set -e
    cd ${REMOTE_DIR}

    # Ensure uv is on PATH (may have been installed in a previous run)
    export PATH="\$HOME/.local/bin:\$PATH"

    # Install system build deps for kenlm (needs Python.h + compression libs)
    echo '--- Installing system build dependencies ---'
    apt-get update -qq && apt-get install -y -qq \
        python3-dev build-essential cmake \
        zlib1g-dev libbz2-dev liblzma-dev \
        > /dev/null 2>&1
    echo '  system deps installed.'

    # Install uv if missing
    if ! command -v uv &>/dev/null; then
        echo '--- Installing uv ---'
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="\$HOME/.local/bin:\$PATH"
    fi

    # Purge stale bytecode so Python picks up freshly-synced .py files
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

    echo '--- Running uv sync ---'
    uv sync

    # Patch pl_bolts for PL 2.x compatibility.
    # pl_bolts 0.7 eagerly imports PL 1.x-only modules throughout the package.
    # We only need LinearWarmupCosineAnnealingLR (a standalone _LRScheduler).
    # Neuter the __init__ files that trigger broken transitive imports.
    PB_DIR=\$(find .venv -type d -name pl_bolts -path '*/site-packages/*' 2>/dev/null | head -1)
    if [ -n "\$PB_DIR" ]; then
        echo '--- Patching pl_bolts for PL 2.x ---'
        echo '# Patched for PL 2.x' > "\$PB_DIR/__init__.py"
        echo '# Patched for PL 2.x' > "\$PB_DIR/optimizers/__init__.py"
        echo '# Patched for PL 2.x' > "\$PB_DIR/utils/__init__.py"
        echo '# Patched for PL 2.x' > "\$PB_DIR/callbacks/__init__.py"
    fi
    echo '--- Dependencies ready ---'
SETUP_EOF
echo ""

# ---------------------------------------------------------------------------
# Phase 4: Launch training (or stop if dry run)
# ---------------------------------------------------------------------------
if [[ "$DRY_RUN" == "1" ]]; then
    echo "=== DRY_RUN=1 — Skipping training. Code is synced and deps installed. ==="
    echo "  To train manually:  ssh $SSH_OPTS ${REMOTE} 'cd ${REMOTE_DIR} && export PATH=\$HOME/.local/bin:\$PATH && uv run python scripts/train_batched.py ${MODE} --model ${MODEL}'"
    exit 0
fi

echo "=== Phase 4/4: Starting training ==="
echo "  Command: uv run python scripts/train_batched.py ${MODE} --model ${MODEL}"
echo ""
ssh -t $SSH_OPTS "$REMOTE" << TRAIN_EOF
    set -e
    cd ${REMOTE_DIR}
    export PATH="\$HOME/.local/bin:\$PATH"

    # Show GPU info
    echo '--- GPU info ---'
    python3 -c "import torch; print(f'GPUs: {torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}')" 2>/dev/null || echo '(torch not yet importable — will be after uv run)'
    echo ''

    # Train
    uv run python scripts/train_batched.py ${MODE} --model ${MODEL}
TRAIN_EOF

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Training complete!                                      "
echo "║  Pull logs:  rsync -avz ${REMOTE}:${REMOTE_DIR}/logs/ ./logs/"
echo "╚══════════════════════════════════════════════════════════╝"
