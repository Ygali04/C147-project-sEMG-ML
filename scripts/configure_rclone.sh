#!/usr/bin/env bash
# configure_rclone.sh — One-time setup for rclone with the Backblaze B2 remote.
#
# Prerequisites:
#   brew install rclone   (macOS)
#   sudo apt install rclone   (Linux)
#
# This script creates (or overwrites) the [b2-c147] remote in your
# rclone config file using the environment variables B2_KEY_ID and
# B2_APPLICATION_KEY.
#
# Usage:
#   source .env          # load credentials
#   bash scripts/configure_rclone.sh

set -euo pipefail

# Load .env if present
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

if [ -z "${B2_KEY_ID:-}" ] || [ -z "${B2_APPLICATION_KEY:-}" ]; then
    echo "ERROR: B2_KEY_ID and B2_APPLICATION_KEY must be set."
    echo "  Copy .env.example to .env and fill in your credentials."
    exit 1
fi

REMOTE_NAME="b2-c147"
ENDPOINT="s3.us-west-004.backblazeb2.com"

echo "Configuring rclone remote '${REMOTE_NAME}' ..."

rclone config create "${REMOTE_NAME}" s3 \
    provider="Other" \
    env_auth="false" \
    access_key_id="${B2_KEY_ID}" \
    secret_access_key="${B2_APPLICATION_KEY}" \
    endpoint="${ENDPOINT}" \
    acl="private"

echo ""
echo "Done! Test with:"
echo "  rclone ls ${REMOTE_NAME}:C147-project"
echo ""
echo "Useful commands:"
echo "  rclone ls ${REMOTE_NAME}:C147-project/emg2qwerty/"
echo "  rclone sync ${REMOTE_NAME}:C147-project/emg2qwerty/ data/emg2qwerty/ --progress"
echo "  rclone mount ${REMOTE_NAME}:C147-project/emg2qwerty/ data/emg2qwerty/ --read-only"
