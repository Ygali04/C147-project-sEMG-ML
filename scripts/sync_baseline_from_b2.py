#!/usr/bin/env python3
"""Sync the baseline user's sessions from Backblaze B2 into local data/.

This script is intentionally narrow: it does not upload anything and it does
not train. It only resolves the baseline profile from the existing B2 registry
and downloads the required HDF5 files for user 89335547.
"""

from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv

# Ensure src/ is on sys.path when running as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from emg2qwerty.pipeline.config import B2Config, DownloadMode, TrainBatchConfig
from emg2qwerty.pipeline.trainer import BatchTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    load_dotenv()

    key_id = os.environ.get("B2_KEY_ID", "")
    app_key = os.environ.get("B2_APPLICATION_KEY", "")
    if not key_id or not app_key:
        raise SystemExit(
            "B2_KEY_ID and B2_APPLICATION_KEY must be set. Copy .env.example to .env and fill in your credentials."
        )

    config = TrainBatchConfig(
        mode=DownloadMode.BASELINE,
        b2=B2Config(key_id=key_id, application_key=app_key),
        batch_size_profiles=1,
    )
    trainer = BatchTrainer(config)

    profiles = trainer.resolve_profiles()
    if not profiles:
        raise SystemExit("No baseline profile found in the B2 registry.")

    user_id, b2_keys = next(iter(profiles.items()))
    local_paths = trainer.sync_profile(user_id, b2_keys)

    log.info("Synced %d baseline session files for user %s.", len(local_paths), user_id)
    for path in local_paths:
        print(path)


if __name__ == "__main__":
    main()
