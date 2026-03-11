#!/usr/bin/env python3
"""Batched training script — train models across user profiles stored in
Backblaze B2.

Usage::

    # Train on baseline profile (user 89335547)
    uv run python scripts/train_batched.py --baseline

    # Train on first 10 profiles from the registry
    uv run python scripts/train_batched.py --test

    # Train on ALL profiles in the registry
    uv run python scripts/train_batched.py --all
"""

from __future__ import annotations

import logging
import os
import sys

import click
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


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--baseline",
    "mode",
    flag_value="baseline",
    default=True,
    help="Train on baseline profile (user 89335547).",
)
@click.option(
    "--test",
    "mode",
    flag_value="test",
    help="Train on first 10 user profiles from the registry.",
)
@click.option(
    "--all",
    "mode",
    flag_value="all",
    help="Train on ALL user profiles in the registry.",
)
@click.option(
    "--checkpoint",
    type=str,
    default=None,
    help="Optional checkpoint path to resume from.",
)
@click.option(
    "--batch-size-profiles",
    type=int,
    default=1,
    show_default=True,
    help="Number of user profiles per training batch.",
)
def main(mode: str, checkpoint: str | None, batch_size_profiles: int) -> None:
    """Train models across user profiles stored in Backblaze B2."""
    load_dotenv()

    key_id = os.environ.get("B2_KEY_ID", "")
    app_key = os.environ.get("B2_APPLICATION_KEY", "")
    if not key_id or not app_key:
        click.echo(
            "ERROR: B2_KEY_ID and B2_APPLICATION_KEY must be set.\n"
            "  Copy .env.example to .env and fill in your credentials.",
            err=True,
        )
        raise SystemExit(1)

    config = TrainBatchConfig(
        mode=DownloadMode(mode),
        b2=B2Config(key_id=key_id, application_key=app_key),
        batch_size_profiles=batch_size_profiles,
        checkpoint=checkpoint,
    )

    trainer = BatchTrainer(config)
    results = trainer.run()

    # Summary table
    if results:
        click.echo("\n=== Training Summary ===")
        for user_id, success in results.items():
            status = "OK" if success else "FAILED"
            click.echo(f"  {user_id}: {status}")


if __name__ == "__main__":
    main()
