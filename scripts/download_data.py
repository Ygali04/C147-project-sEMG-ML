#!/usr/bin/env python3
"""Download EMG session HDF5 files from Meta's public S3 bucket into
Backblaze B2.

Usage::

    # Download baseline (single user 89335547, 18 sessions)
    uv run python scripts/download_data.py --baseline

    # Download 10 random user profiles
    uv run python scripts/download_data.py --test

    # Download ALL profiles (~200 GB)
    uv run python scripts/download_data.py --all

    # Dry-run — show what would be downloaded without uploading
    uv run python scripts/download_data.py --baseline --dry-run
"""

from __future__ import annotations

import logging
import os
import sys

import click
from dotenv import load_dotenv

# Ensure src/ is on sys.path when running as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from emg2qwerty.pipeline.config import B2Config, DownloadConfig, DownloadMode, SourceS3Config
from emg2qwerty.pipeline.downloader import EMGDownloader

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
    help="Download baseline profile (user 89335547, 18 sessions).",
)
@click.option(
    "--test",
    "mode",
    flag_value="test",
    help="Download 10 random user profiles.",
)
@click.option(
    "--all",
    "mode",
    flag_value="all",
    help="Download ALL user profiles (~200 GB).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Resolve sessions but skip actual uploads.",
)
@click.option(
    "--n-test-users",
    type=int,
    default=10,
    show_default=True,
    help="Number of random users to sample in --test mode.",
)
@click.option(
    "--seed",
    type=int,
    default=1501,
    show_default=True,
    help="Random seed for user sampling.",
)
def main(mode: str, dry_run: bool, n_test_users: int, seed: int) -> None:
    """Download EMG session files from Meta S3 → Backblaze B2."""
    # Load .env for credentials
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

    config = DownloadConfig(
        mode=DownloadMode(mode),
        n_test_users=n_test_users,
        seed=seed,
        dry_run=dry_run,
        b2=B2Config(key_id=key_id, application_key=app_key),
        source=SourceS3Config(),
    )

    downloader = EMGDownloader(config)
    uploaded = downloader.run()

    # Summary
    if uploaded:
        click.echo(f"\nUploaded {len(uploaded)} files:")
        for rec in uploaded:
            click.echo(f"  {rec.b2_key}  ({rec.size_bytes / 1e6:.1f} MB)")
    elif not dry_run:
        click.echo("\nAll files already present in B2 — nothing to upload.")


if __name__ == "__main__":
    main()
