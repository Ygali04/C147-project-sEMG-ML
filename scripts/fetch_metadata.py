"""One-shot script: stream the Meta tar.gz and extract metadata.csv.

metadata.csv is the very last member (#1,136) of the archive, so the entire
308 GB stream must be consumed to reach it. This only needs to run once — the
extracted CSV is committed to the repo and used by all subsequent pipeline
code.

Usage
-----
    uv run python scripts/fetch_metadata.py [--out data/metadata.csv]
"""

from __future__ import annotations

import logging
import sys
import tarfile
from pathlib import Path

import click
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

TAR_URL = "https://fb-ctrl-oss.s3.amazonaws.com/emg2qwerty/emg2qwerty-data-2021-08.tar.gz"


@click.command()
@click.option(
    "--out",
    default="data/metadata.csv",
    show_default=True,
    help="Destination path for metadata.csv",
)
def main(out: str) -> None:
    dest = Path(out)
    if dest.exists():
        log.info("metadata.csv already exists at %s — nothing to do.", dest)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)

    log.info("Streaming %s", TAR_URL)
    log.info("metadata.csv is the last of 1,136 members — the full 308 GB must be consumed.")
    log.info("Progress is logged every 100 members. Estimated time: 2–6 h depending on bandwidth.")

    # Connect timeout 60 s; no read timeout for the multi-hour stream
    response = requests.get(TAR_URL, stream=True, timeout=(60, None))
    response.raise_for_status()

    count = 0
    found = False

    try:
        with tarfile.open(fileobj=response.raw, mode="r|gz") as tar:
            for member in tar:
                count += 1

                if count % 100 == 0:
                    log.info("  … %d members scanned, last: %s", count, member.name)

                if member.name.endswith("metadata.csv"):
                    log.info("Found metadata.csv at member #%d (%d bytes) — extracting …", count, member.size)
                    f = tar.extractfile(member)
                    if f is None:
                        log.error("Could not extract metadata.csv!")
                        sys.exit(1)
                    dest.write_bytes(f.read())
                    log.info("Saved to %s (%.1f KB)", dest, dest.stat().st_size / 1024)
                    found = True
                    break

    except Exception:
        log.exception("Stream error after %d members", count)
        sys.exit(1)
    finally:
        response.close()

    if not found:
        log.error("metadata.csv was not found in the archive after scanning %d members!", count)
        sys.exit(1)

    log.info("Done. Scanned %d members total.", count)


if __name__ == "__main__":
    main()
