"""Integration test — end-to-end pipeline verification with real cloud endpoints.

Tests hit **real** services:
  • Meta public S3 (``fb-ctrl-oss``) — anonymous HTTPS streaming of the tar.gz
  • Backblaze B2 (``C147-project``) — authenticated S3-compatible read/write

Run with::

    uv run pytest tests/test_integration_baseline.py -m integration -v -s

The ``-s`` flag shows download progress and logging in real time.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tarfile
from pathlib import Path

import boto3
import h5py
import pytest
import requests
from dotenv import load_dotenv

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Skip entire module unless explicitly opted-in
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TAR_GZ_URL = "https://fb-ctrl-oss.s3.amazonaws.com/emg2qwerty/emg2qwerty-data-2021-08.tar.gz"
TAR_PREFIX = "emg2qwerty-data-2021-08/"
B2_ENDPOINT = "https://s3.us-west-004.backblazeb2.com"
B2_REGION = "us-west-004"
B2_BUCKET = "C147-project"
DATA_DIR = Path("data")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _load_env():
    """Load .env so B2 credentials are available."""
    load_dotenv(override=True)
    key_id = os.environ.get("B2_KEY_ID", "")
    app_key = os.environ.get("B2_APPLICATION_KEY", "")
    if not key_id or not app_key:
        pytest.skip("B2_KEY_ID / B2_APPLICATION_KEY not set — skipping integration tests")


@pytest.fixture(scope="module")
def b2_client():
    """Authenticated B2 S3 client."""
    return boto3.client(
        "s3",
        endpoint_url=B2_ENDPOINT,
        region_name=B2_REGION,
        aws_access_key_id=os.environ["B2_KEY_ID"],
        aws_secret_access_key=os.environ["B2_APPLICATION_KEY"],
    )


# ---------------------------------------------------------------------------
# 1. B2 connectivity (fast — ~2 s)
# ---------------------------------------------------------------------------


class TestB2Connectivity:
    """Prove we can read and write to the B2 bucket."""

    TEST_KEY = "_integration_test_probe.txt"

    def test_put_get_delete(self, b2_client) -> None:
        payload = b"integration-test-payload"

        # PUT
        b2_client.put_object(Bucket=B2_BUCKET, Key=self.TEST_KEY, Body=payload)
        log.info("PUT %s → OK", self.TEST_KEY)

        # GET
        resp = b2_client.get_object(Bucket=B2_BUCKET, Key=self.TEST_KEY)
        body = resp["Body"].read()
        assert body == payload, f"GET mismatch: {body!r}"
        log.info("GET %s → OK (content matches)", self.TEST_KEY)

        # DELETE
        b2_client.delete_object(Bucket=B2_BUCKET, Key=self.TEST_KEY)
        log.info("DELETE %s → OK", self.TEST_KEY)


# ---------------------------------------------------------------------------
# 2. Meta tar.gz reachability (fast — ~1 s)
# ---------------------------------------------------------------------------


class TestMetaTarGzReachability:
    """Verify that the tar.gz archive is reachable via HTTPS."""

    def test_head_request(self) -> None:
        resp = requests.head(TAR_GZ_URL, allow_redirects=True, timeout=15)
        assert resp.status_code == 200, f"HEAD returned {resp.status_code}"
        content_length = int(resp.headers.get("Content-Length", 0))
        assert content_length > 100_000_000_000, f"Unexpectedly small: {content_length}"
        log.info("tar.gz reachable: %.1f GB", content_length / 1e9)


# ---------------------------------------------------------------------------
# 3. End-to-end pipeline proof: tar.gz → local → B2 → local  (1 file, ~30 s)
# ---------------------------------------------------------------------------


class TestPipelineRoundTrip:
    """Extract the very first HDF5 from the tar.gz, save it locally,
    upload to B2, download back, and verify the round-trip."""

    def test_extract_first_hdf5_roundtrip(self, b2_client) -> None:
        """Stream the tar.gz until we find the first .hdf5 file, then:
        1. Save it locally under data/
        2. Upload it to B2
        3. Download it from B2 to a different path
        4. Compare checksums
        5. Verify it's a valid HDF5
        """
        # --- Stream and extract the first HDF5 ---
        log.info("Streaming tar.gz to extract first HDF5 file...")
        resp = requests.get(TAR_GZ_URL, stream=True, timeout=60)
        resp.raise_for_status()

        extracted_name: str | None = None
        extracted_data: bytes | None = None

        try:
            with tarfile.open(fileobj=resp.raw, mode="r|gz") as tar:
                for member in tar:
                    if member.isfile() and member.name.endswith(".hdf5"):
                        fileobj = tar.extractfile(member)
                        assert fileobj is not None, f"Cannot extract {member.name}"
                        extracted_data = fileobj.read()
                        extracted_name = member.name
                        log.info(
                            "Extracted: %s (%.1f MB)",
                            extracted_name,
                            len(extracted_data) / 1e6,
                        )
                        break
        finally:
            resp.close()

        assert extracted_name is not None, "No HDF5 file found in tar.gz"
        assert extracted_data is not None
        assert len(extracted_data) > 0

        # Derive a clean filename
        bare_name = extracted_name
        if bare_name.startswith(TAR_PREFIX):
            bare_name = bare_name[len(TAR_PREFIX) :]

        # --- 1. Save locally ---
        local_path = DATA_DIR / "integration_test" / bare_name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(extracted_data)
        assert local_path.exists()
        assert local_path.stat().st_size == len(extracted_data)
        log.info("Saved locally: %s", local_path)

        # --- 2. Upload to B2 ---
        b2_key = f"_integration_test/{bare_name}"
        b2_client.put_object(
            Bucket=B2_BUCKET,
            Key=b2_key,
            Body=extracted_data,
            ContentType="application/x-hdf5",
        )
        log.info("Uploaded to B2: %s", b2_key)

        # --- 3. Download from B2 ---
        roundtrip_path = DATA_DIR / "integration_test" / f"roundtrip_{bare_name}"
        roundtrip_path.parent.mkdir(parents=True, exist_ok=True)
        b2_client.download_file(B2_BUCKET, b2_key, str(roundtrip_path))
        assert roundtrip_path.exists()
        log.info("Downloaded from B2: %s", roundtrip_path)

        # --- 4. Compare checksums ---
        original_md5 = hashlib.md5(extracted_data).hexdigest()
        roundtrip_md5 = hashlib.md5(roundtrip_path.read_bytes()).hexdigest()
        assert original_md5 == roundtrip_md5, f"MD5 mismatch: original={original_md5} vs roundtrip={roundtrip_md5}"
        log.info("Checksum match: %s ✓", original_md5)

        # --- 5. Verify valid HDF5 ---
        with h5py.File(local_path, "r") as f:
            keys = list(f.keys())
            assert len(keys) > 0, "HDF5 file has no top-level groups/datasets"
            log.info("HDF5 validated: top-level keys = %s", keys)

        # --- Cleanup B2 test object ---
        b2_client.delete_object(Bucket=B2_BUCKET, Key=b2_key)
        log.info("Cleaned up B2 test object: %s", b2_key)

        # --- Cleanup local test files ---
        roundtrip_path.unlink(missing_ok=True)
        local_path.unlink(missing_ok=True)
        # Remove empty dirs
        for d in [local_path.parent, roundtrip_path.parent]:
            if d.exists() and not any(d.iterdir()):
                d.rmdir()
        log.info("Cleaned up local test files.")
