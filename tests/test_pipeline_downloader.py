"""Tests for ``emg2qwerty.pipeline.downloader`` — EMGDownloader.

Uses moto to mock both the source (Meta) and destination (B2) S3 buckets.

Covers:
- resolve_sessions for BASELINE mode
- resolve_sessions for TEST mode (n distinct users)
- resolve_sessions for ALL mode
- dry_run does not upload
- dedup skips existing files
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import boto3
import pandas as pd
import pytest
from moto import mock_aws

from emg2qwerty.pipeline.config import B2Config, DownloadConfig, DownloadMode, SourceS3Config
from emg2qwerty.pipeline.downloader import EMGDownloader
from emg2qwerty.pipeline.registry import FileRegistry, make_record


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCE_BUCKET = "fb-baml-public"
SOURCE_PREFIX = "emg2qwerty"
B2_BUCKET = "test-b2-bucket"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


def _make_metadata_csv() -> bytes:
    """Create a minimal metadata.csv with 3 users, each with 4 sessions."""
    rows = []
    for user_id in [89335547, 11111111, 22222222]:
        for i in range(4):
            rows.append(
                {
                    "user": user_id,
                    "session": f"2021-01-01-{user_id}-session{i}",
                    "quality_check_tags": "[]",
                }
            )
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


@pytest.fixture()
def mock_s3():
    """Set up moto-mocked source and B2 buckets with seed data."""
    with mock_aws():
        # Source bucket
        source = boto3.client("s3", region_name="us-east-1")
        source.create_bucket(Bucket=SOURCE_BUCKET)
        source.put_object(
            Bucket=SOURCE_BUCKET,
            Key=f"{SOURCE_PREFIX}/metadata.csv",
            Body=_make_metadata_csv(),
        )
        # Seed a few HDF5 "files" (small payloads for testing)
        for user_id in [89335547, 11111111, 22222222]:
            for i in range(4):
                key = f"{SOURCE_PREFIX}/{user_id}/2021-01-01-{user_id}-session{i}.hdf5"
                source.put_object(
                    Bucket=SOURCE_BUCKET,
                    Key=key,
                    Body=b"FAKE_HDF5_DATA",
                )

        # B2 bucket
        b2 = boto3.client("s3", region_name="us-east-1")
        b2.create_bucket(Bucket=B2_BUCKET)

        yield {"source": source, "b2": b2}


def _make_config(
    mode: DownloadMode = DownloadMode.BASELINE,
    dry_run: bool = False,
    n_test_users: int = 2,
    data_root: Path | None = None,
) -> DownloadConfig:
    return DownloadConfig(
        mode=mode,
        n_test_users=n_test_users,
        seed=42,
        data_root=data_root or Path("/tmp/emg_test_data"),
        dry_run=dry_run,
        b2=B2Config(
            endpoint="s3.us-east-1.backblazeb2.com",
            region="us-east-1",
            bucket_name=B2_BUCKET,
            bucket_id="test-id",
            key_id="testing",
            application_key="testing",
        ),
        source=SourceS3Config(
            bucket=SOURCE_BUCKET,
            prefix=SOURCE_PREFIX,
            region="us-east-1",
            anonymous=False,  # moto doesn't need unsigned
        ),
    )


# ---------------------------------------------------------------------------
# Session resolution tests
# ---------------------------------------------------------------------------


class TestResolveSessions:
    def test_baseline_mode(self, mock_s3: dict, tmp_path: Path) -> None:
        """BASELINE mode uses single_user.yaml, not metadata.csv."""
        config = _make_config(mode=DownloadMode.BASELINE, data_root=tmp_path)
        downloader = EMGDownloader(config)
        # Patch the baseline loader to return known sessions
        with patch.object(downloader, "_load_baseline_sessions") as mock_load:
            mock_load.return_value = [
                ("89335547", "session-a"),
                ("89335547", "session-b"),
            ]
            sessions = downloader.resolve_sessions()
            assert len(sessions) == 2
            assert all(u == "89335547" for u, _ in sessions)

    def test_test_mode(self, mock_s3: dict, tmp_path: Path) -> None:
        """TEST mode returns exactly n_test_users distinct users."""
        config = _make_config(mode=DownloadMode.TEST, n_test_users=2, data_root=tmp_path)
        downloader = EMGDownloader(config)
        # Pre-cache metadata
        metadata_csv = _make_metadata_csv()
        (tmp_path / "metadata.csv").write_bytes(metadata_csv)

        sessions = downloader.resolve_sessions()
        users = {u for u, _ in sessions}
        assert len(users) == 2

    def test_all_mode(self, mock_s3: dict, tmp_path: Path) -> None:
        """ALL mode returns every session in metadata."""
        config = _make_config(mode=DownloadMode.ALL, data_root=tmp_path)
        downloader = EMGDownloader(config)
        metadata_csv = _make_metadata_csv()
        (tmp_path / "metadata.csv").write_bytes(metadata_csv)

        sessions = downloader.resolve_sessions()
        assert len(sessions) == 12  # 3 users * 4 sessions


# ---------------------------------------------------------------------------
# Dry-run test
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_no_upload(self, mock_s3: dict, tmp_path: Path) -> None:
        """dry_run=True should resolve sessions but upload nothing."""
        config = _make_config(mode=DownloadMode.ALL, dry_run=True, data_root=tmp_path)
        downloader = EMGDownloader(config)
        metadata_csv = _make_metadata_csv()
        (tmp_path / "metadata.csv").write_bytes(metadata_csv)

        # Inject mocked clients (moto-backed)
        downloader._source_client = mock_s3["source"]
        downloader._b2_client = mock_s3["b2"]

        uploaded = downloader.run()
        assert uploaded == []

        # Verify nothing was uploaded to B2
        objects = mock_s3["b2"].list_objects_v2(Bucket=B2_BUCKET)
        assert objects.get("KeyCount", 0) == 0


# ---------------------------------------------------------------------------
# Deduplication test
# ---------------------------------------------------------------------------


class TestDedup:
    def test_skips_existing(self, mock_s3: dict, tmp_path: Path) -> None:
        """Pre-populated registry entries should be skipped."""
        config = _make_config(mode=DownloadMode.ALL, data_root=tmp_path)
        downloader = EMGDownloader(config)
        metadata_csv = _make_metadata_csv()
        (tmp_path / "metadata.csv").write_bytes(metadata_csv)

        downloader._source_client = mock_s3["source"]
        downloader._b2_client = mock_s3["b2"]

        # Pre-populate registry with 4 of 12 sessions (user 89335547)
        registry = FileRegistry(
            b2_client=mock_s3["b2"],
            bucket_name=B2_BUCKET,
            registry_key=config.registry_key,
        )
        registry.load()
        for i in range(4):
            registry.add(
                make_record(
                    source_key=f"{SOURCE_PREFIX}/89335547/2021-01-01-89335547-session{i}.hdf5",
                    b2_key=f"emg2qwerty/89335547/2021-01-01-89335547-session{i}.hdf5",
                    size_bytes=100,
                    etag='"existing"',
                )
            )
        registry.save()

        # Force downloader to use fresh registry
        downloader._registry = None

        uploaded = downloader.run()
        # Only 8 of 12 should be uploaded (4 already exist)
        assert len(uploaded) == 8
