"""Tests for ``emg2qwerty.pipeline.downloader`` — EMGDownloader.

The downloader streams a gzip-compressed tar archive from Meta's public S3
via HTTPS.  Here we mock ``requests.get`` with an in-memory tar.gz so all
tests run offline and fast.

Covers:
- resolve_sessions for BASELINE mode (via single_user.yaml)
- resolve_sessions for TEST mode (n distinct users)
- resolve_sessions for ALL mode
- dry_run does not upload or write files
- dedup skips files already in registry + on disk
- files are saved locally AND uploaded to B2
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

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

B2_BUCKET = "test-b2-bucket"
TAR_PREFIX = "emg2qwerty-data-2021-08"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_tar_gz(sessions: list[str], prefix: str = TAR_PREFIX) -> bytes:
    """Create a gzip-compressed tar archive with fake HDF5 files."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for session in sessions:
            name = f"{prefix}/{session}.hdf5"
            data = f"FAKE_HDF5_{session}".encode()
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _make_metadata_csv(users_and_sessions: dict[int, list[str]]) -> bytes:
    """Build metadata.csv bytes from a dict of user → session list."""
    rows = []
    for user_id, sessions in users_and_sessions.items():
        for s in sessions:
            rows.append({"user": user_id, "session": s, "quality_check_tags": "[]"})
    return pd.DataFrame(rows).to_csv(index=False).encode()


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


@pytest.fixture()
def mock_b2():
    """Create a moto-backed B2 bucket."""
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=B2_BUCKET)
        yield client


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
            tar_gz_url="https://fake.example.com/data.tar.gz",
            tar_prefix=TAR_PREFIX,
        ),
    )


# ---------------------------------------------------------------------------
# Session resolution tests (no network needed)
# ---------------------------------------------------------------------------

BASELINE_SESSIONS = [
    ("89335547", "2021-06-03-1622765527-keystrokes-dca-study@1-0efbe614"),
    ("89335547", "2021-06-04-1622862148-keystrokes-dca-study@1-0efbe614"),
]


class TestResolveSessions:
    def test_baseline_mode(self, tmp_path: Path) -> None:
        """BASELINE mode uses single_user.yaml via _load_baseline_sessions."""
        config = _make_config(mode=DownloadMode.BASELINE, data_root=tmp_path)
        downloader = EMGDownloader(config)
        with patch.object(downloader, "_load_baseline_sessions", return_value=BASELINE_SESSIONS):
            sessions = downloader.resolve_sessions()
        assert len(sessions) == 2
        assert all(u == "89335547" for u, _ in sessions)

    def test_test_mode(self, tmp_path: Path) -> None:
        """TEST mode samples n_test_users from metadata.csv."""
        config = _make_config(mode=DownloadMode.TEST, n_test_users=2, data_root=tmp_path)
        metadata = _make_metadata_csv(
            {
                89335547: ["s1", "s2"],
                11111111: ["s3", "s4"],
                22222222: ["s5", "s6"],
            }
        )
        (tmp_path / "metadata.csv").write_bytes(metadata)

        downloader = EMGDownloader(config)
        sessions = downloader.resolve_sessions()
        users = {u for u, _ in sessions}
        assert len(users) == 2

    def test_all_mode(self, tmp_path: Path) -> None:
        """ALL mode returns every session in metadata."""
        config = _make_config(mode=DownloadMode.ALL, data_root=tmp_path)
        metadata = _make_metadata_csv(
            {
                89335547: ["s1", "s2"],
                11111111: ["s3", "s4"],
                22222222: ["s5", "s6"],
            }
        )
        (tmp_path / "metadata.csv").write_bytes(metadata)

        downloader = EMGDownloader(config)
        sessions = downloader.resolve_sessions()
        assert len(sessions) == 6


# ---------------------------------------------------------------------------
# Dry-run test
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_no_upload_no_local_files(self, mock_b2, tmp_path: Path) -> None:
        """dry_run=True should resolve sessions but create no files."""
        sessions = [("89335547", "ses1"), ("89335547", "ses2")]

        config = _make_config(mode=DownloadMode.BASELINE, dry_run=True, data_root=tmp_path)
        downloader = EMGDownloader(config)
        downloader._b2_client = mock_b2

        with patch.object(downloader, "resolve_sessions", return_value=sessions):
            uploaded = downloader.run()

        assert uploaded == []
        # No local files should have been created
        assert not list(tmp_path.rglob("*.hdf5"))
        # No objects in B2 (other than maybe the registry)
        objs = mock_b2.list_objects_v2(Bucket=B2_BUCKET)
        data_keys = [o["Key"] for o in objs.get("Contents", []) if not o["Key"].endswith(".json")]
        assert data_keys == []


# ---------------------------------------------------------------------------
# Full run (mocked HTTPS stream + mocked B2)
# ---------------------------------------------------------------------------


class TestRun:
    def test_extracts_and_uploads(self, mock_b2, tmp_path: Path) -> None:
        """run() should save matching HDF5 locally AND upload to B2."""
        sessions = [("89335547", "ses1"), ("89335547", "ses2")]
        tar_data = _build_tar_gz(["ses1", "ses2", "other_file"])

        config = _make_config(mode=DownloadMode.BASELINE, data_root=tmp_path)
        downloader = EMGDownloader(config)
        downloader._b2_client = mock_b2

        # Mock the requests.get to return our fake tar.gz
        mock_response = MagicMock()
        mock_response.raw = io.BytesIO(tar_data)
        mock_response.raise_for_status = MagicMock()
        mock_response.close = MagicMock()

        with (
            patch.object(downloader, "resolve_sessions", return_value=sessions),
            patch("emg2qwerty.pipeline.downloader.requests.get", return_value=mock_response),
        ):
            uploaded = downloader.run()

        # Should have uploaded 2 files
        assert len(uploaded) == 2

        # Check local files exist
        for _, ses in sessions:
            local = tmp_path / "emg2qwerty" / "89335547" / f"{ses}.hdf5"
            assert local.exists(), f"Local file missing: {local}"
            assert local.read_bytes() == f"FAKE_HDF5_{ses}".encode()

        # Check B2 objects exist
        objs = mock_b2.list_objects_v2(Bucket=B2_BUCKET)
        b2_keys = [o["Key"] for o in objs.get("Contents", []) if o["Key"].endswith(".hdf5")]
        assert len(b2_keys) == 2
        assert "emg2qwerty/89335547/ses1.hdf5" in b2_keys
        assert "emg2qwerty/89335547/ses2.hdf5" in b2_keys


# ---------------------------------------------------------------------------
# Deduplication test
# ---------------------------------------------------------------------------


class TestDedup:
    def test_skips_existing(self, mock_b2, tmp_path: Path) -> None:
        """Files already in registry AND on local disk should be skipped."""
        sessions = [("89335547", "ses1"), ("89335547", "ses2")]
        tar_data = _build_tar_gz(["ses1", "ses2"])

        config = _make_config(mode=DownloadMode.BASELINE, data_root=tmp_path)
        downloader = EMGDownloader(config)
        downloader._b2_client = mock_b2

        # Pre-populate registry with ses1
        registry = FileRegistry(b2_client=mock_b2, bucket_name=B2_BUCKET, registry_key=config.registry_key)
        registry.load()
        registry.add(
            make_record(
                source_key=f"{TAR_PREFIX}/ses1.hdf5",
                b2_key="emg2qwerty/89335547/ses1.hdf5",
                size_bytes=100,
                etag='"existing"',
            )
        )
        registry.save()

        # Also create the local file for ses1 (both must exist to skip)
        local_ses1 = tmp_path / "emg2qwerty" / "89335547" / "ses1.hdf5"
        local_ses1.parent.mkdir(parents=True, exist_ok=True)
        local_ses1.write_bytes(b"FAKE_HDF5_ses1")

        # Mock tar.gz stream
        mock_response = MagicMock()
        mock_response.raw = io.BytesIO(tar_data)
        mock_response.raise_for_status = MagicMock()
        mock_response.close = MagicMock()

        with (
            patch.object(downloader, "resolve_sessions", return_value=sessions),
            patch("emg2qwerty.pipeline.downloader.requests.get", return_value=mock_response),
        ):
            uploaded = downloader.run()

        # Only ses2 should be uploaded (ses1 was already in registry + local)
        assert len(uploaded) == 1
        assert uploaded[0].b2_key == "emg2qwerty/89335547/ses2.hdf5"

        # ses2 should exist locally
        local_ses2 = tmp_path / "emg2qwerty" / "89335547" / "ses2.hdf5"
        assert local_ses2.exists()
