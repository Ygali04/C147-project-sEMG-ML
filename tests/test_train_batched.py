"""Tests for ``emg2qwerty.pipeline.trainer`` — BatchTrainer.

Uses moto-mocked S3 to simulate B2 registry and profile resolution.

Covers:
- resolve_profiles reads from registry
- baseline mode filters to single user
- skips profiles with no data
- Hydra override generation
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import boto3
import pytest
from moto import mock_aws

from emg2qwerty.pipeline.config import B2Config, DownloadMode, TrainBatchConfig
from emg2qwerty.pipeline.registry import FileRegistry, make_record
from emg2qwerty.pipeline.trainer import BatchTrainer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

B2_BUCKET = "test-b2-bucket"
REGISTRY_KEY = "emg2qwerty_registry.json"


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
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=B2_BUCKET)
        yield client


def _seed_registry(client, users: dict[str, int]) -> None:
    """Seed the registry with ``users`` mapping ``{user_id: n_sessions}``."""
    registry = FileRegistry(
        b2_client=client,
        bucket_name=B2_BUCKET,
        registry_key=REGISTRY_KEY,
    )
    registry.load()
    for user_id, n in users.items():
        for i in range(n):
            registry.add(
                make_record(
                    source_key=f"emg2qwerty/{user_id}/session{i}.hdf5",
                    b2_key=f"emg2qwerty/{user_id}/session{i}.hdf5",
                    size_bytes=1024,
                    etag=f'"etag-{user_id}-{i}"',
                )
            )
    registry.save()


def _make_config(
    mode: DownloadMode = DownloadMode.BASELINE,
    n_test_users: int = 10,
) -> TrainBatchConfig:
    return TrainBatchConfig(
        mode=mode,
        n_test_users=n_test_users,
        b2=B2Config(
            endpoint="s3.us-east-1.backblazeb2.com",
            region="us-east-1",
            bucket_name=B2_BUCKET,
            bucket_id="test-id",
            key_id="testing",
            application_key="testing",
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestResolveProfiles:
    def test_reads_registry(self, mock_b2) -> None:
        """resolve_profiles returns all users from the registry."""
        _seed_registry(mock_b2, {"user_a": 3, "user_b": 2, "user_c": 1})
        config = _make_config(mode=DownloadMode.ALL)
        trainer = BatchTrainer(config)
        trainer._b2_client = mock_b2

        profiles = trainer.resolve_profiles()
        assert len(profiles) == 3
        assert "user_a" in profiles
        assert len(profiles["user_a"]) == 3

    def test_baseline_mode(self, mock_b2) -> None:
        """BASELINE mode returns only user 89335547."""
        _seed_registry(mock_b2, {"89335547": 4, "other_user": 2})
        config = _make_config(mode=DownloadMode.BASELINE)
        trainer = BatchTrainer(config)
        trainer._b2_client = mock_b2

        profiles = trainer.resolve_profiles()
        assert list(profiles.keys()) == ["89335547"]
        assert len(profiles["89335547"]) == 4

    def test_baseline_missing(self, mock_b2) -> None:
        """BASELINE mode with missing user returns empty dict."""
        _seed_registry(mock_b2, {"other_user": 2})
        config = _make_config(mode=DownloadMode.BASELINE)
        trainer = BatchTrainer(config)
        trainer._b2_client = mock_b2

        profiles = trainer.resolve_profiles()
        assert profiles == {}

    def test_test_mode(self, mock_b2) -> None:
        """TEST mode returns first n_test_users."""
        _seed_registry(
            mock_b2,
            {
                "user_a": 1,
                "user_b": 1,
                "user_c": 1,
                "user_d": 1,
                "user_e": 1,
            },
        )
        config = _make_config(mode=DownloadMode.TEST, n_test_users=3)
        trainer = BatchTrainer(config)
        trainer._b2_client = mock_b2

        profiles = trainer.resolve_profiles()
        assert len(profiles) == 3


class TestHydraOverrides:
    def test_build_overrides(self) -> None:
        overrides = BatchTrainer._build_hydra_overrides(
            user_id="89335547",
            session_paths=[Path("data/session0.hdf5")],
            model="bilstm_ctc",
        )
        assert "user=89335547" in overrides
        assert "model=bilstm_ctc" in overrides

    def test_build_overrides_with_checkpoint(self) -> None:
        overrides = BatchTrainer._build_hydra_overrides(
            user_id="89335547",
            session_paths=[Path("data/session0.hdf5")],
            model="cnn_bilstm_ctc",
            checkpoint="logs/best.ckpt",
        )
        assert "checkpoint=logs/best.ckpt" in overrides
        assert "model=cnn_bilstm_ctc" in overrides


class TestBatchTrainerRun:
    def test_no_profiles_warns(self, mock_b2) -> None:
        """If no profiles in registry, run() returns empty dict."""
        # Empty registry
        _seed_registry(mock_b2, {})
        config = _make_config(mode=DownloadMode.ALL)
        trainer = BatchTrainer(config)
        trainer._b2_client = mock_b2

        results = trainer.run()
        assert results == {}

    def test_skips_when_sync_fails(self, mock_b2) -> None:
        """If sync_profile returns empty, that user is marked failed."""
        _seed_registry(mock_b2, {"89335547": 2})
        config = _make_config(mode=DownloadMode.BASELINE)
        trainer = BatchTrainer(config)
        trainer._b2_client = mock_b2

        # Mock sync_profile to return empty (simulating download failure)
        with patch.object(trainer, "sync_profile", return_value=[]):
            results = trainer.run()
            assert results["89335547"] is False
