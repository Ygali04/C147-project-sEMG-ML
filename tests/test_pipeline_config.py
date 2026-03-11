"""Tests for ``emg2qwerty.pipeline.config`` Pydantic models.

Covers:
- B2Config: valid construction, missing fields, endpoint validation
- SourceS3Config: defaults
- DownloadMode: all enum values
- DownloadConfig: defaults, nested validation
- TrainBatchConfig: batch_size_profiles > 0
- Hypothesis property test: random endpoint strings
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from emg2qwerty.pipeline.config import (
    B2Config,
    DownloadConfig,
    DownloadMode,
    SourceS3Config,
    TrainBatchConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def valid_b2_kwargs() -> dict:
    return {
        "endpoint": "s3.us-west-004.backblazeb2.com",
        "region": "us-west-004",
        "bucket_name": "C147-project",
        "bucket_id": "857ef5759126dac299c10e13",
        "key_id": "test-key-id",
        "application_key": "test-app-key",
    }


# ---------------------------------------------------------------------------
# B2Config
# ---------------------------------------------------------------------------


class TestB2Config:
    def test_valid(self, valid_b2_kwargs: dict) -> None:
        cfg = B2Config(**valid_b2_kwargs)
        assert cfg.endpoint == "s3.us-west-004.backblazeb2.com"
        assert cfg.bucket_name == "C147-project"

    def test_missing_key_id(self, valid_b2_kwargs: dict) -> None:
        del valid_b2_kwargs["key_id"]
        with pytest.raises(ValidationError):
            B2Config(**valid_b2_kwargs)

    def test_missing_application_key(self, valid_b2_kwargs: dict) -> None:
        del valid_b2_kwargs["application_key"]
        with pytest.raises(ValidationError):
            B2Config(**valid_b2_kwargs)

    def test_invalid_endpoint(self, valid_b2_kwargs: dict) -> None:
        valid_b2_kwargs["endpoint"] = "not-a-valid-endpoint.com"
        with pytest.raises(ValidationError, match="does not match expected B2 pattern"):
            B2Config(**valid_b2_kwargs)

    def test_endpoint_must_start_with_s3(self, valid_b2_kwargs: dict) -> None:
        valid_b2_kwargs["endpoint"] = "x3.us-west-004.backblazeb2.com"
        with pytest.raises(ValidationError):
            B2Config(**valid_b2_kwargs)


# ---------------------------------------------------------------------------
# SourceS3Config
# ---------------------------------------------------------------------------


class TestSourceS3Config:
    def test_defaults(self) -> None:
        cfg = SourceS3Config()
        assert cfg.bucket == "fb-baml-public"
        assert cfg.prefix == "emg2qwerty"
        assert cfg.region == "us-east-1"
        assert cfg.anonymous is True

    def test_custom_values(self) -> None:
        cfg = SourceS3Config(bucket="my-bucket", prefix="custom/prefix")
        assert cfg.bucket == "my-bucket"
        assert cfg.prefix == "custom/prefix"


# ---------------------------------------------------------------------------
# DownloadMode
# ---------------------------------------------------------------------------


class TestDownloadMode:
    def test_baseline(self) -> None:
        assert DownloadMode("baseline") == DownloadMode.BASELINE

    def test_test(self) -> None:
        assert DownloadMode("test") == DownloadMode.TEST

    def test_all(self) -> None:
        assert DownloadMode("all") == DownloadMode.ALL

    def test_invalid(self) -> None:
        with pytest.raises(ValueError):
            DownloadMode("invalid")


# ---------------------------------------------------------------------------
# DownloadConfig
# ---------------------------------------------------------------------------


class TestDownloadConfig:
    def test_defaults(self, valid_b2_kwargs: dict) -> None:
        cfg = DownloadConfig(b2=B2Config(**valid_b2_kwargs))
        assert cfg.mode == DownloadMode.BASELINE
        assert cfg.n_test_users == 10
        assert cfg.seed == 1501
        assert cfg.dry_run is False
        assert cfg.registry_key == "emg2qwerty_registry.json"

    def test_test_mode(self, valid_b2_kwargs: dict) -> None:
        cfg = DownloadConfig(
            mode=DownloadMode.TEST,
            n_test_users=5,
            b2=B2Config(**valid_b2_kwargs),
        )
        assert cfg.mode == DownloadMode.TEST
        assert cfg.n_test_users == 5

    def test_n_test_users_must_be_positive(self, valid_b2_kwargs: dict) -> None:
        with pytest.raises(ValidationError):
            DownloadConfig(
                n_test_users=0,
                b2=B2Config(**valid_b2_kwargs),
            )


# ---------------------------------------------------------------------------
# TrainBatchConfig
# ---------------------------------------------------------------------------


class TestTrainBatchConfig:
    def test_defaults(self, valid_b2_kwargs: dict) -> None:
        cfg = TrainBatchConfig(b2=B2Config(**valid_b2_kwargs))
        assert cfg.mode == DownloadMode.BASELINE
        assert cfg.batch_size_profiles == 1
        assert cfg.checkpoint is None

    def test_batch_size_must_be_positive(self, valid_b2_kwargs: dict) -> None:
        with pytest.raises(ValidationError):
            TrainBatchConfig(
                batch_size_profiles=0,
                b2=B2Config(**valid_b2_kwargs),
            )

    def test_with_checkpoint(self, valid_b2_kwargs: dict) -> None:
        cfg = TrainBatchConfig(
            b2=B2Config(**valid_b2_kwargs),
            checkpoint="logs/2026-03-11/best.ckpt",
        )
        assert cfg.checkpoint == "logs/2026-03-11/best.ckpt"


# ---------------------------------------------------------------------------
# Hypothesis property test: B2 endpoint validation
# ---------------------------------------------------------------------------


@given(
    region=st.text(
        alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789-"),
        min_size=1,
        max_size=20,
    )
)
@settings(max_examples=50)
def test_b2_endpoint_valid_regions(region: str) -> None:
    """Any endpoint matching ``s3.<region>.backblazeb2.com`` should pass
    validation."""
    endpoint = f"s3.{region}.backblazeb2.com"
    cfg = B2Config(
        endpoint=endpoint,
        region=region,
        bucket_name="test",
        bucket_id="abc123",
        key_id="k",
        application_key="s",
    )
    assert cfg.endpoint == endpoint


@given(
    prefix=st.text(
        alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz.:-"),
        min_size=1,
        max_size=30,
    ).filter(lambda s: not s.startswith("s3.") or not s.endswith(".backblazeb2.com"))
)
@settings(max_examples=50)
def test_b2_endpoint_rejects_invalid(prefix: str) -> None:
    """Random strings that do NOT match the B2 pattern should be rejected."""
    with pytest.raises(ValidationError):
        B2Config(
            endpoint=prefix,
            region="us-west-004",
            bucket_name="test",
            bucket_id="abc123",
            key_id="k",
            application_key="s",
        )
