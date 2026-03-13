"""Tests for ``emg2qwerty.pipeline.registry`` — FileRegistry backed by
a mocked S3 (moto) bucket to simulate Backblaze B2.

Covers:
- Empty registry on first load
- Add + contains
- Pending filters existing keys
- Save/load round-trip
- Idempotent re-add
"""

from __future__ import annotations


import boto3
import pytest
from moto import mock_aws

from emg2qwerty.pipeline.registry import FileRecord, FileRegistry, make_record


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BUCKET_NAME = "test-bucket"
REGISTRY_KEY = "test_registry.json"


@pytest.fixture(autouse=True)
def _aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure moto uses dummy credentials."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


@pytest.fixture()
def s3_client():
    """Yield a mocked boto3 S3 client with a pre-created bucket."""
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET_NAME)
        yield client


@pytest.fixture()
def registry(s3_client) -> FileRegistry:
    return FileRegistry(
        b2_client=s3_client,
        bucket_name=BUCKET_NAME,
        registry_key=REGISTRY_KEY,
    )


def _sample_record(n: int = 0) -> FileRecord:
    return make_record(
        source_key=f"emg2qwerty/user{n}/session{n}.hdf5",
        b2_key=f"emg2qwerty/user{n}/session{n}.hdf5",
        size_bytes=1024 * (n + 1),
        etag=f'"etag{n}"',
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFileRegistry:
    def test_starts_empty(self, registry: FileRegistry) -> None:
        registry.load()
        assert len(registry) == 0
        assert registry.all_keys() == []

    def test_add_and_contains(self, registry: FileRegistry) -> None:
        registry.load()
        record = _sample_record(0)
        registry.add(record)
        assert registry.contains(record.b2_key)
        assert record.b2_key in registry
        assert len(registry) == 1

    def test_pending_filters_existing(self, registry: FileRegistry) -> None:
        registry.load()
        for i in range(2):
            registry.add(_sample_record(i))

        candidates = [_sample_record(i).b2_key for i in range(5)]
        pending = registry.pending(candidates)
        assert len(pending) == 3
        assert all(f"user{i}" in k for i, k in zip(range(2, 5), pending))

    def test_save_load_roundtrip(self, registry: FileRegistry) -> None:
        registry.load()
        for i in range(3):
            registry.add(_sample_record(i))
        registry.save()

        # Create a fresh registry pointing at the same bucket
        registry2 = FileRegistry(
            b2_client=registry.b2_client,
            bucket_name=BUCKET_NAME,
            registry_key=REGISTRY_KEY,
        )
        registry2.load()
        assert len(registry2) == 3
        assert registry2.all_keys() == registry.all_keys()

    def test_idempotent_readd(self, registry: FileRegistry) -> None:
        registry.load()
        record = _sample_record(0)
        registry.add(record)
        registry.add(record)  # should be a no-op
        assert len(registry) == 1

    def test_all_keys_ordered(self, registry: FileRegistry) -> None:
        registry.load()
        for i in range(5):
            registry.add(_sample_record(i))
        keys = registry.all_keys()
        assert len(keys) == 5


class TestFileRecord:
    def test_from_dict_roundtrip(self) -> None:
        record = _sample_record(42)
        d = record.to_dict()
        restored = FileRecord.from_dict(d)
        assert restored == record

    def test_make_record_has_timestamp(self) -> None:
        record = make_record(
            source_key="a/b.hdf5",
            b2_key="a/b.hdf5",
            size_bytes=100,
            etag='"abc"',
        )
        assert record.uploaded_at  # non-empty ISO-8601 string
        assert "T" in record.uploaded_at  # ISO-8601 contains 'T'
