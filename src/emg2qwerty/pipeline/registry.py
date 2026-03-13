"""File registry backed by a JSON object in Backblaze B2.

The registry tracks which HDF5 session files have already been uploaded to
the B2 bucket so that subsequent pipeline runs can skip them (deduplication).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from typing import Any

import botocore.exceptions

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FileRecord:
    """Metadata for a single file that has been uploaded to B2."""

    source_key: str
    b2_key: str
    size_bytes: int
    etag: str
    uploaded_at: str  # ISO-8601

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FileRecord":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FileRegistry:
    """A thin JSON-backed registry stored as an object in a B2 bucket.

    Usage::

        registry = FileRegistry(b2_client, "C147-project", "emg2qwerty_registry.json")
        registry.load()          # pull current state from B2
        if not registry.contains("emg2qwerty/89335547/session.hdf5"):
            registry.add(record)
        registry.save()          # push updated state back to B2
    """

    b2_client: Any  # boto3 S3 client
    bucket_name: str
    registry_key: str
    _records: dict[str, FileRecord] = field(default_factory=dict, init=False, repr=False)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Download the registry JSON from B2 into memory.

        If the registry object does not yet exist in the bucket, start
        with an empty registry (first run).
        """
        try:
            response = self.b2_client.get_object(
                Bucket=self.bucket_name,
                Key=self.registry_key,
            )
            payload = json.loads(response["Body"].read().decode("utf-8"))
            self._records = {key: FileRecord.from_dict(rec) for key, rec in payload.items()}
            log.info(
                "Loaded registry with %d records from s3://%s/%s",
                len(self._records),
                self.bucket_name,
                self.registry_key,
            )
        except botocore.exceptions.ClientError as exc:
            error_code = exc.response["Error"]["Code"]
            if error_code in ("404", "NoSuchKey"):
                log.info(
                    "No existing registry found in s3://%s/%s — starting fresh", self.bucket_name, self.registry_key
                )
                self._records = {}
            else:
                raise

    def save(self) -> None:
        """Serialize the in-memory registry and upload to B2."""
        payload = {key: rec.to_dict() for key, rec in self._records.items()}
        body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self.b2_client.put_object(
            Bucket=self.bucket_name,
            Key=self.registry_key,
            Body=BytesIO(body),
            ContentType="application/json",
        )
        log.info(
            "Saved registry with %d records to s3://%s/%s", len(self._records), self.bucket_name, self.registry_key
        )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def contains(self, b2_key: str) -> bool:
        """Return ``True`` if *b2_key* has already been uploaded."""
        return b2_key in self._records

    def add(self, record: FileRecord) -> None:
        """Register *record*.  Idempotent — re-adding an existing key is a no-op."""
        if record.b2_key in self._records:
            log.debug("Key %s already in registry — skipping", record.b2_key)
            return
        self._records[record.b2_key] = record

    def pending(self, candidates: list[str]) -> list[str]:
        """Return the subset of *candidates* (b2 keys) that are **not** yet
        in the registry."""
        return [k for k in candidates if k not in self._records]

    def all_keys(self) -> list[str]:
        """Return all registered B2 keys."""
        return list(self._records.keys())

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, b2_key: str) -> bool:
        return self.contains(b2_key)


def make_record(
    source_key: str,
    b2_key: str,
    size_bytes: int,
    etag: str,
) -> FileRecord:
    """Convenience factory that stamps ``uploaded_at`` with the current UTC
    time."""
    return FileRecord(
        source_key=source_key,
        b2_key=b2_key,
        size_bytes=size_bytes,
        etag=etag,
        uploaded_at=datetime.now(timezone.utc).isoformat(),
    )
