"""EMGDownloader — streams HDF5 session files from Meta's public S3 bucket
directly into a Backblaze B2 bucket, with zero local disk usage.

The download is driven by ``metadata.csv`` which lists every user / session
pair in the dataset.  The three download modes (``--baseline``, ``--test``,
``--all``) select different subsets of that manifest.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import boto3
import botocore
import botocore.config
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from emg2qwerty.pipeline.config import DownloadConfig, DownloadMode
from emg2qwerty.pipeline.registry import FileRecord, FileRegistry, make_record

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

# Baseline user hard-coded from config/user/single_user.yaml
_BASELINE_USER = "89335547"


class EMGDownloader:
    """Orchestrates data transfer from Meta's public S3 → Backblaze B2.

    Parameters
    ----------
    config : DownloadConfig
        Fully validated pipeline configuration (Pydantic model).
    """

    def __init__(self, config: DownloadConfig) -> None:
        self.config = config
        self._source_client: boto3.client | None = None
        self._b2_client: boto3.client | None = None
        self._registry: FileRegistry | None = None

    # ------------------------------------------------------------------
    # boto3 client factories
    # ------------------------------------------------------------------

    def _make_source_client(self) -> boto3.client:
        """Create an anonymous S3 client for Meta's public bucket."""
        if self._source_client is None:
            cfg = botocore.config.Config(
                signature_version=botocore.UNSIGNED,
                region_name=self.config.source.region,
            )
            self._source_client = boto3.client("s3", config=cfg)
        return self._source_client

    def _make_b2_client(self) -> boto3.client:
        """Create an authenticated S3 client for Backblaze B2."""
        if self._b2_client is None:
            self._b2_client = boto3.client(
                "s3",
                endpoint_url=f"https://{self.config.b2.endpoint}",
                region_name=self.config.b2.region,
                aws_access_key_id=self.config.b2.key_id,
                aws_secret_access_key=self.config.b2.application_key,
            )
        return self._b2_client

    def _get_registry(self) -> FileRegistry:
        """Return the file registry, loading it from B2 on first call."""
        if self._registry is None:
            self._registry = FileRegistry(
                b2_client=self._make_b2_client(),
                bucket_name=self.config.b2.bucket_name,
                registry_key=self.config.registry_key,
            )
            self._registry.load()
        return self._registry

    # ------------------------------------------------------------------
    # Metadata & session resolution
    # ------------------------------------------------------------------

    def fetch_metadata(self) -> pd.DataFrame:
        """Download ``metadata.csv`` from the source S3 bucket into a
        DataFrame.  Falls back to a local cache under ``config.data_root``
        if present."""
        cache_path = self.config.data_root / "metadata.csv"
        if cache_path.exists():
            log.info("Using cached metadata from %s", cache_path)
            return pd.read_csv(cache_path)

        source_key = f"{self.config.source.prefix}/metadata.csv"
        log.info("Downloading metadata from s3://%s/%s", self.config.source.bucket, source_key)
        client = self._make_source_client()
        response = client.get_object(Bucket=self.config.source.bucket, Key=source_key)
        raw = response["Body"].read()

        # Cache locally for future runs
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(raw)
        log.info("Cached metadata to %s", cache_path)

        return pd.read_csv(io.BytesIO(raw))

    def _load_baseline_sessions(self) -> list[tuple[str, str]]:
        """Read baseline sessions directly from ``config/user/single_user.yaml``."""
        config_path = Path("config/user/single_user.yaml")
        if not config_path.exists():
            raise FileNotFoundError(f"Baseline config not found at {config_path}. Run from the project root directory.")
        with open(config_path) as f:
            data = yaml.safe_load(f)

        sessions: list[tuple[str, str]] = []
        for split in ("train", "val", "test"):
            for entry in data.get("dataset", {}).get(split, []) or []:
                sessions.append((str(entry["user"]), entry["session"]))
        return sessions

    def resolve_sessions(self, metadata: pd.DataFrame | None = None) -> list[tuple[str, str]]:
        """Return ``(user, session)`` pairs for the configured download mode.

        * **BASELINE** — the 18 sessions of user ``89335547`` from
          ``config/user/single_user.yaml``.
        * **TEST** — ``n_test_users`` randomly sampled users (all their
          sessions).
        * **ALL** — every user/session pair in ``metadata.csv``.
        """
        mode = self.config.mode

        if mode == DownloadMode.BASELINE:
            return self._load_baseline_sessions()

        # TEST and ALL both need metadata
        if metadata is None:
            metadata = self.fetch_metadata()

        if mode == DownloadMode.ALL:
            return list(zip(metadata["user"].astype(str), metadata["session"].astype(str)))

        if mode == DownloadMode.TEST:
            rng = np.random.RandomState(self.config.seed)
            unique_users = metadata["user"].unique()
            n = min(self.config.n_test_users, len(unique_users))
            sampled_users = set(rng.choice(unique_users, size=n, replace=False))
            subset = metadata[metadata["user"].isin(sampled_users)]
            return list(zip(subset["user"].astype(str), subset["session"].astype(str)))

        raise ValueError(f"Unknown download mode: {mode}")  # pragma: no cover

    # ------------------------------------------------------------------
    # Streaming transfer
    # ------------------------------------------------------------------

    @staticmethod
    def _b2_key(user: str, session: str) -> str:
        """Compute the B2 object key for a given user/session pair."""
        return f"emg2qwerty/{user}/{session}.hdf5"

    def _source_key(self, user: str, session: str) -> str:
        """Compute the source S3 object key."""
        return f"{self.config.source.prefix}/{user}/{session}.hdf5"

    def stream_one(self, user: str, session: str) -> FileRecord:
        """Stream a single HDF5 file from source S3 → B2.

        Returns a :class:`FileRecord` with size/etag metadata.
        """
        source_key = self._source_key(user, session)
        b2_key = self._b2_key(user, session)

        source = self._make_source_client()
        b2 = self._make_b2_client()

        # Stream from source
        response = source.get_object(
            Bucket=self.config.source.bucket,
            Key=source_key,
        )
        body = response["Body"]
        content_length = response.get("ContentLength", 0)

        # Upload to B2
        b2.upload_fileobj(
            Fileobj=body,
            Bucket=self.config.b2.bucket_name,
            Key=b2_key,
        )

        # Get the uploaded object's metadata for the record
        head = b2.head_object(
            Bucket=self.config.b2.bucket_name,
            Key=b2_key,
        )

        return make_record(
            source_key=source_key,
            b2_key=b2_key,
            size_bytes=head.get("ContentLength", content_length),
            etag=head.get("ETag", ""),
        )

    # ------------------------------------------------------------------
    # Main entry-point
    # ------------------------------------------------------------------

    def run(self) -> list[FileRecord]:
        """Execute the full download pipeline:

        1. Resolve the list of sessions for the chosen mode.
        2. Load the B2 registry and filter out already-uploaded files.
        3. Stream each pending file from source S3 → B2.
        4. Update and save the registry.

        Returns the list of newly uploaded :class:`FileRecord` instances.
        """
        log.info("Download mode: %s", self.config.mode.value)

        # 1. Resolve sessions
        metadata = None
        if self.config.mode != DownloadMode.BASELINE:
            metadata = self.fetch_metadata()
        sessions = self.resolve_sessions(metadata)
        log.info("Resolved %d sessions", len(sessions))

        # 2. Filter via registry
        registry = self._get_registry()
        b2_keys = [self._b2_key(u, s) for u, s in sessions]
        pending_keys = set(registry.pending(b2_keys))
        pending_sessions = [(u, s) for (u, s), k in zip(sessions, b2_keys) if k in pending_keys]
        log.info(
            "%d sessions already in B2, %d pending",
            len(sessions) - len(pending_sessions),
            len(pending_sessions),
        )

        if self.config.dry_run:
            log.info("[DRY RUN] Would upload %d files — exiting.", len(pending_sessions))
            return []

        # 3. Stream
        uploaded: list[FileRecord] = []
        for user, session in tqdm(pending_sessions, desc="Uploading", unit="file"):
            try:
                record = self.stream_one(user, session)
                registry.add(record)
                uploaded.append(record)
                log.info("Uploaded %s (%d bytes)", record.b2_key, record.size_bytes)
            except Exception:
                log.exception("Failed to upload %s/%s — skipping", user, session)

        # 4. Save registry
        if uploaded:
            registry.save()

        log.info(
            "Done. Uploaded %d / %d files (%.1f MB total).",
            len(uploaded),
            len(pending_sessions),
            sum(r.size_bytes for r in uploaded) / 1e6,
        )
        return uploaded
