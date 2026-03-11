"""EMGDownloader — extracts HDF5 session files from Meta's public tar.gz
archive and saves them locally (+ optionally uploads to Backblaze B2).

The dataset is distributed as a single ~308 GB gzip-compressed tar archive
at ``https://fb-ctrl-oss.s3.amazonaws.com/emg2qwerty/emg2qwerty-data-2021-08.tar.gz``.
Individual S3 object access is **not** available, so we stream through the
entire archive with ``tarfile.open(mode='r|gz')``, extracting only the
sessions we need.

Download modes (``--baseline``, ``--test``, ``--all``) select different
subsets of the ~1 135 session files.
"""

from __future__ import annotations

import logging
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING

import boto3
import numpy as np
import requests
import yaml

from emg2qwerty.pipeline.config import DownloadConfig, DownloadMode
from emg2qwerty.pipeline.registry import FileRecord, FileRegistry, make_record

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

# Baseline user hard-coded from config/user/single_user.yaml
_BASELINE_USER = "89335547"


class EMGDownloader:
    """Orchestrates data extraction from Meta's public tar.gz archive.

    Files are saved locally under ``config.data_root`` and, when B2
    credentials are provided, also uploaded to Backblaze B2.

    Parameters
    ----------
    config : DownloadConfig
        Fully validated pipeline configuration (Pydantic model).
    """

    def __init__(self, config: DownloadConfig) -> None:
        self.config = config
        self._b2_client: boto3.client | None = None
        self._registry: FileRegistry | None = None

    # ------------------------------------------------------------------
    # boto3 client factory (B2 only — Meta is accessed via HTTPS)
    # ------------------------------------------------------------------

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
    # Session resolution
    # ------------------------------------------------------------------

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

    def resolve_sessions(self) -> list[tuple[str, str]]:
        """Return ``(user, session)`` pairs for the configured download mode.

        * **BASELINE** — the 18 sessions of user ``89335547`` from
          ``config/user/single_user.yaml``.
        * **TEST** — ``n_test_users`` randomly sampled users (all their
          sessions).  Requires ``metadata.csv`` in ``data_root``.
        * **ALL** — every user/session pair in ``metadata.csv``.
        """
        mode = self.config.mode

        if mode == DownloadMode.BASELINE:
            return self._load_baseline_sessions()

        # TEST and ALL need metadata.csv (must be pre-downloaded or available)
        import pandas as pd

        meta_path = self.config.data_root / "metadata.csv"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"metadata.csv not found at {meta_path}. Download it manually or run --baseline first."
            )
        metadata = pd.read_csv(meta_path)

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
    # Key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _b2_key(user: str, session: str) -> str:
        """Compute the B2 object key for a given user/session pair."""
        return f"emg2qwerty/{user}/{session}.hdf5"

    def _local_path(self, user: str, session: str) -> Path:
        """Compute the local file path for a given user/session pair."""
        return self.config.data_root / "emg2qwerty" / user / f"{session}.hdf5"

    # ------------------------------------------------------------------
    # Tar member → session name matching
    # ------------------------------------------------------------------

    def _match_member(self, member_name: str, wanted: set[str]) -> tuple[str, str] | None:
        """Try to match a tar member name against our wanted session set.

        Returns ``(user, session)`` if found, else ``None``.
        The tar contains entries like::

            emg2qwerty-data-2021-08/<session>.hdf5

        And we want to match against session names like::

            2021-06-03-1622765527-keystrokes-dca-study@1-<uuid>
        """
        # Strip the top-level tar directory and .hdf5 extension
        name = member_name
        prefix = self.config.source.tar_prefix + "/"
        if name.startswith(prefix):
            name = name[len(prefix) :]
        if name.endswith(".hdf5"):
            name = name[: -len(".hdf5")]

        if name in wanted:
            # Find the corresponding (user, session) from our resolved list
            # The session name is the key in `wanted`
            return None  # caller handles lookup
        return None

    # ------------------------------------------------------------------
    # Streaming extraction
    # ------------------------------------------------------------------

    def run(self) -> list[FileRecord]:
        """Execute the full download pipeline:

        1. Resolve the list of sessions for the chosen mode.
        2. Load the B2 registry and filter out already-downloaded files.
        3. Stream the tar.gz, extracting matching files locally + to B2.
        4. Update and save the registry.

        Returns the list of newly saved :class:`FileRecord` instances.
        """
        log.info("Download mode: %s", self.config.mode.value)

        # 1. Resolve sessions
        sessions = self.resolve_sessions()
        log.info("Resolved %d sessions", len(sessions))

        # Build lookup: session_name -> (user, session)
        session_lookup: dict[str, tuple[str, str]] = {}
        for user, session in sessions:
            session_lookup[session] = (user, session)

        # 2. Filter via registry — skip files already in B2 AND on local disk
        registry = self._get_registry()
        pending_sessions: dict[str, tuple[str, str]] = {}
        for session_name, (user, session) in session_lookup.items():
            b2_key = self._b2_key(user, session)
            local_path = self._local_path(user, session)
            if registry.contains(b2_key) and local_path.exists():
                continue
            pending_sessions[session_name] = (user, session)

        log.info(
            "%d sessions already complete, %d pending",
            len(sessions) - len(pending_sessions),
            len(pending_sessions),
        )

        if not pending_sessions:
            log.info("All sessions already downloaded. Nothing to do.")
            return []

        if self.config.dry_run:
            log.info("[DRY RUN] Would download %d files — exiting.", len(pending_sessions))
            for session_name in sorted(pending_sessions):
                user, session = pending_sessions[session_name]
                log.info("  %s → %s", session_name, self._local_path(user, session))
            return []

        # 3. Stream the tar.gz
        tar_url = self.config.source.tar_gz_url
        tar_prefix = self.config.source.tar_prefix + "/"
        log.info("Streaming tar.gz from %s ...", tar_url)
        log.info("Looking for %d session(s) — this may take a while for large archives.", len(pending_sessions))

        response = requests.get(tar_url, stream=True, timeout=60)
        response.raise_for_status()

        uploaded: list[FileRecord] = []
        found_count = 0

        try:
            with tarfile.open(fileobj=response.raw, mode="r|gz") as tar:
                for member in tar:
                    if not member.isfile():
                        continue

                    # Strip tar prefix to get the bare filename
                    name = member.name
                    if name.startswith(tar_prefix):
                        name = name[len(tar_prefix) :]

                    # Strip .hdf5 to compare with session names
                    if name.endswith(".hdf5"):
                        session_name = name[: -len(".hdf5")]
                    else:
                        continue

                    if session_name not in pending_sessions:
                        continue

                    # Found a match!
                    user, session = pending_sessions[session_name]
                    found_count += 1
                    log.info(
                        "[%d/%d] Extracting %s (%.1f MB)...",
                        found_count,
                        len(pending_sessions),
                        name,
                        member.size / 1e6,
                    )

                    # Extract file content
                    fileobj = tar.extractfile(member)
                    if fileobj is None:
                        log.warning("Could not extract %s — skipping", name)
                        continue
                    file_data = fileobj.read()

                    # Save locally
                    local_path = self._local_path(user, session)
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    local_path.write_bytes(file_data)
                    log.info("  Saved locally: %s (%.1f MB)", local_path, len(file_data) / 1e6)

                    # Upload to B2
                    b2_key = self._b2_key(user, session)
                    try:
                        b2 = self._make_b2_client()
                        b2.put_object(
                            Bucket=self.config.b2.bucket_name,
                            Key=b2_key,
                            Body=file_data,
                            ContentType="application/x-hdf5",
                        )
                        head = b2.head_object(
                            Bucket=self.config.b2.bucket_name,
                            Key=b2_key,
                        )
                        record = make_record(
                            source_key=member.name,
                            b2_key=b2_key,
                            size_bytes=head.get("ContentLength", len(file_data)),
                            etag=head.get("ETag", ""),
                        )
                        registry.add(record)
                        uploaded.append(record)
                        log.info("  Uploaded to B2: %s", b2_key)
                    except Exception:
                        log.exception("  Failed to upload %s to B2 — file saved locally only", b2_key)

                    # Remove from pending so we can stop early
                    del pending_sessions[session_name]
                    if not pending_sessions:
                        log.info("All %d target sessions extracted!", found_count)
                        break

        except Exception:
            log.exception("Error streaming tar.gz")
        finally:
            response.close()

        # 4. Save registry
        if uploaded:
            registry.save()

        log.info(
            "Done. Extracted %d / %d files (%.1f MB locally, %.1f MB to B2).",
            found_count,
            len(sessions),
            sum(
                self._local_path(u, s).stat().st_size
                for u, s in session_lookup.values()
                if self._local_path(u, s).exists()
            )
            / 1e6,
            sum(r.size_bytes for r in uploaded) / 1e6,
        )
        return uploaded
