"""EMGDownloader — extracts HDF5 session files from Meta's public tar.gz
archive and saves them locally (+ optionally uploads to Backblaze B2).

The dataset is distributed as a single ~308 GB gzip-compressed tar archive
at ``https://fb-ctrl-oss.s3.amazonaws.com/emg2qwerty/emg2qwerty-data-2021-08.tar.gz``.
Individual S3 object access is **not** available, so we stream through the
entire archive with ``tarfile.open(mode='r|gz')``, extracting only the
sessions we need.

Download modes (``--baseline``, ``--test``, ``--all``) select different
subsets of the 1,135 session files.

``metadata.csv`` layout
-----------------------
The archive contains one ``metadata.csv`` as the very last member (#1136).
Once extracted (``scripts/fetch_metadata.py``) it is committed to the repo
under ``data/metadata.csv`` so that every subsequent pipeline run can use
it without touching the archive again.  Columns::

    user,session,condition,duration_mins,num_keystrokes,num_prompts,quality_check_tags

* ``user``    — numeric user-id (e.g. ``89335547``)
* ``session`` — bare session name, matching the HDF5 stem inside the tar
                (e.g. ``2021-06-03-1622765527-keystrokes-dca-study@1-<uuid>``)
"""

from __future__ import annotations

import logging
import tarfile
from pathlib import Path
from typing import Callable

import boto3
import numpy as np
import requests
import yaml

from emg2qwerty.pipeline.config import DownloadConfig, DownloadMode
from emg2qwerty.pipeline.registry import FileRecord, FileRegistry, make_record

log = logging.getLogger(__name__)

# Baseline user hard-coded from config/user/single_user.yaml
_BASELINE_USER = "89335547"

# Canonical location of the committed metadata file
_METADATA_FILENAME = "metadata.csv"


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
    # metadata.csv helpers
    # ------------------------------------------------------------------

    def _load_metadata(self):  # -> pd.DataFrame
        """Load the committed ``data/metadata.csv``.

        Raises
        ------
        FileNotFoundError
            If the file has not yet been extracted.  Run
            ``uv run python scripts/fetch_metadata.py`` once to obtain it.
        """
        import pandas as pd

        meta_path = self.config.data_root / _METADATA_FILENAME
        if not meta_path.exists():
            raise FileNotFoundError(
                f"metadata.csv not found at {meta_path}.\n"
                "Extract it first:\n"
                "    uv run python scripts/fetch_metadata.py\n"
                "This streams the full 308 GB archive once and saves the 166 KB CSV.\n"
                "After that, commit data/metadata.csv to the repo."
            )
        return pd.read_csv(meta_path, dtype={"user": str})

    # ------------------------------------------------------------------
    # Session resolution for BASELINE mode
    # ------------------------------------------------------------------

    def _load_baseline_sessions(self) -> set[str]:
        """Read baseline session names from ``config/user/single_user.yaml``.

        Returns a set of bare session names (no ``.hdf5``) to match against
        tar member filenames.
        """
        config_path = Path("config/user/single_user.yaml")
        if not config_path.exists():
            raise FileNotFoundError(f"Baseline config not found at {config_path}. Run from the project root directory.")
        with open(config_path) as f:
            data = yaml.safe_load(f)

        sessions: set[str] = set()
        for split in ("train", "val", "test"):
            for entry in data.get("dataset", {}).get(split, []) or []:
                sessions.add(str(entry["session"]))
        return sessions

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
    # Member filter factories
    # ------------------------------------------------------------------

    def _make_filter(self) -> Callable[[str], tuple[str, str] | None]:
        """Return a callable that maps a bare member filename → (user, session) or None.

        The filter encapsulates all download-mode logic so the streaming loop
        stays simple and mode-agnostic.

        Returns
        -------
        filter_fn
            ``filter_fn(name)`` receives the bare filename (stem stripped of
            the top-level tar directory) and returns ``(user, session)`` when
            the file should be downloaded, or ``None`` to skip it.
        """
        mode = self.config.mode

        # ---- BASELINE ---------------------------------------------------
        if mode == DownloadMode.BASELINE:
            wanted: set[str] = self._load_baseline_sessions()
            log.info("BASELINE mode — targeting %d sessions for user %s", len(wanted), _BASELINE_USER)

            def _baseline_filter(name: str) -> tuple[str, str] | None:
                if not name.endswith(".hdf5"):
                    return None
                session = name[: -len(".hdf5")]
                if session in wanted:
                    return (_BASELINE_USER, session)
                return None

            return _baseline_filter

        # ---- ALL / TEST — need metadata.csv ----------------------------
        metadata = self._load_metadata()

        if mode == DownloadMode.ALL:
            # Build a dict: session_name → user (string)
            session_to_user: dict[str, str] = dict(zip(metadata["session"].astype(str), metadata["user"].astype(str)))
            log.info(
                "ALL mode — targeting all %d sessions across %d users",
                len(session_to_user),
                metadata["user"].nunique(),
            )

            def _all_filter(name: str) -> tuple[str, str] | None:
                if not name.endswith(".hdf5"):
                    return None
                session = name[: -len(".hdf5")]
                user = session_to_user.get(session)
                if user is None:
                    return None
                return (user, session)

            return _all_filter

        if mode == DownloadMode.TEST:
            rng = np.random.RandomState(self.config.seed)
            unique_users = metadata["user"].astype(str).unique()
            n = min(self.config.n_test_users, len(unique_users))
            sampled_users: set[str] = set(rng.choice(unique_users, size=n, replace=False))
            subset = metadata[metadata["user"].astype(str).isin(sampled_users)]
            session_to_user_test: dict[str, str] = dict(zip(subset["session"].astype(str), subset["user"].astype(str)))
            log.info(
                "TEST mode — targeting %d sessions from %d sampled users (seed=%d)",
                len(session_to_user_test),
                len(sampled_users),
                self.config.seed,
            )

            def _test_filter(name: str) -> tuple[str, str] | None:
                if not name.endswith(".hdf5"):
                    return None
                session = name[: -len(".hdf5")]
                user = session_to_user_test.get(session)
                if user is None:
                    return None
                return (user, session)

            return _test_filter

        raise ValueError(f"Unknown download mode: {mode}")  # pragma: no cover

    # ------------------------------------------------------------------
    # Streaming extraction
    # ------------------------------------------------------------------

    def run(self) -> list[FileRecord]:
        """Execute the full download pipeline:

        1. Build a member-filter for the chosen mode (BASELINE/TEST/ALL).
        2. Load the B2 file registry (deduplicate already-uploaded files).
        3. Stream the tar.gz, extracting matching HDF5 files locally + to B2.
        4. Persist the updated registry back to B2.

        Returns the list of newly saved :class:`FileRecord` instances.
        """
        log.info("Download mode: %s", self.config.mode.value)

        # 1. Build filter (resolves sessions / loads metadata)
        member_filter = self._make_filter()

        # 2. Load deduplication registry
        registry = self._get_registry()

        tar_url = self.config.source.tar_gz_url
        tar_prefix = self.config.source.tar_prefix + "/"

        log.info("Streaming tar.gz from %s", tar_url)
        if self.config.dry_run:
            log.info("[DRY RUN] Will not download or upload — listing matches only.")

        # Connect timeout 60 s; no read timeout (stream runs for many hours)
        response = requests.get(tar_url, stream=True, timeout=(60, None))
        response.raise_for_status()

        uploaded: list[FileRecord] = []
        found_count = 0
        skipped_registry = 0

        try:
            with tarfile.open(fileobj=response.raw, mode="r|gz") as tar:
                for member in tar:
                    if not member.isfile():
                        continue

                    # Strip top-level tar directory prefix
                    name = member.name
                    if name.startswith(tar_prefix):
                        name = name[len(tar_prefix) :]

                    # Apply mode-specific filter
                    result = member_filter(name)
                    if result is None:
                        continue

                    user, session = result

                    # Skip files already complete (in registry AND on disk)
                    b2_key = self._b2_key(user, session)
                    local_path = self._local_path(user, session)
                    if registry.contains(b2_key) and local_path.exists():
                        skipped_registry += 1
                        log.debug("  Skipping (already complete): %s", b2_key)
                        continue

                    found_count += 1
                    log.info(
                        "[#%d] %s  user=%s  (%.1f MB)",
                        found_count,
                        name,
                        user,
                        member.size / 1e6,
                    )

                    if self.config.dry_run:
                        log.info("  [DRY RUN] → %s  (B2: %s)", local_path, b2_key)
                        continue

                    # Extract file bytes
                    fileobj = tar.extractfile(member)
                    if fileobj is None:
                        log.warning("  Could not extract %s — skipping", name)
                        continue
                    file_data = fileobj.read()

                    # Save locally
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    local_path.write_bytes(file_data)
                    log.info("  Saved locally: %s (%.1f MB)", local_path, len(file_data) / 1e6)

                    # Upload to B2
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

        except Exception:
            log.exception("Error during tar.gz stream after finding %d file(s)", found_count)
        finally:
            response.close()

        # 4. Persist updated registry
        if uploaded:
            registry.save()

        log.info(
            "Done. Extracted %d new file(s), skipped %d already-complete.",
            found_count if not self.config.dry_run else 0,
            skipped_registry,
        )
        if uploaded:
            log.info("Total uploaded to B2: %.2f GB", sum(r.size_bytes for r in uploaded) / 1e9)

        return uploaded
