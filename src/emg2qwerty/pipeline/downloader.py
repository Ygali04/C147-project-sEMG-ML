"""EMGDownloader — extracts HDF5 session files from Meta's public tar.gz
archive and saves them locally (+ optionally uploads to Backblaze B2).

The dataset is distributed as a single ~308 GB gzip-compressed tar archive
at ``https://fb-ctrl-oss.s3.amazonaws.com/emg2qwerty/emg2qwerty-data-2021-08.tar.gz``.
Individual S3 object access is **not** available, so we stream through the
entire archive with ``tarfile.open(mode='r|gz')``, extracting only the
sessions we need.

Design: metadata.csv as a byproduct, never a prerequisite
----------------------------------------------------------
``metadata.csv`` is the very last member (#1136) of the archive.  Rather than
requiring it to exist before a run, the downloader operates in one of two
states:

**Metadata available** (``data/metadata.csv`` exists in the repo):
    Numeric user-IDs are read from the CSV.  Session → user lookup is exact.
    ``--test`` mode samples users by their canonical numeric ID.

**Metadata not yet available** (first run, or fresh clone):
    User identifiers are parsed directly from the tar member filename:

    * ``…-keystrokes-71409769.hdf5``          → user ``71409769``
    * ``…-keystrokes-dca-study@1-{uuid}.hdf5`` → user ``{uuid}``
    * ``…-keystrokes.hdf5``                    → user ``pilot``

    ``--test`` mode uses reservoir sampling over the identifiers seen during
    the stream (no pre-knowledge required).

In **both** states, whenever the stream reaches ``metadata.csv``, the
downloader:

1. Saves it to ``data/metadata.csv`` locally.
2. Uploads it to B2 as ``emg2qwerty/metadata.csv``.
3. Logs a reminder to commit the file to the repo.

This means a single ``make download-all`` run obtains all HDF5 files **and**
``metadata.csv`` in one pass, with no pre-flight requirements.

Parallel uploads
----------------
The gzip stream is inherently sequential.  B2 uploads are independent network
I/O, so they run on a :class:`~concurrent.futures.ThreadPoolExecutor` while
the main thread continues reading the tar stream.  Each worker thread has its
own thread-local boto3 client.  A :class:`threading.Semaphore` caps peak RAM
usage at ``_UPLOAD_WORKERS × 2`` in-flight file buffers (~1–2 GB).
"""

from __future__ import annotations

import io
import logging
import re
import tarfile
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import boto3
from boto3.s3.transfer import TransferConfig
import numpy as np
import requests
import yaml

from emg2qwerty.pipeline.config import DownloadConfig, DownloadMode
from emg2qwerty.pipeline.registry import FileRecord, FileRegistry, make_record

log = logging.getLogger(__name__)

# Baseline user hard-coded from config/user/single_user.yaml
_BASELINE_USER = "89335547"

# Canonical location of the committed metadata file (relative to data_root)
_METADATA_FILENAME = "metadata.csv"

# B2 key under which metadata.csv is also stored in the bucket
_METADATA_B2_KEY = "emg2qwerty/metadata.csv"

# Number of concurrent B2 upload workers.
_UPLOAD_WORKERS = 4

# Multipart threshold / chunk size for boto3 S3 Transfer Manager.
_MULTIPART_THRESHOLD = 64 * 1024 * 1024  # 64 MB
_MULTIPART_CHUNKSIZE = 64 * 1024 * 1024  # 64 MB

_TRANSFER_CFG = TransferConfig(
    multipart_threshold=_MULTIPART_THRESHOLD,
    multipart_chunksize=_MULTIPART_CHUNKSIZE,
    max_concurrency=1,  # one HTTP connection per worker thread
    use_threads=False,  # we manage our own pool
)

# Thread-local storage so each worker thread gets its own boto3 client.
_thread_local = threading.local()

# Regex patterns for parsing user identifiers from HDF5 filenames.
_RE_NUMERIC = re.compile(r"keystrokes-(?P<user>\d{6,12})\.hdf5$")
_RE_UUID = re.compile(
    r"keystrokes-dca-study@\d+-(?P<user>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.hdf5$"
)


def _parse_user_from_name(name: str) -> str:
    """Extract a user identifier from a bare HDF5 tar-member filename.

    Returns the numeric user-id, UUID, or ``"pilot"`` for files without
    a user suffix.
    """
    m = _RE_UUID.search(name) or _RE_NUMERIC.search(name)
    if m:
        return m.group("user")
    if name.endswith(".hdf5"):
        return "pilot"
    return "unknown"


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
        self._registry: FileRegistry | None = None

    # ------------------------------------------------------------------
    # boto3 client — thread-local so workers never share a connection
    # ------------------------------------------------------------------

    def _make_b2_client(self) -> boto3.client:
        """Return the boto3 S3 client for the current thread."""
        if not hasattr(_thread_local, "b2_client"):
            _thread_local.b2_client = boto3.client(
                "s3",
                endpoint_url=f"https://{self.config.b2.endpoint}",
                region_name=self.config.b2.region,
                aws_access_key_id=self.config.b2.key_id,
                aws_secret_access_key=self.config.b2.application_key,
            )
        return _thread_local.b2_client

    def _get_registry(self) -> FileRegistry:
        if self._registry is None:
            self._registry = FileRegistry(
                b2_client=self._make_b2_client(),
                bucket_name=self.config.b2.bucket_name,
                registry_key=self.config.registry_key,
            )
            self._registry.load()
        return self._registry

    # ------------------------------------------------------------------
    # metadata.csv — opportunistic load / capture
    # ------------------------------------------------------------------

    @property
    def _metadata_local_path(self) -> Path:
        return self.config.data_root / _METADATA_FILENAME

    def _try_load_metadata(self):  # -> pd.DataFrame | None
        """Load ``data/metadata.csv`` if it exists; return None otherwise."""
        if not self._metadata_local_path.exists():
            return None
        import pandas as pd

        log.info("Loaded metadata from %s", self._metadata_local_path)
        return pd.read_csv(self._metadata_local_path, dtype={"user": str})

    def _capture_metadata(self, raw_bytes: bytes) -> None:
        """Save metadata.csv locally and upload it to B2.

        Called opportunistically when the tar stream reaches the
        ``metadata.csv`` member (always the last file, #1136).
        """
        # Save locally
        self._metadata_local_path.parent.mkdir(parents=True, exist_ok=True)
        self._metadata_local_path.write_bytes(raw_bytes)
        log.info(
            "Captured metadata.csv → saved locally (%d KB). Please commit data/metadata.csv to the repo.",
            len(raw_bytes) // 1024,
        )

        # Upload to B2
        try:
            b2 = self._make_b2_client()
            b2.put_object(
                Bucket=self.config.b2.bucket_name,
                Key=_METADATA_B2_KEY,
                Body=raw_bytes,
                ContentType="text/csv",
            )
            log.info("Uploaded metadata.csv to B2: %s", _METADATA_B2_KEY)
        except Exception:
            log.exception("Failed to upload metadata.csv to B2 — local copy is safe")

    # ------------------------------------------------------------------
    # Session resolution for BASELINE mode
    # ------------------------------------------------------------------

    def _load_baseline_sessions(self) -> set[str]:
        """Read baseline session names from ``config/user/single_user.yaml``."""
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
        return f"emg2qwerty/{user}/{session}.hdf5"

    def _local_path(self, user: str, session: str) -> Path:
        return self.config.data_root / "emg2qwerty" / user / f"{session}.hdf5"

    # ------------------------------------------------------------------
    # Member filter factories
    # ------------------------------------------------------------------

    def _make_filter(self, metadata) -> Callable[[str], tuple[str, str] | None]:
        """Return a filter: bare filename → (user, session) | None.

        Parameters
        ----------
        metadata
            A ``pd.DataFrame`` with columns ``user`` and ``session`` if
            ``data/metadata.csv`` was found, or ``None`` for first-run mode.
        """
        mode = self.config.mode

        # ---- BASELINE — always reads single_user.yaml, no metadata needed ----
        if mode == DownloadMode.BASELINE:
            wanted = self._load_baseline_sessions()
            log.info("BASELINE mode — %d sessions for user %s", len(wanted), _BASELINE_USER)

            def _baseline_filter(name: str) -> tuple[str, str] | None:
                if not name.endswith(".hdf5"):
                    return None
                session = name[: -len(".hdf5")]
                return (_BASELINE_USER, session) if session in wanted else None

            return _baseline_filter

        # ---- ALL / TEST — two sub-paths depending on metadata availability ----

        if metadata is not None:
            # ---- Metadata available: exact numeric user-ID lookup ----
            session_to_user: dict[str, str] = dict(zip(metadata["session"].astype(str), metadata["user"].astype(str)))

            if mode == DownloadMode.ALL:
                log.info(
                    "ALL mode (metadata loaded) — %d sessions / %d users",
                    len(session_to_user),
                    metadata["user"].nunique(),
                )

                def _all_meta_filter(name: str) -> tuple[str, str] | None:
                    if not name.endswith(".hdf5"):
                        return None
                    session = name[: -len(".hdf5")]
                    user = session_to_user.get(session)
                    return (user, session) if user is not None else None

                return _all_meta_filter

            if mode == DownloadMode.TEST:
                rng = np.random.RandomState(self.config.seed)
                unique_users = metadata["user"].astype(str).unique()
                n = min(self.config.n_test_users, len(unique_users))
                sampled: set[str] = set(rng.choice(unique_users, size=n, replace=False))
                subset = metadata[metadata["user"].astype(str).isin(sampled)]
                test_lookup: dict[str, str] = dict(zip(subset["session"].astype(str), subset["user"].astype(str)))
                log.info(
                    "TEST mode (metadata loaded) — %d sessions from %d users (seed=%d)",
                    len(test_lookup),
                    len(sampled),
                    self.config.seed,
                )

                def _test_meta_filter(name: str) -> tuple[str, str] | None:
                    if not name.endswith(".hdf5"):
                        return None
                    session = name[: -len(".hdf5")]
                    user = test_lookup.get(session)
                    return (user, session) if user is not None else None

                return _test_meta_filter

        else:
            # ---- No metadata: parse user ID from filename ----
            if mode == DownloadMode.ALL:
                log.info(
                    "ALL mode (no metadata.csv yet) — streaming everything; "
                    "user IDs parsed from filenames. metadata.csv will be captured "
                    "at the end of the stream and saved for future runs."
                )

                def _all_nomd_filter(name: str) -> tuple[str, str] | None:
                    if not name.endswith(".hdf5"):
                        return None
                    user = _parse_user_from_name(name)
                    session = name[: -len(".hdf5")]
                    return (user, session)

                return _all_nomd_filter

            if mode == DownloadMode.TEST:
                # Reservoir sampling — select n_test_users unique identifiers
                # as we encounter them in the stream.
                n_users = self.config.n_test_users
                rng = np.random.RandomState(self.config.seed)
                accepted: set[str] = set()
                reservoir: list[tuple[float, str]] = []

                log.info(
                    "TEST mode (no metadata.csv yet) — reservoir-sampling %d users from the stream (seed=%d)",
                    n_users,
                    self.config.seed,
                )

                def _test_nomd_filter(name: str) -> tuple[str, str] | None:
                    if not name.endswith(".hdf5"):
                        return None
                    user = _parse_user_from_name(name)
                    session = name[: -len(".hdf5")]

                    if user in accepted:
                        return (user, session)

                    priority = rng.random()
                    if len(accepted) < n_users:
                        accepted.add(user)
                        reservoir.append((priority, user))
                        return (user, session)

                    # Evict worst (highest priority value = lowest quality)
                    worst = max(reservoir, key=lambda x: x[0])
                    if priority < worst[0]:
                        reservoir.remove(worst)
                        accepted.discard(worst[1])
                        reservoir.append((priority, user))
                        accepted.add(user)
                        return (user, session)

                    return None

                return _test_nomd_filter

        raise ValueError(f"Unknown download mode: {mode}")  # pragma: no cover

    # ------------------------------------------------------------------
    # B2 upload worker (runs in a thread-pool thread)
    # ------------------------------------------------------------------

    def _upload_one(
        self,
        source_key: str,
        b2_key: str,
        file_data: bytes,
        semaphore: threading.Semaphore,
    ) -> FileRecord:
        """Upload *file_data* to B2 via multipart and return a FileRecord.

        The *semaphore* is released when the upload completes so the main
        thread can immediately read the next file from the tar stream.
        """
        try:
            b2 = self._make_b2_client()
            b2.upload_fileobj(
                io.BytesIO(file_data),
                self.config.b2.bucket_name,
                b2_key,
                ExtraArgs={"ContentType": "application/x-hdf5"},
                Config=_TRANSFER_CFG,
            )
            head = b2.head_object(Bucket=self.config.b2.bucket_name, Key=b2_key)
            record = make_record(
                source_key=source_key,
                b2_key=b2_key,
                size_bytes=head.get("ContentLength", len(file_data)),
                etag=head.get("ETag", ""),
            )
            log.info("  ✓ B2: %s (%.1f MB)", b2_key, len(file_data) / 1e6)
            return record
        finally:
            semaphore.release()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> list[FileRecord]:
        """Execute the full download pipeline in a single tar.gz pass.

        Steps
        -----
        1.  Attempt to load ``data/metadata.csv`` (may not exist yet).
        2.  Build a mode-specific member filter (exact lookup if metadata
            available; filename-parsing / reservoir-sampling otherwise).
        3.  Load the B2 file registry for deduplication.
        4.  Stream the tar.gz:

            * For each matching ``.hdf5`` member → save locally, submit
              upload to the thread pool (non-blocking).
            * When ``metadata.csv`` is reached → capture it locally + B2.

        5.  Wait for all pending uploads; persist the registry.

        Returns the list of newly saved :class:`FileRecord` instances.
        """
        log.info("Download mode: %s", self.config.mode.value)

        # 1. Try to load existing metadata (graceful if absent)
        metadata = self._try_load_metadata()
        if metadata is None:
            log.info(
                "data/metadata.csv not found — will parse user IDs from filenames. "
                "metadata.csv will be saved automatically when encountered in the stream."
            )

        # 2. Build member filter
        member_filter = self._make_filter(metadata)

        # 3. Load deduplication registry
        registry = self._get_registry()
        registry_lock = threading.Lock()

        tar_url = self.config.source.tar_gz_url
        tar_prefix = self.config.source.tar_prefix + "/"

        log.info("Streaming %s", tar_url)
        log.info(
            "Parallel B2 uploads: %d workers, multipart threshold %d MB",
            _UPLOAD_WORKERS,
            _MULTIPART_THRESHOLD // (1024 * 1024),
        )
        if self.config.save_local:
            log.info("--save-local: HDF5 files will also be written to %s", self.config.data_root)
        else:
            log.info("Streaming directly to B2 — no local HDF5 copies will be made.")
        if self.config.dry_run:
            log.info("[DRY RUN] No files will be written or uploaded.")

        # Semaphore: cap in-flight file data at _UPLOAD_WORKERS*2 buffers (~1-2 GB)
        in_flight = threading.Semaphore(_UPLOAD_WORKERS * 2)

        uploaded: list[FileRecord] = []
        futures: list[Future] = []
        found_count = 0
        skipped_count = 0

        # Connect timeout 60 s; no read timeout (stream runs for many hours)
        response = requests.get(tar_url, stream=True, timeout=(60, None))
        response.raise_for_status()

        try:
            with ThreadPoolExecutor(max_workers=_UPLOAD_WORKERS) as pool:
                with tarfile.open(fileobj=response.raw, mode="r|gz") as tar:
                    for member in tar:
                        if not member.isfile():
                            continue

                        # Strip top-level tar directory prefix
                        name = member.name
                        if name.startswith(tar_prefix):
                            name = name[len(tar_prefix) :]

                        # ---- Opportunistic metadata.csv capture ----
                        if name == _METADATA_FILENAME:
                            f = tar.extractfile(member)
                            if f is not None and not self.config.dry_run:
                                self._capture_metadata(f.read())
                            continue

                        # ---- HDF5 session filter ----
                        result = member_filter(name)
                        if result is None:
                            continue

                        user, session = result

                        # Skip files already in the B2 registry
                        b2_key = self._b2_key(user, session)
                        if registry.contains(b2_key):
                            skipped_count += 1
                            log.debug("  Skip (already in B2): %s", b2_key)
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
                            log.info("  [DRY RUN] → B2: %s", b2_key)
                            continue

                        # Read bytes from the tar stream (sequential, must block)
                        fileobj = tar.extractfile(member)
                        if fileobj is None:
                            log.warning("  Could not extract %s — skipping", name)
                            continue
                        file_data = fileobj.read()

                        # Optionally write to local disk (--save-local flag)
                        if self.config.save_local:
                            local_path = self._local_path(user, session)
                            local_path.parent.mkdir(parents=True, exist_ok=True)
                            local_path.write_bytes(file_data)
                            log.info("  Saved locally: %s (%.1f MB)", local_path, len(file_data) / 1e6)

                        # Submit B2 upload (non-blocking)
                        in_flight.acquire()
                        fut = pool.submit(self._upload_one, member.name, b2_key, file_data, in_flight)
                        futures.append(fut)

                        # Drain completed futures promptly to free byte buffers
                        still_pending: list[Future] = []
                        for f in futures:
                            if f.done():
                                try:
                                    record = f.result()
                                    with registry_lock:
                                        registry.add(record)
                                        uploaded.append(record)
                                        try:
                                            registry.save()
                                        except Exception:
                                            log.exception(
                                                "Registry save failed after uploading %s — will retry at end",
                                                record.b2_key,
                                            )
                                except Exception:
                                    log.exception("Upload task failed")
                            else:
                                still_pending.append(f)
                        futures = still_pending

                # Pool exit: wait for remaining uploads
                log.info("Stream complete — waiting for %d pending upload(s)…", len(futures))
                for f in as_completed(futures):
                    try:
                        record = f.result()
                        with registry_lock:
                            registry.add(record)
                            uploaded.append(record)
                            try:
                                registry.save()
                            except Exception:
                                log.exception(
                                    "Registry save failed after uploading %s — will retry at end", record.b2_key
                                )
                    except Exception:
                        log.exception("Upload task failed")

        except Exception:
            log.exception("Stream error after %d file(s) found", found_count)
        finally:
            response.close()

        # Final registry flush — ensures consistency even if per-file saves had transient errors
        if uploaded:
            try:
                registry.save()
            except Exception:
                log.exception("Final registry save failed — %d records may not be persisted", len(uploaded))

        log.info(
            "Done. %d new file(s) uploaded, %d skipped (already complete).",
            found_count if not self.config.dry_run else 0,
            skipped_count,
        )
        if uploaded:
            log.info("Total to B2: %.2f GB", sum(r.size_bytes for r in uploaded) / 1e9)

        return uploaded
