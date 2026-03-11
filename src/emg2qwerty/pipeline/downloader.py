"""EMGDownloader — extracts HDF5 session files from Meta's public tar.gz
archive and saves them locally (+ optionally uploads to Backblaze B2).

The dataset is distributed as a single ~308 GB gzip-compressed tar archive
at ``https://fb-ctrl-oss.s3.amazonaws.com/emg2qwerty/emg2qwerty-data-2021-08.tar.gz``.
Individual S3 object access is **not** available, so we stream through the
entire archive with ``tarfile.open(mode='r|gz')``, extracting only the
sessions we need.

Download modes (``--baseline``, ``--test``, ``--all``) select different
subsets of the ~1 135 session files.

User-ID parsing
---------------
The tar contains two filename conventions:

* **Old format** (2020 pilot data)::

      emg2qwerty-data-2021-08/2020-08-13-1597357485-keystrokes-71409769.hdf5
                                                                  ^^^^^^^^
                                                                  user-id

* **New format** (2021 DCA study)::

      emg2qwerty-data-2021-08/2021-06-03-1622765527-keystrokes-dca-study@1-<uuid>.hdf5
                                                                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                                                             user UUID (used as user-id)

* **Pilot** (no user suffix)::

      emg2qwerty-data-2021-08/2020-08-13-1597354281-keystrokes.hdf5
      → stored under user-id ``pilot``

Because a separate ``metadata.csv`` is not publicly accessible outside the
tar, user identity is recovered entirely from the filename.
"""

from __future__ import annotations

import logging
import re
import tarfile
from pathlib import Path
from typing import Callable

import boto3
import numpy as np
import yaml

import requests

from emg2qwerty.pipeline.config import DownloadConfig, DownloadMode
from emg2qwerty.pipeline.registry import FileRecord, FileRegistry, make_record

log = logging.getLogger(__name__)

# Baseline user hard-coded from config/user/single_user.yaml
_BASELINE_USER = "89335547"

# Regex patterns for extracting user-id from HDF5 filenames
# Group "user" captures the identifier
_RE_OLD = re.compile(r"keystrokes-(?P<user>\d{6,12})\.hdf5$")
_RE_NEW = re.compile(
    r"keystrokes-dca-study@\d+-(?P<user>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.hdf5$"
)


def parse_user_from_name(name: str) -> str:
    """Extract a user identifier from a tar member filename.

    Parameters
    ----------
    name:
        Bare filename (without the tar prefix directory), e.g.
        ``2021-06-03-1622765527-keystrokes-dca-study@1-<uuid>.hdf5``.

    Returns
    -------
    str
        The user identifier, or ``"pilot"`` for sessions with no user suffix.
    """
    m = _RE_NEW.search(name) or _RE_OLD.search(name)
    if m:
        return m.group("user")
    if name.endswith(".hdf5"):
        return "pilot"
    return "unknown"


def parse_session_from_name(name: str) -> str:
    """Strip ``.hdf5`` from a bare filename to get the session name."""
    if name.endswith(".hdf5"):
        return name[: -len(".hdf5")]
    return name


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
        """Return a callable that maps a bare member name → (user, session) or None.

        The filter encapsulates all download-mode logic so the streaming loop
        stays simple.
        """
        mode = self.config.mode

        if mode == DownloadMode.BASELINE:
            wanted: set[str] = self._load_baseline_sessions()
            log.info("BASELINE mode — targeting %d sessions for user %s", len(wanted), _BASELINE_USER)

            def _baseline_filter(name: str) -> tuple[str, str] | None:
                session = parse_session_from_name(name)
                if session in wanted:
                    return (_BASELINE_USER, session)
                return None

            return _baseline_filter

        if mode == DownloadMode.ALL:
            log.info("ALL mode — extracting every session in the archive")

            def _all_filter(name: str) -> tuple[str, str] | None:
                if not name.endswith(".hdf5"):
                    return None
                user = parse_user_from_name(name)
                session = parse_session_from_name(name)
                return (user, session)

            return _all_filter

        if mode == DownloadMode.TEST:
            rng = np.random.RandomState(self.config.seed)
            n_users = self.config.n_test_users
            # We'll dynamically select users as we stream; users seen first
            # are accepted up to n_users total.
            # Use a stable random decision: assign each new user a random
            # priority; keep the n_users with the lowest priority.
            #
            # Implementation: reservoir sampling over users as they appear.
            accepted_users: set[str] = set()
            reservoir: list[tuple[float, str]] = []  # (priority, user)

            def _test_filter(name: str) -> tuple[str, str] | None:
                if not name.endswith(".hdf5"):
                    return None
                user = parse_user_from_name(name)
                session = parse_session_from_name(name)

                if user in accepted_users:
                    return (user, session)

                # New user — reservoir sample
                if len(accepted_users) < n_users:
                    accepted_users.add(user)
                    reservoir.append((rng.random(), user))
                    return (user, session)

                # Reservoir is full — compare priority
                priority = rng.random()
                max_entry = max(reservoir, key=lambda x: x[0])
                if priority < max_entry[0]:
                    # Evict the highest-priority (worst) user
                    reservoir.remove(max_entry)
                    accepted_users.discard(max_entry[1])
                    reservoir.append((priority, user))
                    accepted_users.add(user)
                    return (user, session)

                return None  # User not selected

            return _test_filter

        raise ValueError(f"Unknown download mode: {mode}")  # pragma: no cover

    # ------------------------------------------------------------------
    # Streaming extraction
    # ------------------------------------------------------------------

    def run(self) -> list[FileRecord]:
        """Execute the full download pipeline:

        1. Build a member-filter for the chosen mode (BASELINE/TEST/ALL).
        2. Load the B2 registry.
        3. Stream the tar.gz, extracting matching files locally + to B2.
        4. Update and save the registry.

        Returns the list of newly saved :class:`FileRecord` instances.
        """
        log.info("Download mode: %s", self.config.mode.value)

        # 1. Build filter
        member_filter = self._make_filter()

        # 2. Load registry
        registry = self._get_registry()

        tar_url = self.config.source.tar_gz_url
        tar_prefix = self.config.source.tar_prefix + "/"
        log.info("Streaming tar.gz from %s ...", tar_url)
        log.info("Connect timeout = 60 s; no read timeout (stream may run for many hours).")

        if self.config.dry_run:
            log.info("[DRY RUN] Scanning archive — will not download or upload anything.")

        # Connect timeout = 60s; no read timeout (stream runs for many hours)
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

                    # Strip tar prefix directory
                    name = member.name
                    if name.startswith(tar_prefix):
                        name = name[len(tar_prefix) :]

                    # Apply the mode-specific filter
                    result = member_filter(name)
                    if result is None:
                        continue

                    user, session = result

                    # Skip files already in registry (both B2 and local)
                    b2_key = self._b2_key(user, session)
                    local_path = self._local_path(user, session)
                    if registry.contains(b2_key) and local_path.exists():
                        skipped_registry += 1
                        log.debug("  Skipping (already in registry): %s", b2_key)
                        continue

                    found_count += 1
                    log.info(
                        "[#%d] %s → user=%s  (%.1f MB)",
                        found_count,
                        name,
                        user,
                        member.size / 1e6,
                    )

                    if self.config.dry_run:
                        log.info("  [DRY RUN] Would save to %s and B2 key %s", local_path, b2_key)
                        continue

                    # Extract file content
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
            log.exception("Error streaming tar.gz")
        finally:
            response.close()

        # 4. Save registry
        if uploaded:
            registry.save()

        log.info(
            "Done. Extracted %d new file(s), skipped %d already complete.",
            found_count if not self.config.dry_run else 0,
            skipped_registry,
        )
        if uploaded:
            total_bytes = sum(r.size_bytes for r in uploaded)
            log.info("Total uploaded to B2: %.1f GB", total_bytes / 1e9)
        return uploaded
