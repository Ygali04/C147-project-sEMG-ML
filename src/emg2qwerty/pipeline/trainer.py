"""BatchTrainer — orchestrates training across multiple user profiles by
reading available sessions from the B2 file registry and invoking the
existing Hydra-based training entry-point for each user batch.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import boto3

from emg2qwerty.pipeline.config import DownloadMode, TrainBatchConfig
from emg2qwerty.pipeline.registry import FileRegistry

log = logging.getLogger(__name__)


class BatchTrainer:
    """Load user profiles from the B2 registry and run training per-profile.

    Parameters
    ----------
    config : TrainBatchConfig
        Fully validated batched-training configuration (Pydantic model).
    """

    def __init__(self, config: TrainBatchConfig) -> None:
        self.config = config
        self._b2_client: Any = None
        self._registry: FileRegistry | None = None

    # ------------------------------------------------------------------
    # B2 / registry helpers
    # ------------------------------------------------------------------

    def _make_b2_client(self) -> Any:
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
        if self._registry is None:
            self._registry = FileRegistry(
                b2_client=self._make_b2_client(),
                bucket_name=self.config.b2.bucket_name,
                registry_key="emg2qwerty_registry.json",
            )
            self._registry.load()
        return self._registry

    # ------------------------------------------------------------------
    # Profile resolution
    # ------------------------------------------------------------------

    def resolve_profiles(self) -> dict[str, list[str]]:
        """Return ``{user_id: [b2_key, ...]}`` from the registry, filtered
        by the configured download mode.

        * **BASELINE** — only user ``89335547``.
        * **TEST** — first ``n_test_users`` users from the registry.
        * **ALL** — every user in the registry.
        """
        registry = self._get_registry()
        all_keys = registry.all_keys()

        # Group by user
        profiles: dict[str, list[str]] = defaultdict(list)
        for key in all_keys:
            # keys look like emg2qwerty/<user>/<session>.hdf5
            parts = key.split("/")
            if len(parts) >= 3:
                user_id = parts[1]
                profiles[user_id].append(key)

        mode = self.config.mode

        if mode == DownloadMode.BASELINE:
            baseline_user = "89335547"
            if baseline_user not in profiles:
                log.warning(
                    "Baseline user %s not found in registry. Run download_data.py --baseline first.",
                    baseline_user,
                )
                return {}
            return {baseline_user: profiles[baseline_user]}

        if mode == DownloadMode.TEST:
            users = sorted(profiles.keys())
            n = min(self.config.n_test_users, len(users))
            selected = users[:n]
            return {u: profiles[u] for u in selected}

        if mode == DownloadMode.ALL:
            return dict(profiles)

        raise ValueError(f"Unknown mode: {mode}")  # pragma: no cover

    # ------------------------------------------------------------------
    # Data sync (B2 → local)
    # ------------------------------------------------------------------

    def sync_profile(self, user_id: str, b2_keys: list[str]) -> list[Path]:
        """Download HDF5 files for *user_id* from B2 into ``local_data_dir``.

        Returns a list of local paths to the downloaded files.
        Skips files that already exist locally.
        """
        client = self._make_b2_client()
        local_paths: list[Path] = []

        for b2_key in b2_keys:
            # b2_key = emg2qwerty/<user>/<session>.hdf5
            local_path = self.config.local_data_dir / b2_key
            local_paths.append(local_path)

            if local_path.exists():
                log.debug("Already local: %s", local_path)
                continue

            local_path.parent.mkdir(parents=True, exist_ok=True)
            log.info("Downloading %s → %s", b2_key, local_path)
            try:
                client.download_file(
                    Bucket=self.config.b2.bucket_name,
                    Key=b2_key,
                    Filename=str(local_path),
                )
            except Exception:
                log.exception("Failed to download %s", b2_key)
                if local_path.exists():
                    local_path.unlink()
                local_paths.pop()

        return local_paths

    # ------------------------------------------------------------------
    # Training invocation
    # ------------------------------------------------------------------

    @staticmethod
    def _build_hydra_overrides(
        user_id: str,
        session_paths: list[Path],
        checkpoint: str | None = None,
    ) -> list[str]:
        """Build Hydra CLI override strings for a single user."""
        overrides = [f"user={user_id}"]

        if checkpoint:
            overrides.append(f"checkpoint={checkpoint}")

        return overrides

    def train_profile(self, user_id: str, session_paths: list[Path]) -> bool:
        """Launch training for a single user profile via subprocess.

        Returns ``True`` if the training subprocess exits successfully.
        """
        overrides = self._build_hydra_overrides(
            user_id=user_id,
            session_paths=session_paths,
            checkpoint=self.config.checkpoint,
        )

        cmd = [
            sys.executable,
            "-m",
            "emg2qwerty.train",
            *overrides,
        ]

        log.info("Training user %s: %s", user_id, " ".join(cmd))
        result = subprocess.run(cmd, cwd=Path.cwd())

        if result.returncode != 0:
            log.error("Training failed for user %s (exit code %d)", user_id, result.returncode)
            return False

        log.info("Training completed for user %s", user_id)
        return True

    # ------------------------------------------------------------------
    # Main entry-point
    # ------------------------------------------------------------------

    def run(self) -> dict[str, bool]:
        """Execute the full batched-training pipeline:

        1. Resolve which user profiles to train from the B2 registry.
        2. Sync each profile's HDF5 files to local disk.
        3. Run training for each profile.

        Returns ``{user_id: success_bool}``.
        """
        profiles = self.resolve_profiles()
        if not profiles:
            log.warning("No profiles to train.")
            return {}

        log.info("Training %d profile(s) in mode=%s", len(profiles), self.config.mode.value)

        results: dict[str, bool] = {}
        for user_id, b2_keys in profiles.items():
            log.info(
                "--- Profile %s: %d session(s) ---",
                user_id,
                len(b2_keys),
            )

            # Sync data locally
            session_paths = self.sync_profile(user_id, b2_keys)
            if not session_paths:
                log.warning("No data available for user %s — skipping.", user_id)
                results[user_id] = False
                continue

            # Train
            success = self.train_profile(user_id, session_paths)
            results[user_id] = success

        # Summary
        ok = sum(1 for v in results.values() if v)
        log.info("Done. %d / %d profiles trained successfully.", ok, len(results))
        return results
