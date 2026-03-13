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
from typing import Any, ClassVar

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

        log.info(
            "Registry has %d keys across %d unique users. Sample users: %s",
            len(all_keys),
            len(profiles),
            list(profiles.keys())[:5],
        )

        mode = self.config.mode

        if mode == DownloadMode.BASELINE:
            baseline_user = "89335547"
            if baseline_user not in profiles:
                # Fallback: the download may have stored sessions under UUID
                # keys (ALL mode without metadata.csv).  Use metadata.csv to
                # find the UUID and match by session name.
                log.info(
                    "Baseline user %s not found by numeric ID — searching registry by session name via metadata.csv …",
                    baseline_user,
                )
                matched_keys = self._find_baseline_keys_by_session(all_keys)
                if matched_keys:
                    log.info(
                        "Found %d session(s) for baseline user via session-name matching.",
                        len(matched_keys),
                    )
                    return {baseline_user: matched_keys}

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
    # Baseline session matching (fallback when UUID keys are used)
    # ------------------------------------------------------------------

    @staticmethod
    def _find_baseline_keys_by_session(all_keys: list[str]) -> list[str]:
        """Find registry keys that match the baseline user's sessions.

        When the download was done in ALL mode without ``metadata.csv``,
        user 89335547's sessions are stored under their UUID prefix instead
        of the numeric user ID.  This method reads the session names from
        ``config/user/single_user.yaml`` and matches them against registry
        keys by checking if the session name appears anywhere in the key.
        """
        import yaml

        config_path = Path("config/user/single_user.yaml")
        if not config_path.exists():
            log.warning("Cannot find %s — unable to match baseline sessions.", config_path)
            return []

        with open(config_path) as f:
            data = yaml.safe_load(f)

        sessions: set[str] = set()
        for split in ("train", "val", "test"):
            for entry in data.get("dataset", {}).get(split, []) or []:
                sessions.add(str(entry["session"]))

        matched: list[str] = []
        for key in all_keys:
            # key = emg2qwerty/<user_or_uuid>/<session>.hdf5
            # Extract session name from the key's filename
            filename = key.rsplit("/", 1)[-1]  # <session>.hdf5
            session = filename.removesuffix(".hdf5")
            if session in sessions:
                matched.append(key)

        return matched

    # ------------------------------------------------------------------
    # Data sync (B2 → local)
    # ------------------------------------------------------------------

    def sync_profile(self, user_id: str, b2_keys: list[str]) -> list[Path]:
        """Download HDF5 files for *user_id* from B2 into ``local_data_dir``.

        Returns a list of local paths to the downloaded files.
        Skips files that already exist locally.

        Also creates symlinks at ``data/<session>.hdf5`` so that Hydra's
        ``dataset.root`` (which points to ``data/``) can find the files
        by their session name alone.
        """
        client = self._make_b2_client()
        local_paths: list[Path] = []

        for b2_key in b2_keys:
            # b2_key = emg2qwerty/<user>/<session>.hdf5
            local_path = self.config.local_data_dir / b2_key
            local_paths.append(local_path)

            if local_path.exists():
                log.debug("Already local: %s", local_path)
            else:
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
                    continue

            # Create a symlink at data/<session>.hdf5 → data/emg2qwerty/…/<session>.hdf5
            # so Hydra's dataset.root can find files by session name alone.
            filename = local_path.name  # <session>.hdf5
            symlink_path = self.config.local_data_dir / filename
            if not symlink_path.exists():
                try:
                    symlink_path.symlink_to(local_path.resolve())
                    log.debug("Symlinked %s → %s", symlink_path, local_path)
                except OSError:
                    log.debug("Could not create symlink %s (may already exist)", symlink_path)

        return local_paths

    # ------------------------------------------------------------------
    # Training invocation
    # ------------------------------------------------------------------

    # Maps user IDs to their Hydra config-group names in config/user/.
    _USER_CONFIG_MAP: ClassVar[dict[str, str]] = {
        "89335547": "single_user",
    }

    # Per-model CLI overrides that must beat base.yaml's ``_self_`` priority.
    # Hydra's ``_self_`` is last in the defaults list, so config-group values
    # (e.g. batch_size in model/t5_ctc.yaml) get overridden by base.yaml.
    # CLI overrides always win, so we push critical knobs here.
    _MODEL_CLI_OVERRIDES: ClassVar[dict[str, list[str]]] = {
        "t5_ctc": [
            "batch_size=4",  # 4×8 GPUs = 32 effective
            "optimizer._target_=torch.optim.AdamW",  # AdamW + weight decay
            "optimizer.lr=3e-4",  # transformer LR
            "+optimizer.weight_decay=0.01",  # + prefix: new key
            "+optimizer.betas=[0.9,0.98]",  # research: β₂=0.98
            "lr_scheduler.scheduler.warmup_epochs=5",  # ~600 steps warmup
            "lr_scheduler.scheduler.warmup_start_lr=1e-5",  # start with real gradients
            "+trainer.precision=16-mixed",  # fp16 mixed precision
        ],
    }

    @staticmethod
    def _build_hydra_overrides(
        user_id: str,
        session_paths: list[Path],
        model: str,
        checkpoint: str | None = None,
        model: str = "tds_conv_ctc",
    ) -> list[str]:
        """Build Hydra CLI override strings for a single user."""
        # Resolve the Hydra config-group name for this user.
        # Known users map to their dedicated YAML; unknown users fall back
        # to ``generic`` (the multi-user config).
        user_cfg = BatchTrainer._USER_CONFIG_MAP.get(user_id, "generic")
        overrides = [
            f"user={user_cfg}",
            f"model={model}",
            # Disable the cluster config group entirely — the base.yaml
            # defaults to "local" which requires hydra-submitit-launcher.
            # We run directly via PyTorch Lightning's auto-detected strategy.
            "~cluster",
        ]

        # Add model-specific CLI overrides (beat base.yaml's _self_ priority)
        overrides.extend(BatchTrainer._MODEL_CLI_OVERRIDES.get(model, []))

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
            model=self.config.model,
            checkpoint=self.config.checkpoint,
            model=self.config.model,
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
