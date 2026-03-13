"""Pydantic configuration models for the data-download and batched-training
pipelines.

All credentials are expected to come from environment variables
(``B2_KEY_ID``, ``B2_APPLICATION_KEY``) so that they never appear in
version-controlled files.
"""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Backblaze B2 (S3-compatible) destination
# ---------------------------------------------------------------------------


class B2Config(BaseModel):
    """Connection parameters for a Backblaze B2 bucket accessed through the
    S3-compatible API."""

    endpoint: str = Field(
        default="s3.us-west-004.backblazeb2.com",
        description="B2 S3-compatible endpoint (without scheme).",
    )
    region: str = Field(
        default="us-west-004",
        description="B2 region identifier.",
    )
    bucket_name: str = Field(
        default="C147-project",
        description="Name of the B2 bucket.",
    )
    bucket_id: str = Field(
        default="857ef5759126dac299c10e13",
        description="Hex bucket identifier assigned by B2.",
    )
    key_id: str = Field(
        description="B2 application key ID (from env var B2_KEY_ID).",
    )
    application_key: str = Field(
        description="B2 application key secret (from env var B2_APPLICATION_KEY).",
    )

    @model_validator(mode="after")
    def _validate_endpoint(self) -> "B2Config":
        pattern = r"^s3\.[a-z0-9-]+\.backblazeb2\.com$"
        if not re.match(pattern, self.endpoint):
            raise ValueError(
                f"endpoint '{self.endpoint}' does not match expected B2 pattern 's3.<region>.backblazeb2.com'"
            )
        return self


# ---------------------------------------------------------------------------
# Source (Meta public S3 bucket)
# ---------------------------------------------------------------------------


class SourceS3Config(BaseModel):
    """Read-only access to Meta's public ``emg2qwerty`` dataset.

    Individual object access is denied; the data is distributed as a single
    gzip-compressed tar archive (~308 GB).  We stream through it with
    ``tarfile.open(mode='r|gz')`` and extract only the files we need.
    """

    tar_gz_url: str = Field(
        default="https://fb-ctrl-oss.s3.amazonaws.com/emg2qwerty/emg2qwerty-data-2021-08.tar.gz",
        description="HTTPS URL of the gzip-compressed tar archive.",
    )
    tar_prefix: str = Field(
        default="emg2qwerty-data-2021-08",
        description="Top-level directory inside the tar archive.",
    )


# ---------------------------------------------------------------------------
# Download mode enum
# ---------------------------------------------------------------------------


class DownloadMode(str, Enum):
    """Which subset of the dataset to download."""

    BASELINE = "baseline"
    TEST = "test"
    ALL = "all"


# ---------------------------------------------------------------------------
# Download pipeline config
# ---------------------------------------------------------------------------


class DownloadConfig(BaseModel):
    """Top-level configuration for ``scripts/download_data.py``."""

    mode: DownloadMode = Field(
        default=DownloadMode.BASELINE,
        description="Which subset of profiles to download.",
    )
    n_test_users: int = Field(
        default=10,
        ge=1,
        description="Number of random users to sample in TEST mode.",
    )
    seed: int = Field(
        default=1501,
        description="Random seed for deterministic user sampling.",
    )
    data_root: Path = Field(
        default=Path("data"),
        description="Local directory for metadata cache.",
    )
    registry_key: str = Field(
        default="emg2qwerty_registry.json",
        description="Object key in B2 where the file registry is stored.",
    )
    dry_run: bool = Field(
        default=False,
        description="If True, resolve sessions but skip actual uploads.",
    )
    save_local: bool = Field(
        default=False,
        description=(
            "If True, also write each HDF5 file to local disk under data_root. "
            "By default files are streamed directly to B2 and never touch local disk "
            "(aside from data/metadata.csv which is always cached locally)."
        ),
    )
    b2: B2Config = Field(description="Backblaze B2 destination configuration.")
    source: SourceS3Config = Field(
        default_factory=SourceS3Config,
        description="Meta public S3 source configuration.",
    )


# ---------------------------------------------------------------------------
# Batched training config
# ---------------------------------------------------------------------------


class TrainBatchConfig(BaseModel):
    """Top-level configuration for ``scripts/train_batched.py``."""

    mode: DownloadMode = Field(
        default=DownloadMode.BASELINE,
        description="Which subset of profiles to train on.",
    )
    n_test_users: int = Field(
        default=10,
        ge=1,
        description="Number of random users (TEST mode).",
    )
    seed: int = Field(
        default=1501,
        description="Random seed for deterministic user sampling.",
    )
    b2: B2Config = Field(description="Backblaze B2 configuration for data access.")
    batch_size_profiles: int = Field(
        default=1,
        ge=1,
        description="How many user profiles to train in one batch.",
    )
    hydra_config_path: str = Field(
        default="config/base.yaml",
        description="Path to the Hydra base config.",
    )
    checkpoint: str | None = Field(
        default=None,
        description="Optional path to a training checkpoint to resume from.",
    )
    local_data_dir: Path = Field(
        default=Path("data"),
        description="Local directory where HDF5 files are synced for training.",
    )
    model: str = Field(
        default="tds_conv_ctc",
        description="Hydra model config name (e.g. tds_conv_ctc, tds_conv_ctc, bilstm_ctc, t5_ctc).",
    )
