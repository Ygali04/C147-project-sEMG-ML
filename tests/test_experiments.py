from __future__ import annotations

from pathlib import Path

import pytest

from emg2qwerty.experiments import (
    PREPROCESSING_SPECS,
    build_train_command,
    channel_ablation_specs,
    evenly_spaced_channels,
    filter_specs,
)


def test_evenly_spaced_channels_returns_expected_count() -> None:
    channels = evenly_spaced_channels(total_channels=16, keep_channels=6)

    assert len(channels) == 6
    assert channels == sorted(channels)
    assert len(set(channels)) == 6
    assert channels[0] == 0
    assert channels[-1] == 15


def test_evenly_spaced_channels_rejects_invalid_count() -> None:
    with pytest.raises(ValueError):
        evenly_spaced_channels(total_channels=16, keep_channels=0)


def test_channel_ablation_specs_emit_channel_override() -> None:
    specs = channel_ablation_specs([4])

    assert len(specs) == 1
    assert specs[0].name == "04_channels_per_wrist"
    assert "transforms=log_spectrogram_channel_mask" in specs[0].overrides
    assert any(override.startswith("channel_subset.channels=[") for override in specs[0].overrides)


def test_filter_specs_selects_by_name() -> None:
    filtered = filter_specs(PREPROCESSING_SPECS, ["no_augmentation", "specaug_only"])

    assert [spec.name for spec in filtered] == ["no_augmentation", "specaug_only"]


def test_build_train_command_includes_output_dir_and_overrides() -> None:
    spec = PREPROCESSING_SPECS[0]

    command = build_train_command(
        python_executable="python",
        experiment_root=Path("logs/experiments/20260313_120000"),
        spec=spec,
        user="single_user",
        model="tds_conv_ctc",
        accelerator="cpu",
        checkpoint="logs/best.ckpt",
        extra_overrides=("train=False",),
    )

    assert command[:3] == ["python", "-m", "emg2qwerty.train"]
    assert "user=single_user" in command
    assert "model=tds_conv_ctc" in command
    assert "trainer.accelerator=cpu" in command
    assert "checkpoint=logs/best.ckpt" in command
    assert "train=False" in command
    assert "hydra.run.dir=logs/experiments/20260313_120000/preprocessing/baseline_full_aug" in command