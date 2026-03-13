from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class ExperimentSpec:
    suite: str
    name: str
    description: str
    overrides: tuple[str, ...]


PREPROCESSING_SPECS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec(
        suite="preprocessing",
        name="baseline_full_aug",
        description="Default preprocessing stack with band rotation, temporal jitter, and SpecAugment.",
        overrides=("transforms=log_spectrogram",),
    ),
    ExperimentSpec(
        suite="preprocessing",
        name="no_augmentation",
        description="Only tensor conversion and log-spectrogram generation.",
        overrides=("transforms=log_spectrogram_no_augmentation",),
    ),
    ExperimentSpec(
        suite="preprocessing",
        name="band_rotation_only",
        description="Isolate random electrode-band rotation without temporal or spectral masking.",
        overrides=("transforms=log_spectrogram_band_rotation_only",),
    ),
    ExperimentSpec(
        suite="preprocessing",
        name="temporal_jitter_only",
        description="Isolate left-right temporal alignment jitter.",
        overrides=("transforms=log_spectrogram_temporal_jitter_only",),
    ),
    ExperimentSpec(
        suite="preprocessing",
        name="specaug_only",
        description="Isolate spectrogram time/frequency masking.",
        overrides=("transforms=log_spectrogram_specaug_only",),
    ),
    ExperimentSpec(
        suite="preprocessing",
        name="heavy_specaug",
        description="Stress-test stronger time/frequency masking on top of log-spectrograms.",
        overrides=("transforms=log_spectrogram_heavy_specaug",),
    ),
)


def hydra_list(values: Sequence[int | str]) -> str:
    items = ",".join(str(value) for value in values)
    return f"[{items}]"


def evenly_spaced_channels(total_channels: int, keep_channels: int) -> list[int]:
    if keep_channels <= 0:
        raise ValueError("keep_channels must be positive")
    if keep_channels > total_channels:
        raise ValueError("keep_channels cannot exceed total_channels")
    if keep_channels == total_channels:
        return list(range(total_channels))

    raw = np.linspace(0, total_channels - 1, num=keep_channels)
    selected: list[int] = []
    used: set[int] = set()

    for position in raw:
        candidate = int(round(float(position)))
        if candidate not in used:
            selected.append(candidate)
            used.add(candidate)
            continue

        for delta in range(1, total_channels):
            lower = candidate - delta
            upper = candidate + delta
            if lower >= 0 and lower not in used:
                selected.append(lower)
                used.add(lower)
                break
            if upper < total_channels and upper not in used:
                selected.append(upper)
                used.add(upper)
                break

    return sorted(selected)


def channel_ablation_specs(
    channels_per_wrist: Iterable[int],
    *,
    total_channels_per_wrist: int = 16,
    transforms_config: str = "log_spectrogram_channel_mask",
) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    for count in channels_per_wrist:
        indices = evenly_spaced_channels(total_channels_per_wrist, count)
        specs.append(
            ExperimentSpec(
                suite="channel_ablation",
                name=f"{count:02d}_channels_per_wrist",
                description=(
                    f"Keep {count} of {total_channels_per_wrist} electrodes per wrist "
                    f"using evenly spaced channel indices {indices}."
                ),
                overrides=(
                    f"transforms={transforms_config}",
                    f"channel_subset.channels={hydra_list(indices)}",
                ),
            )
        )
    return specs


def filter_specs(specs: Sequence[ExperimentSpec], names: Sequence[str] | None) -> list[ExperimentSpec]:
    if not names:
        return list(specs)

    allowed = set(names)
    return [spec for spec in specs if spec.name in allowed]


def build_train_command(
    *,
    python_executable: str,
    experiment_root: Path,
    spec: ExperimentSpec,
    user: str,
    model: str,
    accelerator: str,
    checkpoint: str | None = None,
    extra_overrides: Sequence[str] = (),
) -> list[str]:
    output_dir = experiment_root / spec.suite / spec.name
    command = [
        python_executable,
        "-m",
        "emg2qwerty.train",
        f"user={user}",
        f"model={model}",
        f"trainer.accelerator={accelerator}",
        f"hydra.run.dir={output_dir.as_posix()}",
        *spec.overrides,
    ]
    if checkpoint:
        command.append(f"checkpoint={checkpoint}")
    command.extend(extra_overrides)
    return command