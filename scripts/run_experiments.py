#!/usr/bin/env python3
"""Launch preprocessing and channel-ablation experiment suites.

Examples::

    uv run python scripts/run_experiments.py preprocessing --dry-run
    uv run python scripts/run_experiments.py channels --channels-per-wrist 4 --channels-per-wrist 8 --dry-run
    uv run python scripts/run_experiments.py all --execute --extra-override trainer.devices=1
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import click

# Ensure src/ is on sys.path when running as a script.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from emg2qwerty.experiments import (
    PREPROCESSING_SPECS,
    ExperimentSpec,
    build_train_command,
    channel_ablation_specs,
    filter_specs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
DEFAULT_CHANNELS_PER_WRIST = (1, 2, 4, 8, 12, 16)


def _timestamped_root(output_root: Path) -> Path:
    return output_root / datetime.now().strftime("%Y%m%d_%H%M%S")


def _format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _run_specs(
    specs: list[ExperimentSpec],
    *,
    user: str,
    model: str,
    accelerator: str,
    checkpoint: str | None,
    output_root: Path,
    execute: bool,
    python_executable: str,
    extra_overrides: tuple[str, ...],
) -> None:
    run_root = _timestamped_root(output_root)
    click.echo(f"Planned output root: {run_root}")

    # Make local src/ importable for subprocesses when the package is not
    # installed into the active Python environment.
    child_env = os.environ.copy()
    existing_pythonpath = child_env.get("PYTHONPATH")
    src_path = str(SRC_ROOT)
    child_env["PYTHONPATH"] = (
        f"{src_path}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else src_path
    )

    failures: list[str] = []
    for spec in specs:
        command = build_train_command(
            python_executable=python_executable,
            experiment_root=run_root,
            spec=spec,
            user=user,
            model=model,
            accelerator=accelerator,
            checkpoint=checkpoint,
            extra_overrides=extra_overrides,
        )
        click.echo(f"\n[{spec.suite}/{spec.name}] {spec.description}")
        click.echo(_format_command(command))

        if not execute:
            continue

        result = subprocess.run(command, cwd=REPO_ROOT, env=child_env)
        if result.returncode != 0:
            failures.append(spec.name)

    if failures:
        raise SystemExit(f"Experiment runs failed: {', '.join(failures)}")


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """Run reproducible Hydra experiments for preprocessing and channel sweeps."""


@cli.command()
@click.option("--user", default="single_user", show_default=True, help="Hydra user config to train with.")
@click.option(
    "--model",
    type=click.Choice(["tds_conv_ctc", "bilstm_ctc", "cnn_bilstm_ctc", "whisper_ctc"]),
    default="tds_conv_ctc",
    show_default=True,
    help="Hydra model config to train with.",
)
@click.option("--accelerator", default="gpu", show_default=True, help="Trainer accelerator override.")
@click.option("--checkpoint", default=None, help="Optional checkpoint override.")
@click.option(
    "--experiment",
    "selected_experiments",
    type=click.Choice([spec.name for spec in PREPROCESSING_SPECS]),
    multiple=True,
    help="Only run the named preprocessing experiment(s).",
)
@click.option(
    "--output-root",
    type=click.Path(path_type=Path),
    default=Path("logs/experiments"),
    show_default=True,
    help="Parent directory for Hydra outputs.",
)
@click.option("--execute/--dry-run", default=False, show_default=True, help="Execute commands instead of printing them.")
@click.option(
    "--python-executable",
    default=sys.executable,
    show_default=True,
    help="Python executable used to invoke training.",
)
@click.option(
    "--extra-override",
    multiple=True,
    help="Additional Hydra override(s), e.g. trainer.devices=1 or train=False.",
)
def preprocessing(
    user: str,
    model: str,
    accelerator: str,
    checkpoint: str | None,
    selected_experiments: tuple[str, ...],
    output_root: Path,
    execute: bool,
    python_executable: str,
    extra_override: tuple[str, ...],
) -> None:
    """Compare augmentation and preprocessing variants."""
    specs = filter_specs(PREPROCESSING_SPECS, selected_experiments)
    _run_specs(
        specs,
        user=user,
        model=model,
        accelerator=accelerator,
        checkpoint=checkpoint,
        output_root=output_root,
        execute=execute,
        python_executable=python_executable,
        extra_overrides=extra_override,
    )


@cli.command(name="channels")
@click.option("--user", default="single_user", show_default=True, help="Hydra user config to train with.")
@click.option(
    "--model",
    type=click.Choice(["tds_conv_ctc", "bilstm_ctc", "cnn_bilstm_ctc", "whisper_ctc"]),
    default="tds_conv_ctc",
    show_default=True,
    help="Hydra model config to train with.",
)
@click.option("--accelerator", default="gpu", show_default=True, help="Trainer accelerator override.")
@click.option("--checkpoint", default=None, help="Optional checkpoint override.")
@click.option(
    "--channels-per-wrist",
    type=int,
    multiple=True,
    help="Number of active electrodes to keep per wrist. Defaults to 1,2,4,8,12,16.",
)
@click.option(
    "--output-root",
    type=click.Path(path_type=Path),
    default=Path("logs/experiments"),
    show_default=True,
    help="Parent directory for Hydra outputs.",
)
@click.option("--execute/--dry-run", default=False, show_default=True, help="Execute commands instead of printing them.")
@click.option(
    "--python-executable",
    default=sys.executable,
    show_default=True,
    help="Python executable used to invoke training.",
)
@click.option(
    "--extra-override",
    multiple=True,
    help="Additional Hydra override(s), e.g. trainer.devices=1 or train=False.",
)
def channels(
    user: str,
    model: str,
    accelerator: str,
    checkpoint: str | None,
    channels_per_wrist: tuple[int, ...],
    output_root: Path,
    execute: bool,
    python_executable: str,
    extra_override: tuple[str, ...],
) -> None:
    """Run CER-vs-channel-count ablations with symmetric per-wrist masking."""
    selected_counts = channels_per_wrist or DEFAULT_CHANNELS_PER_WRIST
    specs = channel_ablation_specs(selected_counts)
    _run_specs(
        specs,
        user=user,
        model=model,
        accelerator=accelerator,
        checkpoint=checkpoint,
        output_root=output_root,
        execute=execute,
        python_executable=python_executable,
        extra_overrides=extra_override,
    )


@cli.command()
@click.option("--user", default="single_user", show_default=True, help="Hydra user config to train with.")
@click.option(
    "--model",
    type=click.Choice(["tds_conv_ctc", "bilstm_ctc", "cnn_bilstm_ctc", "whisper_ctc"]),
    default="tds_conv_ctc",
    show_default=True,
    help="Hydra model config to train with.",
)
@click.option("--accelerator", default="gpu", show_default=True, help="Trainer accelerator override.")
@click.option("--checkpoint", default=None, help="Optional checkpoint override.")
@click.option(
    "--channels-per-wrist",
    type=int,
    multiple=True,
    help="Number of active electrodes to keep per wrist. Defaults to 1,2,4,8,12,16.",
)
@click.option(
    "--output-root",
    type=click.Path(path_type=Path),
    default=Path("logs/experiments"),
    show_default=True,
    help="Parent directory for Hydra outputs.",
)
@click.option("--execute/--dry-run", default=False, show_default=True, help="Execute commands instead of printing them.")
@click.option(
    "--python-executable",
    default=sys.executable,
    show_default=True,
    help="Python executable used to invoke training.",
)
@click.option(
    "--extra-override",
    multiple=True,
    help="Additional Hydra override(s), e.g. trainer.devices=1 or train=False.",
)
def all(
    user: str,
    model: str,
    accelerator: str,
    checkpoint: str | None,
    channels_per_wrist: tuple[int, ...],
    output_root: Path,
    execute: bool,
    python_executable: str,
    extra_override: tuple[str, ...],
) -> None:
    """Run both preprocessing and channel-count suites."""
    specs = list(PREPROCESSING_SPECS) + channel_ablation_specs(channels_per_wrist or DEFAULT_CHANNELS_PER_WRIST)
    _run_specs(
        specs,
        user=user,
        model=model,
        accelerator=accelerator,
        checkpoint=checkpoint,
        output_root=output_root,
        execute=execute,
        python_executable=python_executable,
        extra_overrides=extra_override,
    )


if __name__ == "__main__":
    cli()