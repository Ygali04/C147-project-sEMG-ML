from __future__ import annotations

import argparse
from pathlib import Path

from emg2qwerty.training_curves import load_training_history, save_training_curve_plot


def _resolve_metrics_path(run_path: Path) -> Path | None:
    if run_path.is_file() and run_path.name == "metrics.csv":
        return run_path

    if not run_path.is_dir():
        return None

    candidates = list(run_path.glob("lightning_logs/version_*/metrics.csv"))
    if not candidates:
        return None

    versioned_candidates: list[tuple[int, Path]] = []
    fallback_candidates: list[Path] = []

    for path in candidates:
        dirname = path.parent.name
        prefix = "version_"
        if dirname.startswith(prefix):
            version_str = dirname[len(prefix) :]
            try:
                version = int(version_str)
            except ValueError:
                fallback_candidates.append(path)
            else:
                versioned_candidates.append((version, path))
        else:
            fallback_candidates.append(path)

    if versioned_candidates:
        # Select the metrics file from the highest numeric version directory.
        return max(versioned_candidates, key=lambda item: item[0])[1]

    if fallback_candidates:
        # Fallback to lexicographic order if no numeric versions could be parsed.
        return sorted(fallback_candidates)[-1]

    return None


def _resolve_output_path(run_path: Path, output_name: str) -> Path:
    if run_path.is_file():
        # When given a metrics.csv file produced by PyTorch Lightning, the expected
        # layout is .../lightning_logs/version_*/metrics.csv and we want to write the
        # output into the run directory above lightning_logs.
        if (
            run_path.name == "metrics.csv"
            and run_path.parent.name.startswith("version_")
            and run_path.parent.parent.name == "lightning_logs"
        ):
            return run_path.parent.parent.parent / output_name
        # For any other file layout, place the output alongside the given file.
        return run_path.parent / output_name
    return run_path / output_name


def regenerate_plot(run_path: Path, output_name: str) -> tuple[bool, str]:
    metrics_path = _resolve_metrics_path(run_path)
    if metrics_path is None:
        return False, f"missing metrics.csv under {run_path}"

    history = load_training_history(metrics_path)
    if not history["epochs"]:
        return False, f"no epoch-level train/val metrics found in {metrics_path}"

    output_path = _resolve_output_path(run_path, output_name)
    save_training_curve_plot(
        output_path=output_path,
        title=run_path.name,
        epochs=history["epochs"],
        train_losses=history["train_losses"],
        val_losses=history["val_losses"],
        train_cers=history["train_cers"],
        val_cers=history["val_cers"],
    )
    return True, str(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate training_progress.png from prior Lightning metrics.csv files.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Run directories or metrics.csv files to process.",
    )
    parser.add_argument(
        "--output-name",
        default="training_progress.png",
        help="Output image filename written into each run directory.",
    )

    args = parser.parse_args()

    had_error = False
    for raw_path in args.paths:
        run_path = raw_path.expanduser().resolve()
        ok, message = regenerate_plot(run_path, args.output_name)
        status = "wrote" if ok else "skipped"
        print(f"[{status}] {message}")
        had_error = had_error or not ok

    return 1 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())