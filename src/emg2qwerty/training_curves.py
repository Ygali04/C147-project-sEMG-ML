from __future__ import annotations

import csv
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def metric_to_float(value) -> float | None:
    if value is None:
        return None

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        value = value.item()

    if value == "":
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_training_history(metrics_path: Path) -> dict[str, list[float]]:
    """Load epoch-level train/val loss and CER from a Lightning metrics CSV."""

    epoch_history: dict[int, dict[str, float]] = {}

    with metrics_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            epoch_value = metric_to_float(row.get("epoch"))
            if epoch_value is None:
                continue

            epoch = int(epoch_value)
            series = epoch_history.setdefault(epoch, {})
            for metric_name in ("train/loss", "val/loss", "train/CER", "val/CER"):
                metric_value = metric_to_float(row.get(metric_name))
                if metric_value is not None:
                    series[metric_name] = metric_value

    epochs = sorted(epoch_history)
    return {
        "epochs": epochs,
        "train_losses": [epoch_history[epoch].get("train/loss", float("nan")) for epoch in epochs],
        "val_losses": [epoch_history[epoch].get("val/loss", float("nan")) for epoch in epochs],
        "train_cers": [epoch_history[epoch].get("train/CER", float("nan")) for epoch in epochs],
        "val_cers": [epoch_history[epoch].get("val/CER", float("nan")) for epoch in epochs],
    }


def save_training_curve_plot(
    output_path: Path,
    title: str,
    epochs: list[int],
    train_losses: list[float],
    val_losses: list[float],
    train_cers: list[float],
    val_cers: list[float],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, facecolor="white")

    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(True, color="black", alpha=0.2, linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.0)

    axes[0].plot(epochs, train_losses, label="train/loss", linewidth=2)
    axes[0].plot(epochs, val_losses, label="val/loss", linewidth=2)
    axes[0].set_title("Loss vs Epoch")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(loc="upper right")

    axes[1].plot(epochs, train_cers, label="train/CER", linewidth=2)
    axes[1].plot(epochs, val_cers, label="val/CER", linewidth=2)
    axes[1].set_title("CER vs Epoch")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("CER (%)")
    axes[1].set_ylim(bottom=0, top=100)
    axes[1].legend(loc="upper right")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)