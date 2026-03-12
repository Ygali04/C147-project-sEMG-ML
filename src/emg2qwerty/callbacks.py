from __future__ import annotations

from pathlib import Path

import matplotlib
import pytorch_lightning as pl


matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TrainingCurvePlotCallback(pl.callbacks.Callback):
    """Save per-epoch training curves (train loss and val CER) to a PNG file."""

    def __init__(self, filename: str = "training_progress.png") -> None:
        super().__init__()
        self.filename = filename
        self.epochs: list[int] = []
        self.train_losses: list[float] = []
        self.val_cers: list[float] = []

    @staticmethod
    def _metric_to_float(value) -> float | None:
        if value is None:
            return None

        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "item"):
            value = value.item()

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics
        train_loss = self._metric_to_float(metrics.get("train/loss"))
        val_cer = self._metric_to_float(metrics.get("val/CER"))

        if train_loss is None and val_cer is None:
            return

        self.epochs.append(int(trainer.current_epoch))
        self.train_losses.append(train_loss if train_loss is not None else float("nan"))
        self.val_cers.append(val_cer if val_cer is not None else float("nan"))

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.global_rank != 0 or not self.epochs:
            return

        run_dir = Path(trainer.default_root_dir)
        output_path = run_dir / self.filename

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        axes[0].plot(self.epochs, self.train_losses, color="tab:blue", marker="o", linewidth=1.5)
        axes[0].set_ylabel("Train Loss")
        axes[0].grid(alpha=0.3)

        axes[1].plot(self.epochs, self.val_cers, color="tab:red", marker="o", linewidth=1.5)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Val CER")
        axes[1].grid(alpha=0.3)

        fig.suptitle(type(pl_module).__name__)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
