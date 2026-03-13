from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl

from emg2qwerty.training_curves import metric_to_float, save_training_curve_plot

class TrainingCurvePlotCallback(pl.callbacks.Callback):
    """Save per-epoch train/val loss and CER curves to a PNG file."""

    def __init__(self, filename: str = "training_progress.png") -> None:
        super().__init__()
        self.filename = filename
        self.epochs: list[int] = []
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.train_cers: list[float] = []
        self.val_cers: list[float] = []

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics
        train_loss = metric_to_float(metrics.get("train/loss"))
        val_loss = metric_to_float(metrics.get("val/loss"))
        train_cer = metric_to_float(metrics.get("train/CER"))
        val_cer = metric_to_float(metrics.get("val/CER"))

        if all(metric is None for metric in (train_loss, val_loss, train_cer, val_cer)):
            return

        self.epochs.append(int(trainer.current_epoch))
        self.train_losses.append(train_loss if train_loss is not None else float("nan"))
        self.val_losses.append(val_loss if val_loss is not None else float("nan"))
        self.train_cers.append(train_cer if train_cer is not None else float("nan"))
        self.val_cers.append(val_cer if val_cer is not None else float("nan"))

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.global_rank != 0 or not self.epochs:
            return

        run_dir = Path(trainer.default_root_dir)
        output_path = run_dir / self.filename
        save_training_curve_plot(
            output_path=output_path,
            title=type(pl_module).__name__,
            epochs=self.epochs,
            train_losses=self.train_losses,
            val_losses=self.val_losses,
            train_cers=self.train_cers,
            val_cers=self.val_cers,
        )
