# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import MetricCollection

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.inference import build_window_specs, merge_log_prob_chunks, resize_log_probs_to_input_length
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import (
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TDSConvEncoder,
)
from emg2qwerty.transforms import Transform


class WindowedEMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        train_transform: Transform[np.ndarray, torch.Tensor],
        val_transform: Transform[np.ndarray, torch.Tensor],
        test_transform: Transform[np.ndarray, torch.Tensor],
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.train_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=True,
                )
                for hdf5_path in self.train_sessions
            ]
        )
        self.val_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.val_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=False,
                )
                for hdf5_path in self.val_sessions
            ]
        )
        self.test_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.test_transform,
                    # Feed the entire session at once without windowing/padding
                    # at test time for more realism
                    window_length=None,
                    padding=(0, 0),
                    jitter=False,
                )
                for hdf5_path in self.test_sessions
            ]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        # Test dataset does not involve windowing and entire sessions are
        # fed at once. Limit batch size to 1 to fit within GPU memory and
        # avoid any influence of padding (while collating multiple batch items)
        # in test scores.
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )


class CTCModuleBase(pl.LightningModule):
    def _setup_inference(self, inference: DictConfig | None) -> None:
        inference = inference or {}
        self._inference_policy = str(inference.get("policy", "full_session"))
        self._inference_window_length = inference.get("window_length")
        self._inference_stride = inference.get("stride")
        self._inference_trim_margin = int(inference.get("trim_margin", 0))
        self._inference_apply_on = str(inference.get("apply_on", "test"))

    def _setup_ctc(self, decoder: DictConfig) -> None:
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)

        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {f"{phase}_metrics": metrics.clone(prefix=f"{phase}/") for phase in ["train", "val", "test"]}
        )

    def _forward_for_ctc(self, inputs: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        del input_lengths
        return self.forward(inputs)

    def _augment_loss(self, phase: str, loss: torch.Tensor, emissions: torch.Tensor) -> torch.Tensor:
        del phase, emissions
        return loss

    def _use_windowed_inference(self, phase: str, batch_size: int) -> bool:
        return (
            phase == self._inference_apply_on
            and self._inference_policy != "full_session"
            and batch_size == 1
            and self._inference_window_length is not None
        )

    def _build_test_window_specs(self, full_length: int) -> list:
        window_length = int(self._inference_window_length)
        stride = int(self._inference_stride or window_length)
        return build_window_specs(
            total_length=full_length,
            window_length=window_length,
            stride=stride,
            trim_margin=self._inference_trim_margin,
        )

    def _run_windowed_chunk_decode(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> list[LabelData]:
        full_length = int(input_lengths[0].item())
        window_specs = self._build_test_window_specs(full_length)
        predictions: list[LabelData] = []

        for spec in window_specs:
            chunk_inputs = inputs[spec.start : spec.end]
            chunk_input_lengths = torch.tensor([spec.length], dtype=input_lengths.dtype, device=input_lengths.device)
            chunk_emissions = self._forward_for_ctc(chunk_inputs, chunk_input_lengths)
            resized = resize_log_probs_to_input_length(chunk_emissions, spec.length)
            kept = resized[spec.keep_start : spec.keep_end]

            self.decoder.reset()
            chunk_prediction = self.decoder.decode(
                emissions=kept[:, 0].detach().cpu().numpy(),
                timestamps=np.arange(spec.start + spec.keep_start, spec.start + spec.keep_end),
                finish=True,
            )
            predictions.append(chunk_prediction)

        if not predictions:
            return [LabelData.from_labels([])]

        merged_prediction = predictions[0]
        for prediction in predictions[1:]:
            merged_prediction = merged_prediction + prediction
        return [merged_prediction]

    def _run_windowed_logit_merge(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        full_length = int(input_lengths[0].item())
        window_specs = self._build_test_window_specs(full_length)
        chunk_log_probs: list[torch.Tensor] = []

        for spec in window_specs:
            chunk_inputs = inputs[spec.start : spec.end]
            chunk_input_lengths = torch.tensor([spec.length], dtype=input_lengths.dtype, device=input_lengths.device)
            chunk_emissions = self._forward_for_ctc(chunk_inputs, chunk_input_lengths)
            chunk_log_probs.append(chunk_emissions)

        merged = merge_log_prob_chunks(chunk_log_probs, window_specs, full_length)
        merged_lengths = torch.tensor([merged.shape[0]], dtype=input_lengths.dtype, device=input_lengths.device)
        return merged, merged_lengths

    def _run_inference_policy(
        self,
        phase: str,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, list[LabelData] | None]:
        batch_size = len(input_lengths)
        if not self._use_windowed_inference(phase, batch_size):
            emissions = self._forward_for_ctc(inputs, input_lengths)
            emission_lengths = self._compute_emission_lengths(
                input_lengths=input_lengths,
                input_timesteps=inputs.shape[0],
                emission_timesteps=emissions.shape[0],
            )
            return emissions, emission_lengths, None

        if self._inference_policy == "windowed_chunk_decode":
            predictions = self._run_windowed_chunk_decode(inputs, input_lengths)
            return None, None, predictions
        if self._inference_policy == "windowed_logits_merge":
            emissions, emission_lengths = self._run_windowed_logit_merge(inputs, input_lengths)
            return emissions, emission_lengths, None

        raise ValueError(f"Unsupported inference policy: {self._inference_policy}")

    def _step(self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions, emission_lengths, predictions = self._run_inference_policy(phase, inputs, input_lengths)

        loss: torch.Tensor | None = None
        if emissions is not None and emission_lengths is not None:
            loss = self.ctc_loss(
                log_probs=emissions,  # (T, N, num_classes)
                targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
                input_lengths=emission_lengths,  # (N,)
                target_lengths=target_lengths,  # (N,)
            )
            loss = self._augment_loss(phase, loss, emissions)

        if predictions is None:
            assert emissions is not None and emission_lengths is not None
            predictions = self.decoder.decode_batch(
                emissions=emissions.detach().cpu().numpy(),
                emission_lengths=emission_lengths.detach().cpu().numpy(),
            )

        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        if loss is not None:
            self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
            return loss

        fallback_loss = torch.zeros((), device=inputs.device)
        return fallback_loss

    def _compute_emission_lengths(
        self,
        input_lengths: torch.Tensor,
        input_timesteps: int,
        emission_timesteps: int,
    ) -> torch.Tensor:
        # Default: account for optional fixed temporal shrinkage in the encoder stack.
        T_diff = input_timesteps - emission_timesteps
        return (input_lengths - T_diff).clamp_min(1)

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )

    def _run_lstm_with_cudnn_fallback(
        self,
        encoder: nn.LSTM,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x = x.contiguous()
        try:
            encoder.flatten_parameters()
            return encoder(x)
        except RuntimeError as exc:
            # Some sequence shapes/strides can fail in cuDNN LSTM even after
            # making inputs contiguous. Fallback to the native PyTorch kernel.
            if x.is_cuda and "CUDNN_STATUS_NOT_SUPPORTED" in str(exc):
                with torch.backends.cudnn.flags(enabled=False):
                    return encoder(x)
            raise


class TDSConvCTCModule(CTCModuleBase):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        inference: DictConfig | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self._setup_inference(inference)

        num_features = self.NUM_BANDS * mlp_features[-1]

        # Model
        # inputs: (T, N, bands=2, electrode_channels=16, freq)
        self.model = nn.Sequential(
            # (T, N, bands=2, C=16, freq)
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            # (T, N, bands=2, mlp_features[-1])
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            # (T, N, num_features)
            nn.Flatten(start_dim=2),
            TDSConvEncoder(
                num_features=num_features,
                block_channels=block_channels,
                kernel_width=kernel_width,
            ),
            # (T, N, num_classes)
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        self._setup_ctc(decoder)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


class T5CTCModule(CTCModuleBase):
    """Transformer encoder + CTC head for EMG-to-text prediction.

    Reuses the same EMG front-end as :class:`TDSConvCTCModule`
    (SpectrogramNorm → MultiBandRotationInvariantMLP → Flatten) and replaces
    the TDS convolutional encoder with a standard PyTorch TransformerEncoder
    with sinusoidal positional encoding.
    The CTC loss, decoder, and metrics are identical to the baseline."""

    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        d_kv: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        inference: DictConfig | None = None,
        use_cnn: bool = True,
        blank_penalty_epochs: int = 40,
        blank_alpha_max: float = 50.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self._setup_inference(inference)

        self._use_cnn = use_cnn
        self._BLANK_PENALTY_EPOCHS = blank_penalty_epochs
        self._BLANK_ALPHA_MAX = blank_alpha_max

        num_features = self.NUM_BANDS * mlp_features[-1]  # 2 * 384 = 768

        # --- EMG front-end (reused from baseline) ---
        self.spec_norm = SpectrogramNorm(
            channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS,
        )
        self.mlp = MultiBandRotationInvariantMLP(
            in_features=in_features,
            mlp_features=mlp_features,
            num_bands=self.NUM_BANDS,
        )
        self.flatten = nn.Flatten(start_dim=2)

        # --- Project to transformer d_model ---
        self.input_proj = nn.Linear(num_features, d_model)

        # --- Temporal CNN featurizer (optional) ---
        # Research: CNN on sEMG before transformer gives ~8 CER improvement.
        # Conv1d expects (N, C, T), so we transpose around the conv block.
        if use_cnn:
            self.temporal_cnn = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
                nn.Dropout(0.1),
            )
        self.proj_norm = nn.LayerNorm(d_model)

        # --- Sinusoidal positional encoding ---
        max_len = 16000
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_encoding", pe.unsqueeze(1))  # (max_len, 1, d_model)
        self.pe_dropout = nn.Dropout(0.1)

        # --- Standard PyTorch Transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=0.1,
            activation="gelu",
            batch_first=False,  # time-first: (T, N, d_model)
            norm_first=True,  # Pre-LN for better gradient flow
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # --- CTC output head ---
        self.output_proj = nn.Linear(d_model, charset().num_classes)

        # Anti-blank initialization: discourage blank collapse at init.
        with torch.no_grad():
            nn.init.uniform_(self.output_proj.weight, -0.01, 0.01)
            blank_idx = charset().null_class
            num_chars = charset().num_classes - 1
            self.output_proj.bias.fill_(math.log(1.0 / num_chars))
            self.output_proj.bias[blank_idx] = -5.0

        self._setup_ctc(decoder)

    def _forward_for_ctc(self, inputs: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        return self.forward(inputs, input_lengths)

    def _augment_loss(self, phase: str, loss: torch.Tensor, emissions: torch.Tensor) -> torch.Tensor:
        if phase != "train":
            return loss

        blank_idx = charset().null_class
        current_epoch = self.current_epoch
        if current_epoch < self._BLANK_PENALTY_EPOCHS:
            blank_prob = emissions[:, :, blank_idx].exp().mean()
            alpha = self._BLANK_ALPHA_MAX * (1.0 - current_epoch / self._BLANK_PENALTY_EPOCHS)
            loss = loss + alpha * blank_prob
            self.log("train/blank_prob", blank_prob, prog_bar=True, sync_dist=True)
            self.log("train/blank_alpha", alpha, prog_bar=False, sync_dist=True)
        return loss

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, bands=2, C=16, freq)
        x = self.spec_norm(inputs)  # (T, N, 2, 16, freq)
        x = self.mlp(x)  # (T, N, 2, 384)
        x = self.flatten(x)  # (T, N, 768)
        x = self.input_proj(x)  # (T, N, d_model)

        # --- Optional temporal CNN featurizer ---
        if self._use_cnn:
            x = x.permute(1, 2, 0)  # (N, C, T)
            x = self.temporal_cnn(x)  # (N, C, T)
            x = x.permute(2, 0, 1)  # (T, N, C)
        x = self.proj_norm(x)

        T_len = x.shape[0]

        # Add sinusoidal positional encoding.
        # At test time, full sessions can exceed the pre-computed buffer length,
        # so we dynamically extend the encoding if necessary.
        if T_len <= self.pos_encoding.shape[0]:
            pe = self.pos_encoding[:T_len]
        else:
            d_model = self.pos_encoding.shape[2]
            pe = torch.zeros(T_len, 1, d_model, device=x.device)
            position = torch.arange(0, T_len, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float, device=x.device) * (-math.log(10000.0) / d_model)
            )
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        x = self.pe_dropout(x + pe)

        # Build key_padding_mask: True = ignore (PyTorch convention)
        key_padding_mask = torch.arange(T_len, device=x.device).unsqueeze(0) >= input_lengths.unsqueeze(1)  # (N, T)

        x = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)

        return F.log_softmax(self.output_proj(x), dim=-1)  # (T, N, num_classes)


class BiLSTMCTCModule(CTCModuleBase):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        hidden_size: int,
        num_layers: int,
        dropout: float,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        inference: DictConfig | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self._setup_inference(inference)

        num_features = self.NUM_BANDS * mlp_features[-1]

        self.frontend = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),
        )

        self.encoder = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=False,
        )
        self.classifier = nn.Linear(hidden_size * 2, charset().num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self._setup_ctc(decoder)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.frontend(inputs)
        x, _ = self._run_lstm_with_cudnn_fallback(self.encoder, x)
        x = self.classifier(x)
        return self.log_softmax(x)


class CNNBiLSTMCTCModule(CTCModuleBase):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        conv_channels: Sequence[int],
        conv_kernel_size: int,
        conv_dropout: float,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        inference: DictConfig | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self._setup_inference(inference)

        if conv_kernel_size % 2 == 0:
            raise ValueError("conv_kernel_size must be odd to preserve sequence length.")

        num_features = self.NUM_BANDS * mlp_features[-1]

        self.frontend = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),
        )

        channels = [num_features, *conv_channels]
        conv_blocks: list[nn.Module] = []
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            conv_blocks.extend(
                [
                    nn.Conv1d(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=conv_kernel_size,
                        padding=conv_kernel_size // 2,
                    ),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(),
                    nn.Dropout(conv_dropout),
                ]
            )
        self.temporal_conv = nn.Sequential(*conv_blocks)

        self.encoder = nn.LSTM(
            input_size=channels[-1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=False,
        )
        self.classifier = nn.Linear(hidden_size * 2, charset().num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self._setup_ctc(decoder)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.frontend(inputs)
        x = x.permute(1, 2, 0)  # (T, N, C) -> (N, C, T)
        x = self.temporal_conv(x)
        x = x.permute(2, 0, 1)  # (N, C, T) -> (T, N, C)
        x, _ = self._run_lstm_with_cudnn_fallback(self.encoder, x)
        x = self.classifier(x)
        return self.log_softmax(x)


class WhisperCTCModule(CTCModuleBase):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        whisper_model_name: str,
        projection_dropout: float,
        freeze_whisper_encoder: bool,
        unfrozen_encoder_layers: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        inference: DictConfig | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self._setup_inference(inference)

        try:
            whisper_model_cls = import_module("transformers").WhisperModel
        except ImportError as exc:
            raise ImportError(
                "WhisperCTCModule requires the optional 'transformers' dependency. "
                "Install it with: pip install transformers"
            ) from exc

        num_features = self.NUM_BANDS * mlp_features[-1]

        self.frontend = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),
        )

        self.whisper = whisper_model_cls.from_pretrained(whisper_model_name)
        self.encoder = self.whisper.encoder
        self.whisper_input_frames = self.encoder.config.max_source_positions * 2
        self.input_projection = nn.Sequential(
            nn.Linear(num_features, self.encoder.config.num_mel_bins),
            nn.Dropout(projection_dropout),
        )
        self.classifier = nn.Linear(self.encoder.config.d_model, charset().num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        if freeze_whisper_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

            if unfrozen := max(0, unfrozen_encoder_layers):
                for layer in self.encoder.layers[-unfrozen:]:
                    for param in layer.parameters():
                        param.requires_grad = True

        self._setup_ctc(decoder)

    def _compute_emission_lengths(
        self,
        input_lengths: torch.Tensor,
        input_timesteps: int,
        emission_timesteps: int,
    ) -> torch.Tensor:
        # Whisper encoder uses a conv stride of 2 in the temporal frontend.
        lengths = (input_lengths + 1) // 2
        return lengths.clamp_min(1).clamp_max(emission_timesteps)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.frontend(inputs)  # (T, N, C)
        x = x.permute(1, 0, 2)  # (N, T, C)
        x = self.input_projection(x)
        x = x.permute(0, 2, 1)  # (N, mel_bins, T)
        T = x.shape[-1]
        if T < self.whisper_input_frames:
            pad = self.whisper_input_frames - T
            x = nn.functional.pad(x, (0, pad), mode="constant", value=0.0)
        elif T > self.whisper_input_frames:
            x = x[..., : self.whisper_input_frames]
        encoder_hidden = self.encoder(input_features=x).last_hidden_state
        logits = self.classifier(encoder_hidden)  # (N, T', classes)
        return self.log_softmax(logits).permute(1, 0, 2)  # (T', N, classes)
