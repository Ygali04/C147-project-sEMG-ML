"""Ensemble evaluation: average log-probabilities from multiple models before
decoding. This leverages the fact that different architectures make different
errors, so their ensemble produces a better posterior.

Usage:
    uv run python -m emg2qwerty.ensemble_eval \
        --models model1_config:ckpt1.ckpt model2_config:ckpt2.ckpt \
        --decoder ctc_beam \
        --inference windowed_logits_merge \
        [--user single_user]
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from emg2qwerty import transforms
from emg2qwerty.data import LabelData
from emg2qwerty.inference import build_window_specs, merge_log_prob_chunks
from emg2qwerty.transforms import Transform

# Patch torch.load for Lightning compat
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

log = logging.getLogger(__name__)


def _load_model(
    model_config_name: str,
    checkpoint_path: str,
    base_config: DictConfig,
    device: torch.device,
) -> torch.nn.Module:
    """Load a single model from config + checkpoint."""
    # Build a config with the model override
    config_dir = str(Path(__file__).resolve().parent.parent.parent / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="base",
            overrides=[f"model={model_config_name}"],
        )

    # Import and instantiate the module class
    module = instantiate(
        cfg.module,
        optimizer=base_config.optimizer,
        lr_scheduler=base_config.lr_scheduler,
        decoder=base_config.decoder,
        inference=base_config.get("inference"),
        _recursive_=False,
    )
    # Load checkpoint weights
    module = type(module).load_from_checkpoint(
        checkpoint_path,
        optimizer=base_config.optimizer,
        lr_scheduler=base_config.lr_scheduler,
        decoder=base_config.decoder,
        inference=base_config.get("inference"),
        weights_only=False,
    )
    module.eval()
    module.to(device)
    return module


def ensemble_forward(
    models: list[torch.nn.Module],
    inputs: torch.Tensor,
    input_lengths: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Average log-probabilities from multiple models.

    Each model's forward() returns log_softmax outputs of shape (T_i, N, C).
    We interpolate to the same time dimension and average in probability space.
    """
    all_log_probs = []
    max_T = 0

    with torch.no_grad():
        for model in models:
            log_probs = model.forward(inputs.to(device), input_lengths.to(device))
            all_log_probs.append(log_probs)  # (T_i, N, C)
            max_T = max(max_T, log_probs.shape[0])

    # Interpolate all to the same time dimension and average
    aligned = []
    for lp in all_log_probs:
        if lp.shape[0] != max_T:
            # (T, N, C) -> (N, C, T) for interpolate -> (N, C, max_T) -> (max_T, N, C)
            lp_nct = lp.permute(1, 2, 0)
            lp_nct = F.interpolate(lp_nct, size=max_T, mode="linear", align_corners=False)
            lp = lp_nct.permute(2, 0, 1)
        aligned.append(lp)

    # Average in log-prob space (log-sum-exp for numerical stability)
    # log(avg(p1, p2)) = log((exp(lp1) + exp(lp2))/2)
    stacked = torch.stack(aligned, dim=0)  # (M, T, N, C)
    avg_log_probs = torch.logsumexp(stacked, dim=0) - math.log(len(models))

    return avg_log_probs


def main():
    parser = argparse.ArgumentParser(description="Ensemble model evaluation")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="model_config:checkpoint_path pairs, e.g. 'cnn_bilstm_ctc:/path/to/ckpt'",
    )
    parser.add_argument("--decoder", default="ctc_beam", help="Decoder config name")
    parser.add_argument("--inference", default="full_session", help="Inference config name")
    parser.add_argument("--user", default="single_user", help="User config name")
    parser.add_argument("--device", default="cuda:0", help="Device to run on")
    parser.add_argument(
        "--window-length", type=int, default=None, help="Window length for windowed inference (overrides config)"
    )
    parser.add_argument("--stride", type=int, default=None, help="Stride for windowed inference (overrides config)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Parse model specs
    model_specs = []
    for spec in args.models:
        parts = spec.split(":")
        if len(parts) != 2:
            log.error(f"Invalid model spec '{spec}'. Expected 'config:checkpoint'")
            sys.exit(1)
        model_specs.append((parts[0], parts[1]))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load base config
    config_dir = str(Path(__file__).resolve().parent.parent.parent / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        base_config = compose(
            config_name="base",
            overrides=[
                f"user={args.user}",
                f"decoder={args.decoder}",
                f"inference={args.inference}",
            ],
        )

    # Build transform and data paths
    cwd = str(Path(__file__).resolve().parent.parent.parent)
    dataset_root = Path(cwd) / "data"

    def _full_session_paths(dataset: ListConfig) -> list[Path]:
        sessions = [session["session"] for session in dataset]
        return [dataset_root / f"{session}.hdf5" for session in sessions]

    def _build_transform(configs: Sequence[DictConfig]) -> Transform[Any, Any]:
        return transforms.Compose([instantiate(cfg) for cfg in configs])

    # Load all models
    log.info(f"Loading {len(model_specs)} models for ensemble...")
    models = []
    for i, (model_cfg, ckpt_path) in enumerate(model_specs):
        log.info(f"  [{i+1}/{len(model_specs)}] {model_cfg} <- {ckpt_path}")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            model_config = compose(
                config_name="base",
                overrides=[
                    f"model={model_cfg}",
                    f"user={args.user}",
                    f"decoder={args.decoder}",
                    f"inference={args.inference}",
                ],
            )
        module = instantiate(
            model_config.module,
            optimizer=model_config.optimizer,
            lr_scheduler=model_config.lr_scheduler,
            decoder=model_config.decoder,
            inference=model_config.get("inference"),
            _recursive_=False,
        )
        module = type(module).load_from_checkpoint(
            ckpt_path,
            optimizer=model_config.optimizer,
            lr_scheduler=model_config.lr_scheduler,
            decoder=model_config.decoder,
            inference=model_config.get("inference"),
            weights_only=False,
        )
        module.eval()
        module.to(device)
        models.append(module)

    # Build decoder from the first model (all share the same decoder)
    decoder = models[0].decoder

    # Build data module
    datamodule = instantiate(
        base_config.datamodule,
        batch_size=base_config.batch_size,
        num_workers=base_config.num_workers,
        train_sessions=_full_session_paths(base_config.dataset.train),
        val_sessions=_full_session_paths(base_config.dataset.val),
        test_sessions=_full_session_paths(base_config.dataset.test),
        train_transform=_build_transform(base_config.transforms.train),
        val_transform=_build_transform(base_config.transforms.val),
        test_transform=_build_transform(base_config.transforms.test),
        _convert_="object",
    )
    datamodule.setup("test")

    # Evaluate on val and test
    from emg2qwerty.metrics import CharacterErrorRates

    for phase, dataloader in [
        ("val", datamodule.val_dataloader()),
        ("test", datamodule.test_dataloader()),
    ]:
        metrics = CharacterErrorRates()

        for batch_idx, batch in enumerate(dataloader):
            inputs = batch["inputs"].to(device)
            targets = batch["targets"]
            input_lengths = batch["input_lengths"].to(device)
            target_lengths = batch["target_lengths"]

            # Check if we need windowed inference for test
            use_windowed = phase == "test" and args.inference != "full_session" and args.window_length is not None

            if use_windowed:
                window_length = args.window_length
                stride = args.stride or window_length // 2
                full_T = inputs.shape[0]

                windows = build_window_specs(
                    total_length=full_T,
                    window_length=window_length,
                    stride=stride,
                    trim_margin=500,
                )

                all_chunks = []
                for ws in windows:
                    chunk_inputs = inputs[ws.start : ws.end]
                    chunk_lengths = torch.tensor([ws.length], device=device)
                    chunk_log_probs = ensemble_forward(models, chunk_inputs, chunk_lengths, device)
                    all_chunks.append(chunk_log_probs[:, 0, :].cpu())

                merged = merge_log_prob_chunks(all_chunks, windows, full_T)
                avg_log_probs = merged.unsqueeze(1)  # (T, 1, C)
                emission_lengths = torch.tensor([avg_log_probs.shape[0]])
            else:
                avg_log_probs = ensemble_forward(models, inputs, input_lengths, device)
                emission_lengths = input_lengths.cpu()

            # Decode
            predictions = decoder.decode_batch(
                emissions=avg_log_probs.detach().cpu().numpy(),
                emission_lengths=emission_lengths.detach().cpu().numpy(),
            )

            # Compute metrics
            N = len(target_lengths)
            tgt = targets.detach().cpu().numpy()
            tgt_lens = target_lengths.detach().cpu().numpy()
            for i in range(N):
                target = LabelData.from_labels(tgt[: tgt_lens[i], i])
                metrics.update(prediction=predictions[i], target=target)

        results = metrics.compute()
        print(f"\n{'='*60}")
        print(f"  {phase.upper()} Ensemble Results ({len(models)} models)")
        print(f"{'='*60}")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
