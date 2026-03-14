from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class WindowSpec:
    start: int
    end: int
    keep_start: int
    keep_end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def build_window_specs(
    total_length: int,
    window_length: int,
    stride: int,
    trim_margin: int = 0,
) -> list[WindowSpec]:
    if total_length <= 0:
        raise ValueError("total_length must be positive")
    if window_length <= 0 or stride <= 0:
        raise ValueError("window_length and stride must be positive")

    if window_length >= total_length:
        return [WindowSpec(0, total_length, 0, total_length)]

    last_start = max(total_length - window_length, 0)
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)

    specs: list[WindowSpec] = []
    for idx, start in enumerate(starts):
        end = min(start + window_length, total_length)
        local_length = end - start

        keep_start = trim_margin if idx > 0 else 0
        keep_end = local_length - trim_margin if idx < len(starts) - 1 else local_length

        # Avoid dropping the entire chunk if trim is too aggressive.
        if keep_end <= keep_start:
            keep_start = 0
            keep_end = local_length

        specs.append(
            WindowSpec(
                start=start,
                end=end,
                keep_start=keep_start,
                keep_end=keep_end,
            )
        )
    return specs


def resize_log_probs_to_input_length(log_probs: torch.Tensor, target_length: int) -> torch.Tensor:
    """Upsample chunk log-probabilities to the chunk input length.

    This makes window merge logic operate in the input-time domain, which keeps
    the chunk stitching generic across models with different temporal shrinkage.
    """
    if log_probs.ndim != 3:
        raise ValueError("log_probs must have shape (T, N, C)")
    if target_length <= 0:
        raise ValueError("target_length must be positive")

    if log_probs.shape[0] == target_length:
        return log_probs

    probs = log_probs.exp().permute(1, 2, 0)  # (N, C, T)
    resized = F.interpolate(probs, size=target_length, mode="linear", align_corners=False)
    resized = resized.permute(2, 0, 1)  # (T, N, C)
    resized = resized / resized.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return resized.log()


def merge_log_prob_chunks(
    chunk_log_probs: list[torch.Tensor],
    window_specs: list[WindowSpec],
    full_length: int,
) -> torch.Tensor:
    if len(chunk_log_probs) != len(window_specs):
        raise ValueError("chunk_log_probs and window_specs must have equal length")
    if not chunk_log_probs:
        raise ValueError("chunk_log_probs cannot be empty")

    batch_size = chunk_log_probs[0].shape[1]
    num_classes = chunk_log_probs[0].shape[2]
    device = chunk_log_probs[0].device

    merged_probs = torch.zeros(full_length, batch_size, num_classes, device=device)
    coverage = torch.zeros(full_length, batch_size, 1, device=device)

    for chunk_log_prob, spec in zip(chunk_log_probs, window_specs):
        resized = resize_log_probs_to_input_length(chunk_log_prob, spec.length)
        kept = resized[spec.keep_start : spec.keep_end].exp()
        global_start = spec.start + spec.keep_start
        global_end = spec.start + spec.keep_end
        merged_probs[global_start:global_end] += kept
        coverage[global_start:global_end] += 1.0

    merged_probs = merged_probs / coverage.clamp_min(1.0)
    merged_probs = merged_probs / merged_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return merged_probs.log()
