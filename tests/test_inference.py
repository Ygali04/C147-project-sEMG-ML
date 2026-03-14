from __future__ import annotations

import torch

from emg2qwerty.inference import build_window_specs, merge_log_prob_chunks, resize_log_probs_to_input_length


def test_build_window_specs_covers_full_range() -> None:
    specs = build_window_specs(total_length=20, window_length=8, stride=4, trim_margin=2)
    covered = set()
    for spec in specs:
        covered.update(range(spec.start + spec.keep_start, spec.start + spec.keep_end))

    assert covered == set(range(20))


def test_resize_log_probs_to_input_length_preserves_shape() -> None:
    log_probs = torch.log_softmax(torch.randn(5, 1, 4), dim=-1)
    resized = resize_log_probs_to_input_length(log_probs, target_length=11)

    assert resized.shape == (11, 1, 4)
    probs = resized.exp()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(11, 1), atol=1e-5)


def test_merge_log_prob_chunks_returns_normalized_log_probs() -> None:
    specs = build_window_specs(total_length=16, window_length=8, stride=4, trim_margin=2)
    chunks = [torch.log_softmax(torch.randn(8, 1, 6), dim=-1) for _ in specs]

    merged = merge_log_prob_chunks(chunks, specs, full_length=16)

    assert merged.shape == (16, 1, 6)
    probs = merged.exp()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(16, 1), atol=1e-5)
