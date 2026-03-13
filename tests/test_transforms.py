from __future__ import annotations

import torch

from emg2qwerty.transforms import ElectrodeChannelSubset


def test_electrode_channel_subset_masks_inactive_channels() -> None:
    tensor = torch.arange(2 * 2 * 4, dtype=torch.float32).reshape(2, 2, 4)

    transform = ElectrodeChannelSubset(channels=[1, 3], keep_shape=True)
    masked = transform(tensor)

    assert masked.shape == tensor.shape
    assert torch.equal(masked[..., [1, 3]], tensor[..., [1, 3]])
    assert torch.count_nonzero(masked[..., [0, 2]]) == 0


def test_electrode_channel_subset_can_compact_channels() -> None:
    tensor = torch.arange(2 * 2 * 4, dtype=torch.float32).reshape(2, 2, 4)

    transform = ElectrodeChannelSubset(channels=[0, 2], keep_shape=False)
    compact = transform(tensor)

    assert compact.shape == (2, 2, 2)
    assert torch.equal(compact, tensor[..., [0, 2]])