from __future__ import annotations

import torch
from torch import Tensor


def per_atom_to_per_sample_index(index_per_atom: Tensor, batch: Tensor) -> Tensor:
    if index_per_atom.dim() != 1:
        raise RuntimeError("index_per_atom must be a 1D tensor")
    if batch.dim() != 1:
        raise RuntimeError("batch must be a 1D tensor")
    if index_per_atom.shape[0] != batch.shape[0]:
        raise RuntimeError("index_per_atom and batch must have the same length")
    if batch.numel() == 0:
        return torch.zeros(0, dtype=index_per_atom.dtype, device=index_per_atom.device)

    batch_long = batch.to(dtype=torch.long, device=index_per_atom.device)
    num_samples = int(torch.max(batch_long).item()) + 1
    index_per_sample = torch.zeros(
        num_samples, dtype=index_per_atom.dtype, device=index_per_atom.device
    )
    index_per_sample.scatter_(0, batch_long, index_per_atom)
    rebuilt_index_per_atom = torch.index_select(index_per_sample, 0, batch_long)
    if bool(torch.any(rebuilt_index_per_atom != index_per_atom)):
        raise RuntimeError(
            "All per-atom indices inside one batch item must be identical"
        )

    return index_per_sample


def per_sample_to_per_atom_index(index_per_sample: Tensor, batch: Tensor) -> Tensor:
    if index_per_sample.dim() != 1:
        raise RuntimeError("index_per_sample must be a 1D tensor")
    if batch.dim() != 1:
        raise RuntimeError("batch must be a 1D tensor")
    if batch.numel() == 0:
        return torch.zeros(
            0, dtype=index_per_sample.dtype, device=index_per_sample.device
        )
    if int(torch.max(batch).item()) >= index_per_sample.shape[0]:
        raise RuntimeError("batch contains a sample index that is out of bounds")

    return torch.index_select(index_per_sample, 0, batch.to(dtype=torch.long))
