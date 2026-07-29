from __future__ import annotations

import torch


def scatter_add(
    x: torch.Tensor, idx_i: torch.Tensor, dim_size: int, dim: int = 0
) -> torch.Tensor:
    return _scatter_add(x, idx_i, dim_size, dim)


def _scatter_add(
    x: torch.Tensor, idx_i: torch.Tensor, dim_size: int, dim: int = 0
) -> torch.Tensor:
    shape = list(x.shape)
    shape[dim] = dim_size
    tmp = torch.zeros(shape, dtype=x.dtype, device=x.device)
    y = tmp.index_add(dim, idx_i, x)
    return y
