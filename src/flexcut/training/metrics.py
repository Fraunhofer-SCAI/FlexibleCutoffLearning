from __future__ import annotations

from typing import Optional

from torch import Tensor
from torchmetrics import MaxMetric, MeanAbsoluteError, MeanSquaredError


class MAE(MeanAbsoluteError):
    def __init__(self, per_atom: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.per_atom = per_atom

    def update(self, pred: Tensor, target: Tensor, n_atoms: Optional[Tensor] = None):
        if self.per_atom:
            if n_atoms is None:
                raise ValueError("n_atoms is required for per-atom metrics.")
            pred = pred / n_atoms
            target = target / n_atoms
        super().update(pred, target)


class RMSE(MeanSquaredError):
    def __init__(self, per_atom: bool = False, **kwargs):
        super().__init__(squared=False, **kwargs)
        self.per_atom = per_atom

    def update(self, pred: Tensor, target: Tensor, n_atoms: Optional[Tensor] = None):
        if self.per_atom:
            if n_atoms is None:
                raise ValueError("n_atoms is required for per-atom metrics.")
            pred = pred / n_atoms
            target = target / n_atoms
        super().update(pred, target)


class MaxAE(MaxMetric):
    def __init__(self, per_atom: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.per_atom = per_atom

    def update(self, pred: Tensor, target: Tensor, n_atoms: Optional[Tensor] = None):
        if self.per_atom:
            if n_atoms is None:
                raise ValueError("n_atoms is required for per-atom metrics.")
            pred = pred / n_atoms
            target = target / n_atoms
        super().update((pred - target).abs())
