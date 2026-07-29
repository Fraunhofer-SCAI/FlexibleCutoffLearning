from __future__ import annotations

from copy import deepcopy
from typing import Callable, Dict, Optional

import torch
from torch import Tensor

from .metrics import MAE, MaxAE, RMSE


class PropertyTask:
    def __init__(
        self,
        name: str,
        target_key: str,
        loss_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        loss_weight: float = 1.0,
        loss_per_atom: bool = False,
        local: bool = False,
        metrics: Optional[Dict[str, torch.nn.Module]] = None,
    ):
        self.name = name
        self.target_key = target_key
        self.loss_fn = loss_fn or torch.nn.SmoothL1Loss()
        self.loss_weight = loss_weight
        self.loss_per_atom = loss_per_atom
        self.local = local
        self.metrics = metrics or {
            "MAE": MAE(),
            "RMSE": RMSE(),
            "MAX_AE": MaxAE(),
        }

    def clone_metrics(self) -> Dict[str, torch.nn.Module]:
        return {
            metric_name: deepcopy(metric)
            for metric_name, metric in self.metrics.items()
        }

    def prediction_tensor(self, prediction: Dict[str, Tensor]) -> Tensor:
        return prediction[self.name]

    def target_tensor(self, target: Dict[str, Tensor]) -> Tensor:
        return target[self.target_key]

    def compute_loss(
        self,
        prediction: Dict[str, Tensor],
        target: Dict[str, Tensor],
        n_atoms: Optional[Tensor] = None,
    ) -> Tensor:
        pred_value = self.prediction_tensor(prediction)
        target_value = self.target_tensor(target)

        if self.loss_per_atom and not self.local:
            if n_atoms is None:
                raise ValueError("n_atoms is required for per-atom losses.")
            pred_value = pred_value / n_atoms
            target_value = target_value / n_atoms

        return self.loss_weight * self.loss_fn(pred_value, target_value)


class EnergyTask(PropertyTask):
    def __init__(self, loss_per_atom: bool = True, **kwargs):
        super().__init__(
            name="energy",
            target_key="energy",
            loss_per_atom=loss_per_atom,
            metrics={
                "MAE": MAE(),
                "RMSE": RMSE(),
                "MAX_AE": MaxAE(),
                "MAE_per_atom": MAE(per_atom=True),
                "RMSE_per_atom": RMSE(per_atom=True),
                "MAX_AE_per_atom": MaxAE(per_atom=True),
            },
            **kwargs,
        )


class ForcesTask(PropertyTask):
    def __init__(self, **kwargs):
        super().__init__(
            name="forces",
            target_key="forces",
            local=True,
            **kwargs,
        )


class StressTask(PropertyTask):
    def __init__(self, **kwargs):
        super().__init__(
            name="stress",
            target_key="stress",
            **kwargs,
        )
