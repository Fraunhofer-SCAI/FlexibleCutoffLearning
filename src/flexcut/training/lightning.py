from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, Optional, Union

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch_geometric.data import Batch, Data

from ..data.transforms import FLEXIBLE_CUTOFF_KEY, Neighbourhoods
from .adapter import Adapter, MlipAdapter
from .tasks import PropertyTask


AVERAGE_EDGES_PER_ATOM_KEY = "average_edges_per_atom"
GRAPH_WEIGHTED_COST_AGGREGATION = "per_graph_mean"
ATOM_WEIGHTED_COST_AGGREGATION = "per_atom_mean"
VALID_COST_AGGREGATIONS = {
    GRAPH_WEIGHTED_COST_AGGREGATION,
    ATOM_WEIGHTED_COST_AGGREGATION,
}
DEFAULT_REPORTED_TASK_METRICS = {
    "energy": ("MAE", "RMSE", "MAE_per_atom", "RMSE_per_atom"),
    "forces": ("MAE", "RMSE"),
}


def _set_model_outputs(model: torch.nn.Module, tasks: Iterable[PropertyTask]) -> None:
    compute_forces = any(task.name == "forces" for task in tasks)
    compute_stress = any(task.name == "stress" for task in tasks)
    model.compute_forces = compute_forces
    model.compute_stress = compute_stress


def _forward_model(
    model: torch.nn.Module,
    adapter: Adapter,
    data: Union[Dict[str, Tensor], Data],
) -> Dict[str, Tensor]:
    model_input = adapter(data)
    needs_grad_outputs = bool(
        getattr(model, "compute_forces", False)
        or getattr(model, "compute_stress", False)
    )
    if needs_grad_outputs and not torch.is_grad_enabled():
        with torch.enable_grad():
            return model.forward(**model_input)
    return model.forward(**model_input)


def _prepare_target(data: Union[Dict[str, Tensor], Data]) -> Dict[str, Tensor]:
    return {key: value for key, value in data.items() if isinstance(value, Tensor)}


def _num_atoms(batch: Tensor) -> Tensor:
    return torch.unique(batch, return_counts=True)[1].detach()


def _batch_size(batch: Tensor) -> int:
    return int(torch.unique(batch).numel())


def _weighted_loss(
    tasks: Iterable[PropertyTask],
    prediction: Dict[str, Tensor],
    target: Dict[str, Tensor],
) -> Tensor:
    n_atoms = _num_atoms(target["batch"])
    total_loss = torch.zeros((), device=target["batch"].device)
    for task in tasks:
        total_loss = total_loss + task.compute_loss(
            prediction=prediction,
            target=target,
            n_atoms=n_atoms,
        )
    return total_loss


class MlipLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        tasks: Iterable[PropertyTask],
        scheduler: Optional[LRScheduler] = None,
        adapter: Optional[Adapter] = None,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tasks = list(tasks)
        self.adapter = adapter or MlipAdapter()

        if len(self.tasks) == 0:
            raise ValueError("At least one task is required.")

        _set_model_outputs(self.model, self.tasks)
        self.metrics: Dict[str, Dict[str, Dict[str, torch.nn.Module]]] = {}
        self.reset_metrics()

    def reset_metrics(self) -> None:
        self.metrics = {
            subset: {task.name: task.clone_metrics() for task in self.tasks}
            for subset in ("val", "test")
        }

    def setup(self, stage: Optional[str] = None) -> None:
        for subset_metrics in self.metrics.values():
            for task_metrics in subset_metrics.values():
                for metric in task_metrics.values():
                    metric.to(self.device)

    def forward(self, data: Union[Dict[str, Tensor], Data]) -> Dict[str, Tensor]:
        return _forward_model(self.model, self.adapter, data)

    def _prepare_target(
        self, data: Union[Dict[str, Tensor], Data]
    ) -> Dict[str, Tensor]:
        return _prepare_target(data)

    def _num_atoms(self, batch: Tensor) -> Tensor:
        return _num_atoms(batch)

    def loss(self, prediction: Dict[str, Tensor], target: Dict[str, Tensor]) -> Tensor:
        return _weighted_loss(self.tasks, prediction, target)

    def training_step(
        self, data: Union[Dict[str, Tensor], Data], batch_idx: int
    ) -> Tensor:
        prediction = self.forward(data)
        target = self._prepare_target(data)
        loss = self.loss(prediction, target)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self._batch_size(target["batch"]),
        )
        return loss

    def validation_step(
        self, data: Union[Dict[str, Tensor], Data], batch_idx: int
    ) -> None:
        self._eval_step(data, subset="val")

    def test_step(self, data: Union[Dict[str, Tensor], Data], batch_idx: int) -> None:
        self._eval_step(data, subset="test")

    def _eval_step(self, data: Union[Dict[str, Tensor], Data], subset: str) -> None:
        prediction = self.forward(data)
        target = self._prepare_target(data)
        loss = self.loss(prediction, target)
        n_atoms = self._num_atoms(target["batch"])

        self.log(
            f"{subset}/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=subset == "val",
            batch_size=self._batch_size(target["batch"]),
        )

        for task in self.tasks:
            pred_value = task.prediction_tensor(prediction)
            target_value = task.target_tensor(target)
            for metric in self.metrics[subset][task.name].values():
                metric(pred_value, target_value, n_atoms=n_atoms)

    def on_validation_epoch_end(self) -> None:
        self._log_metrics("val")

    def on_test_epoch_end(self) -> None:
        self._log_metrics("test")

    def _log_metrics(self, subset: str) -> None:
        for task_name, task_metrics in self.metrics[subset].items():
            for metric_name, metric in task_metrics.items():
                self.log(
                    f"{subset}/{task_name}/{metric_name}",
                    metric.compute(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                metric.reset()

    def _batch_size(self, batch: Tensor) -> int:
        return _batch_size(batch)

    def configure_optimizers(self):
        if self.scheduler is None:
            return [self.optimizer]

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "monitor": self.metric_for_monitoring(),
                "name": "lr_scheduler",
            },
        }

    def metric_for_monitoring(self) -> str:
        return "val/loss"


class CutoffCalibrationLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        tasks: Iterable[PropertyTask],
        lambda_cost: float,
        *,
        atomic_numbers: Optional[Iterable[int]] = None,
        initial_cutoffs_by_atomic_number: Optional[Dict[int, float]] = None,
        cutoff_bounds: tuple[float, float] = (3.5, 7.0),
        learning_rate: float = 1e-2,
        neighbourlist_implementation: str = "pymatgen",
        cost_aggregation: str = GRAPH_WEIGHTED_COST_AGGREGATION,
        task_weights: Optional[Dict[str, float]] = None,
        adapter: Optional[Adapter] = None,
    ):
        super().__init__()
        self.model = model
        self.base_tasks = list(tasks)
        if len(self.base_tasks) == 0:
            raise ValueError("At least one task is required.")
        self.tasks = self._override_task_weights(self.base_tasks, task_weights)
        self.lambda_cost = float(lambda_cost)
        self.learning_rate = float(learning_rate)
        self.cutoff_bounds = (float(cutoff_bounds[0]), float(cutoff_bounds[1]))
        if self.cutoff_bounds[0] <= 0 or self.cutoff_bounds[1] <= self.cutoff_bounds[0]:
            raise ValueError("cutoff_bounds must satisfy 0 < low < high.")
        self.cost_aggregation = self._validate_cost_aggregation(cost_aggregation)

        inferred_atomic_numbers = self._infer_atomic_numbers(atomic_numbers)
        self.atomic_numbers = inferred_atomic_numbers
        self.register_buffer(
            "atomic_numbers_tensor",
            torch.tensor(self.atomic_numbers, dtype=torch.long),
            persistent=False,
        )

        initial_cutoffs = self._initial_cutoff_tensor(initial_cutoffs_by_atomic_number)
        self.register_buffer(
            "initial_cutoffs_tensor",
            initial_cutoffs.clone(),
            persistent=False,
        )
        self.cutoff_parameters = torch.nn.Parameter(initial_cutoffs.clone())
        self.adapter = adapter or MlipAdapter()
        self.neighbourhoods = Neighbourhoods(
            cutoff=self.cutoff_bounds[1],
            implementation=neighbourlist_implementation,
        )
        self.latest_report: Dict[str, Any] = {}
        self.split_summaries: Dict[str, Dict[str, float]] = {}
        self.split_task_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._split_totals: Dict[str, Dict[str, float]] = {}
        self.metrics: Dict[str, Dict[str, Dict[str, torch.nn.Module]]] = {}

        _set_model_outputs(self.model, self.tasks)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.reset_metrics()

    def reset_metrics(self) -> None:
        self.metrics = {
            split: self._fresh_task_metrics()
            for split in ("train", "val", "test")
        }
        self.split_task_metrics = {}

    def _fresh_task_metrics(self) -> Dict[str, Dict[str, torch.nn.Module]]:
        split_metrics: Dict[str, Dict[str, torch.nn.Module]] = {}
        for task in self.tasks:
            task_metrics = self._clone_report_metrics(task)
            if task_metrics:
                split_metrics[task.name] = task_metrics
        return split_metrics

    def _clone_report_metrics(
        self, task: PropertyTask
    ) -> Dict[str, torch.nn.Module]:
        cloned_metrics = task.clone_metrics()
        metric_names = DEFAULT_REPORTED_TASK_METRICS.get(task.name)
        if metric_names is None:
            return cloned_metrics
        return {
            metric_name: cloned_metrics[metric_name]
            for metric_name in metric_names
            if metric_name in cloned_metrics
        }

    def setup(self, stage: Optional[str] = None) -> None:
        for split_metrics in self.metrics.values():
            for task_metrics in split_metrics.values():
                for metric in task_metrics.values():
                    metric.to(self.device)

    def on_fit_start(self) -> None:
        self.reset_metrics()
        self.setup(stage="fit")

    def on_test_start(self) -> None:
        self.setup(stage="test")

    def _update_task_metrics(
        self,
        split: str,
        prediction: Dict[str, Tensor],
        target: Dict[str, Tensor],
        n_atoms: Tensor,
    ) -> None:
        split_metrics = self.metrics.get(split, {})
        for task in self.tasks:
            task_metrics = split_metrics.get(task.name)
            if not task_metrics:
                continue
            pred_value = task.prediction_tensor(prediction)
            target_value = task.target_tensor(target)
            for metric in task_metrics.values():
                metric(pred_value, target_value, n_atoms=n_atoms)

    def _serialize_metric_collection(
        self,
        metrics_by_task: Dict[str, Dict[str, torch.nn.Module]],
    ) -> Dict[str, Dict[str, float]]:
        serialized: Dict[str, Dict[str, float]] = {}
        for task_name, task_metrics in metrics_by_task.items():
            if not task_metrics:
                continue
            serialized[task_name] = {
                metric_name: float(metric.compute().detach().cpu())
                for metric_name, metric in task_metrics.items()
            }
        return serialized

    def _record_split_task_metrics(self, split: str, *, log: bool) -> None:
        split_metrics = self.metrics.get(split, {})
        serialized = self._serialize_metric_collection(split_metrics)
        if serialized:
            self.split_task_metrics[split] = serialized
        if log:
            for task_name, task_metrics in split_metrics.items():
                for metric_name, metric in task_metrics.items():
                    self.log(
                        f"{split}/{task_name}/{metric_name}",
                        metric.compute(),
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                    )
        for task_metrics in split_metrics.values():
            for metric in task_metrics.values():
                metric.reset()

    def _task_metric_delta(
        self,
        before_metrics: Dict[str, Dict[str, float]],
        after_metrics: Dict[str, Dict[str, float]],
    ) -> Dict[str, Dict[str, float]]:
        delta: Dict[str, Dict[str, float]] = {}
        for task_name in sorted(set(before_metrics) | set(after_metrics)):
            before_task_metrics = before_metrics.get(task_name, {})
            after_task_metrics = after_metrics.get(task_name, {})
            delta[task_name] = {
                metric_name: float(after_task_metrics.get(metric_name, 0.0))
                - float(before_task_metrics.get(metric_name, 0.0))
                for metric_name in sorted(
                    set(before_task_metrics) | set(after_task_metrics)
                )
            }
        return delta

    def _split_task_metric_delta(
        self,
        before_metrics: Dict[str, Dict[str, Dict[str, float]]],
        after_metrics: Dict[str, Dict[str, Dict[str, float]]],
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        delta: Dict[str, Dict[str, Dict[str, float]]] = {}
        for split in sorted(set(before_metrics) | set(after_metrics)):
            delta[split] = self._task_metric_delta(
                before_metrics.get(split, {}),
                after_metrics.get(split, {}),
            )
        return delta

    def _infer_atomic_numbers(
        self, atomic_numbers: Optional[Iterable[int]]
    ) -> list[int]:
        if atomic_numbers is not None:
            numbers = sorted({int(number) for number in atomic_numbers})
            if len(numbers) == 0:
                raise ValueError("atomic_numbers must not be empty.")
            return numbers

        candidates = [
            getattr(self.model, "atomic_numbers", None),
            getattr(getattr(self.model, "model", None), "atomic_numbers", None),
            getattr(getattr(self.model, "backbone", None), "atomic_numbers", None),
        ]
        for candidate in candidates:
            if candidate is not None:
                return sorted({int(number) for number in candidate})
        raise ValueError(
            "atomic_numbers must be provided when the model does not expose them."
        )

    def _initial_cutoff_tensor(
        self, initial_cutoffs_by_atomic_number: Optional[Dict[int, float]]
    ) -> Tensor:
        if initial_cutoffs_by_atomic_number is None:
            midpoint = 0.5 * (self.cutoff_bounds[0] + self.cutoff_bounds[1])
            values = [midpoint for _ in self.atomic_numbers]
        else:
            values = []
            for atomic_number in self.atomic_numbers:
                if atomic_number not in initial_cutoffs_by_atomic_number:
                    raise KeyError(
                        f"Missing initial cutoff for atomic number {atomic_number}."
                    )
                values.append(float(initial_cutoffs_by_atomic_number[atomic_number]))

        cutoffs = torch.tensor(values, dtype=torch.float32)
        if torch.any(cutoffs <= 0):
            raise ValueError("Initial cutoffs must be strictly positive.")
        return cutoffs

    def current_cutoffs(self) -> Tensor:
        if torch.any(self.cutoff_parameters <= 0):
            raise RuntimeError(
                "Optimized cutoffs must remain strictly positive. "
                "Adjust the learning rate or initialization if optimization crosses zero."
            )
        return self.cutoff_parameters

    def cutoffs_by_atomic_number(self) -> Dict[int, float]:
        return {
            atomic_number: float(cutoff)
            for atomic_number, cutoff in zip(
                self.atomic_numbers,
                self.current_cutoffs().detach().cpu().tolist(),
            )
        }

    def initial_cutoffs_by_atomic_number(self) -> Dict[int, float]:
        return {
            atomic_number: float(cutoff)
            for atomic_number, cutoff in zip(
                self.atomic_numbers,
                self.initial_cutoffs_tensor.detach().cpu().tolist(),
            )
        }

    def effective_task_weights(self) -> Dict[str, float]:
        return {task.name: float(task.loss_weight) for task in self.tasks}

    def _override_task_weights(
        self,
        tasks: Iterable[PropertyTask],
        task_weights: Optional[Dict[str, float]],
    ) -> list[PropertyTask]:
        overridden_tasks = [deepcopy(task) for task in tasks]
        if task_weights is None:
            return overridden_tasks
        for task in overridden_tasks:
            if task.name in task_weights:
                task.loss_weight = float(task_weights[task.name])
        return overridden_tasks

    def _validate_cost_aggregation(self, cost_aggregation: str) -> str:
        normalized = str(cost_aggregation).strip().lower()
        if normalized not in VALID_COST_AGGREGATIONS:
            options = ", ".join(sorted(VALID_COST_AGGREGATIONS))
            raise ValueError(
                f"cost_aggregation must be one of: {options}."
            )
        return normalized

    def _apply_current_cutoffs(
        self, data: Union[Data, Batch]
    ) -> Batch:
        data_list = data.to_data_list() if isinstance(data, Batch) else [data]
        current_cutoffs = self.current_cutoffs()
        processed_samples: list[Data] = []

        for sample in data_list:
            species = sample["z"].long()
            atomic_numbers = self.atomic_numbers_tensor.to(device=species.device)
            cutoff_indices = torch.searchsorted(atomic_numbers, species)
            safe_indices = cutoff_indices.clamp(max=atomic_numbers.numel() - 1)
            valid_species = (cutoff_indices < atomic_numbers.numel()) & (
                atomic_numbers[safe_indices] == species
            )
            if not torch.all(valid_species):
                missing_atomic_number = int(species[~valid_species][0].item())
                raise KeyError(
                    f"Missing calibrated cutoff for atomic number {missing_atomic_number}."
                )

            per_node_cutoffs = current_cutoffs.to(
                device=sample["pos"].device,
                dtype=sample["pos"].dtype,
            )[cutoff_indices]

            sample[FLEXIBLE_CUTOFF_KEY] = per_node_cutoffs
            processed_samples.append(self.neighbourhoods(sample))

        return Batch.from_data_list(processed_samples)

    def forward(self, data: Union[Dict[str, Tensor], Data]) -> Dict[str, Tensor]:
        return _forward_model(self.model, self.adapter, data)

    def _mean_per_graph(self, values: Tensor, batch: Tensor) -> Tensor:
        num_graphs = _batch_size(batch)
        graph_sums = torch.zeros(num_graphs, device=values.device, dtype=values.dtype)
        graph_sums.index_add_(0, batch, values)
        atoms_per_graph = torch.bincount(batch, minlength=num_graphs).to(values.dtype)
        return graph_sums / atoms_per_graph.clamp_min(1.0)

    def _retained_edges_per_atom(self, batch: Batch) -> Tensor:
        num_graphs = _batch_size(batch.batch)
        atoms_per_graph = torch.bincount(batch.batch, minlength=num_graphs).to(
            batch.pos.dtype
        )
        if batch.edge_index.shape[1] == 0:
            return torch.zeros((), device=batch.pos.device, dtype=batch.pos.dtype)
        edge_graph = batch.batch[batch.edge_index[0]]
        edges_per_graph = torch.bincount(edge_graph, minlength=num_graphs).to(
            batch.pos.dtype
        )
        return (edges_per_graph / atoms_per_graph.clamp_min(1.0)).mean()

    def _cost_proxy(self, batch: Batch) -> Tensor:
        cutoff_volume = batch[FLEXIBLE_CUTOFF_KEY].pow(3.0)
        if self.cost_aggregation == ATOM_WEIGHTED_COST_AGGREGATION:
            return cutoff_volume.mean()
        per_graph = self._mean_per_graph(cutoff_volume, batch.batch)
        return per_graph.mean()

    def _evaluate_batch(
        self, data: Union[Data, Batch]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Batch, Dict[str, Tensor], Dict[str, Tensor], Tensor]:
        batch = self._apply_current_cutoffs(data)
        prediction = self.forward(batch)
        target = _prepare_target(batch)
        n_atoms = _num_atoms(target["batch"])
        epsilon = _weighted_loss(self.tasks, prediction, target)
        cost = self._cost_proxy(batch)
        avg_edges_per_atom = self._retained_edges_per_atom(batch)
        objective = epsilon + self.lambda_cost * cost
        return (
            objective,
            epsilon,
            cost,
            avg_edges_per_atom,
            batch,
            prediction,
            target,
            n_atoms,
        )

    def _reset_split(self, split: str) -> None:
        self._split_totals[split] = {
            "objective": 0.0,
            "epsilon": 0.0,
            "cost": 0.0,
            AVERAGE_EDGES_PER_ATOM_KEY: 0.0,
            "weight": 0.0,
        }

    def _update_split_summary(
        self,
        split: str,
        objective: Tensor,
        epsilon: Tensor,
        cost: Tensor,
        avg_edges_per_atom: Tensor,
        batch: Batch,
    ) -> None:
        if split not in self._split_totals:
            self._reset_split(split)
        weight = float(_batch_size(batch.batch))
        totals = self._split_totals[split]
        totals["objective"] += float(objective.detach()) * weight
        totals["epsilon"] += float(epsilon.detach()) * weight
        totals["cost"] += float(cost.detach()) * weight
        totals[AVERAGE_EDGES_PER_ATOM_KEY] += (
            float(avg_edges_per_atom.detach()) * weight
        )
        totals["weight"] += weight

    def _finalize_split(self, split: str) -> None:
        totals = self._split_totals.get(split)
        if totals is None or totals["weight"] == 0.0:
            return
        weight = totals["weight"]
        self.split_summaries[split] = {
            "objective": totals["objective"] / weight,
            "epsilon": totals["epsilon"] / weight,
            "cost": totals["cost"] / weight,
            AVERAGE_EDGES_PER_ATOM_KEY: totals[AVERAGE_EDGES_PER_ATOM_KEY] / weight,
        }
        self.latest_report = self.build_report(primary_split=split)

    def on_train_epoch_start(self) -> None:
        self._reset_split("train")

    def on_validation_epoch_start(self) -> None:
        self._reset_split("val")

    def on_test_epoch_start(self) -> None:
        self._reset_split("test")

    def training_step(
        self, data: Union[Dict[str, Tensor], Data], batch_idx: int
    ) -> Tensor:
        (
            objective,
            epsilon,
            cost,
            avg_edges_per_atom,
            batch,
            prediction,
            target,
            n_atoms,
        ) = self._evaluate_batch(data)
        self._update_split_summary(
            "train", objective, epsilon, cost, avg_edges_per_atom, batch
        )
        self._update_task_metrics("train", prediction, target, n_atoms)
        self.log(
            "train/objective",
            objective,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=_batch_size(batch.batch),
        )
        self.log("train/epsilon", epsilon, on_step=False, on_epoch=True)
        self.log("train/cost", cost, on_step=False, on_epoch=True)
        self.log(
            "train/average_retained_edges_per_atom",
            avg_edges_per_atom,
            on_step=False,
            on_epoch=True,
        )
        return objective

    def validation_step(
        self, data: Union[Dict[str, Tensor], Data], batch_idx: int
    ) -> None:
        (
            objective,
            epsilon,
            cost,
            avg_edges_per_atom,
            batch,
            prediction,
            target,
            n_atoms,
        ) = self._evaluate_batch(data)
        self._update_split_summary("val", objective, epsilon, cost, avg_edges_per_atom, batch)
        self._update_task_metrics("val", prediction, target, n_atoms)
        self.log(
            "val/objective",
            objective,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=_batch_size(batch.batch),
        )
        self.log("val/epsilon", epsilon, on_step=False, on_epoch=True)
        self.log("val/cost", cost, on_step=False, on_epoch=True)
        self.log(
            "val/average_retained_edges_per_atom",
            avg_edges_per_atom,
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, data: Union[Dict[str, Tensor], Data], batch_idx: int) -> None:
        (
            objective,
            epsilon,
            cost,
            avg_edges_per_atom,
            batch,
            prediction,
            target,
            n_atoms,
        ) = self._evaluate_batch(data)
        self._update_split_summary(
            "test", objective, epsilon, cost, avg_edges_per_atom, batch
        )
        self._update_task_metrics("test", prediction, target, n_atoms)
        self.log(
            "test/objective",
            objective,
            on_step=False,
            on_epoch=True,
            batch_size=_batch_size(batch.batch),
        )
        self.log("test/epsilon", epsilon, on_step=False, on_epoch=True)
        self.log("test/cost", cost, on_step=False, on_epoch=True)
        self.log(
            "test/average_retained_edges_per_atom",
            avg_edges_per_atom,
            on_step=False,
            on_epoch=True,
        )

    def on_train_epoch_end(self) -> None:
        self._record_split_task_metrics("train", log=True)

    def on_validation_epoch_end(self) -> None:
        self._record_split_task_metrics("val", log=True)
        self._finalize_split("val")

    def on_test_epoch_end(self) -> None:
        self._record_split_task_metrics("test", log=True)
        self._finalize_split("test")

    def on_train_end(self) -> None:
        self._finalize_split("train")
        if not self.latest_report:
            self.latest_report = self.build_report(primary_split="train")

    def summarize_dataloaders(
        self,
        dataloaders: Dict[str, Iterable[Union[Data, Batch]]],
        primary_split: Optional[str] = None,
    ) -> Dict[str, Any]:
        split_summaries: Dict[str, Dict[str, float]] = {}
        split_task_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
        was_training = self.training
        self.eval()
        try:
            for split, dataloader in dataloaders.items():
                totals = {
                    "objective": 0.0,
                    "epsilon": 0.0,
                    "cost": 0.0,
                    AVERAGE_EDGES_PER_ATOM_KEY: 0.0,
                    "weight": 0.0,
                }
                local_metrics = self._fresh_task_metrics()
                for task_metrics in local_metrics.values():
                    for metric in task_metrics.values():
                        metric.to(self.device)
                for data in dataloader:
                    if hasattr(data, "to"):
                        data = data.to(self.device)
                    with torch.no_grad():
                        (
                            objective,
                            epsilon,
                            cost,
                            avg_edges_per_atom,
                            batch,
                            prediction,
                            target,
                            n_atoms,
                        ) = (
                            self._evaluate_batch(data)
                        )
                    weight = float(_batch_size(batch.batch))
                    totals["objective"] += float(objective.detach()) * weight
                    totals["epsilon"] += float(epsilon.detach()) * weight
                    totals["cost"] += float(cost.detach()) * weight
                    totals[AVERAGE_EDGES_PER_ATOM_KEY] += (
                        float(avg_edges_per_atom.detach()) * weight
                    )
                    totals["weight"] += weight
                    for task in self.tasks:
                        task_metrics = local_metrics.get(task.name)
                        if not task_metrics:
                            continue
                        pred_value = task.prediction_tensor(prediction)
                        target_value = task.target_tensor(target)
                        for metric in task_metrics.values():
                            metric(pred_value, target_value, n_atoms=n_atoms)
                if totals["weight"] == 0.0:
                    continue
                split_summaries[split] = {
                    "objective": totals["objective"] / totals["weight"],
                    "epsilon": totals["epsilon"] / totals["weight"],
                    "cost": totals["cost"] / totals["weight"],
                    AVERAGE_EDGES_PER_ATOM_KEY: totals[AVERAGE_EDGES_PER_ATOM_KEY]
                    / totals["weight"],
                }
                split_task_metrics[split] = self._serialize_metric_collection(
                    local_metrics
                )
        finally:
            self.train(was_training)

        return self.build_report(
            primary_split=primary_split,
            split_summaries=split_summaries,
            split_task_metrics=split_task_metrics,
        )

    def build_report(
        self,
        primary_split: Optional[str] = None,
        *,
        split_summaries: Optional[Dict[str, Dict[str, float]]] = None,
        split_task_metrics: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
        cutoffs_by_atomic_number: Optional[Dict[int, float]] = None,
    ) -> Dict[str, Any]:
        summaries = deepcopy(
            self.split_summaries if split_summaries is None else split_summaries
        )
        task_metric_summaries = deepcopy(
            self.split_task_metrics
            if split_task_metrics is None
            else split_task_metrics
        )
        split_name = primary_split
        if split_name is None:
            for candidate in ("val", "test", "train"):
                if candidate in summaries:
                    split_name = candidate
                    break
        if split_name is None:
            current_cost = torch.mean(self.current_cutoffs().pow(3.0)).detach().cpu()
            primary_summary = {
                "objective": float(self.lambda_cost * current_cost),
                "epsilon": 0.0,
                "cost": float(current_cost),
                AVERAGE_EDGES_PER_ATOM_KEY: 0.0,
            }
        else:
            primary_summary = summaries.get(split_name, {})
        primary_task_metrics = (
            task_metric_summaries.get(split_name, {}) if split_name is not None else {}
        )

        if cutoffs_by_atomic_number is None:
            cutoffs = self.cutoffs_by_atomic_number()
        else:
            cutoffs = {
                int(atomic_number): float(cutoff)
                for atomic_number, cutoff in cutoffs_by_atomic_number.items()
            }

        return {
            "cutoffs_by_atomic_number": cutoffs,
            "lambda": self.lambda_cost,
            "cost_aggregation": self.cost_aggregation,
            "objective": primary_summary.get("objective", 0.0),
            "epsilon": primary_summary.get("epsilon", 0.0),
            "cost": primary_summary.get("cost", 0.0),
            AVERAGE_EDGES_PER_ATOM_KEY: primary_summary.get(
                AVERAGE_EDGES_PER_ATOM_KEY, 0.0
            ),
            "task_metrics": primary_task_metrics,
            "effective_task_weights": self.effective_task_weights(),
            "split_summaries": summaries,
            "split_task_metrics": task_metric_summaries,
            "cutoff_bounds": {
                "low": self.cutoff_bounds[0],
                "high": self.cutoff_bounds[1],
            },
            "primary_split": split_name,
        }

    def build_comparison_report(
        self,
        before_report: Dict[str, Any],
        primary_split: Optional[str] = None,
    ) -> Dict[str, Any]:
        after_report = self.build_report(primary_split=primary_split)
        scalar_keys = (
            "objective",
            "epsilon",
            "cost",
            AVERAGE_EDGES_PER_ATOM_KEY,
        )
        optimization_delta = {
            key: float(after_report.get(key, 0.0)) - float(before_report.get(key, 0.0))
            for key in scalar_keys
        }

        before_split_summaries = before_report.get("split_summaries", {})
        after_split_summaries = after_report.get("split_summaries", {})
        split_summary_delta: Dict[str, Dict[str, float]] = {}
        for split in sorted(set(before_split_summaries) | set(after_split_summaries)):
            before_summary = before_split_summaries.get(split, {})
            after_summary = after_split_summaries.get(split, {})
            split_summary_delta[split] = {
                key: float(after_summary.get(key, 0.0)) - float(before_summary.get(key, 0.0))
                for key in scalar_keys
            }
        optimization_delta["split_summaries"] = split_summary_delta
        optimization_delta["task_metrics"] = self._task_metric_delta(
            before_report.get("task_metrics", {}),
            after_report.get("task_metrics", {}),
        )
        optimization_delta["split_task_metrics"] = self._split_task_metric_delta(
            before_report.get("split_task_metrics", {}),
            after_report.get("split_task_metrics", {}),
        )

        before_cutoffs = before_report.get("cutoffs_by_atomic_number", {})
        after_cutoffs = after_report.get("cutoffs_by_atomic_number", {})
        cutoff_delta: Dict[int, float] = {}
        for atomic_number in sorted(
            {int(key) for key in before_cutoffs} | {int(key) for key in after_cutoffs}
        ):
            cutoff_delta[atomic_number] = float(after_cutoffs.get(atomic_number, 0.0)) - float(
                before_cutoffs.get(atomic_number, 0.0)
            )
        optimization_delta["cutoffs_by_atomic_number"] = cutoff_delta

        after_report["before_optimization"] = deepcopy(before_report)
        after_report["optimization_delta"] = optimization_delta
        return after_report

    def configure_optimizers(self):
        return [Adam([self.cutoff_parameters], lr=self.learning_rate)]

    def metric_for_monitoring(self) -> str:
        return "val/objective"
