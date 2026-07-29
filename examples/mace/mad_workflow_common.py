from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

from flexcut import EnergyTask, ForcesTask, MACEWrapper, MlipLightningModule
from flexcut import load_dataset, make_dataloaders

REPO_ROOT = Path(__file__).resolve().parents[2]
MAD_DATA_DIR = REPO_ROOT / "MAD-data"
MAD_PROCESSED_DIR = MAD_DATA_DIR / "processed"
DEFAULT_TRAIN_DATASET = MAD_PROCESSED_DIR / "mad-train-all.hdf5"
DEFAULT_VAL_DATASET = MAD_PROCESSED_DIR / "mad-val-all.hdf5"
DEFAULT_TEST_DATASET = MAD_PROCESSED_DIR / "mad-test-all.hdf5"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "examples" / "mace" / "runs" / "mad_multistage"

PAPER_PRETRAIN_CUTOFF = 6.0
PAPER_FCL_CUTOFF_LOW = 3.5
PAPER_FCL_CUTOFF_HIGH = 7.0
PAPER_PRETRAIN_AVG_NEIGHBORS = 65.4
PAPER_FCL_AVG_NEIGHBORS = 50.0
PAPER_CALIBRATION_SUBSETS = (
    "MC3D",
    "MC2D",
    "SHIFTML-molcrys",
    "SHIFTML-molfrags",
)
PAPER_LAMBDA_VALUES = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)


def mad_subset_dataset_path(split: str, subset: str) -> Path:
    return (
        MAD_PROCESSED_DIR
        / f"{split}_split_by_subset"
        / f"mad-{split}-{subset}.hdf5"
    )


DEFAULT_CALIBRATION_DATASET = mad_subset_dataset_path("val", "MC2D")
DEFAULT_EVALUATION_DATASET = mad_subset_dataset_path("test", "MC2D")


def format_lambda_value(value: float) -> str:
    formatted = f"{float(value):.0e}"
    return formatted.replace("e-0", "e-").replace("e+0", "e+")


def seed_everything(seed: int) -> None:
    pl.seed_everything(seed, workers=True)


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Any) -> None:
    ensure_output_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def average_edges_per_atom(dataset: Iterable[object]) -> float:
    values: list[float] = []
    for sample in dataset:
        if not hasattr(sample, "pos") or not hasattr(sample, "edge_index"):
            continue
        num_atoms = int(sample.pos.shape[0])
        if num_atoms == 0:
            continue
        num_edges = int(sample.edge_index.shape[1])
        values.append(num_edges / num_atoms)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def average_edges_per_atom_by_cutoff(
    datasets_by_cutoff: Mapping[float, Iterable[object]],
) -> dict[str, float]:
    return {
        str(float(cutoff)): average_edges_per_atom(dataset)
        for cutoff, dataset in datasets_by_cutoff.items()
    }


def build_default_tasks(
    *,
    energy_weight: float = 0.1,
    force_weight: float = 1.0,
):
    return [
        EnergyTask(loss_fn=torch.nn.L1Loss(), loss_weight=energy_weight),
        ForcesTask(loss_fn=torch.nn.L1Loss(), loss_weight=force_weight),
    ]


def build_scale_shift_mace_wrapper(
    *,
    cutoff: float,
    num_bessel: int,
    num_polynomial_cutoff: int,
    max_ell: int,
    num_interactions: int,
    hidden_irreps: str,
    mlp_irreps: str,
    correlation: int,
    atomic_numbers: Sequence[int],
    avg_num_neighbors: float,
) -> MACEWrapper:
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    import numpy as np
    from e3nn import o3
    from mace.modules.blocks import RealAgnosticInteractionBlock
    from mace.modules.blocks import RealAgnosticResidualInteractionBlock
    from mace.modules.models import ScaleShiftMACE

    model = ScaleShiftMACE(
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
        r_max=float(cutoff),
        num_bessel=int(num_bessel),
        num_polynomial_cutoff=int(num_polynomial_cutoff),
        max_ell=int(max_ell),
        interaction_cls=RealAgnosticResidualInteractionBlock,
        interaction_cls_first=RealAgnosticInteractionBlock,
        num_interactions=int(num_interactions),
        num_elements=len(atomic_numbers),
        hidden_irreps=o3.Irreps(hidden_irreps),
        MLP_irreps=o3.Irreps(mlp_irreps),
        atomic_energies=np.zeros(len(atomic_numbers)),
        avg_num_neighbors=float(avg_num_neighbors),
        atomic_numbers=[int(number) for number in atomic_numbers],
        correlation=int(correlation),
        gate=torch.nn.functional.silu,
    )
    return MACEWrapper(model=model)


def set_avg_num_neighbors(model: torch.nn.Module, avg_num_neighbors: float) -> None:
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        backbone = getattr(model, "model", None)
    if backbone is None:
        backbone = model

    interactions = getattr(backbone, "interactions", None)
    if interactions is None:
        return
    for interaction in interactions:
        if hasattr(interaction, "avg_num_neighbors"):
            interaction.avg_num_neighbors = float(avg_num_neighbors)


def load_mad_dataset(
    dataset_path: Path,
    *,
    cutoff: float,
    precision: int,
    transforms: Iterable[object] | None = None,
    neighbourlist_implementation: str = "pymatgen",
):
    return load_dataset(
        dataset_path,
        cutoff=cutoff,
        precision=precision,
        transforms=transforms,
        neighbourlist_implementation=neighbourlist_implementation,
    )


def make_split_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    *,
    batch_size: int,
    num_workers: int,
):
    return make_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def build_training_module(
    model: torch.nn.Module,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: Any = None,
    energy_weight: float,
    force_weight: float,
) -> MlipLightningModule:
    return MlipLightningModule(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        tasks=build_default_tasks(
            energy_weight=energy_weight,
            force_weight=force_weight,
        ),
    )


def build_trainer(
    *,
    accelerator: str,
    devices: int,
    max_epochs: int,
    precision: int,
    default_root_dir: Path,
    callbacks: Sequence[pl.Callback] | None = None,
) -> pl.Trainer:
    callback_list = list(callbacks or [])
    return pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        precision=precision,
        inference_mode=False,
        logger=False,
        enable_checkpointing=any(
            isinstance(callback, ModelCheckpoint) for callback in callback_list
        ),
        enable_model_summary=False,
        callbacks=callback_list,
        default_root_dir=str(default_root_dir),
    )
