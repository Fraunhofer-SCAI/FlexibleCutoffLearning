from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from .datasets import HDF5Dataset
from .transforms import Compose, Neighbourhoods, Rename


def load_dataset(
    filepath: str | Path,
    *,
    cutoff: float,
    rename_map: Optional[dict[str, str]] = None,
    precision: int = 32,
    transforms: Optional[Iterable[object]] = None,
    neighbourlist_implementation: str = "pymatgen",
) -> HDF5Dataset:
    dataset = HDF5Dataset.from_hdf5(
        str(filepath),
        precision=precision,
        transform=Compose([]),
    )

    if rename_map:
        missing_keys = sorted(set(rename_map) - set(dataset.keys))
        if missing_keys:
            raise KeyError(
                f"Could not find keys {missing_keys} in dataset at {filepath}."
            )
        dataset.transform.append(Rename(key_map=rename_map))

    for transform in transforms or []:
        dataset.transform.append(transform)

    dataset.transform.append(
        Neighbourhoods(
            cutoff=cutoff,
            implementation=neighbourlist_implementation,
        )
    )
    return dataset


def split_dataset(
    dataset: HDF5Dataset,
    *,
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int = 42,
):
    total = train_size + val_size + test_size
    if total > 1.0:
        raise ValueError("train_size + val_size + test_size must be <= 1.0")

    ensbids = dataset.get_ensbids()
    if ensbids is not None:
        grouped_indices: dict[str, list[int]] = {}
        for index, ensbid in enumerate(ensbids):
            grouped_indices.setdefault(ensbid, []).append(index)

        ensemble_ids = list(grouped_indices)
        rng = np.random.default_rng(seed)
        rng.shuffle(ensemble_ids)

        n_train = math.floor(train_size * len(ensemble_ids))
        n_val = math.floor(val_size * len(ensemble_ids))
        n_test = math.floor(test_size * len(ensemble_ids))

        train_ids = ensemble_ids[:n_train]
        val_ids = ensemble_ids[n_train : n_train + n_val]
        test_ids = ensemble_ids[n_train + n_val : n_train + n_val + n_test]

        train_indices = [idx for ensbid in train_ids for idx in grouped_indices[ensbid]]
        val_indices = [idx for ensbid in val_ids for idx in grouped_indices[ensbid]]
        test_indices = [idx for ensbid in test_ids for idx in grouped_indices[ensbid]]
    else:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)
        n_train = math.floor(train_size * len(dataset))
        n_val = math.floor(val_size * len(dataset))
        n_test = math.floor(test_size * len(dataset))
        train_indices = indices[:n_train].tolist()
        val_indices = indices[n_train : n_train + n_val].tolist()
        test_indices = indices[n_train + n_val : n_train + n_val + n_test].tolist()

    return (
        dataset[train_indices],
        dataset[val_indices],
        dataset[test_indices],
    )


def make_dataloaders(
    trainset,
    valset,
    testset,
    *,
    batch_size: int,
    num_workers: int = 0,
):
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader
