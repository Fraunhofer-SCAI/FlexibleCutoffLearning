from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from warnings import warn

import h5py
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.data import BaseData
from torch_geometric.transforms import BaseTransform
from tqdm import tqdm

def data_list_to_hdf5(
    data_list: List[Data], filepath: str, exclude_keys: Optional[List[str]] = None
) -> None:
    _exclude_keys = exclude_keys if exclude_keys is not None else []
    data, slices = InMemoryDataset.collate(data_list)

    with h5py.File(filepath, "w") as f:
        data_group = f.create_group("data")
        slices_group = f.create_group("slices")
        additional = f.create_group("additional")
        encode_utf8_keys = []
        for key, value in data.items():
            if (
                "/" in key
            ):  # when saving, change / to : to avoid that the key will be interpreted as a path.
                escaped_key = key.replace("/", ":")
                warn(
                    f"Found '/' in key={key}. This will cause problems when saving to hdf5. "
                    f"Renaming the key to {escaped_key}. "
                    f"Make sure to use {escaped_key} and not {key} after loading the data from disk."
                )
            else:
                escaped_key = key
            if key in _exclude_keys:
                continue
            if isinstance(value, list):
                if isinstance(value[0], str):
                    value = [v.encode("utf-8") for v in value]
                    # save those keys that correspond to str and were encoded using utf-8.
                    # Need to encode those keys themselves.
                    encode_utf8_keys.append(key.encode("utf-8"))
                value = np.array(value)
            elif isinstance(value, torch.Tensor):
                value = value.numpy()
            slice_value = slices[key].numpy()
            data_group.create_dataset(escaped_key, data=value)
            slices_group.create_dataset(escaped_key, data=slice_value)
        # save keys for entries that were encoded with utf8
        additional.create_dataset("utf-8-encoded", data=np.array(encode_utf8_keys))


class HDF5Dataset(InMemoryDataset):
    """
    Class to load custom hdf5 dataset files. With the following hierarchy:
    file:
        group 'data'
            - dataset 'pos' (coordinates, shape=[n_nodes,3])
            - dataset 'z' (atomic numbers, shape=[n_nodes])
            - dataset 'edge_index' (optional, shape=[2,n_edges])
            - dataset 'cell' (optional, shape=[n_config,3,3])
            - dataset 'shifts' (optional, shape=[n_edges,3])
            - dataset '<property_1>'
            - ...
        group 'slices' (stores index pointers for each property)
            - dataset 'pos' (slices for coordinates)
            - dataset 'z' (slices for atomic numbers)
            - dataset 'edge_index' (optional)
            - dataset 'cell' (optional)
            - dataset 'shifts' (optional)
            - dataset '<property_1>'
            - ...

    n_config: Number of all configurations in the dataset (number of connected components in the graph)
    n_nodes: Number of all atoms in all configurations
    n_edges: Number of all edges in all configurations

    If edge_index (and shifts in the periodic case) are not provided, they can be computed on the fly by providing a
    neighbourlist calculator to transform.
    """

    def __init__(
        self,
        data: BaseData,
        slices: dict[str, Tensor],
        transform: Optional[BaseTransform] = None,
    ):
        super().__init__(root=str(Path.cwd()), transform=transform)
        self.data, self.slices = data, slices

    @staticmethod
    def from_hdf5(
        filepath: str,
        precision: int = 32,
        transform: Optional[BaseTransform] = None,
    ) -> "HDF5Dataset":
        data: dict[str, object] = {}
        slices: dict[str, Tensor] = {}
        with h5py.File(filepath, "r") as handle:
            if "additional" in handle and "utf-8-encoded" in handle["additional"]:
                decode_utf8 = [
                    key.decode("utf-8")
                    for key in handle["additional"]["utf-8-encoded"][:]
                ]
            else:
                decode_utf8 = []

            for key in handle["data"].keys():
                np_data = handle["data"][key][:]
                np_slices = handle["slices"][key][:]

                if np_data.dtype == np.uint64:
                    np_data = np_data.astype(np.int64)
                if np_slices.dtype == np.uint64:
                    np_slices = np_slices.astype(np.int64)

                if np_data.dtype == np.float64 and precision == 32:
                    np_data = np_data.astype(np.float32)
                elif np_data.dtype == np.float32 and precision == 64:
                    warn(
                        "You are trying to load the dataset with 64-bit precision, but the on-disk data is float32. "
                        "Casting will be applied."
                    )
                    np_data = np_data.astype(np.float64)

                if key in decode_utf8:
                    data[key] = [item.decode("utf-8") for item in np_data.tolist()]
                else:
                    data[key] = torch.from_numpy(np_data)
                slices[key] = torch.from_numpy(np_slices)

        return HDF5Dataset(
            data=Data.from_dict(data),
            slices=slices,
            transform=transform,
        )

    @staticmethod
    def from_ase(
        filepaths: List[str],
        file_format: Optional[str] = None,
        precision: int = 32,
        index: Optional[slice] = ":",
        transform: Optional[BaseTransform] = None,
    ) -> HDF5Dataset:

        from ase.io import read

        assert precision in [32, 64], "Precision must be either 32 or 64!"

        data = {}
        slices = {}
        dtype = torch.double if precision == 64 else torch.float32

        for filepath in filepaths:
            atoms_list = read(filepath, index=index, format=file_format)
            for i, atoms in enumerate(atoms_list):
                atom_data = {
                    "pos": torch.tensor(atoms.positions, dtype=dtype),
                    "z": torch.tensor(atoms.numbers, dtype=torch.long).view(-1),
                }

                for key, value in atom_data.items():
                    if key not in data:
                        data[key] = []
                        slices[key] = [0]
                    data[key].append(value)
                    increment_slice = value.shape[0]
                    slice_index = slices[key][-1] + increment_slice
                    slices[key].append(slice_index)

        for key in data:
            data[key] = torch.cat(data[key], dim=0)
            slices[key] = torch.tensor(slices[key], dtype=torch.long)

        dataset = HDF5Dataset(
            data=Data.from_dict(data),
            slices=slices,
            transform=transform,
        )
        return dataset

    def to_hdf5(self, filepath: str, exclude_keys: Optional[List[str]] = None):

        data_list = [
            d
            for d in tqdm(
                self,
                desc="Iterate over dataset and apply lazy transformations before saving.",
            )
        ]

        data_list_to_hdf5(
            data_list=data_list, filepath=filepath, exclude_keys=exclude_keys
        )


    @property
    def keys(self):
        storage = self._data if self._data is not None else self.data
        return storage.keys()


    def get_ensbids(self):
        if "ensbid" in self._data.keys():
            ensbids = np.array(self._data["ensbid"])[self.indices()].tolist()
        else:
            ensbids = None
        return ensbids

