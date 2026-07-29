from __future__ import annotations

from typing import Literal, Mapping, Optional
from warnings import warn

import ase.geometry
import ase.neighborlist
import numpy as np
import torch
from pymatgen.optimization.neighbors import find_points_in_spheres
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from torch_geometric.transforms import BaseTransform as PygBaseTransform
from torch_geometric.transforms import Compose as PygCompose

from ..utils import data_keys

FLEXIBLE_CUTOFF_KEY = "flexible_cutoff_per_node"
EDGE_CUTOFF_KEY = "edge_cutoff"


class Compose(PygCompose):
    def append(self, transform: PygBaseTransform) -> None:
        self.transforms.append(transform)


class Rename(PygBaseTransform):
    def __init__(self, key_map: dict[str, str]):
        super().__init__()
        self.key_map = key_map

    def forward(self, data: Data) -> Data:
        for old_key, new_key in self.key_map.items():
            if old_key == new_key:
                continue
            if new_key in data_keys(data):
                warn(
                    f"Renaming '{old_key}' to '{new_key}' but '{new_key}' already exists in the data. Overwriting existing data."
                )
            if old_key in data_keys(data):
                data[new_key] = data[old_key]
                del data[old_key]
        return data


class FlexibleCutoffTransform(PygBaseTransform):
    def _validate_single_sample(self, data: Data) -> list[str]:
        keys = data_keys(data)
        if "pos" not in keys:
            raise AssertionError("Could not find data.pos")
        if "batch" in keys:
            batch_size = torch.unique(data["batch"]).numel()
            if batch_size != 1:
                raise AssertionError(
                    f"{self.__class__.__name__} can only be applied to single samples, not mini-batches."
                )
        return keys

    def _coerce_cutoffs(
        self, values: Tensor, num_nodes: int, dtype: torch.dtype
    ) -> Tensor:
        if values.ndim == 2 and values.shape[1] == 1:
            values = values.view(-1)
        if values.ndim != 1:
            raise AssertionError(
                f"{FLEXIBLE_CUTOFF_KEY} must be a rank-1 tensor or shape (N, 1)."
            )
        if values.shape[0] != num_nodes:
            raise AssertionError(
                f"{FLEXIBLE_CUTOFF_KEY} must provide one value per node."
            )
        if torch.any(values <= 0):
            raise AssertionError(
                f"{FLEXIBLE_CUTOFF_KEY} values must be strictly positive."
            )
        return values.to(dtype=dtype)


class SampleFlexibleCutoff(FlexibleCutoffTransform):
    def __init__(
        self,
        low: float = 3.5,
        high: float = 7.0,
        mode: Literal["uniform", "inverse_cubic"] = "uniform",
        homogenity: Literal[
            "per_node", "per_system", "per_element", "mixed"
        ] = "per_node",
        max_stddev_per_system: Optional[float] = None,
    ):
        super().__init__()
        if low <= 0:
            raise AssertionError("low must be > 0")
        if high <= low:
            raise AssertionError("high must be > low")
        if mode not in ("uniform", "inverse_cubic"):
            raise AssertionError(f"Unknown sampling mode {mode}")
        self.low = float(low)
        self.high = float(high)
        self.mode = mode
        self.homogenity = homogenity
        self.max_stddev_per_system = max_stddev_per_system
        if max_stddev_per_system is not None and homogenity == "per_system":
            warn(
                "max_stddev_per_system has no effect when homogenity='per_system'."
            )

    def _sample(
        self,
        low: float,
        high: float,
        size: tuple[int, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if self.mode == "uniform":
            return torch.empty(size, device=device, dtype=dtype).uniform_(low, high)

        low_tensor = torch.as_tensor(low, device=device, dtype=dtype)
        high_tensor = torch.as_tensor(high, device=device, dtype=dtype)
        u = torch.rand(size, device=device, dtype=dtype)
        inv_low_sq = 1.0 / (low_tensor * low_tensor)
        inv_high_sq = 1.0 / (high_tensor * high_tensor)
        denom = inv_low_sq - u * (inv_low_sq - inv_high_sq)
        return 1.0 / torch.sqrt(denom)

    def forward(self, data: Data) -> Data:
        self._validate_single_sample(data)
        homogenity = (
            np.random.choice(["per_node", "per_system", "per_element"])
            if self.homogenity == "mixed"
            else self.homogenity
        )

        if homogenity == "per_system":
            size = (1,)
        elif homogenity == "per_element":
            size = (int(torch.unique(data["z"]).numel()),)
        else:
            size = (int(data["pos"].shape[0]),)

        cutoffs = self._sample(
            self.low,
            self.high,
            size,
            device=data["pos"].device,
            dtype=data["pos"].dtype,
        )

        if self.max_stddev_per_system is not None and homogenity != "per_system":
            stddev = float(cutoffs.std()) if cutoffs.numel() > 1 else 0.0
            if stddev > self.max_stddev_per_system:
                scale = self.max_stddev_per_system / stddev
                cutoffs = (cutoffs - cutoffs.mean()) * scale + cutoffs.mean()

        if homogenity == "per_element":
            _, inverse_species = torch.unique(
                data["z"],
                sorted=True,
                return_inverse=True,
            )
            per_node_cutoffs = cutoffs[inverse_species]
        elif homogenity == "per_system":
            per_node_cutoffs = cutoffs.repeat(data["pos"].shape[0])
        else:
            per_node_cutoffs = cutoffs

        data[FLEXIBLE_CUTOFF_KEY] = self._coerce_cutoffs(
            per_node_cutoffs,
            num_nodes=int(data["pos"].shape[0]),
            dtype=data["pos"].dtype,
        )
        return data


class ElementwiseFlexibleCutoff(FlexibleCutoffTransform):
    def __init__(self, cutoffs_by_atomic_number: Mapping[int, float]):
        super().__init__()
        if len(cutoffs_by_atomic_number) == 0:
            raise ValueError("cutoffs_by_atomic_number must not be empty.")
        self.cutoffs_by_atomic_number = {
            int(key): float(value) for key, value in cutoffs_by_atomic_number.items()
        }
        if any(value <= 0 for value in self.cutoffs_by_atomic_number.values()):
            raise ValueError("Element-wise cutoffs must be strictly positive.")
        self.atomic_numbers = tuple(sorted(self.cutoffs_by_atomic_number))
        self.cutoff_values = tuple(
            self.cutoffs_by_atomic_number[atomic_number]
            for atomic_number in self.atomic_numbers
        )

    def forward(self, data: Data) -> Data:
        self._validate_single_sample(data)
        keys = data_keys(data)
        if "z" not in keys:
            raise AssertionError("Could not find data.z")

        species = data["z"].long()
        atomic_numbers = torch.tensor(
            self.atomic_numbers,
            device=species.device,
            dtype=species.dtype,
        )
        cutoff_indices = torch.searchsorted(atomic_numbers, species)
        safe_indices = cutoff_indices.clamp(max=atomic_numbers.numel() - 1)
        valid_species = (cutoff_indices < atomic_numbers.numel()) & (
            atomic_numbers[safe_indices] == species
        )
        if not torch.all(valid_species):
            missing_atomic_number = int(species[~valid_species][0].item())
            raise KeyError(f"Missing cutoff for atomic number {missing_atomic_number}.")
        per_node = torch.tensor(
            self.cutoff_values,
            device=species.device,
            dtype=data["pos"].dtype,
        )[cutoff_indices]

        data[FLEXIBLE_CUTOFF_KEY] = self._coerce_cutoffs(
            per_node,
            num_nodes=int(data["pos"].shape[0]),
            dtype=data["pos"].dtype,
        )
        return data


def _ase_neighbor_list(
    pos: Tensor,
    cutoff: float,
    cell: Optional[Tensor],
    pbc: Tensor,
) -> tuple[Tensor, Tensor]:
    np_pos = pos.detach().cpu().numpy()
    out_device = pos.device
    out_dtype = pos.dtype

    if cell is not None:
        np_cell = cell.detach().cpu().numpy()
        completed_cell = ase.geometry.complete_cell(np_cell)
        pbc_tuple = tuple(bool(flag) for flag in pbc.tolist())
    else:
        completed_cell = np.zeros((3, 3), dtype=np_pos.dtype)
        pbc_tuple = (False, False, False)

    src, dst, shifts = ase.neighborlist.primitive_neighbor_list(
        "ijS",
        pbc_tuple,
        completed_cell,
        np_pos,
        cutoff=float(cutoff),
        self_interaction=False,
        use_scaled_positions=False,
    )

    edge_index = torch.vstack(
        [torch.as_tensor(src, dtype=torch.long), torch.as_tensor(dst, dtype=torch.long)]
    ).to(out_device)

    if cell is None or not any(pbc_tuple):
        shift_vectors = np.zeros((len(src), 3), dtype=np_pos.dtype)
    else:
        shift_vectors = shifts @ completed_cell

    return edge_index, torch.as_tensor(
        shift_vectors,
        dtype=out_dtype,
        device=out_device,
    )


def _pymatgen_neighbor_list(
    pos: Tensor,
    cutoff: float,
    cell: Optional[Tensor],
    pbc: Tensor,
) -> tuple[Tensor, Tensor]:
    np_pbc = pbc.long().detach().cpu().numpy() if pbc.any() else np.array([0, 0, 0])
    np_cell = (
        cell.view(3, 3).detach().cpu().numpy().astype(np.float64)
        if cell is not None
        else np.eye(3, dtype=np.float64) * 1e5
    )
    np_pos = pos.detach().cpu().numpy().astype(np.float64)

    src_id, dst_id, images, bond_dist = find_points_in_spheres(
        np_pos,
        np_pos,
        r=cutoff,
        pbc=np_pbc,
        lattice=np_cell,
        tol=1e-8,
        min_r=0.1,
    )

    exclude_self = (src_id != dst_id) | (bond_dist > 1e-8)
    src_id = src_id[exclude_self]
    dst_id = dst_id[exclude_self]
    shifts = images[exclude_self] @ np_cell

    edge_index = torch.from_numpy(np.vstack([src_id, dst_id])).long().to(pos.device)
    shift_tensor = torch.from_numpy(shifts).to(device=pos.device, dtype=pos.dtype)
    return edge_index, shift_tensor


class Neighbourhoods(PygBaseTransform):
    def __init__(
        self,
        cutoff: float,
        ignore_existing_edges: bool = True,
        implementation: Literal["pymatgen", "ase", "radius_graph"] = "pymatgen",
    ):
        super().__init__()
        self.cutoff = cutoff
        self.ignore_existing_edges = ignore_existing_edges
        self.implementation = implementation

    def _compute_distance_vectors(
        self,
        pos: Tensor,
        edge_index: Tensor,
        shifts: Optional[Tensor] = None,
    ) -> Tensor:
        vectors = pos[edge_index[1]] - pos[edge_index[0]]
        if shifts is not None:
            vectors = vectors + shifts
        return vectors

    def _apply_cutoff_mask(
        self,
        pos: Tensor,
        edge_index: Tensor,
        shifts: Optional[Tensor] = None,
        cutoff: Optional[Tensor | float] = None,
    ) -> tuple[Tensor, Optional[Tensor], Tensor]:
        distances = torch.norm(
            self._compute_distance_vectors(pos, edge_index, shifts),
            dim=1,
            p=2,
        )
        if cutoff is None:
            cutoff_tensor = torch.full_like(distances, fill_value=self.cutoff)
        elif isinstance(cutoff, Tensor):
            cutoff_tensor = cutoff.to(device=distances.device, dtype=distances.dtype)
            if cutoff_tensor.ndim == 0:
                cutoff_tensor = cutoff_tensor.expand_as(distances)
        else:
            cutoff_tensor = torch.full_like(distances, fill_value=float(cutoff))
        edge_mask = distances <= cutoff_tensor
        masked_shifts = shifts[edge_mask] if shifts is not None else None
        return edge_index[..., edge_mask], masked_shifts, cutoff_tensor[edge_mask]

    def _get_flexible_cutoffs(self, data: Data) -> Optional[Tensor]:
        keys = data_keys(data)
        if FLEXIBLE_CUTOFF_KEY not in keys:
            return None
        values = data[FLEXIBLE_CUTOFF_KEY]
        if not isinstance(values, Tensor):
            raise AssertionError(f"{FLEXIBLE_CUTOFF_KEY} must be a torch.Tensor.")
        if values.ndim == 2 and values.shape[1] == 1:
            values = values.view(-1)
        if values.ndim != 1:
            raise AssertionError(
                f"{FLEXIBLE_CUTOFF_KEY} must be a rank-1 tensor or shape (N, 1)."
            )
        if values.shape[0] != data["pos"].shape[0]:
            raise AssertionError(
                f"{FLEXIBLE_CUTOFF_KEY} must provide one value per node."
            )
        if torch.any(values <= 0):
            raise AssertionError(
                f"{FLEXIBLE_CUTOFF_KEY} values must be strictly positive."
            )
        return values.to(device=data["pos"].device, dtype=data["pos"].dtype)

    def _build_edges(
        self,
        pos: Tensor,
        cell: Optional[Tensor],
        pbc: Tensor,
        cutoff: Optional[float] = None,
    ) -> tuple[Tensor, Tensor]:
        effective_cutoff = self.cutoff if cutoff is None else float(cutoff)
        if self.implementation == "radius_graph" and not pbc.any():
            edge_index = radius_graph(
                x=pos,
                r=effective_cutoff,
                batch=None,
                max_num_neighbors=320,
            ).long()
            shifts = torch.zeros(
                edge_index.shape[1],
                3,
                device=pos.device,
                dtype=pos.dtype,
            )
            return edge_index, shifts
        if self.implementation == "ase":
            return _ase_neighbor_list(
                pos=pos, cutoff=effective_cutoff, cell=cell, pbc=pbc
            )
        return _pymatgen_neighbor_list(
            pos=pos, cutoff=effective_cutoff, cell=cell, pbc=pbc
        )

    def forward(self, data: Data) -> Data:
        keys = data_keys(data)
        if "pos" not in keys:
            raise AssertionError("Could not find data.pos")
        if "batch" in keys:
            batch_size = torch.unique(data["batch"]).numel()
            if batch_size != 1:
                raise AssertionError(
                    "Neighbourhoods can only be applied to single samples, not mini-batches."
                )

        if "cell" in keys and data["cell"] is not None:
            pbc = data["pbc"] if "pbc" in keys else torch.ones(3, dtype=torch.bool)
            if pbc.dtype != torch.bool or pbc.shape[0] != 3:
                raise AssertionError("pbc must be a torch.bool tensor of shape (3,)")
            cell = data["cell"].view(3, 3)
        else:
            pbc = torch.zeros(3, dtype=torch.bool, device=data["pos"].device)
            cell = None

        flexible_cutoffs = self._get_flexible_cutoffs(data)

        if "edge_index" in keys and not self.ignore_existing_edges:
            edge_index = data["edge_index"]
            shifts = data["shifts"] if "shifts" in keys else None
            if flexible_cutoffs is None:
                edge_index, shifts, edge_cutoff = self._apply_cutoff_mask(
                    data["pos"], edge_index, shifts
                )
            else:
                pair_cutoff = 0.5 * (
                    flexible_cutoffs[edge_index[0]] + flexible_cutoffs[edge_index[1]]
                )
                edge_index, shifts, edge_cutoff = self._apply_cutoff_mask(
                    data["pos"], edge_index, shifts, cutoff=pair_cutoff
                )
            if shifts is None:
                shifts = torch.zeros(
                    edge_index.shape[1],
                    3,
                    device=data["pos"].device,
                    dtype=data["pos"].dtype,
                )
        else:
            if flexible_cutoffs is None:
                edge_index, shifts = self._build_edges(data["pos"], cell, pbc)
                edge_cutoff = torch.full(
                    (edge_index.shape[1],),
                    fill_value=self.cutoff,
                    device=data["pos"].device,
                    dtype=data["pos"].dtype,
                )
            else:
                safe_cutoff = float(flexible_cutoffs.max().item())
                edge_index, shifts = self._build_edges(
                    data["pos"],
                    cell,
                    pbc,
                    cutoff=safe_cutoff,
                )
                pair_cutoff = 0.5 * (
                    flexible_cutoffs[edge_index[0]] + flexible_cutoffs[edge_index[1]]
                )
                edge_index, shifts, edge_cutoff = self._apply_cutoff_mask(
                    data["pos"],
                    edge_index,
                    shifts,
                    cutoff=pair_cutoff,
                )

        data["edge_index"] = edge_index
        data["shifts"] = shifts
        data[EDGE_CUTOFF_KEY] = edge_cutoff
        return data
