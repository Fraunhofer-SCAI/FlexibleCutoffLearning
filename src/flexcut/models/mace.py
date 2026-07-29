from __future__ import annotations

import os
from typing import Any, Dict
from warnings import warn

import torch
from torch import Tensor

from .base import WrapperBase
from .flexible_mace import CutoffFlexibleScaleShiftMACE

_MACE_IMPORT_ERROR: Exception | None = None
_E3NN_IMPORT_ERROR: Exception | None = None
_mace = None
_MACEModel: Any = None
_ScaleShiftMACE: Any = None

try:
    import mace as _mace  # type: ignore[no-redef]
    from mace.modules.models import MACE as _MACEModel  # type: ignore[no-redef]
    from mace.modules.models import ScaleShiftMACE as _ScaleShiftMACE  # type: ignore[no-redef]
except Exception as exc:  # pragma: no cover - optional dependency
    _MACE_IMPORT_ERROR = exc


def is_mace_available() -> bool:
    return _MACE_IMPORT_ERROR is None


def _require_mace() -> None:
    if _MACE_IMPORT_ERROR is not None:
        raise ImportError(
            "mace is not installed. Install the optional dependency, e.g. `pip install -e '.[mace]'`."
        ) from _MACE_IMPORT_ERROR


class MACEWrapper(WrapperBase):
    def __init__(self, model: torch.nn.Module):
        _require_mace()
        mace_version = getattr(_mace, "__version__", None)
        if mace_version not in (None,) and not str(mace_version).startswith("0.3."):
            warn(
                f"MACEWrapper was developed against mace 0.3.x, found {mace_version}."
            )
        backbone = self._unwrap_backbone(model)
        if not isinstance(backbone, (_MACEModel, _ScaleShiftMACE)):
            raise AssertionError("Model is not a supported MACE backbone instance.")

        cutoff_value = backbone.r_max.item() if hasattr(backbone.r_max, "item") else backbone.r_max
        num_interactions = (
            backbone.num_interactions.item()
            if hasattr(backbone.num_interactions, "item")
            else backbone.num_interactions
        )
        cutoff = float(cutoff_value)
        depth = int(num_interactions)
        super().__init__(
            model=model,
            cutoff=cutoff,
            depth=depth,
            layer_cutoffs=[cutoff for _ in range(depth)],
            supports_node_energies=True,
            handles_batches=True,
            hyperparameters=None,
        )
        self._pass_dataset_index = False
        self.backbone = backbone
        self.atomic_numbers = [int(number) for number in backbone.atomic_numbers]
        onehot_weights = torch.zeros(
            max(self.atomic_numbers) + 1,
            len(self.atomic_numbers),
        )
        for idx, atomic_number in enumerate(self.atomic_numbers):
            onehot_weights[atomic_number][idx] = 1
        self.onehot = torch.nn.Embedding.from_pretrained(onehot_weights)

    @staticmethod
    def _unwrap_backbone(model: torch.nn.Module) -> torch.nn.Module:
        backbone = getattr(model, "backbone_model", None)
        return model if backbone is None else backbone

    @classmethod
    def load_from_pretrained(
        cls,
        path: str,
        *,
        flexible_cutoffs: bool = False,
        mixing_rule: str = "arithmetic",
        cutoff_embedding_dim: int = 32,
        radial_hidden_dim: int = 64,
        r_max: float | None = None,
    ) -> "MACEWrapper":
        _require_mace()
        model = torch.load(path, weights_only=False)
        if flexible_cutoffs:
            backbone = cls._unwrap_backbone(model)
            model = CutoffFlexibleScaleShiftMACE(
                backbone_model=backbone,
                mixing_rule=mixing_rule,
                cutoff_embedding_dim=cutoff_embedding_dim,
                radial_hidden_dim=radial_hidden_dim,
                r_max=r_max,
            )
        return cls(model=model)

    def with_flexible_cutoffs(
        self,
        *,
        mixing_rule: str = "arithmetic",
        cutoff_embedding_dim: int = 32,
        radial_hidden_dim: int = 64,
        r_max: float | None = None,
    ) -> "MACEWrapper":
        return self.__class__(
            model=CutoffFlexibleScaleShiftMACE(
                backbone_model=self.backbone,
                mixing_rule=mixing_rule,
                cutoff_embedding_dim=cutoff_embedding_dim,
                radial_hidden_dim=radial_hidden_dim,
                r_max=r_max,
            )
        )

    def compile_for_torchscript(self) -> torch.nn.Module:
        global _E3NN_IMPORT_ERROR
        try:
            os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
            import e3nn.util.jit as e3nn_jit
        except Exception as exc:  # pragma: no cover - optional dependency
            _E3NN_IMPORT_ERROR = exc
            raise ImportError(
                "e3nn is required to compile MACE models for TorchScript."
            ) from exc
        return e3nn_jit.compile(self)

    def model_forward(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        coordinates = inputs["coordinates"]
        species = inputs["species"]
        edge_index = inputs["edge_index"]
        shifts = inputs["shifts"]
        batch = inputs["batch"]
        cell = inputs.get("cell")
        dataset_index = inputs.get("dataset_index")

        one_hot = self.onehot(species)
        _, counts = torch.unique(batch, return_counts=True)
        ptr_parts = [torch.tensor([0], dtype=counts.dtype, device=counts.device)]
        for count in counts:
            ptr_parts.append(ptr_parts[-1] + count)
        ptr = torch.cat(ptr_parts)

        head = torch.zeros(len(counts), dtype=torch.long, device=batch.device)
        if dataset_index is not None and self._pass_dataset_index:
            head = dataset_index.to(dtype=torch.long)

        data = {
            "positions": coordinates,
            "edge_index": edge_index,
            "node_attrs": one_hot,
            "shifts": shifts,
            "batch": batch,
            "cell": cell if cell is not None else torch.zeros(len(counts), 3, 3),
            "ptr": ptr,
            "head": head,
        }
        if "flexible_cutoff_per_node" in inputs:
            data["flexible_cutoff_per_node"] = inputs["flexible_cutoff_per_node"]
        if "edge_cutoff" in inputs:
            data["edge_cutoff"] = inputs["edge_cutoff"]

        output = self.model.forward(
            data,
            training=True,
            compute_force=False,
            compute_stress=False,
        )
        energy = output["energy"]
        node_energy = output["node_energy"]
        if energy is None or node_energy is None:
            raise RuntimeError("MACE model did not return energy and node_energy")

        return {
            "energy": energy.flatten().to(coordinates.dtype),
            "node_energies": node_energy.to(coordinates.dtype),
        }
