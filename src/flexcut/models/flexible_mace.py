from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor
from torch.nn import Linear, Sequential, SiLU

from ..data.transforms import FLEXIBLE_CUTOFF_KEY

_MACE_IMPORT_ERROR: Exception | None = None
_E3NN_IMPORT_ERROR: Exception | None = None
_RadialEmbeddingBlock = None
_ScaleShiftMACE = None
_extract_config_mace_model = None
_scatter_sum = None
_get_atomic_virials_stresses = None
_get_outputs = None
_prepare_graph = None
_o3 = None

# e3nn<0.6 can trip over torch>=2.6 defaulting torch.load(weights_only=True).
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

try:
    from e3nn import o3 as _o3_import
except Exception as exc:  # pragma: no cover - optional dependency
    _E3NN_IMPORT_ERROR = exc
else:  # pragma: no cover - optional dependency
    _o3 = _o3_import

try:
    from mace.modules.blocks import RadialEmbeddingBlock as _RadialEmbeddingBlockImport
    from mace.modules.models import ScaleShiftMACE as _ScaleShiftMACEImport
    from mace.modules.utils import (
        get_atomic_virials_stresses as _get_atomic_virials_stresses_import,
    )
    from mace.modules.utils import get_outputs as _get_outputs_import
    from mace.modules.utils import prepare_graph as _prepare_graph_import
    from mace.tools.scatter import scatter_sum as _scatter_sum_import
    from mace.tools.scripts_utils import (
        extract_config_mace_model as _extract_config_mace_model_import,
    )
except Exception as exc:  # pragma: no cover - optional dependency
    _MACE_IMPORT_ERROR = exc
else:  # pragma: no cover - optional dependency
    _RadialEmbeddingBlock = _RadialEmbeddingBlockImport
    _ScaleShiftMACE = _ScaleShiftMACEImport
    _get_atomic_virials_stresses = _get_atomic_virials_stresses_import
    _get_outputs = _get_outputs_import
    _prepare_graph = _prepare_graph_import
    _scatter_sum = _scatter_sum_import
    _extract_config_mace_model = _extract_config_mace_model_import


def _require_flexible_mace_dependencies() -> None:
    if _MACE_IMPORT_ERROR is not None:
        raise ImportError(
            "mace is required for flexible MACE support. Install the optional dependency, e.g. `pip install -e '.[mace]'`."
        ) from _MACE_IMPORT_ERROR
    if _E3NN_IMPORT_ERROR is not None:
        raise ImportError(
            "e3nn is required for flexible MACE support. Install the optional dependency, e.g. `pip install -e '.[mace]'`."
        ) from _E3NN_IMPORT_ERROR

class GenericJointEmbedding(torch.nn.Module):
    def __init__(
        self,
        *,
        base_dim: int,
        embedding_specs: Optional[Dict[str, Any]],
        out_dim: Optional[int] = None,
    ):
        super().__init__()
        self.base_dim = base_dim
        self.out_dim = out_dim or base_dim

        items = list(embedding_specs.items()) if embedding_specs is not None else []
        self.names: List[str] = [name for name, _ in items]
        self.per_graph: List[bool] = []
        self.is_categorical: List[bool] = []
        self.offsets: List[int] = []
        self.emb_dims: List[int] = []
        self.embedders = torch.nn.ModuleList()

        for _, spec in items:
            emb_dim = int(spec["emb_dim"])
            feature_type = str(spec["type"])
            self.per_graph.append(str(spec["per"]) == "graph")
            self.is_categorical.append(feature_type == "categorical")
            self.offsets.append(int(spec.get("offset", 0)))
            self.emb_dims.append(emb_dim)

            if feature_type == "categorical":
                self.embedders.append(
                    torch.nn.Embedding(int(spec["num_classes"]), emb_dim)
                )
            elif feature_type == "continuous":
                input_dim = int(spec["in_dim"])
                use_bias = bool(spec.get("use_bias", True))
                self.embedders.append(
                    Sequential(
                        Linear(input_dim, emb_dim, bias=use_bias),
                        SiLU(),
                        Linear(emb_dim, emb_dim, bias=use_bias),
                    )
                )
            else:
                raise ValueError(f"Unknown type {feature_type} for feature embedding.")

        self.project = Sequential(
            Linear(int(sum(self.emb_dims)), int(self.out_dim), bias=False),
            SiLU(),
        )

    def forward(
        self,
        batch: torch.Tensor,
        features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        embeddings: List[torch.Tensor] = []
        for index, embedder in enumerate(self.embedders):
            feature = features[self.names[index]]
            if self.per_graph[index]:
                feature = feature[batch].unsqueeze(-1)
            if self.is_categorical[index]:
                feature = (feature + self.offsets[index]).long().squeeze(-1)
            embeddings.append(embedder(feature))
        return self.project(torch.cat(embeddings, dim=-1))


class FlexibleRadialEmbeddingBlock(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        radial_type: str = "bessel",
        distance_transform: str = "None",
        apply_cutoff: bool = True,
        hidden_dim: int = 32,
    ):
        _require_flexible_mace_dependencies()
        super().__init__()
        self.cutoff_fn = _RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
            distance_transform=distance_transform,
            apply_cutoff=False,
        ).cutoff_fn
        self.base = _RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
            distance_transform=distance_transform,
            apply_cutoff=False,
        )
        self._apply_cutoff = apply_cutoff
        self.radial_post_processing_nn = Sequential(
            Linear(num_bessel + 1, hidden_dim),
            SiLU(),
            Linear(hidden_dim, num_bessel),
        )
        self.r_max = r_max

    @staticmethod
    def _calculate_envelope(
        edge_lengths: Tensor,
        cutoff_per_edge: Tensor,
        p: Tensor,
    ) -> Tensor:
        r_over_r_max = edge_lengths / cutoff_per_edge
        envelope = (
            1.0
            - ((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(r_over_r_max, p)
            + p * (p + 2.0) * torch.pow(r_over_r_max, p + 1)
            - (p * (p + 1.0) / 2.0) * torch.pow(r_over_r_max, p + 2)
        )
        return envelope * (edge_lengths < cutoff_per_edge)

    def forward(
        self,
        edge_lengths: Tensor,
        node_attrs: Tensor,
        edge_index: Tensor,
        atomic_numbers: Tensor,
        cutoff_per_edge: Tensor,
    ):
        if edge_lengths.shape[0] != cutoff_per_edge.shape[0]:
            raise AssertionError(
                "Cutoff radius must be defined per edge."
            )

        envelope = self._calculate_envelope(
            edge_lengths=edge_lengths,
            cutoff_per_edge=cutoff_per_edge,
            p=self.cutoff_fn.p.to(edge_lengths.dtype),
        )
        radial, _ = self.base.forward(
            edge_lengths,
            node_attrs,
            edge_index,
            atomic_numbers,
        )
        radial = self.radial_post_processing_nn(
            torch.cat([radial, cutoff_per_edge], dim=1)
        )
        if not self._apply_cutoff:
            return radial, envelope
        return radial * envelope, None


class CutoffFlexibleScaleShiftMACE(torch.nn.Module):
    def __init__(
        self,
        backbone_model: _ScaleShiftMACE,  # type: ignore[valid-type]
        mixing_rule: str = "arithmetic",
        cutoff_embedding_dim: int = 64,
        radial_hidden_dim: int = 128,
        r_max: Optional[float] = None,
    ):
        super().__init__()
        _require_flexible_mace_dependencies()
        if not isinstance(backbone_model, _ScaleShiftMACE):
            raise AssertionError("backbone_model must be a ScaleShiftMACE instance.")

        model_config = _extract_config_mace_model(backbone_model)
        self.backbone_model = backbone_model
        self.r_max = backbone_model.r_max if r_max is None else r_max
        self.num_interactions = backbone_model.num_interactions
        self.atomic_numbers = backbone_model.atomic_numbers
        self.mixing_rule = mixing_rule
        self.radial_embedding = FlexibleRadialEmbeddingBlock(
            r_max=self.r_max,
            num_bessel=model_config["num_bessel"],
            num_polynomial_cutoff=model_config["num_polynomial_cutoff"],
            radial_type=model_config.get("radial_type", "bessel"),
            distance_transform=model_config["distance_transform"],
            apply_cutoff=(
                backbone_model.apply_cutoff
                if hasattr(backbone_model, "apply_cutoff")
                else True
            ),
            hidden_dim=radial_hidden_dim,
        )

        embedding_size = self.backbone_model.node_embedding.linear.irreps_out.count(
            _o3.Irrep(0, 1)
        )
        self.embedding_names = [FLEXIBLE_CUTOFF_KEY]
        self.joint_embedding = GenericJointEmbedding(
            base_dim=embedding_size,
            embedding_specs={
                FLEXIBLE_CUTOFF_KEY: {
                    "in_dim": 1,
                    "emb_dim": cutoff_embedding_dim,
                    "use_bias": True,
                    "type": "continuous",
                    "per": "atom",
                }
            },
            out_dim=embedding_size,
        )

    def mixing_fn(self, flexible_cutoff_per_node: Tensor, edge_index: Tensor) -> Tensor:
        if self.mixing_rule == "geometric":
            mixed = torch.sqrt(
                flexible_cutoff_per_node[edge_index[0]]
                * flexible_cutoff_per_node[edge_index[1]]
            )
        elif self.mixing_rule == "arithmetic":
            mixed = 0.5 * (
                flexible_cutoff_per_node[edge_index[0]]
                + flexible_cutoff_per_node[edge_index[1]]
            )
        elif self.mixing_rule == "central_atom":
            mixed = flexible_cutoff_per_node[edge_index[0]]
        else:
            raise NotImplementedError(
                f"Unsupported flexible cutoff mixing rule: {self.mixing_rule}."
            )
        return mixed.view(-1, 1)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        compute_edge_forces: bool = False,
        compute_atomic_stresses: bool = False,
        lammps_mliap: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if FLEXIBLE_CUTOFF_KEY not in data:
            raise AssertionError(
                f"{FLEXIBLE_CUTOFF_KEY} is required for flexible MACE forward passes."
            )

        ctx = _prepare_graph(
            data,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            lammps_mliap=lammps_mliap,
        )
        is_lammps = ctx.is_lammps
        num_atoms_arange = ctx.num_atoms_arange.to(torch.int64)
        num_graphs = ctx.num_graphs
        displacement = ctx.displacement
        positions = ctx.positions
        vectors = ctx.vectors
        lengths = ctx.lengths
        cell = ctx.cell
        node_heads = ctx.node_heads.to(torch.int64)
        interaction_kwargs = ctx.interaction_kwargs
        lammps_natoms = interaction_kwargs.lammps_natoms
        lammps_class = interaction_kwargs.lammps_class

        flexible_cutoff_per_node = data[FLEXIBLE_CUTOFF_KEY].view(-1)
        flexible_cutoff_per_edge = self.mixing_fn(
            flexible_cutoff_per_node, data["edge_index"]
        )

        mask = torch.less_equal(lengths, flexible_cutoff_per_edge).squeeze(1)
        data["edge_index"] = data["edge_index"][:, mask]
        data["shifts"] = data["shifts"][mask]
        lengths = lengths[mask]
        vectors = vectors[mask]
        flexible_cutoff_per_edge = flexible_cutoff_per_edge[mask]

        node_e0 = self.backbone_model.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, node_heads
        ]
        e0 = _scatter_sum(
            src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs
        ).to(vectors.dtype)

        node_feats = self.backbone_model.node_embedding(data["node_attrs"])
        edge_attrs = self.backbone_model.spherical_harmonics(vectors)
        edge_feats, cutoff = self.radial_embedding(
            lengths,
            data["node_attrs"],
            data["edge_index"],
            self.backbone_model.atomic_numbers,
            cutoff_per_edge=flexible_cutoff_per_edge,
        )

        if hasattr(self.backbone_model, "pair_repulsion"):
            pair_node_energy = self.backbone_model.pair_repulsion_fn(
                lengths,
                data["node_attrs"],
                data["edge_index"],
                self.backbone_model.atomic_numbers,
            )
            if is_lammps:
                pair_node_energy = pair_node_energy[: lammps_natoms[0]]
        else:
            pair_node_energy = torch.zeros_like(node_e0)

        node_feats = node_feats + self.joint_embedding(
            data["batch"],
            {FLEXIBLE_CUTOFF_KEY: data[FLEXIBLE_CUTOFF_KEY].view(-1, 1)},
        )

        node_es_list = [pair_node_energy]
        node_feats_list: List[torch.Tensor] = []
        for index, (interaction, product) in enumerate(
            zip(self.backbone_model.interactions, self.backbone_model.products)
        ):
            node_attrs_slice = data["node_attrs"]
            if is_lammps and index > 0:
                node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            node_feats, sc = interaction(
                node_attrs=node_attrs_slice,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                cutoff=cutoff,
                first_layer=index == 0,
                lammps_class=lammps_class,
                lammps_natoms=lammps_natoms,
            )
            if is_lammps and index == 0:
                node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=node_attrs_slice,
            )
            node_feats_list.append(node_feats)

        for index, readout in enumerate(self.backbone_model.readouts):
            feat_index = -1 if len(self.backbone_model.readouts) == 1 else index
            node_es_list.append(
                readout(node_feats_list[feat_index], node_heads)[
                    num_atoms_arange, node_heads
                ]
            )

        node_inter_es = torch.sum(torch.stack(node_es_list, dim=0), dim=0)
        node_inter_es = self.backbone_model.scale_shift(node_inter_es, node_heads)
        inter_e = _scatter_sum(node_inter_es, data["batch"], dim=-1, dim_size=num_graphs)
        total_energy = e0 + inter_e
        node_energy = node_e0.clone().double() + node_inter_es.clone().double()

        forces, virials, stress, hessian, edge_forces = _get_outputs(
            energy=inter_e,
            positions=positions,
            displacement=displacement,
            vectors=vectors,
            cell=cell,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
            compute_edge_forces=compute_edge_forces or compute_atomic_stresses,
        )

        atomic_virials: Optional[torch.Tensor] = None
        atomic_stresses: Optional[torch.Tensor] = None
        if compute_atomic_stresses and edge_forces is not None:
            atomic_virials, atomic_stresses = _get_atomic_virials_stresses(
                edge_forces=edge_forces,
                edge_index=data["edge_index"],
                vectors=vectors,
                num_atoms=positions.shape[0],
                batch=data["batch"],
                cell=cell,
            )

        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "interaction_energy": inter_e,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "hessian": hessian,
            "node_feats": torch.cat(node_feats_list, dim=-1),
            "edge_forces": edge_forces,
            "atomic_virials": atomic_virials,
            "atomic_stresses": atomic_stresses,
        }
