from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from warnings import warn

import torch
from torch import Tensor

from ..ops.scatter import scatter_add


@torch.jit.script
def gradient(
    y: torch.Tensor, x: List[torch.Tensor], create_graph: bool, retain_graph: bool
) -> List[torch.Tensor]:
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(y)]
    grads = torch.autograd.grad(
        [y],
        x,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    new_grads = []
    for grad in grads:
        if grad is None:
            raise RuntimeError("grad was None")
        new_grads.append(grad)
    return new_grads


class StressAndForcesOutput(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        energy: torch.Tensor,
        coordinates: torch.Tensor,
        cell: Optional[torch.Tensor],
        deformation: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if deformation is not None and cell is not None:
            grads = gradient(
                energy,
                [coordinates, deformation],
                create_graph=self.training,
                retain_graph=self.training,
            )
            spat = torch.cross(cell[:, 0, :], cell[:, 1, :], dim=1) * cell[:, 2, :]
            volume = torch.sum(spat, dim=1).abs().unsqueeze(1).unsqueeze(1)
            stress = torch.div(grads[1], volume)
        else:
            grads = gradient(
                energy,
                [coordinates],
                create_graph=self.training,
                retain_graph=self.training,
            )
            stress = None
        forces = (-1) * grads[0]
        return forces, stress


class WrapperBase(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        cutoff: float,
        depth: int,
        layer_cutoffs: Optional[List[float]] = None,
        supports_node_energies: bool = False,
        handles_batches: bool = False,
        hyperparameters: Optional[Dict] = None,
    ):
        super().__init__()
        self._model = model
        self.cutoff = cutoff
        self.depth = depth
        if layer_cutoffs is not None:
            if max(layer_cutoffs) > cutoff:
                raise AssertionError(
                    "The cutoffs specified for each layer exceed the global cutoff."
                )
            if len(layer_cutoffs) != depth:
                raise AssertionError(
                    "Length of layer_cutoffs does not match the specified depth."
                )
        self.layer_cutoffs = layer_cutoffs
        self.supports_node_energies = supports_node_energies
        self.compute_forces = True
        self.compute_stress = True
        self.output_module = StressAndForcesOutput()
        self._hyperparameters = hyperparameters
        self.handles_batches = handles_batches

    @property
    def model(self):
        return self._model

    @model.setter
    @torch.jit.unused
    def model(self, model: Optional[torch.nn.Module] = None):
        if model is not None and not isinstance(model, torch.nn.Module):
            raise AssertionError("The passed model must be a torch.nn.Module instance.")
        if model is not None:
            self._model = model

    @property
    @torch.jit.unused
    def hyperparameters(self) -> Optional[Dict]:
        return self._hyperparameters

    def model_forward(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        raise NotImplementedError

    def forward(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        coordinates = inputs["coordinates"]
        species = inputs["species"]
        edge_index = inputs["edge_index"]
        batch = inputs["batch"]

        shifts = torch.jit.annotate(Optional[Tensor], None)
        cell = torch.jit.annotate(Optional[Tensor], None)
        atom_mask = torch.jit.annotate(Optional[Tensor], None)
        dataset_index = torch.jit.annotate(Optional[Tensor], None)

        if "shifts" in inputs:
            shifts = inputs["shifts"]
        if "cell" in inputs:
            cell = inputs["cell"]
        if "atom_mask" in inputs:
            atom_mask = inputs["atom_mask"]
        if "dataset_index" in inputs:
            dataset_index = inputs["dataset_index"]

        batch_size = int(torch.unique(batch).numel())
        if not self.handles_batches and batch_size > 1:
            raise AssertionError(
                "A non-trivial batch tensor was passed to a model that cannot handle batches."
            )

        torch.set_grad_enabled(True)
        coordinates.requires_grad_()

        if shifts is None:
            edge_shifts = torch.zeros(
                edge_index.shape[1],
                3,
                device=coordinates.device,
                dtype=coordinates.dtype,
            )
        else:
            edge_shifts = shifts

        if self.compute_stress:
            if cell is None:
                deformation = None
                warn(
                    "Stress computation was requested but no cell was provided. Stress will not be computed."
                )
            else:
                deformation = (
                    torch.eye(3, dtype=coordinates.dtype, device=coordinates.device)
                    .repeat((cell.shape[0], 1))
                    .reshape(-1, 3, 3)
                )
                deformation.requires_grad_()
                coord_basis = torch.nn.functional.embedding(
                    batch,
                    weight=torch.eye(
                        batch.max().long().item() + 1,
                        dtype=coordinates.dtype,
                        device=coordinates.device,
                    ),
                )
                coordinates = torch.einsum(
                    "bij,qj,qb->qi", deformation, coordinates, coord_basis
                )
                edge_batch = batch[edge_index[0]]
                if edge_shifts.shape[0] > 0:
                    edge_basis = torch.nn.functional.embedding(
                        edge_batch,
                        weight=torch.eye(
                            batch.max().item() + 1,
                            dtype=coordinates.dtype,
                            device=coordinates.device,
                        ),
                    )
                    edge_shifts = torch.einsum(
                        "bij,qj,qb->qi", deformation, edge_shifts, edge_basis
                    )
                cell = torch.bmm(cell, deformation)
        else:
            deformation = None

        model_inputs: Dict[str, Tensor] = {}
        for key, value in inputs.items():
            if isinstance(value, Tensor):
                model_inputs[key] = value

        model_inputs["coordinates"] = coordinates
        model_inputs["species"] = species
        model_inputs["edge_index"] = edge_index
        model_inputs["batch"] = batch
        model_inputs["shifts"] = edge_shifts
        if cell is not None:
            model_inputs["cell"] = cell
        if dataset_index is not None:
            model_inputs["dataset_index"] = dataset_index

        model_output = self.model_forward(model_inputs)

        if atom_mask is not None:
            if not self.supports_node_energies:
                raise AssertionError(
                    "Model does not return atom-wise energies and does not support masking."
                )
            atom_mask = atom_mask.to(coordinates.dtype)
            total_energy_for_grad = scatter_add(
                x=model_output["node_energies"], idx_i=batch, dim_size=batch_size
            )
            masked_node_energies = atom_mask * model_output["node_energies"]
            model_output["node_energies"] = masked_node_energies
            energy = scatter_add(
                x=masked_node_energies, idx_i=batch, dim_size=batch_size
            )
        else:
            energy = model_output["energy"]
            total_energy_for_grad = model_output["energy"]

        output_forces = model_output.get("forces")
        output_stress = model_output.get("stress")

        compute_forces = output_forces is None and self.compute_forces
        compute_stress = (
            output_stress is None and self.compute_stress and cell is not None
        )

        if compute_forces or compute_stress:
            total_energy_for_grad = total_energy_for_grad + 0.0 * coordinates[0][0]
            forces, stress = self.output_module(
                energy=total_energy_for_grad,
                coordinates=coordinates,
                cell=cell,
                deformation=deformation,
            )
        else:
            forces = output_forces
            stress = output_stress

        output_dict = {"energy": energy}
        if isinstance(model_output.get("node_energies"), Tensor):
            output_dict["node_energies"] = model_output["node_energies"]
        if isinstance(forces, Tensor):
            output_dict["forces"] = forces
        if isinstance(stress, Tensor):
            output_dict["stress"] = stress

        for key, value in model_output.items():
            if key not in output_dict and isinstance(value, Tensor):
                output_dict[key] = value

        return output_dict
