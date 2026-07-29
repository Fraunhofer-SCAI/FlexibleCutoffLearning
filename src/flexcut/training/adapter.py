from __future__ import annotations

from typing import Any, Dict, Union

from torch_geometric.data import Data

from ..utils import data_keys


class Adapter:
    def __call__(self, data: Union[Dict[str, Any], Data]) -> Dict[str, Any]:
        raise NotImplementedError


class MlipAdapter(Adapter):
    """Map PyG batches to the minimal MLIP model input contract."""

    def __call__(self, data: Union[Dict[str, Any], Data]) -> Dict[str, Any]:
        keys = data_keys(data)
        inputs: Dict[str, Any] = {
            "coordinates": data["pos"],
            "species": data["z"].long(),
            "edge_index": data["edge_index"],
            "batch": data["batch"],
        }

        if "shifts" in keys:
            inputs["shifts"] = data["shifts"]
        if "cell" in keys:
            inputs["cell"] = data["cell"]

        for key, value in data.items():
            if key not in inputs:
                inputs[key] = value

        return {"inputs": inputs}
