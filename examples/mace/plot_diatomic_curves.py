from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from ase.data import atomic_numbers, chemical_symbols
import torch
from torch_geometric.data import Data

from flexcut import ElementwiseFlexibleCutoff, MACEWrapper, Neighbourhoods
from flexcut.training.adapter import MlipAdapter

DEFAULT_CUTOFFS = [3.5, 4.0, 5.0, 6.0, 7.0]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "diatomic_curves.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot diatomic energy curves for a trained MACE model at multiple cutoff "
            "values."
        )
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to a saved stage-1 backbone (.model) or stage-2 wrapper (.pt).",
    )
    parser.add_argument(
        "--pair",
        nargs=2,
        metavar=("ELEMENT1", "ELEMENT2"),
        default=None,
        help="Diatomic pair to evaluate, for example `--pair O O` or `--pair C H`.",
    )
    parser.add_argument(
        "--cutoffs",
        type=float,
        nargs="+",
        default=list(DEFAULT_CUTOFFS),
        help="Cutoff values to compare in Angstrom.",
    )
    parser.add_argument(
        "--distance-min",
        type=float,
        default=3.6,
        help="Minimum bond distance in Angstrom.",
    )
    parser.add_argument(
        "--distance-max",
        type=float,
        default=7.5,
        help="Maximum bond distance in Angstrom.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=300,
        help="Number of sampled distances.",
    )
    parser.add_argument(
        "--reference-mode",
        choices=("none", "max-distance", "minimum"),
        default="max-distance",
        help=(
            "Optional energy shift per curve. `max-distance` subtracts the last point, "
            "which is useful for dissociation-style plots."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Target path for the plot image.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional CSV export with raw sampled energies.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title. Defaults to an auto-generated title.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Saved plot resolution.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device to use, for example `cpu`, `cuda`, or `auto`.",
    )
    parser.add_argument(
        "--wrap-flexible-cutoffs",
        action="store_true",
        help=(
            "Wrap a raw `.model` backbone with the flexible-cutoff adapter before "
            "evaluation. Useful when you want to compare cutoffs above the backbone's "
            "native r_max."
        ),
    )
    parser.add_argument(
        "--cutoff-embedding-dim",
        type=int,
        default=64,
        help="Only used together with --wrap-flexible-cutoffs.",
    )
    parser.add_argument(
        "--radial-hidden-dim",
        type=int,
        default=128,
        help="Only used together with --wrap-flexible-cutoffs.",
    )
    return parser.parse_args()


def atomic_number_from_token(token: str) -> int:
    token = token.strip()
    if not token:
        raise ValueError("Empty element token is not allowed.")
    if token.isdigit():
        return int(token)

    normalized = token[0].upper() + token[1:].lower()
    atomic_number = atomic_numbers.get(normalized)
    if atomic_number is None:
        raise ValueError(f"Unknown element symbol: {token}")
    return int(atomic_number)


def format_pair_label(pair: tuple[int, int]) -> str:
    return f"{chemical_symbols[pair[0]]}-{chemical_symbols[pair[1]]}"


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _as_float(value: object) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def model_support_cutoff(model: MACEWrapper) -> float:
    if hasattr(model.model, "r_max"):
        return _as_float(model.model.r_max)
    return float(model.cutoff)


def is_flexible_model(model: MACEWrapper) -> bool:
    return hasattr(model.model, "mixing_rule")


def load_model(
    model_path: Path,
    *,
    wrap_flexible_cutoffs: bool,
    max_cutoff: float,
    cutoff_embedding_dim: int,
    radial_hidden_dim: int,
) -> MACEWrapper:
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    if wrap_flexible_cutoffs:
        return MACEWrapper.load_from_pretrained(
            str(model_path),
            flexible_cutoffs=True,
            cutoff_embedding_dim=cutoff_embedding_dim,
            radial_hidden_dim=radial_hidden_dim,
            r_max=max_cutoff,
        )

    loaded = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(loaded, MACEWrapper):
        return loaded
    if isinstance(loaded, torch.nn.Module):
        return MACEWrapper(model=loaded)
    raise TypeError(
        f"Unsupported model artifact at {model_path}: {type(loaded).__name__}."
    )


def resolve_pair(model: MACEWrapper, pair_tokens: list[str] | None) -> tuple[int, int]:
    if pair_tokens is None:
        if not getattr(model, "atomic_numbers", None):
            raise ValueError("Could not infer a default pair from the loaded model.")
        first_atomic_number = int(model.atomic_numbers[0])
        return first_atomic_number, first_atomic_number

    pair = tuple(atomic_number_from_token(token) for token in pair_tokens)
    supported_atomic_numbers = {int(number) for number in model.atomic_numbers}
    unsupported = sorted(set(pair) - supported_atomic_numbers)
    if unsupported:
        unsupported_symbols = ", ".join(chemical_symbols[number] for number in unsupported)
        raise ValueError(
            "The loaded model does not contain embeddings for: "
            f"{unsupported_symbols}."
        )
    return pair[0], pair[1]


def build_diatomic_data(
    pair: tuple[int, int],
    distance: float,
    *,
    dtype: torch.dtype,
    model_cutoff: float,
    cutoff: float,
    use_flexible_cutoffs: bool,
) -> Data:
    data = Data(
        pos=torch.tensor(
            [[0.0, 0.0, 0.0], [float(distance), 0.0, 0.0]],
            dtype=dtype,
        ),
        z=torch.tensor(pair, dtype=torch.long),
        batch=torch.zeros(2, dtype=torch.long),
    )

    if use_flexible_cutoffs:
        data = ElementwiseFlexibleCutoff(
            {int(pair[0]): float(cutoff), int(pair[1]): float(cutoff)}
        )(data)

    return Neighbourhoods(cutoff=float(model_cutoff))(data)


def predict_energy(
    model: MACEWrapper,
    adapter: MlipAdapter,
    pair: tuple[int, int],
    distance: float,
    *,
    cutoff: float,
    model_cutoff: float,
    device: torch.device,
    dtype: torch.dtype,
    use_flexible_cutoffs: bool,
) -> float:
    data = build_diatomic_data(
        pair,
        distance,
        dtype=dtype,
        model_cutoff=model_cutoff,
        cutoff=cutoff,
        use_flexible_cutoffs=use_flexible_cutoffs,
    ).to(device)
    prediction = model.forward(**adapter(data))
    return float(prediction["energy"].detach().cpu().reshape(-1)[0].item())


def apply_reference(energies: list[float], mode: str) -> list[float]:
    if mode == "none":
        return list(energies)
    if mode == "max-distance":
        reference = energies[-1]
    elif mode == "minimum":
        reference = min(energies)
    else:
        raise ValueError(f"Unsupported reference mode: {mode}")
    return [energy - reference for energy in energies]


def write_csv(
    rows: Iterable[dict[str, float | str]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("pair", "cutoff", "distance", "energy"),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_plot(
    curves: dict[float, list[float]],
    distances: list[float],
    *,
    pair_label: str,
    output_path: Path,
    title: str,
    dpi: int,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
    color_map = plt.get_cmap("viridis")
    cutoff_values = list(curves)
    for index, cutoff in enumerate(cutoff_values):
        color = color_map(index / max(len(cutoff_values) - 1, 1))
        ax.plot(
            distances,
            curves[cutoff],
            linewidth=2.0,
            color=color,
            label=f"cutoff = {cutoff:.2f} A",
        )

    ax.set_title(title)
    ax.set_xlabel(f"Distance {pair_label} / A")
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.6, alpha=0.35)
    ax.legend(frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.distance_min <= 0:
        raise ValueError("--distance-min must be positive.")
    if args.distance_max <= args.distance_min:
        raise ValueError("--distance-max must be larger than --distance-min.")
    if args.num_points < 2:
        raise ValueError("--num-points must be at least 2.")
    if len(args.cutoffs) == 0:
        raise ValueError("Provide at least one cutoff value.")

    device = choose_device(args.device)
    requested_max_cutoff = max(float(value) for value in args.cutoffs)
    model = load_model(
        args.model_path,
        wrap_flexible_cutoffs=args.wrap_flexible_cutoffs,
        max_cutoff=requested_max_cutoff,
        cutoff_embedding_dim=args.cutoff_embedding_dim,
        radial_hidden_dim=args.radial_hidden_dim,
    )
    model = model.to(device)
    model.eval()
    model.compute_forces = False
    model.compute_stress = False

    pair = resolve_pair(model, args.pair)
    pair_label = format_pair_label(pair)
    use_flexible_cutoffs = is_flexible_model(model)
    supported_cutoff = model_support_cutoff(model)
    if requested_max_cutoff > supported_cutoff + 1e-8 and not use_flexible_cutoffs:
        raise ValueError(
            "Requested cutoffs exceed the model support cutoff. Either lower "
            "--cutoffs or pass --wrap-flexible-cutoffs when loading a raw backbone."
        )

    adapter = MlipAdapter()
    dtype = next(model.parameters()).dtype
    distances_tensor = torch.linspace(
        args.distance_min,
        args.distance_max,
        args.num_points,
        dtype=torch.float64,
    )
    distances = [float(value) for value in distances_tensor.tolist()]

    curves: dict[float, list[float]] = {}
    rows: list[dict[str, float | str]] = []
    with torch.no_grad():
        for cutoff in [float(value) for value in args.cutoffs]:
            energies = [
                predict_energy(
                    model,
                    adapter,
                    pair,
                    distance,
                    cutoff=cutoff,
                    model_cutoff=max(supported_cutoff, cutoff),
                    device=device,
                    dtype=dtype,
                    use_flexible_cutoffs=use_flexible_cutoffs,
                )
                for distance in distances
            ]
            referenced_energies = apply_reference(energies, args.reference_mode)
            curves[cutoff] = referenced_energies
            rows.extend(
                {
                    "pair": pair_label,
                    "cutoff": cutoff,
                    "distance": distance,
                    "energy": energy,
                }
                for distance, energy in zip(distances, referenced_energies)
            )

    title = args.title or f"Diatomic curve for {pair_label}"
    ylabel = (
        "Energy / eV"
        if args.reference_mode == "none"
        else "Relative energy / eV"
    )
    make_plot(
        curves,
        distances,
        pair_label=pair_label,
        output_path=args.output,
        title=title,
        dpi=args.dpi,
        ylabel=ylabel,
    )
    print(f"Saved plot to {args.output}")

    if args.csv_output is not None:
        write_csv(rows, args.csv_output)
        print(f"Saved CSV to {args.csv_output}")


if __name__ == "__main__":
    main()