from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader

from mad_workflow_common import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VAL_DATASET,
    average_edges_per_atom,
    ensure_output_dir,
    load_mad_dataset,
    set_avg_num_neighbors,
)
from flexcut import ElementwiseFlexibleCutoff, MACEWrapper
from flexcut.training.adapter import MlipAdapter

DEFAULT_CUTOFFS = [3.0, 4.0, 5.0, 6.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot validation-set force RMSE for different global cutoff radii."
        )
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to a stage-1 backbone (.model) or stage-2 wrapper (.pt).",
    )
    parser.add_argument(
        "--val-dataset",
        type=Path,
        default=DEFAULT_VAL_DATASET,
        help="Validation dataset used for the cutoff sweep.",
    )
    parser.add_argument(
        "--cutoffs",
        type=float,
        nargs="+",
        default=list(DEFAULT_CUTOFFS),
        help="Global cutoff radii to evaluate in Angstrom.",
    )
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--precision", type=int, choices=(32, 64), default=32)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device to use, for example `cpu`, `cuda`, or `auto`.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "stage2" / "val_forces_rmse_vs_cutoff.png",
        help="Where to save the generated plot.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional CSV export with cutoff, RMSE, and average edges per atom.",
    )
    parser.add_argument(
        "--title",
        default="Validation Forces RMSE vs Cutoff Radius",
        help="Plot title.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--avg-num-neighbors",
        type=float,
        default=None,
        help="Optional override for the model's avg_num_neighbors setting.",
    )
    parser.add_argument(
        "--wrap-flexible-cutoffs",
        action="store_true",
        help=(
            "Wrap a raw `.model` backbone with the flexible-cutoff adapter before "
            "evaluation. Use this when you need global cutoffs above the backbone's "
            "native support radius."
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


def build_uniform_cutoff_dataset(
    dataset_path: Path,
    *,
    model: MACEWrapper,
    cutoff: float,
    precision: int,
) -> object:
    supported_cutoff = model_support_cutoff(model)
    use_flexible_cutoffs = is_flexible_model(model)

    if use_flexible_cutoffs:
        return load_mad_dataset(
            dataset_path,
            cutoff=max(supported_cutoff, cutoff),
            precision=precision,
            transforms=[
                ElementwiseFlexibleCutoff(
                    {
                        int(atomic_number): float(cutoff)
                        for atomic_number in model.atomic_numbers
                    }
                )
            ],
        )

    if cutoff > supported_cutoff + 1e-8:
        raise ValueError(
            "Requested cutoff exceeds the fixed model support radius. Either lower "
            "--cutoffs or pass --wrap-flexible-cutoffs when loading a raw backbone."
        )
    return load_mad_dataset(
        dataset_path,
        cutoff=cutoff,
        precision=precision,
    )


def evaluate_force_rmse(
    model: MACEWrapper,
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> float:
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    adapter = MlipAdapter()
    squared_error_sum = 0.0
    element_count = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            prediction = model.forward(**adapter(batch))
            target = batch["forces"]
            squared_error = torch.square(prediction["forces"] - target)
            squared_error_sum += float(squared_error.sum().detach().cpu().item())
            element_count += int(target.numel())

    if element_count == 0:
        raise ValueError("Validation dataset did not contain any force targets.")
    return math.sqrt(squared_error_sum / element_count)


def write_csv(rows: list[dict[str, float]], output_path: Path) -> None:
    ensure_output_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("cutoff", "forces_rmse", "average_edges_per_atom"),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_plot(
    rows: list[dict[str, float]],
    output_path: Path,
    *,
    title: str,
    dpi: int,
) -> None:
    x_values = [row["cutoff"] for row in rows]
    y_values = [row["forces_rmse"] for row in rows]

    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    ax.plot(
        x_values,
        y_values,
        marker="o",
        markersize=7,
        linewidth=2,
        color="#0b6e4f",
    )
    ax.set_title(title)
    ax.set_xlabel("Global cutoff radius / A")
    ax.set_ylabel("Validation forces RMSE")
    ax.grid(True, linewidth=0.6, alpha=0.35)

    ensure_output_dir(output_path.parent)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if len(args.cutoffs) == 0:
        raise ValueError("Provide at least one cutoff value.")

    cutoff_values = sorted(float(value) for value in args.cutoffs)
    device = choose_device(args.device)
    model = load_model(
        args.model_path,
        wrap_flexible_cutoffs=args.wrap_flexible_cutoffs,
        max_cutoff=max(cutoff_values),
        cutoff_embedding_dim=args.cutoff_embedding_dim,
        radial_hidden_dim=args.radial_hidden_dim,
    )
    if args.avg_num_neighbors is not None:
        set_avg_num_neighbors(model, args.avg_num_neighbors)

    model = model.to(device)
    model.eval()
    model.compute_forces = True
    model.compute_stress = False

    rows: list[dict[str, float]] = []
    for cutoff in cutoff_values:
        dataset = build_uniform_cutoff_dataset(
            args.val_dataset,
            model=model,
            cutoff=cutoff,
            precision=args.precision,
        )
        rows.append(
            {
                "cutoff": cutoff,
                "forces_rmse": evaluate_force_rmse(
                    model,
                    dataset,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    device=device,
                ),
                "average_edges_per_atom": average_edges_per_atom(dataset),
            }
        )

    make_plot(rows, args.output, title=args.title, dpi=args.dpi)
    print(f"Saved plot to {args.output}")
    if args.csv_output is not None:
        write_csv(rows, args.csv_output)
        print(f"Saved CSV to {args.csv_output}")


if __name__ == "__main__":
    main()