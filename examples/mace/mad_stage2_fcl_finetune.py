from __future__ import annotations

import argparse
from pathlib import Path

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset

from mad_workflow_common import (
    average_edges_per_atom,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TEST_DATASET,
    DEFAULT_TRAIN_DATASET,
    DEFAULT_VAL_DATASET,
    PAPER_FCL_AVG_NEIGHBORS,
    PAPER_FCL_CUTOFF_HIGH,
    PAPER_FCL_CUTOFF_LOW,
    build_trainer,
    build_training_module,
    ensure_output_dir,
    load_mad_dataset,
    make_split_dataloaders,
    save_json,
    set_avg_num_neighbors,
    seed_everything,
)
from flexcut import ElementwiseFlexibleCutoff, MACEWrapper, SampleFlexibleCutoff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2: continue MAD training with flexible cutoff learning."
    )
    parser.add_argument("--train-dataset", type=Path, default=DEFAULT_TRAIN_DATASET)
    parser.add_argument("--val-dataset", type=Path, default=DEFAULT_VAL_DATASET)
    parser.add_argument("--test-dataset", type=Path, default=DEFAULT_TEST_DATASET)
    parser.add_argument(
        "--foundation-model-path",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "stage1" / "foundation.model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "stage2",
    )
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--precision", type=int, choices=(32, 64), default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-8)
    parser.add_argument("--energy-weight", type=float, default=0.1)
    parser.add_argument("--force-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--cutoff-low", type=float, default=PAPER_FCL_CUTOFF_LOW)
    parser.add_argument("--cutoff-high", type=float, default=PAPER_FCL_CUTOFF_HIGH)
    parser.add_argument(
        "--avg-num-neighbors",
        type=float,
        default=PAPER_FCL_AVG_NEIGHBORS,
    )
    parser.add_argument(
        "--sampling-homogenity",
        choices=("per_node", "per_system", "per_element", "mixed"),
        default="per_node",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=("uniform", "inverse_cubic"),
        default="uniform",
    )
    parser.add_argument(
        "--validation-cutoffs",
        type=float,
        nargs="+",
        default=[3.0, 4.0, 5.0, 6.0],
    )
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=20)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--early-stopping-patience", type=int, default=50)
    return parser.parse_args()


def build_uniform_cutoff_dataset(
    dataset_path: Path,
    *,
    cutoff: float,
    precision: int,
    atomic_numbers: list[int],
    validation_cutoffs: list[float],
):
    datasets = []
    for value in validation_cutoffs:
        datasets.append(
            load_mad_dataset(
                dataset_path,
                cutoff=cutoff,
                precision=precision,
                transforms=[
                    ElementwiseFlexibleCutoff(
                        {
                            int(atomic_number): float(value)
                            for atomic_number in atomic_numbers
                        }
                    )
                ],
            )
        )
    return ConcatDataset(datasets)


def build_uniform_cutoff_datasets(
    dataset_path: Path,
    *,
    cutoff: float,
    precision: int,
    atomic_numbers: list[int],
    validation_cutoffs: list[float],
) -> dict[float, object]:
    return {
        float(value): load_mad_dataset(
            dataset_path,
            cutoff=cutoff,
            precision=precision,
            transforms=[
                ElementwiseFlexibleCutoff(
                    {
                        int(atomic_number): float(value)
                        for atomic_number in atomic_numbers
                    }
                )
            ],
        )
        for value in validation_cutoffs
    }


def evaluate_metrics_by_cutoff(
    trainer,
    lightning_module,
    datasets_by_cutoff: dict[float, object],
    *,
    batch_size: int,
    num_workers: int,
    subset: str,
) -> dict[str, dict[str, float]]:
    metrics_by_cutoff: dict[str, dict[str, float]] = {}
    for cutoff, dataset in datasets_by_cutoff.items():
        average_edges = average_edges_per_atom(dataset)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        if subset == "val":
            metrics = trainer.validate(
                lightning_module, dataloaders=dataloader, verbose=False
            )
        elif subset == "test":
            metrics = trainer.test(
                lightning_module, dataloaders=dataloader, verbose=False
            )
        else:
            raise ValueError(f"Unsupported subset {subset}.")
        cutoff_metrics = dict(metrics[0]) if metrics else {}
        cutoff_metrics[f"{subset}/average_edges_per_atom"] = average_edges
        metrics_by_cutoff[str(cutoff)] = cutoff_metrics
    return metrics_by_cutoff


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    seed_everything(args.seed)

    model = MACEWrapper.load_from_pretrained(
        str(args.foundation_model_path),
        cutoff_embedding_dim=64,
        radial_hidden_dim=128,
        flexible_cutoffs=True,
        r_max=7.0,
    )
    set_avg_num_neighbors(model, args.avg_num_neighbors)
    superset_cutoff = max(float(model.cutoff), args.cutoff_high)

    transforms = [
        SampleFlexibleCutoff(
            low=args.cutoff_low,
            high=args.cutoff_high,
            homogenity=args.sampling_homogenity,
            mode=args.sampling_mode,
        )
    ]
    train_dataset = load_mad_dataset(
        args.train_dataset,
        cutoff=superset_cutoff,
        precision=args.precision,
        transforms=transforms,
    )
    val_datasets_by_cutoff = build_uniform_cutoff_datasets(
        args.val_dataset,
        cutoff=superset_cutoff,
        precision=args.precision,
        atomic_numbers=[int(number) for number in model.atomic_numbers],
        validation_cutoffs=[float(value) for value in args.validation_cutoffs],
    )
    test_datasets_by_cutoff = build_uniform_cutoff_datasets(
        args.test_dataset,
        cutoff=superset_cutoff,
        precision=args.precision,
        atomic_numbers=[int(number) for number in model.atomic_numbers],
        validation_cutoffs=[float(value) for value in args.validation_cutoffs],
    )
    val_dataset = ConcatDataset(list(val_datasets_by_cutoff.values()))
    test_dataset = ConcatDataset(list(test_datasets_by_cutoff.values()))
    train_loader, val_loader, test_loader = make_split_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    lightning_module = build_training_module(
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        energy_weight=args.energy_weight,
        force_weight=args.force_weight,
    )
    best_checkpoint = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch:03d}",
        auto_insert_metric_name=False,
    )
    trainer = build_trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        precision=args.precision,
        default_root_dir=output_dir / "lightning",
        callbacks=[
            best_checkpoint,
            EarlyStopping(
                monitor="val/loss", mode="min", patience=args.early_stopping_patience
            ),
        ],
    )
    trainer.fit(
        lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    best_checkpoint_epoch = None
    if best_checkpoint.best_model_path:
        checkpoint = torch.load(best_checkpoint.best_model_path, map_location="cpu")
        if checkpoint.get("epoch") is not None:
            best_checkpoint_epoch = int(checkpoint["epoch"])
        lightning_module.load_state_dict(checkpoint["state_dict"])

    validation_metrics_by_cutoff = evaluate_metrics_by_cutoff(
        trainer,
        lightning_module,
        val_datasets_by_cutoff,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset="val",
    )
    test_metrics_by_cutoff = evaluate_metrics_by_cutoff(
        trainer,
        lightning_module,
        test_datasets_by_cutoff,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset="test",
    )

    flexible_wrapper_path = output_dir / "fcl_wrapper.pt"
    torch.save(lightning_module.model, flexible_wrapper_path)
    torch.save(lightning_module.model.backbone, output_dir / "fcl_backbone.model")

    save_json(
        output_dir / "summary.json",
        {
            "flexible_wrapper_path": str(flexible_wrapper_path),
            "foundation_model_path": str(args.foundation_model_path),
            "best_checkpoint_path": best_checkpoint.best_model_path,
            "best_val_loss": (
                float(best_checkpoint.best_model_score)
                if best_checkpoint.best_model_score is not None
                else None
            ),
            "best_val_loss_epoch": best_checkpoint_epoch,
            "cutoff_low": args.cutoff_low,
            "cutoff_high": args.cutoff_high,
            "sampling_homogenity": args.sampling_homogenity,
            "sampling_mode": args.sampling_mode,
            "validation_cutoffs": list(args.validation_cutoffs),
            "max_epochs": args.max_epochs,
            "validation_metrics_by_cutoff": validation_metrics_by_cutoff,
            "test_metrics_by_cutoff": test_metrics_by_cutoff,
        },
    )
    print(f"Saved flexible wrapper to {flexible_wrapper_path}")


if __name__ == "__main__":
    main()
