from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim.lr_scheduler import MultiStepLR

from mad_workflow_common import (
    average_edges_per_atom,
    DEFAULT_OUTPUT_DIR,
    PAPER_PRETRAIN_AVG_NEIGHBORS,
    PAPER_PRETRAIN_CUTOFF,
    DEFAULT_TEST_DATASET,
    DEFAULT_TRAIN_DATASET,
    DEFAULT_VAL_DATASET,
    build_scale_shift_mace_wrapper,
    build_trainer,
    build_training_module,
    ensure_output_dir,
    load_mad_dataset,
    make_split_dataloaders,
    save_json,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1: train a fixed-cutoff MACE foundation model on the MAD dataset."
    )
    parser.add_argument("--train-dataset", type=Path, default=DEFAULT_TRAIN_DATASET)
    parser.add_argument("--val-dataset", type=Path, default=DEFAULT_VAL_DATASET)
    parser.add_argument("--test-dataset", type=Path, default=DEFAULT_TEST_DATASET)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "stage1",
    )
    parser.add_argument("--cutoff", type=float, default=PAPER_PRETRAIN_CUTOFF)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--precision", type=int, choices=(32, 64), default=32)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-8)
    parser.add_argument("--energy-weight", type=float, default=0.1)
    parser.add_argument("--force-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num-bessel", type=int, default=10)
    parser.add_argument("--num-polynomial-cutoff", type=int, default=5)
    parser.add_argument("--max-ell", type=int, default=1)
    parser.add_argument("--num-interactions", type=int, default=2)
    parser.add_argument("--hidden-irreps", default="32x0e")
    parser.add_argument("--mlp-irreps", default="32x0e")
    parser.add_argument("--correlation", type=int, default=1)
    parser.add_argument(
        "--avg-num-neighbors",
        type=float,
        default=PAPER_PRETRAIN_AVG_NEIGHBORS,
    )
    parser.add_argument(
        "--lr-milestones",
        type=int,
        nargs="+",
        default=[10, 20, 100],
    )
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument(
        "--atomic-numbers",
        type=int,
        nargs="+",
        default=[i for i in range(1, 101)],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    seed_everything(args.seed)

    model = build_scale_shift_mace_wrapper(
        cutoff=args.cutoff,
        num_bessel=args.num_bessel,
        num_polynomial_cutoff=args.num_polynomial_cutoff,
        max_ell=args.max_ell,
        num_interactions=args.num_interactions,
        hidden_irreps=args.hidden_irreps,
        mlp_irreps=args.mlp_irreps,
        correlation=args.correlation,
        atomic_numbers=args.atomic_numbers,
        avg_num_neighbors=args.avg_num_neighbors,
    )

    train_dataset = load_mad_dataset(
        args.train_dataset,
        cutoff=args.cutoff,
        precision=args.precision,
    )
    val_dataset = load_mad_dataset(
        args.val_dataset,
        cutoff=args.cutoff,
        precision=args.precision,
    )
    test_dataset = load_mad_dataset(
        args.test_dataset,
        cutoff=args.cutoff,
        precision=args.precision,
    )
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
    scheduler = MultiStepLR(
        optimizer,
        milestones=list(args.lr_milestones),
        gamma=args.lr_gamma,
    )
    lightning_module = build_training_module(
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        energy_weight=args.energy_weight,
        force_weight=args.force_weight,
    )
    trainer = build_trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        precision=args.precision,
        default_root_dir=output_dir / "lightning",
    )
    trainer.fit(
        lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    test_metrics = trainer.test(lightning_module, dataloaders=test_loader)

    foundation_model_path = output_dir / "foundation.model"
    torch.save(lightning_module.model.backbone, foundation_model_path)
    save_json(
        output_dir / "summary.json",
        {
            "foundation_model_path": str(foundation_model_path),
            "cutoff": args.cutoff,
            "max_epochs": args.max_epochs,
            "average_edges_per_atom": {
                "train": average_edges_per_atom(train_dataset),
                "val": average_edges_per_atom(val_dataset),
                "test": average_edges_per_atom(test_dataset),
            },
            "test_metrics": test_metrics,
        },
    )
    print(f"Saved foundation model to {foundation_model_path}")


if __name__ == "__main__":
    main()
