from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch

from flexcut import EnergyTask, ForcesTask, MACEWrapper, MlipLightningModule
from flexcut import load_dataset, make_dataloaders, split_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finetune a pretrained MACE model.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--cutoff", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--precision", type=int, choices=(32, 64), default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-size", type=float, default=0.8)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--accelerator", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    model = MACEWrapper.load_from_pretrained(str(args.model_path))
    cutoff = args.cutoff if args.cutoff is not None else model.cutoff

    dataset = load_dataset(
        args.dataset,
        cutoff=cutoff,
        precision=args.precision,
        rename_map={"total_energy_ref": "energy"},
    )
    trainset, valset, testset = split_dataset(
        dataset,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    train_loader, val_loader, test_loader = make_dataloaders(
        trainset,
        valset,
        testset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    optimizer = torch.optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=args.lr,
    )
    lightning_module = MlipLightningModule(
        model=model,
        optimizer=optimizer,
        tasks=[
            EnergyTask(loss_fn=torch.nn.SmoothL1Loss()),
            ForcesTask(loss_fn=torch.nn.L1Loss(), loss_weight=10.0),
        ],
    )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=1,
        max_epochs=args.max_epochs,
        precision=args.precision,
        inference_mode=False,
        default_root_dir=str(Path.cwd() / "runs" / "flexcut" / "mace"),
    )
    trainer.fit(
        lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    trainer.test(lightning_module, dataloaders=test_loader)


if __name__ == "__main__":
    main()
