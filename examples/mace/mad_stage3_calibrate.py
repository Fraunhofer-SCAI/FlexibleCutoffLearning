from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from mad_workflow_common import (
    DEFAULT_OUTPUT_DIR,
    build_default_tasks,
    build_trainer,
    ensure_output_dir,
    format_lambda_value,
    load_mad_dataset,
    mad_subset_dataset_path,
    PAPER_CALIBRATION_SUBSETS,
    PAPER_LAMBDA_VALUES,
    PAPER_PRETRAIN_CUTOFF,
    save_json,
    seed_everything,
)
from flexcut import CutoffCalibrationLightningModule, MACEWrapper, split_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 3: calibrate element-wise cutoff radii for an FCL MAD model."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "stage2" / "fcl_wrapper.pt",
    )
    parser.add_argument(
        "--calibration-dataset",
        type=Path,
        default=None,
        help="Run a single calibration on this dataset instead of the full paper subset sweep.",
    )
    parser.add_argument(
        "--evaluation-dataset",
        type=Path,
        default=None,
        help="Optional evaluation dataset for single-run mode. When omitted, the calibration dataset can be split with train/val/test sizes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "stage3",
    )
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--precision", type=int, choices=(32, 64), default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--lambda-cost", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--initial-cutoff", type=float, default=PAPER_PRETRAIN_CUTOFF)
    parser.add_argument("--energy-weight", type=float, default=0.1)
    parser.add_argument("--force-weight", type=float, default=1.0)
    parser.add_argument(
        "--cost-aggregation",
        choices=("per_graph_mean", "per_atom_mean"),
        default="per_graph_mean",
        help="Aggregate the cutoff^3 cost equally per structure or equally per atom.",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=list(PAPER_CALIBRATION_SUBSETS),
        help="Paper subset names for sweep mode.",
    )
    parser.add_argument(
        "--lambda-values",
        type=float,
        nargs="+",
        default=list(PAPER_LAMBDA_VALUES),
        help="Tradeoff values to sweep during calibration.",
    )
    parser.add_argument(
        "--case-name",
        default=None,
        help="Optional name for single-run mode output directories.",
    )
    parser.add_argument("--train-size", type=float, default=1.0)
    parser.add_argument("--val-size", type=float, default=0.0)
    parser.add_argument("--test-size", type=float, default=0.0)
    return parser.parse_args()


def load_model(path: Path):
    loaded = torch.load(path, weights_only=False)
    if isinstance(loaded, MACEWrapper):
        return loaded
    if isinstance(loaded, torch.nn.Module):
        return MACEWrapper(model=loaded)
    raise TypeError(f"Unsupported model artifact at {path}: {type(loaded)}")


def make_loader(dataset, *, batch_size: int, num_workers: int, shuffle: bool):
    drop_last = bool(shuffle and len(dataset) >= batch_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )


def run_calibration_case(
    *,
    model,
    case_name: str,
    calibration_dataset_path: Path,
    evaluation_dataset_path: Path | None,
    output_dir: Path,
    lambda_cost: float,
    args: argparse.Namespace,
) -> dict:
    run_output_dir = ensure_output_dir(output_dir)
    superset_cutoff = max(float(model.cutoff), float(args.initial_cutoff))
    calibration_dataset = load_mad_dataset(
        calibration_dataset_path,
        cutoff=superset_cutoff,
        precision=args.precision,
    )

    evaluation_dataset = None
    if evaluation_dataset_path is None:
        calibration_train, calibration_val, calibration_test = split_dataset(
            calibration_dataset,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            seed=args.seed,
        )
    else:
        calibration_train = calibration_dataset
        calibration_val = None
        calibration_test = None
        evaluation_dataset = load_mad_dataset(
            evaluation_dataset_path,
            cutoff=superset_cutoff,
            precision=args.precision,
        )

    train_loader = make_loader(
        calibration_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )

    val_loader = None
    test_loader = None
    report_loaders = {
        "train": make_loader(
            calibration_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
    }

    if evaluation_dataset is not None:
        test_loader = make_loader(
            evaluation_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
        report_loaders["test"] = test_loader
        primary_split = "test"
    else:
        if calibration_val is not None and len(calibration_val) > 0:
            val_loader = make_loader(
                calibration_val,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
            )
            report_loaders["val"] = val_loader
        if calibration_test is not None and len(calibration_test) > 0:
            test_loader = make_loader(
                calibration_test,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
            )
            report_loaders["test"] = test_loader
        primary_split = (
            "test"
            if test_loader is not None
            else "val" if val_loader is not None else "train"
        )

    calibration_module = CutoffCalibrationLightningModule(
        model=model,
        tasks=build_default_tasks(
            energy_weight=args.energy_weight,
            force_weight=args.force_weight,
        ),
        lambda_cost=lambda_cost,
        initial_cutoffs_by_atomic_number={
            atomic_number: args.initial_cutoff for atomic_number in model.atomic_numbers
        },
        learning_rate=args.learning_rate,
        neighbourlist_implementation="pymatgen",
        cost_aggregation=args.cost_aggregation,
    )
    trainer = build_trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        precision=args.precision,
        default_root_dir=run_output_dir / "lightning",
    )

    before_report = calibration_module.summarize_dataloaders(
        report_loaders,
        primary_split=primary_split,
    )
    fit_kwargs = {"train_dataloaders": train_loader}
    if val_loader is not None:
        fit_kwargs["val_dataloaders"] = val_loader
    trainer.fit(calibration_module, **fit_kwargs)
    if test_loader is not None:
        trainer.test(calibration_module, dataloaders=test_loader, verbose=False)

    report = calibration_module.build_comparison_report(
        before_report,
        primary_split=primary_split,
    )
    report["case_name"] = case_name
    report["calibration_dataset"] = str(calibration_dataset_path)
    report["evaluation_dataset"] = (
        str(evaluation_dataset_path) if evaluation_dataset_path is not None else None
    )
    report["lambda"] = float(lambda_cost)

    summary_path = run_output_dir / "summary.json"
    save_json(summary_path, report)
    return {
        "summary_path": str(summary_path),
        "calibration_dataset": str(calibration_dataset_path),
        "evaluation_dataset": (
            str(evaluation_dataset_path)
            if evaluation_dataset_path is not None
            else None
        ),
        "primary_split": report.get("primary_split"),
        "objective": report.get("objective"),
        "epsilon": report.get("epsilon"),
        "cost": report.get("cost"),
        "average_edges_per_atom": report.get("average_edges_per_atom"),
        "task_metrics": report.get("task_metrics"),
        "cutoffs_by_atomic_number": report.get("cutoffs_by_atomic_number"),
        "split_summaries": report.get("split_summaries"),
        "split_task_metrics": report.get("split_task_metrics"),
    }


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    seed_everything(args.seed)

    model = load_model(args.model_path)
    if args.calibration_dataset is not None:
        case_name = args.case_name or args.calibration_dataset.stem.replace(
            "mad-val-", ""
        ).replace("mad-test-", "")
        run_result = run_calibration_case(
            model=model,
            case_name=case_name,
            calibration_dataset_path=args.calibration_dataset,
            evaluation_dataset_path=args.evaluation_dataset,
            output_dir=output_dir / "run",
            lambda_cost=args.lambda_cost,
            args=args,
        )
        aggregate_summary = {
            "mode": "single",
            "model_path": str(args.model_path),
            "summary_path": run_result["summary_path"],
            "result": run_result,
        }
    else:
        aggregate_runs: dict[str, dict[str, dict]] = {}
        for subset in args.subsets:
            calibration_dataset_path = mad_subset_dataset_path("train", subset)
            evaluation_dataset_path = mad_subset_dataset_path("test", subset)
            subset_runs: dict[str, dict] = {}
            for lambda_value in args.lambda_values:
                lambda_key = format_lambda_value(lambda_value)
                run_result = run_calibration_case(
                    model=model,
                    case_name=subset,
                    calibration_dataset_path=calibration_dataset_path,
                    evaluation_dataset_path=evaluation_dataset_path,
                    output_dir=output_dir / subset / f"lambda_{lambda_key}",
                    lambda_cost=float(lambda_value),
                    args=args,
                )
                subset_runs[lambda_key] = run_result
            aggregate_runs[subset] = subset_runs
        aggregate_summary = {
            "mode": "paper_subset_lambda_sweep",
            "model_path": str(args.model_path),
            "subsets": list(args.subsets),
            "lambda_values": [
                format_lambda_value(value) for value in args.lambda_values
            ],
            "runs": aggregate_runs,
        }

    summary_path = output_dir / "summary.json"
    save_json(summary_path, aggregate_summary)
    print(f"Saved stage 3 summary to {summary_path}")


if __name__ == "__main__":
    main()
