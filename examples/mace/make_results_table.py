from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


DEFAULT_PLOT_FILENAME = "forces_rmse_vs_average_pairs.png"
TABLE_COLUMNS = (
	"subset",
	"lambda",
	"forces_rmse",
	"forces_mae",
	"energy_rmse",
	"energy_mae",
	"energy_rmse_per_atom",
	"energy_mae_per_atom",
	"average_pairs_per_atom",
)


def parse_args() -> argparse.Namespace:
	script_dir = Path(__file__).resolve().parent
	parser = argparse.ArgumentParser(
		description=(
			"Create a stage-3 summary table and a Figure-5-style plot from explicit "
			"force and energy metrics."
		)
	)
	parser.add_argument(
		"--summary-path",
		type=Path,
		default=script_dir / "summary.json",
		help="Path to the aggregated stage 3 summary.json.",
	)
	parser.add_argument(
		"--plot-output",
		type=Path,
		default=script_dir / DEFAULT_PLOT_FILENAME,
		help="Where to save the force-RMSE-vs-pairs plot.",
	)
	parser.add_argument(
		"--csv-output",
		type=Path,
		default=None,
		help="Optional CSV path for the extracted long-format table.",
	)
	parser.add_argument(
		"--title",
		default="Stage 3 Force RMSE vs Average Pairs per Atom",
		help="Plot title.",
	)
	parser.add_argument(
		"--dpi",
		type=int,
		default=300,
		help="Saved plot resolution.",
	)
	parser.add_argument(
		"--no-plot",
		action="store_true",
		help="Print the table without generating the plot.",
	)
	return parser.parse_args()


def load_summary(summary_path: Path) -> dict:
	with summary_path.open("r", encoding="utf-8") as handle:
		summary = json.load(handle)
	if not isinstance(summary, dict):
		raise ValueError(f"Summary at {summary_path} must be a JSON object.")
	runs = summary.get("runs")
	if not isinstance(runs, dict) or not runs:
		raise ValueError(
			f"Summary at {summary_path} does not contain aggregated stage 3 runs."
		)
	return summary


def ordered_subsets(summary: dict) -> list[str]:
	subset_values = summary.get("subsets")
	if isinstance(subset_values, list) and subset_values:
		return [str(value) for value in subset_values]
	return list(summary["runs"].keys())


def ordered_lambdas(summary: dict, runs: dict[str, dict]) -> list[str]:
	lambda_values = summary.get("lambda_values")
	if isinstance(lambda_values, list) and lambda_values:
		return [str(value) for value in lambda_values]

	lambda_keys: set[str] = set()
	for subset_runs in runs.values():
		lambda_keys.update(str(key) for key in subset_runs.keys())
	return sorted(lambda_keys, key=lambda value: float(value))


def require_metric(
	run: dict,
	subset: str,
	lambda_key: str,
	task: str,
	metric: str,
) -> float:
	task_metrics = run.get("task_metrics")
	if not isinstance(task_metrics, dict):
		raise ValueError(
			"Summary is missing explicit stage 3 task metrics. "
			f"Run stage 3 again with the updated logging before using this script "
			f"({subset}, lambda={lambda_key})."
		)
	task_entry = task_metrics.get(task)
	if not isinstance(task_entry, dict) or metric not in task_entry:
		raise ValueError(
			f"Missing task metric '{task}/{metric}' for subset={subset}, lambda={lambda_key}. "
			"Run stage 3 again with the updated logging."
		)
	return float(task_entry[metric])


def require_average_pairs(run: dict, subset: str, lambda_key: str) -> float:
	average_pairs = run.get("average_edges_per_atom")
	if average_pairs is None:
		raise ValueError(
			f"Missing average_edges_per_atom for subset={subset}, lambda={lambda_key}."
		)
	return float(average_pairs)


def extract_rows(summary: dict) -> list[dict[str, float | str]]:
	runs = summary["runs"]
	subsets = ordered_subsets(summary)
	lambda_keys = ordered_lambdas(summary, runs)
	rows: list[dict[str, float | str]] = []

	for subset in subsets:
		subset_runs = runs.get(subset)
		if not isinstance(subset_runs, dict):
			raise ValueError(f"Subset '{subset}' is missing from the aggregated runs.")
		for lambda_key in lambda_keys:
			run = subset_runs.get(lambda_key)
			if not isinstance(run, dict):
				raise ValueError(
					f"Subset '{subset}' does not contain lambda '{lambda_key}'."
				)
			rows.append(
				{
					"subset": subset,
					"lambda": lambda_key,
					"forces_rmse": require_metric(
						run, subset, lambda_key, "forces", "RMSE"
					),
					"forces_mae": require_metric(
						run, subset, lambda_key, "forces", "MAE"
					),
					"energy_rmse": require_metric(
						run, subset, lambda_key, "energy", "RMSE"
					),
					"energy_mae": require_metric(
						run, subset, lambda_key, "energy", "MAE"
					),
					"energy_rmse_per_atom": require_metric(
						run, subset, lambda_key, "energy", "RMSE_per_atom"
					),
					"energy_mae_per_atom": require_metric(
						run, subset, lambda_key, "energy", "MAE_per_atom"
					),
					"average_pairs_per_atom": require_average_pairs(
						run, subset, lambda_key
					),
				}
			)
	return rows


def format_value(value: float | str) -> str:
	if isinstance(value, str):
		return value
	return f"{value:.6g}"


def print_table(rows: Iterable[dict[str, float | str]]) -> None:
	rows = list(rows)
	widths = {
		column: max(
			len(column),
			*(len(format_value(row[column])) for row in rows),
		)
		for column in TABLE_COLUMNS
	}
	header = "  ".join(column.ljust(widths[column]) for column in TABLE_COLUMNS)
	separator = "  ".join("-" * widths[column] for column in TABLE_COLUMNS)
	print(header)
	print(separator)
	for row in rows:
		print(
			"  ".join(
				format_value(row[column]).ljust(widths[column])
				for column in TABLE_COLUMNS
			)
		)


def write_csv(rows: Iterable[dict[str, float | str]], output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS)
		writer.writeheader()
		for row in rows:
			writer.writerow(row)


def make_plot(
	rows: list[dict[str, float | str]],
	output_path: Path,
	title: str,
	dpi: int,
) -> None:
	subsets = []
	for row in rows:
		subset = str(row["subset"])
		if subset not in subsets:
			subsets.append(subset)

	colors = {
		"MC2D": "#0b6e4f",
		"MC3D": "#c84c09",
		"SHIFTML-molcrys": "#28587b",
		"SHIFTML-molfrags": "#7a306c",
	}
	markers = {
		"MC2D": "o",
		"MC3D": "s",
		"SHIFTML-molcrys": "^",
		"SHIFTML-molfrags": "D",
	}

	fig, ax = plt.subplots(figsize=(8.6, 5.6), constrained_layout=True)
	for subset in subsets:
		subset_rows = [row for row in rows if row["subset"] == subset]
		x_values = [float(row["average_pairs_per_atom"]) for row in subset_rows]
		y_values = [float(row["forces_rmse"]) for row in subset_rows]
		ax.plot(
			x_values,
			y_values,
			marker=markers.get(subset, "o"),
			markersize=7,
			linewidth=2,
			color=colors.get(subset),
			label=subset,
		)
		for row, x_value, y_value in zip(subset_rows, x_values, y_values):
			ax.annotate(
				str(row["lambda"]),
				(x_value, y_value),
				textcoords="offset points",
				xytext=(5, 5),
				fontsize=8,
				color=colors.get(subset, "black"),
			)

	ax.set_title(title)
	ax.set_xlabel("Average number of pairs per atom")
	ax.set_ylabel("Forces RMSE")
	ax.grid(True, linewidth=0.6, alpha=0.35)
	ax.legend(frameon=False)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
	plt.close(fig)


def main() -> None:
	args = parse_args()
	summary = load_summary(args.summary_path)
	rows = extract_rows(summary)
	print_table(rows)
	if args.csv_output is not None:
		write_csv(rows, args.csv_output)
		print(f"\nSaved CSV table to {args.csv_output}")
	if not args.no_plot:
		make_plot(rows, args.plot_output, title=args.title, dpi=args.dpi)
		print(f"\nSaved plot to {args.plot_output}")


if __name__ == "__main__":
	main()
