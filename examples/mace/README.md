# MACE Flexible Cutoff Workflow

Before starting, make sure to convert the original .xyz files to the .hdf5 format as described in `MAD-data/README.md`. 

## Runnable MAD Workflow

The repository includes a concrete multi-stage training built around the preprocessed MAD HDF5 files in `MAD-data/processed`. 

- `examples/mace/mad_stage1_pretrain.py`
- `examples/mace/mad_stage2_fcl_finetune.py`
- `examples/mace/mad_stage3_calibrate.py`
- `examples/mace/run_mad_multistage_workflow.sh`

The bash script orchestrates the full sequence:

```bash
bash examples/mace/run_mad_multistage_workflow.sh
```

By default it writes artifacts under `examples/mace/runs/mad_multistage/`:

- Stage 1 raw foundation model: `stage1/foundation.model`
- Stage 1 summary: `stage1/summary.json`
- Stage 2 trained flexible wrapper: `stage2/fcl_wrapper.pt`
- Stage 2 summary: `stage2/summary.json`
- Stage 3 sweep summary: `stage3/summary.json`

The MAD data is expected under:

- `MAD-data/processed`

By default, the calibration stage runs the sweep over the four reported subsets using the validation split for optimization and the corresponding test split for evaluation:

- `MC3D`
- `MC2D`
- `SHIFTML-molcrys`
- `SHIFTML-molfrags`

For each subset it evaluates all reported tradeoff values:

- `1e-2`
- `1e-3`
- `1e-4`
- `1e-5`
- `1e-6`

The script activates the repo-local `.venv` when present and exports `PYTHONPATH=src` automatically. You can override the epoch counts and runtime settings through environment variables such as `STAGE1_MAX_EPOCHS`, `STAGE2_MAX_EPOCHS`, `STAGE3_MAX_EPOCHS`, `STAGE1_BATCH_SIZE`, `STAGE2_BATCH_SIZE`, `STAGE3_BATCH_SIZE`, `ACCELERATOR`, and `DEVICES`.

By default, the following hyperparameters are used. 

- Stage 1 uses a fixed 6.0 A cutoff for 200 epochs with batch size 50.
- Stage 2 samples flexible cutoffs in [3.5, 7.0] for 500 epochs with batch size 50 and validates on fixed 4.0, 5.0, and 6.0 A cutoffs.
- Stage 3 calibrates each subset for 10 epochs with batch size 20, learning rate 0.003, 6.0 A initialization, and the full `lambda` sweep from `1e-2` to `1e-6`.


For a single calibration run outside the sweep, pass explicit datasets to Stage 3, for example:

```bash
python examples/mace/mad_stage3_calibrate.py \
    --model-path examples/mace/runs/mad_multistage/stage2/fcl_wrapper.pt \
    --calibration-dataset MAD-data/processed/val_split_by_subset/mad-val-MC2D.hdf5 \
    --evaluation-dataset MAD-data/processed/test_split_by_subset/mad-test-MC2D.hdf5 \
    --lambda-cost 1e-4 \
    --output-dir examples/mace/runs/mad_multistage/stage3_single_mc2d
```

## Plot Diatomic Curves

To inspect how a trained model behaves for different cutoff values on a simple dimer, use:

```bash
PYTHONPATH=src python examples/mace/plot_diatomic_curves.py \
    --model-path examples/mace/runs/mad_multistage/stage2/fcl_wrapper.pt \
    --pair O O \
    --cutoffs 3.5 4.0 5.0 6.0 7.0 \
    --output examples/mace/runs/mad_multistage/diatomic_OO.png \
    --csv-output examples/mace/runs/mad_multistage/diatomic_OO.csv
```

For raw fixed-cutoff `.model` backbones, add `--wrap-flexible-cutoffs` if you want to evaluate cutoffs above the backbone's native `r_max`.
By default the script shifts each curve so that the largest sampled distance is at zero energy; switch this with `--reference-mode none` if you want absolute energies. In the paper, this was not necessary, as the MAD dataset reports normalized energies and no isolated atom energies were added to the model. 

## Plot Validation Force RMSE vs Cutoff

To sweep a single model over globally uniform cutoff radii on the validation split and plot the resulting force RMSE values, run:

```bash
PYTHONPATH=src python examples/mace/plot_val_forces_rmse_by_cutoff.py \
    --model-path examples/mace/runs/mad_multistage/stage2/fcl_wrapper.pt \
    --cutoffs 3.5 4.0 5.0 6.0 7.0 \
    --output examples/mace/runs/mad_multistage/stage2/val_forces_rmse_vs_cutoff.png \
    --csv-output examples/mace/runs/mad_multistage/stage2/val_forces_rmse_vs_cutoff.csv
```

For fixed backbones the script can evaluate smaller or equal cutoffs directly. If you need larger cutoff values than the raw backbone supports, load it with `--wrap-flexible-cutoffs` so the sweep uses the flexible-cutoff wrapper with a uniform per-atom cutoff.

## Stage 1: Baseline MACE Training

Train or finetune a standard fixed-cutoff model first. The resulting artifact for later reuse should be the raw `mace-torch` model object saved as a `.model` file.

A typical Stage 1 flow is:

```python
from flexcut import EnergyTask, ForcesTask, MACEWrapper, MlipLightningModule
from flexcut import load_dataset, make_dataloaders, split_dataset

model = MACEWrapper.load_from_pretrained("foundation.model")
dataset = load_dataset(
    "train.h5",
    cutoff=model.cutoff,
    rename_map={"total_energy_ref": "energy"},
)
trainset, valset, testset = split_dataset(
    dataset,
    train_size=0.8,
    val_size=0.1,
    test_size=0.1,
)
train_loader, val_loader, test_loader = make_dataloaders(
    trainset,
    valset,
    testset,
    batch_size=50,
)
lightning_module = MlipLightningModule(
    model=model,
    optimizer=..., 
    tasks=[
        EnergyTask(loss_fn=torch.nn.L1Loss(), loss_weight=0.1),
        ForcesTask(loss_fn=torch.nn.L1Loss(), loss_weight=1.0),
    ],
)
```

Persist the underlying raw backbone model for Stage 2:

```python
torch.save(model.backbone, "stage1-fixed-cutoff.model")
```

## Stage 2: FCL Finetuning

Reload the raw `.model` artifact through the same wrapper entry point, but enable flexible cutoffs and add a cutoff-sampling transform to the dataset.

```python
from flexcut import MACEWrapper, SampleFlexibleCutoff, load_dataset

model = MACEWrapper.load_from_pretrained(
    "stage1-fixed-cutoff.model",
    flexible_cutoffs=True,
)

dataset = load_dataset(
    "train.h5",
    cutoff=model.cutoff,
    rename_map={"total_energy_ref": "energy"},
    transforms=[
        SampleFlexibleCutoff(
            low=3.5,
            high=7.0,
            homogenity="per_node",
        )
    ],
)
```

The flexible cutoff transform writes per-atom values under `flexible_cutoff_per_node`. The existing `Neighbourhoods` transform then builds a superset neighbor list and filters edges with the arithmetic-mean rule

$$
 m_{ij} = \tfrac{1}{2}(r_i + r_j)
$$

keeping only edges with

$$
 ||x_i - x_j||_2 \le m_{ij}.
$$

The retained per-edge mixed cutoffs are stored as `edge_cutoff` and can be consumed by FCL-capable models.

For the runnable MAD workflow, Stage 2 also saves the full trained flexible wrapper as `fcl_wrapper.pt`, because the learned FCL conditioning layers sit outside the raw backbone artifact. For downstream optimization tasks, use the full trained wrapper .

## Stage 3: Element-Wise Cutoff Calibration

After training, optimize element-wise cutoffs with the dedicated calibration module.

```python
import pytorch_lightning as pl
from flexcut import CutoffCalibrationLightningModule, EnergyTask

calibration_module = CutoffCalibrationLightningModule(
    model=model,
    tasks=[EnergyTask(loss_fn=torch.nn.L1Loss(), loss_weight=0.1)],
    lambda_cost=1e-4,
    initial_cutoffs_by_atomic_number={1: 6.0, 6: 6.0, 8: 6.0},
    learning_rate=3e-3,
    cost_aggregation="per_graph_mean",
)

trainer = pl.Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=10,
    inference_mode=False,
)
trainer.fit(
    calibration_module,
    train_dataloaders=calibration_train_loader,
    val_dataloaders=calibration_val_loader,
)

report = calibration_module.build_report()
print(report)
```

Each stage-3 per-run summary is written to `stage3/<subset>/lambda_<value>/summary.json`, and the top-level `stage3/summary.json` aggregates the sweep. The per-run summary dict contains at least these keys:

- `cutoffs_by_atomic_number`
- `lambda`
- `cost_aggregation`
- `objective`
- `epsilon`
- `cost`
- `average_edges_per_atom`
- `effective_task_weights`
- `split_summaries`

`cost_aggregation="per_graph_mean"` gives each structure equal weight in the cutoff cost, while `cost_aggregation="per_atom_mean"` reproduces an atom-weighted average of `cutoff^3` across the batch.


