# Processing the MAD dataset to be compatible with the training workflow

## Download the original MAD 1.0 dataset

Download the `mad-(train|val|test).xyz` files from [Materials Cloud](https://archive.materialscloud.org/records/c4ene-0mv14).

## Process the .xyz files

The `process-mad.py` script converts ExtXYZ files into HDF5 format, which is compatible with the training workflow. The script supports two modes of operation:

### Basic Usage

```bash
python process-mad.py --filepath <input.xyz> --output-dir <output_directory> [--split_by_subset <true|false>]
```

### Arguments

- `--filepath`: Path to the input `.xyz` or `.extxyz` file (required)
- `--output-dir`: Directory to write output HDF5 files (default: current directory)
- `--prefix`: Filename prefix for outputs (default: input file stem)
- `--split_by_subset`: Split data by subset → yields a separate file for each subset (default: false)

### Output Format

The script generates HDF5 files with the following structure:

- **Data group**: Contains atomic coordinates (`pos`), atomic numbers (`z`), cell information (`cell`), periodic boundary conditions (`pbc`), energy, forces, stress, and metadata (subset, split)
- **Slices group**: Stores index pointers for each property to enable efficient batching

### Splitting Strategies

The script supports two splitting strategies:

1. **By subset** (`--split_by_subset true`): 
   - Splits data based on the `subset` property in the ExtXYZ files
   - Creates separate HDF5 files for each subset (e.g., `mad-train-MC2D.hdf5`, `mad-train-MC3D.hdf5`)
   - This is the recommended approach for the MAD dataset as it preserves the original dataset splits

2. **Combined** (`--split_by_subset false`):
   - Combines all data into a single HDF5 file
   - Useful for preprocessing or when subset information is not available

### Example: Processing MAD Training Data

```bash
# Process training data, splitting by subset (recommended)
python process-mad.py \
    --filepath mad-train.xyz \
    --output-dir processed/train_split_by_subset \
    --split_by_subset true

# Process validation data
python process-mad.py \
    --filepath mad-val.xyz \
    --output-dir processed/val_split_by_subset \
    --split_by_subset true

# Process test data
python process-mad.py \
    --filepath mad-test.xyz \
    --output-dir processed/test_split_by_subset \
    --split_by_subset true
```

### Loading the Processed Data

The generated HDF5 files can be loaded using `HDF5Dataset`:

```python
from flexcut.data.datasets import HDF5Dataset

dataset = HDF5Dataset.from_hdf5("processed/train_split_by_subset/mad-train-MC2D.hdf5")
``` 