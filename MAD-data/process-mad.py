import argparse
from pathlib import Path
from ase.stress import voigt_6_to_full_3x3_stress

import torch
from torch_geometric.data import Data
from tqdm import tqdm
from flexcut.data.datasets import HDF5Dataset


class ExtXYZConverter:

    def __init__(self):
        super(ExtXYZConverter, self).__init__()

    def get_data_list(self, filepath: str, padding: bool = False):
        from ase.io import read

        dataset = read(filepath, index=":")
        data_list = []
        for data in tqdm(dataset, desc=f"Loading data from {filepath}"):
            pos = torch.from_numpy(data.positions).double()
            z = torch.from_numpy(data.numbers).long()
            pbc = torch.from_numpy(data.pbc).bool() if hasattr(data, "pbc") else None

            if hasattr(data, "cell"):
                cell = torch.from_numpy(data.cell.array).double().view(1, 3, 3)
            elif padding:
                cell = torch.zeros(1, 3, 3, dtype=torch.double)
            else:
                cell = None

            if cell is not None and torch.allclose(
                cell, torch.zeros_like(cell), atol=1e-10, rtol=1e-6
            ):
                cell = torch.eye(3, dtype=torch.double).view(1, 3, 3) * 1e4

            properties = data.info
            _energy = data.get_potential_energy()  # in eV
            _forces = data.get_forces()  # in eV/Å
            _stress = data.get_stress() if torch.all(pbc) else None  # in eV/Å³

            energy = torch.Tensor([_energy]).double()
            forces = torch.from_numpy(_forces).double() if _forces is not None else None

            if _stress is not None:
                stress = (
                    torch.from_numpy(voigt_6_to_full_3x3_stress(_stress))
                    .view(1, 3, 3)
                    .double()
                )
            elif padding:
                stress = torch.zeros(1, 3, 3, dtype=torch.double)
            else:
                stress = None

            data = Data(
                pos=pos,
                z=z,
                cell=cell,
                pbc=pbc,
                energy=energy,
                stress=stress,
                forces=forces,
                subset=properties.get("subset", ""),
                split=properties.get("split", ""),
            )

            data_list.append(data)
        return data_list


def process_xyz_file(
    xyz_path: Path,
    output_dir: Path,
    prefix: str | None = None,
    split_by_subset: bool = True,
) -> list[Path]:
    converter = ExtXYZConverter()

    data_list = converter.get_data_list(
        filepath=str(xyz_path), padding=not split_by_subset
    )
    if not data_list:
        raise ValueError(f"No data entries found in {xyz_path}")

    # split = data_list[0].get("split") or "split"

    # Group entries by subset
    grouped: dict[str, list[dict]] = {}
    if split_by_subset:
        for d in data_list:
            key = d.get("subset")
            if key is None:
                key = "subset"
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(d)
    else:
        grouped["all"] = data_list

    # Determine output filename prefix
    base_prefix = prefix if prefix else xyz_path.stem

    output_paths: list[Path] = []
    for key, items in grouped.items():
        data, slices = HDF5Dataset.collate(items)
        subset = HDF5Dataset(data=data, slices=slices)
        out_name = f"{base_prefix}-{key}.hdf5"
        out_path = output_dir / out_name
        subset.to_hdf5(filepath=str(out_path))
        output_paths.append(out_path)

    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an ExtXYZ file into HDF5 files grouped by subset."
    )
    parser.add_argument(
        "--filepath",
        help="Path to the input .xyz or .extxyz file",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to write output HDF5 files",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--prefix",
        help="Filename prefix for outputs (default: input file stem)",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--split_by_subset",
        help="Split data by subset -> yields a separate file for each subset",
        type=lambda x: x.lower() == "true",
        default=False,
    )

    args = parser.parse_args()

    xyz_path: Path = args.filepath
    output_dir: Path = args.output_dir
    prefix: str | None = args.prefix
    split_by_subset: bool = args.split_by_subset

    if not xyz_path.exists():
        raise FileNotFoundError(f"Input file not found: {xyz_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = process_xyz_file(
        xyz_path=xyz_path,
        output_dir=output_dir,
        prefix=prefix,
        split_by_subset=split_by_subset,
    )
    for p in outputs:
        print(f"Saved {p}")


if __name__ == "__main__":
    main()
