#!/usr/bin/env python3
"""Read-only structural preflight for the Parse-10M PBMC H5AD.

The script reads categorical observation codes in chunks and never loads the
9.7M x 2,000 expression matrix into memory. It does not compute benchmark
metrics or choose a final experimental split.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np


REQUIRED_OBS = ("donor", "cell_type", "cytokine", "treatment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--min_cells", type=int, default=50)
    parser.add_argument("--chunk_size", type=int, default=1_000_000)
    parser.add_argument("--sample_nnz", type=int, default=100_000)
    return parser.parse_args()


def decode_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def decode_strings(values: np.ndarray) -> list[str]:
    return [str(decode_scalar(value)) for value in values.tolist()]


def categorical(group: h5py.Group) -> tuple[list[str], h5py.Dataset]:
    if "categories" not in group or "codes" not in group:
        raise ValueError(f"Expected categorical group at {group.name}")
    return decode_strings(group["categories"][:]), group["codes"]


def dataset_shape(obj: h5py.Dataset | h5py.Group) -> list[int] | None:
    if isinstance(obj, h5py.Dataset):
        return [int(x) for x in obj.shape]
    if "shape" in obj.attrs:
        return [int(x) for x in obj.attrs["shape"]]
    return None


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return decode_scalar(value)


def write_condition_counts(
    path: Path,
    cube: np.ndarray,
    donors: list[str],
    cell_types: list[str],
    cytokines: list[str],
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("donor", "cell_type", "cytokine", "n_cells"))
        for donor_idx, donor in enumerate(donors):
            for cell_idx, cell_type in enumerate(cell_types):
                for cytokine_idx, cytokine in enumerate(cytokines):
                    writer.writerow(
                        (
                            donor,
                            cell_type,
                            cytokine,
                            int(cube[donor_idx, cell_idx, cytokine_idx]),
                        )
                    )


def write_cell_type_eligibility(
    path: Path,
    cube: np.ndarray,
    donors: list[str],
    cell_types: list[str],
    cytokines: list[str],
    pbs_idx: int,
    min_cells: int,
) -> list[dict[str, Any]]:
    perturbation_idx = np.array(
        [idx for idx in range(len(cytokines)) if idx != pbs_idx], dtype=np.int64
    )
    rows: list[dict[str, Any]] = []
    for cell_idx, cell_type in enumerate(cell_types):
        pbs_counts = cube[:, cell_idx, pbs_idx]
        perturbation_counts = cube[:, cell_idx, :][:, perturbation_idx]
        positive = perturbation_counts[perturbation_counts > 0]
        row = {
            "cell_type": cell_type,
            "total_cells": int(cube[:, cell_idx, :].sum()),
            "pbs_min_across_donors": int(pbs_counts.min()),
            "pbs_median_across_donors": float(np.median(pbs_counts)),
            "pbs_max_across_donors": int(pbs_counts.max()),
            "all_donors_pbs_ge_min": bool(np.all(pbs_counts >= min_cells)),
            "cytokines_present_in_all_donors": int(
                np.sum(np.all(perturbation_counts > 0, axis=0))
            ),
            "cytokines_all_donors_ge_min": int(
                np.sum(np.all(perturbation_counts >= min_cells, axis=0))
            ),
            "positive_condition_median_cells": (
                float(np.median(positive)) if positive.size else 0.0
            ),
        }
        rows.append(row)

    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_target_donor_eligibility(
    path: Path,
    cube: np.ndarray,
    donors: list[str],
    cell_types: list[str],
    cytokines: list[str],
    pbs_idx: int,
    min_cells: int,
) -> list[dict[str, Any]]:
    perturbation_idx = np.array(
        [idx for idx in range(len(cytokines)) if idx != pbs_idx], dtype=np.int64
    )
    rows: list[dict[str, Any]] = []
    for donor_idx, donor in enumerate(donors):
        for cell_idx, cell_type in enumerate(cell_types):
            target = cube[donor_idx, cell_idx, perturbation_idx]
            source_pool = (
                cube[:, cell_idx, :][:, perturbation_idx].sum(axis=0) - target
            )
            both = (target >= min_cells) & (source_pool >= min_cells)
            rows.append(
                {
                    "target_donor": donor,
                    "cell_type": cell_type,
                    "target_pbs_cells": int(cube[donor_idx, cell_idx, pbs_idx]),
                    "target_pbs_ge_min": bool(
                        cube[donor_idx, cell_idx, pbs_idx] >= min_cells
                    ),
                    "cytokines_target_ge_min": int(np.sum(target >= min_cells)),
                    "cytokines_source_pool_ge_min": int(
                        np.sum(source_pool >= min_cells)
                    ),
                    "cytokines_target_and_source_ge_min": int(np.sum(both)),
                }
            )

    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    args = parse_args()
    if args.min_cells <= 0 or args.chunk_size <= 0 or args.sample_nnz <= 0:
        raise ValueError("min_cells, chunk_size, and sample_nnz must be positive")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input, "r") as handle:
        missing_top = [key for key in ("X", "obs", "var") if key not in handle]
        if missing_top:
            raise ValueError(f"Missing top-level H5AD groups: {missing_top}")
        missing_obs = [key for key in REQUIRED_OBS if key not in handle["obs"]]
        if missing_obs:
            raise ValueError(f"Missing required obs columns: {missing_obs}")

        x = handle["X"]
        x_shape = dataset_shape(x)
        if x_shape is None or len(x_shape) != 2:
            raise ValueError("Could not determine X shape")
        n_obs, n_vars = x_shape
        if "data" not in x:
            raise ValueError("Expected sparse X/data dataset")
        nnz = int(x["data"].shape[0])

        category_names: dict[str, list[str]] = {}
        code_datasets: dict[str, h5py.Dataset] = {}
        for key in REQUIRED_OBS:
            names, codes = categorical(handle["obs"][key])
            if codes.shape[0] != n_obs:
                raise ValueError(
                    f"obs/{key}/codes has {codes.shape[0]} rows, expected {n_obs}"
                )
            category_names[key] = names
            code_datasets[key] = codes

        donors = category_names["donor"]
        cell_types = category_names["cell_type"]
        cytokines = category_names["cytokine"]
        treatments = category_names["treatment"]
        if "PBS" not in cytokines or "PBS" not in treatments:
            raise ValueError("PBS is missing from cytokine or treatment categories")
        pbs_cytokine_idx = cytokines.index("PBS")
        pbs_treatment_idx = treatments.index("PBS")

        cube = np.zeros(
            (len(donors), len(cell_types), len(cytokines)), dtype=np.int64
        )
        marginal_counts = {
            key: np.zeros(len(category_names[key]), dtype=np.int64)
            for key in REQUIRED_OBS
        }
        invalid_code_rows = 0
        treatment_cytokine_mismatch = 0

        for start in range(0, n_obs, args.chunk_size):
            stop = min(start + args.chunk_size, n_obs)
            codes = {
                key: np.asarray(dataset[start:stop], dtype=np.int64)
                for key, dataset in code_datasets.items()
            }
            valid = np.ones(stop - start, dtype=bool)
            for key in REQUIRED_OBS:
                key_valid = (codes[key] >= 0) & (
                    codes[key] < len(category_names[key])
                )
                valid &= key_valid
                marginal_counts[key] += np.bincount(
                    codes[key][key_valid], minlength=len(category_names[key])
                )
            invalid_code_rows += int((~valid).sum())
            if not valid.any():
                continue

            donor_code = codes["donor"][valid]
            cell_code = codes["cell_type"][valid]
            cytokine_code = codes["cytokine"][valid]
            flat = (
                (donor_code * len(cell_types) + cell_code) * len(cytokines)
                + cytokine_code
            )
            cube += np.bincount(flat, minlength=cube.size).reshape(cube.shape)

            is_pbs_cytokine = cytokine_code == pbs_cytokine_idx
            is_pbs_treatment = (
                codes["treatment"][valid] == pbs_treatment_idx
            )
            treatment_cytokine_mismatch += int(
                np.sum(is_pbs_cytokine != is_pbs_treatment)
            )

        if int(cube.sum()) != n_obs - invalid_code_rows:
            raise AssertionError("Condition cube does not account for valid rows")

        var_index = decode_strings(handle["var"]["_index"][:])
        if len(var_index) != n_vars:
            raise ValueError(f"var index has {len(var_index)} rows, expected {n_vars}")

        sample_size = min(args.sample_nnz, nnz)
        sample_idx = np.unique(
            np.linspace(0, nnz - 1, sample_size, dtype=np.int64)
        )
        x_sample = np.asarray(x["data"][sample_idx], dtype=np.float64)
        quantiles = np.quantile(x_sample, [0.0, 0.01, 0.5, 0.99, 1.0])
        matrix_sample = {
            "n_sampled_nonzero": int(x_sample.size),
            "min": float(quantiles[0]),
            "q01": float(quantiles[1]),
            "median": float(quantiles[2]),
            "q99": float(quantiles[3]),
            "max": float(quantiles[4]),
            "fraction_integer_like": float(
                np.mean(np.isclose(x_sample, np.rint(x_sample), atol=1e-6))
            ),
        }

        encoding_type = decode_scalar(x.attrs.get("encoding-type", "unknown"))
        uns_keys = sorted(handle["uns"].keys()) if "uns" in handle else []
        obsm_keys = sorted(handle["obsm"].keys()) if "obsm" in handle else []
        layer_keys = sorted(handle["layers"].keys()) if "layers" in handle else []

        condition_path = args.output_dir / "condition_counts.csv"
        write_condition_counts(
            condition_path, cube, donors, cell_types, cytokines
        )
        cell_type_rows = write_cell_type_eligibility(
            args.output_dir / "cell_type_eligibility.csv",
            cube,
            donors,
            cell_types,
            cytokines,
            pbs_cytokine_idx,
            args.min_cells,
        )
        target_rows = write_target_donor_eligibility(
            args.output_dir / "target_donor_eligibility.csv",
            cube,
            donors,
            cell_types,
            cytokines,
            pbs_cytokine_idx,
            args.min_cells,
        )

        report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "input": str(args.input),
            "input_size_bytes": int(os.path.getsize(args.input)),
            "min_cells": args.min_cells,
            "matrix": {
                "shape": x_shape,
                "encoding_type": encoding_type,
                "nnz": nnz,
                "density": float(nnz / (n_obs * n_vars)),
                "data_dtype": str(x["data"].dtype),
                "sample": matrix_sample,
            },
            "obs_categories": category_names,
            "obs_category_counts": {
                key: {
                    name: int(marginal_counts[key][idx])
                    for idx, name in enumerate(category_names[key])
                }
                for key in REQUIRED_OBS
            },
            "invalid_categorical_code_rows": invalid_code_rows,
            "treatment_cytokine_mismatch_rows": treatment_cytokine_mismatch,
            "var": {
                "n_vars": n_vars,
                "n_unique": len(set(var_index)),
                "first_20": var_index[:20],
            },
            "layers": layer_keys,
            "obsm": obsm_keys,
            "uns": uns_keys,
            "cell_type_eligibility": cell_type_rows,
            "target_donor_eligibility": target_rows,
            "outputs": {
                "condition_counts": str(condition_path),
                "cell_type_eligibility": str(
                    args.output_dir / "cell_type_eligibility.csv"
                ),
                "target_donor_eligibility": str(
                    args.output_dir / "target_donor_eligibility.csv"
                ),
            },
        }

    report_path = args.output_dir / "preflight_report.json"
    report_path.write_text(json.dumps(json_ready(report), indent=2) + "\n")

    eligible = [
        row
        for row in report["cell_type_eligibility"]
        if row["all_donors_pbs_ge_min"]
        and row["cytokines_all_donors_ge_min"] > 0
    ]
    best_targets = sorted(
        report["target_donor_eligibility"],
        key=lambda row: row["cytokines_target_and_source_ge_min"],
        reverse=True,
    )[:20]
    summary_lines = [
        "# Parse-10M PBMC structural preflight",
        "",
        f"- Input: `{args.input}`",
        f"- Shape: `{n_obs:,} x {n_vars:,}`; sparse nnz `{nnz:,}`",
        f"- Donors: `{len(donors)}`",
        f"- Cell types: `{len(cell_types)}`",
        f"- Cytokine labels: `{len(cytokines)}` including PBS",
        f"- Invalid categorical rows: `{invalid_code_rows}`",
        f"- Treatment/cytokine mismatches: `{treatment_cytokine_mismatch}`",
        f"- Cell types with PBS >= {args.min_cells} in every donor and at least one cytokine >= {args.min_cells} in every donor: `{len(eligible)}`",
        "",
        "## Eligible cell types",
        "",
    ]
    if eligible:
        summary_lines.extend(
            f"- {row['cell_type']}: {row['cytokines_all_donors_ge_min']} cytokines"
            for row in sorted(
                eligible,
                key=lambda row: row["cytokines_all_donors_ge_min"],
                reverse=True,
            )
        )
    else:
        summary_lines.append("- None under the all-donor threshold.")
    summary_lines.extend(("", "## Strongest target-donor/cell-type coverage", ""))
    summary_lines.extend(
        f"- {row['target_donor']} / {row['cell_type']}: "
        f"{row['cytokines_target_and_source_ge_min']} eligible cytokines; "
        f"target PBS n={row['target_pbs_cells']}"
        for row in best_targets
    )
    summary_lines.extend(
        (
            "",
            "This is a structural report only. It does not freeze a split or support a rebuttal claim.",
            "",
        )
    )
    (args.output_dir / "preflight_summary.md").write_text(
        "\n".join(summary_lines)
    )

    print(f"Wrote {report_path}")
    print(f"Wrote {args.output_dir / 'preflight_summary.md'}")


if __name__ == "__main__":
    main()
