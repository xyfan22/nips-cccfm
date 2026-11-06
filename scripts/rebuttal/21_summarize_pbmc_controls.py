"""Audit and combine Parse-10M control results across sampling seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


METHODS = ("target_control_copy", "shifted_control")
METRICS = (
    "pearson_ctrl",
    "pearson_pert",
    "mse",
    "centroid_accuracy",
    "mmd",
    "energy",
    "centered_energy",
    "moment",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 314])
    parser.add_argument("--expected_donors", type=int, default=12)
    parser.add_argument("--expected_cell_types", type=int, default=6)
    parser.add_argument("--expected_cytokines", type=int, default=88)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    expected_units = args.expected_donors * args.expected_cell_types * args.expected_cytokines

    seed_means: list[dict] = []
    paired_rows: list[dict] = []
    reference_keys: set[tuple[str, str, str]] | None = None
    reference_protocol: dict | None = None

    for seed in args.seeds:
        seed_dir = args.result_root / f"seed{seed}"
        manifest = json.loads((seed_dir / "manifest.json").read_text())
        if manifest.get("smoke_only"):
            raise ValueError(f"{seed_dir} is marked smoke_only")
        if int(manifest["sample_seed"]) != seed:
            raise ValueError(f"{seed_dir}: manifest seed mismatch")
        protocol = {
            key: manifest[key]
            for key in (
                "shape",
                "normalization",
                "target_axis",
                "donors",
                "cell_types",
                "cytokines",
                "n_cells_per_condition",
                "methods",
                "metrics",
                "leakage_rule",
            )
        }
        if reference_protocol is None:
            reference_protocol = protocol
        elif protocol != reference_protocol:
            raise ValueError(f"Protocol differs for seed {seed}")

        rows = read_csv(seed_dir / "per_condition_metrics.csv")
        if len(rows) != expected_units * len(METHODS):
            raise ValueError(
                f"seed {seed}: {len(rows)} rows, expected {expected_units * len(METHODS)}"
            )
        by_method: dict[str, dict[tuple[str, str, str], dict[str, str]]] = {}
        for method in METHODS:
            method_rows = [row for row in rows if row["method"] == method]
            if len(method_rows) != expected_units:
                raise ValueError(f"seed {seed} {method}: incomplete rows")
            mapping = {
                (row["target_donor"], row["cell_type"], row["cytokine"]): row
                for row in method_rows
            }
            if len(mapping) != expected_units:
                raise ValueError(f"seed {seed} {method}: duplicate condition keys")
            by_method[method] = mapping
            values = {}
            for metric in METRICS:
                metric_values = [float(row[metric]) for row in method_rows]
                if not all(math.isfinite(value) for value in metric_values):
                    raise ValueError(f"seed {seed} {method}: non-finite {metric}")
                values[metric] = mean(metric_values)
            seed_means.append({"seed": seed, "method": method, **values})

        keys = set(by_method[METHODS[0]])
        if keys != set(by_method[METHODS[1]]):
            raise ValueError(f"seed {seed}: method condition sets differ")
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise ValueError(f"seed {seed}: condition set differs from other seeds")

        for key in sorted(keys):
            copy_row = by_method["target_control_copy"][key]
            shift_row = by_method["shifted_control"][key]
            centered_gap = abs(
                float(copy_row["centered_energy"])
                - float(shift_row["centered_energy"])
            )
            if centered_gap > 5e-3:
                raise ValueError(
                    f"seed {seed} {key}: centered Energy differs by {centered_gap}; "
                    "the two predictions should differ only by translation"
                )
            paired_rows.append(
                {
                    "seed": seed,
                    "target_donor": key[0],
                    "cell_type": key[1],
                    "cytokine": key[2],
                    "energy_copy_minus_shifted": (
                        float(copy_row["energy"]) - float(shift_row["energy"])
                    ),
                    "pearson_pert_shifted_minus_copy": (
                        float(shift_row["pearson_pert"])
                        - float(copy_row["pearson_pert"])
                    ),
                    "moment_copy_minus_shifted": (
                        float(copy_row["moment"]) - float(shift_row["moment"])
                    ),
                }
            )

    write_csv(
        args.output_dir / "sampling_seed_means.csv",
        ["seed", "method", *METRICS],
        seed_means,
    )
    write_csv(
        args.output_dir / "paired_condition_differences.csv",
        [
            "seed",
            "target_donor",
            "cell_type",
            "cytokine",
            "energy_copy_minus_shifted",
            "pearson_pert_shifted_minus_copy",
            "moment_copy_minus_shifted",
        ],
        paired_rows,
    )

    combined_rows: list[dict] = []
    for method in METHODS:
        method_seed_rows = [row for row in seed_means if row["method"] == method]
        out = {"method": method, "n_sampling_seeds": len(args.seeds)}
        for metric in METRICS:
            values = [float(row[metric]) for row in method_seed_rows]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        combined_rows.append(out)
    combined_fields = ["method", "n_sampling_seeds"]
    for metric in METRICS:
        combined_fields.extend((f"{metric}_mean", f"{metric}_std"))
    write_csv(args.output_dir / "combined_summary.csv", combined_fields, combined_rows)

    scope_rows: list[dict] = []
    cell_types = sorted({row["cell_type"] for row in paired_rows})
    for scope, scope_value, scoped in [
        ("overall", "all", paired_rows),
        *[
            (
                "cell_type",
                cell_type,
                [row for row in paired_rows if row["cell_type"] == cell_type],
            )
            for cell_type in cell_types
        ],
    ]:
        energy_diffs = [float(row["energy_copy_minus_shifted"]) for row in scoped]
        pearson_diffs = [float(row["pearson_pert_shifted_minus_copy"]) for row in scoped]
        moment_diffs = [float(row["moment_copy_minus_shifted"]) for row in scoped]
        scope_rows.append(
            {
                "scope": scope,
                "scope_value": scope_value,
                "n_seed_conditions": len(scoped),
                "mean_energy_copy_minus_shifted": mean(energy_diffs),
                "fraction_copy_lower_energy": mean(value < 0 for value in energy_diffs),
                "mean_pearson_pert_shifted_minus_copy": mean(pearson_diffs),
                "fraction_shifted_higher_pearson_pert": mean(
                    value > 0 for value in pearson_diffs
                ),
                "mean_moment_copy_minus_shifted": mean(moment_diffs),
            }
        )
    write_csv(
        args.output_dir / "paired_difference_summary.csv",
        [
            "scope",
            "scope_value",
            "n_seed_conditions",
            "mean_energy_copy_minus_shifted",
            "fraction_copy_lower_energy",
            "mean_pearson_pert_shifted_minus_copy",
            "fraction_shifted_higher_pearson_pert",
            "mean_moment_copy_minus_shifted",
        ],
        scope_rows,
    )

    report = [
        "# Parse-10M PBMC controls: three-seed audit",
        "",
        f"- Sampling seeds: `{', '.join(map(str, args.seeds))}`",
        f"- Unique donor/cell-type/cytokine units per seed: `{expected_units}`",
        "- Balanced n, protocol identity, finite metrics, and translation-invariance checks: `PASS`",
        "- Values are mean +/- sample std over cell-sampling seeds.",
        "",
        "| method | Pf^pert | Energy | centered Energy | Moment |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in combined_rows:
        report.append(
            f"| {row['method']} | {row['pearson_pert_mean']:.6f} +/- {row['pearson_pert_std']:.6f} | "
            f"{row['energy_mean']:.6f} +/- {row['energy_std']:.6f} | "
            f"{row['centered_energy_mean']:.6f} +/- {row['centered_energy_std']:.6f} | "
            f"{row['moment_mean']:.6f} +/- {row['moment_std']:.6f} |"
        )
    report.extend(
        [
            "",
            "Interpretation boundary: this tests donor transfer in primary PBMC cytokine stimulation. ",
            "It does not establish CRISPRi or small-molecule generality, and without a trained model row it is a controls audit rather than a leaderboard.",
        ]
    )
    (args.output_dir / "audit_report.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
