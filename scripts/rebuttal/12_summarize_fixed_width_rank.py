"""Audit and summarize the rebuttal rank x hidden-width factorial.

The script refuses incomplete cells, mismatched perturbation sets, non-uniform
prediction counts, and configurations whose recorded rank/width differ from
their path.  It reports raw paired-seed contrasts; with three seeds these are
descriptive diagnostics, not significance tests.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


METRICS = (
    "pearson_pert",
    "pearson20_pert",
    "centroid_accuracy",
    "energy",
    "w2_sinkhorn",
    "moment",
)
LOWER_IS_BETTER = {"energy", "w2_sinkhorn", "moment"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--result_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[256, 512])
    parser.add_argument("--widths", type=int, nargs="+", default=[1024, 2048])
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 314])
    parser.add_argument("--expected_perts", type=int, default=577)
    return parser.parse_args()


def tag(rank: int, width: int, seed: int) -> str:
    return f"rank{rank}_hidden{width}_seed{seed}"


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def finite_float(row: dict[str, str], key: str, path: Path) -> float:
    raw = row.get(key, "")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Missing/non-numeric {key} in {path}: {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {key} in {path}: {value}")
    return value


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed_rows: list[dict] = []
    config_rows: list[dict] = []
    pert_reference: list[str] | None = None
    count_reference: dict[str, tuple[int, int]] | None = None

    for rank in args.ranks:
        for width in args.widths:
            for seed in args.seeds:
                name = tag(rank, width, seed)
                run_dir = args.run_root / name
                result_path = args.result_root / f"{name}.csv"
                cfg_path = run_dir / "config.json"
                if not cfg_path.is_file():
                    raise FileNotFoundError(cfg_path)
                cfg = json.loads(cfg_path.read_text())
                expected_cfg = {
                    "rank": rank,
                    "velocity_hidden": width,
                    "seed": seed,
                    "flow_space": "lowrank",
                    "anchor_target_mode": "mean_preserved",
                    "setting": "2line",
                    "target_line": "Jurkat",
                }
                for key, expected in expected_cfg.items():
                    if cfg.get(key) != expected:
                        raise ValueError(
                            f"{cfg_path}: {key}={cfg.get(key)!r}, expected {expected!r}"
                        )
                if cfg.get("velocity_hidden_policy") != "explicit_rebuttal_factorial":
                    raise ValueError(f"{cfg_path}: explicit width policy not recorded")

                rows = read_rows(result_path)
                if len(rows) != args.expected_perts:
                    raise ValueError(
                        f"{result_path}: {len(rows)} rows, expected {args.expected_perts}"
                    )
                perts = [row["pert"] for row in rows]
                if len(set(perts)) != len(perts):
                    raise ValueError(f"Duplicate perturbations in {result_path}")
                if pert_reference is None:
                    pert_reference = perts
                elif perts != pert_reference:
                    raise ValueError(f"Perturbation order/set differs in {result_path}")

                counts: dict[str, tuple[int, int]] = {}
                for row in rows:
                    n_pred = int(row["n_pred_cells"])
                    n_target = int(float(row["n_target_cells"]))
                    if n_pred != n_target:
                        raise ValueError(
                            f"{result_path} {row['pert']}: n_pred={n_pred}, n_target={n_target}"
                        )
                    counts[row["pert"]] = (n_pred, n_target)
                if count_reference is None:
                    count_reference = counts
                elif counts != count_reference:
                    raise ValueError(f"Per-perturbation sample counts differ in {result_path}")

                metric_means = {
                    metric: mean(finite_float(row, metric, result_path) for row in rows)
                    for metric in METRICS
                }
                seed_rows.append(
                    {"rank": rank, "hidden": width, "seed": seed, **metric_means}
                )
                config_rows.append(
                    {
                        "rank": rank,
                        "hidden": width,
                        "seed": seed,
                        "n_params_velocity": int(cfg["n_params_velocity"]),
                        "run_id": cfg["run_id"],
                        "result_file": str(result_path),
                    }
                )

    write_csv(
        args.output_dir / "seed_means.csv",
        ["rank", "hidden", "seed", *METRICS],
        seed_rows,
    )
    write_csv(
        args.output_dir / "config_audit.csv",
        ["rank", "hidden", "seed", "n_params_velocity", "run_id", "result_file"],
        config_rows,
    )

    lookup = {
        (int(row["rank"]), int(row["hidden"]), int(row["seed"])): row
        for row in seed_rows
    }
    cell_rows: list[dict] = []
    for rank in args.ranks:
        for width in args.widths:
            out = {"rank": rank, "hidden": width, "n_seeds": len(args.seeds)}
            for metric in METRICS:
                values = [float(lookup[(rank, width, seed)][metric]) for seed in args.seeds]
                out[f"{metric}_mean"] = mean(values)
                out[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
            cell_rows.append(out)
    cell_fields = ["rank", "hidden", "n_seeds"]
    for metric in METRICS:
        cell_fields.extend((f"{metric}_mean", f"{metric}_std"))
    write_csv(args.output_dir / "factorial_cell_summary.csv", cell_fields, cell_rows)

    contrast_rows: list[dict] = []
    for width in args.widths:
        for seed in args.seeds:
            for metric in METRICS:
                delta = (
                    float(lookup[(512, width, seed)][metric])
                    - float(lookup[(256, width, seed)][metric])
                )
                contrast_rows.append(
                    {
                        "contrast": "rank_512_minus_256",
                        "fixed_axis": "hidden",
                        "fixed_value": width,
                        "seed": seed,
                        "metric": metric,
                        "raw_delta": delta,
                        "positive_means": (
                            "rank512_worse" if metric in LOWER_IS_BETTER else "rank512_better"
                        ),
                    }
                )
    for rank in args.ranks:
        for seed in args.seeds:
            for metric in METRICS:
                delta = (
                    float(lookup[(rank, 2048, seed)][metric])
                    - float(lookup[(rank, 1024, seed)][metric])
                )
                contrast_rows.append(
                    {
                        "contrast": "hidden_2048_minus_1024",
                        "fixed_axis": "rank",
                        "fixed_value": rank,
                        "seed": seed,
                        "metric": metric,
                        "raw_delta": delta,
                        "positive_means": (
                            "hidden2048_worse"
                            if metric in LOWER_IS_BETTER
                            else "hidden2048_better"
                        ),
                    }
                )
    write_csv(
        args.output_dir / "paired_seed_contrasts.csv",
        [
            "contrast",
            "fixed_axis",
            "fixed_value",
            "seed",
            "metric",
            "raw_delta",
            "positive_means",
        ],
        contrast_rows,
    )

    report = [
        "# Fixed-width rank factorial audit",
        "",
        f"- Complete cells: `{len(cell_rows)}`",
        f"- Seeds per cell: `{len(args.seeds)}` ({', '.join(map(str, args.seeds))})",
        f"- Perturbations per run: `{args.expected_perts}`",
        "- Uniform-n and identical perturbation-set checks: `PASS`",
        "- Values below are mean +/- sample std over training seeds.",
        "",
        "| rank | hidden | Energy | W2 | Moment | Pf^pert |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cell_rows:
        report.append(
            "| {rank} | {hidden} | {energy_mean:.6f} +/- {energy_std:.6f} | "
            "{w2_sinkhorn_mean:.6f} +/- {w2_sinkhorn_std:.6f} | "
            "{moment_mean:.6f} +/- {moment_std:.6f} | "
            "{pearson_pert_mean:.6f} +/- {pearson_pert_std:.6f} |".format(**row)
        )
    report.extend(
        [
            "",
            "Interpretation must compare rank at fixed width and width at fixed rank. ",
            "Do not infer a rank-only mechanism from a diagonal comparison.",
        ]
    )
    (args.output_dir / "audit_report.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
