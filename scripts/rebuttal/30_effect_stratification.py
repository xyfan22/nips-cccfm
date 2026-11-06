"""Source-only effect stratification for the Replogle rebuttal analysis.

The strata are frozen from K562/RPE1 cells before Jurkat evaluation rows are
loaded.  This avoids defining favorable subsets from target-line outcomes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from pathlib import Path
from statistics import mean

import h5py
import numpy as np
from scipy.stats import spearmanr


SOURCE_LINES = ("K562", "RPE1")
METRICS = ("pearson_pert", "energy", "w2_sinkhorn", "moment")
LOWER_BETTER = {"energy", "w2_sinkhorn", "moment"}
METHOD_ORDER = (
    "target_control_copy",
    "linear_shifted",
    "exact_shifted",
    "cellflow",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--test_slice", type=Path, required=True)
    parser.add_argument(
        "--method_csv",
        nargs=2,
        action="append",
        metavar=("NAME", "PATH"),
        required=True,
        help="Repeat once per method; required names are fixed in the script.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--chunk_rows", type=int, default=2048)
    parser.add_argument("--bootstrap_reps", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260726)
    return parser.parse_args()


def decode(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def balanced_tertiles(
    perts: list[str], values: dict[str, float], labels: tuple[str, str, str]
) -> dict[str, str]:
    order = sorted(perts, key=lambda pert: (values[pert], pert))
    n = len(order)
    return {
        pert: labels[min(2, (rank * 3) // n)]
        for rank, pert in enumerate(order)
    }


def bootstrap_mean_ci(
    values: list[float], rng: np.random.Generator, reps: int
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    observed = float(array.mean())
    if reps <= 0 or len(array) == 1:
        return observed, observed, observed
    draws = rng.integers(0, len(array), size=(reps, len(array)))
    means = array[draws].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return observed, float(lo), float(hi)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compute_source_descriptors(
    data_path: Path, perts: list[str], chunk_rows: int
) -> tuple[list[dict], dict]:
    with h5py.File(data_path, "r") as handle:
        x = handle["X"]
        if len(x.shape) != 2:
            raise ValueError(f"Expected dense 2-D X, observed {x.shape}")
        x_shape = (int(x.shape[0]), int(x.shape[1]))
        line_categories = decode(handle["obs/cell_line/categories"][:])
        pert_categories = decode(handle["obs/pert/categories"][:])
        line_codes = np.asarray(handle["obs/cell_line/codes"][:], dtype=np.int32)
        pert_codes = np.asarray(handle["obs/pert/codes"][:], dtype=np.int32)
        if x.shape[0] != len(line_codes) or len(line_codes) != len(pert_codes):
            raise ValueError("X/obs length mismatch")
        missing = sorted(set(perts) - set(pert_categories))
        if missing:
            raise ValueError(f"Test perturbations absent from H5AD: {missing[:10]}")
        if "control" not in pert_categories:
            raise ValueError("Control category is absent")

        local_line = np.full(len(line_categories), -1, dtype=np.int16)
        for idx, line in enumerate(SOURCE_LINES):
            local_line[line_categories.index(line)] = idx
        conditions = ["control", *perts]
        condition_to_local = {condition: idx for idx, condition in enumerate(conditions)}
        local_pert = np.full(len(pert_categories), -1, dtype=np.int32)
        for category_idx, category in enumerate(pert_categories):
            if category in condition_to_local:
                local_pert[category_idx] = condition_to_local[category]

        n_groups = len(SOURCE_LINES) * len(conditions)
        sums = np.zeros((n_groups, x.shape[1]), dtype=np.float64)
        counts = np.zeros(n_groups, dtype=np.int64)
        for start in range(0, x.shape[0], chunk_rows):
            stop = min(start + chunk_rows, x.shape[0])
            line_block = line_codes[start:stop]
            pert_block = pert_codes[start:stop]
            valid_codes = (line_block >= 0) & (pert_block >= 0)
            line_local = local_line[np.clip(line_block, 0, len(local_line) - 1)]
            pert_local = local_pert[np.clip(pert_block, 0, len(local_pert) - 1)]
            keep = valid_codes & (line_local >= 0) & (pert_local >= 0)
            if not np.any(keep):
                continue
            group_ids = line_local[keep].astype(np.int64) * len(conditions) + pert_local[keep]
            block = np.asarray(x[start:stop], dtype=np.float32)[keep]
            order = np.argsort(group_ids, kind="stable")
            group_ids = group_ids[order]
            block = block[order]
            unique, first, group_count = np.unique(
                group_ids, return_index=True, return_counts=True
            )
            sums[unique] += np.add.reduceat(block.astype(np.float64), first, axis=0)
            counts[unique] += group_count

    if np.any(counts == 0):
        empty = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"Empty source groups: {empty[:20]}")
    means = sums / counts[:, None]
    descriptors = []
    strength_values: dict[str, float] = {}
    agreement_values: dict[str, float] = {}
    for pert_idx, pert in enumerate(perts, start=1):
        deltas = []
        pert_counts = []
        control_counts = []
        for line_idx in range(len(SOURCE_LINES)):
            base = line_idx * len(conditions)
            delta = means[base + pert_idx] - means[base]
            deltas.append(delta)
            pert_counts.append(int(counts[base + pert_idx]))
            control_counts.append(int(counts[base]))
        line_rms = [float(np.sqrt(np.mean(delta * delta))) for delta in deltas]
        strength = float(mean(line_rms))
        agreement = safe_pearson(deltas[0], deltas[1])
        cosine = safe_cosine(deltas[0], deltas[1])
        heterogeneity = float(np.sqrt(np.mean((deltas[0] - deltas[1]) ** 2)))
        mean_delta = (deltas[0] + deltas[1]) / 2.0
        mean_delta_rms = float(np.sqrt(np.mean(mean_delta * mean_delta)))
        strength_values[pert] = strength
        agreement_values[pert] = agreement
        descriptors.append(
            {
                "pert": pert,
                "source_strength_rms": strength,
                "k562_delta_rms": line_rms[0],
                "rpe1_delta_rms": line_rms[1],
                "cross_line_agreement_pearson": agreement,
                "cross_line_agreement_cosine": cosine,
                "cross_line_heterogeneity_rms": heterogeneity,
                "source_mean_delta_rms": mean_delta_rms,
                "k562_pert_n": pert_counts[0],
                "rpe1_pert_n": pert_counts[1],
                "k562_control_n": control_counts[0],
                "rpe1_control_n": control_counts[1],
            }
        )
    strength_bins = balanced_tertiles(
        perts, strength_values, ("weak", "middle", "strong")
    )
    agreement_bins = balanced_tertiles(
        perts, agreement_values, ("low", "middle", "high")
    )
    for row in descriptors:
        row["strength_bin"] = strength_bins[row["pert"]]
        row["agreement_bin"] = agreement_bins[row["pert"]]
        row["joint_bin"] = f"{row['strength_bin']}__{row['agreement_bin']}"
    metadata = {
        "h5ad_shape": list(x_shape),
        "source_lines": list(SOURCE_LINES),
        "n_perturbations": len(perts),
        "group_count_min": int(counts.min()),
        "group_count_median": float(np.median(counts)),
        "group_count_max": int(counts.max()),
    }
    return descriptors, metadata


def read_method_rows(
    name: str, path: Path, expected_perts: set[str]
) -> dict[str, dict[str, str]]:
    if name not in METHOD_ORDER:
        raise ValueError(f"Unexpected method name {name}; expected {METHOD_ORDER}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(expected_perts):
        raise ValueError(f"{name}: observed {len(rows)} rows")
    mapping = {row["pert"]: row for row in rows}
    if len(mapping) != len(rows) or set(mapping) != expected_perts:
        raise ValueError(f"{name}: duplicate or mismatched perturbation keys")
    for pert, row in mapping.items():
        if row.get("setting") != "2line" or row.get("target_line") != "Jurkat":
            raise ValueError(f"{name}/{pert}: wrong setting or target")
        for metric in METRICS:
            if not math.isfinite(float(row[metric])):
                raise ValueError(f"{name}/{pert}: non-finite {metric}")
    return mapping


def oriented_advantage(method_value: float, copy_value: float, metric: str) -> float:
    return copy_value - method_value if metric in LOWER_BETTER else method_value - copy_value


def main() -> None:
    args = parse_args()
    if args.chunk_rows < 1 or args.bootstrap_reps < 0:
        raise ValueError("Invalid chunk/bootstrap setting")
    method_paths = {name: Path(path) for name, path in args.method_csv}
    if tuple(name for name in METHOD_ORDER if name in method_paths) != METHOD_ORDER:
        raise ValueError(f"Pass each required method exactly once: {METHOD_ORDER}")
    if len(method_paths) != len(METHOD_ORDER):
        raise ValueError("Duplicate or extra --method_csv entries")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    perts = json.loads(args.test_slice.read_text())
    if not isinstance(perts, list) or len(perts) != 577 or len(set(perts)) != 577:
        raise ValueError("Expected exactly 577 unique test perturbations")
    perts = [str(pert) for pert in perts]

    descriptors, source_metadata = compute_source_descriptors(
        args.data, perts, args.chunk_rows
    )
    descriptor_by_pert = {row["pert"]: row for row in descriptors}
    descriptor_fields = list(descriptors[0])

    # Target-line outcomes enter only after source-only descriptors and bins
    # have been computed and frozen in memory.
    methods = {
        name: read_method_rows(name, path, set(perts))
        for name, path in method_paths.items()
    }
    merged_rows = []
    for pert in perts:
        descriptor = descriptor_by_pert[pert]
        for method in METHOD_ORDER:
            input_row = methods[method][pert]
            merged_rows.append(
                {
                    **descriptor,
                    "method": method,
                    **{metric: float(input_row[metric]) for metric in METRICS},
                    "n_target_cells": int(input_row["n_target_cells"]),
                }
            )
    merged_fields = [*descriptor_fields, "method", *METRICS, "n_target_cells"]

    rng = np.random.default_rng(args.bootstrap_seed)
    summary_rows = []
    strata = [
        ("all", "all", perts),
        *[
            ("strength_bin", label, [p for p in perts if descriptor_by_pert[p]["strength_bin"] == label])
            for label in ("weak", "middle", "strong")
        ],
        *[
            ("agreement_bin", label, [p for p in perts if descriptor_by_pert[p]["agreement_bin"] == label])
            for label in ("low", "middle", "high")
        ],
        *[
            ("joint_bin", label, [p for p in perts if descriptor_by_pert[p]["joint_bin"] == label])
            for label in sorted({row["joint_bin"] for row in descriptors})
        ],
    ]
    for stratum, label, members in strata:
        if not members:
            raise ValueError(f"Empty stratum {stratum}/{label}")
        for method in METHOD_ORDER:
            for metric in METRICS:
                values = [float(methods[method][pert][metric]) for pert in members]
                value, lo, hi = bootstrap_mean_ci(values, rng, args.bootstrap_reps)
                summary_rows.append(
                    {
                        "stratifier": stratum,
                        "stratum": label,
                        "n_perts": len(members),
                        "row_type": "method_mean",
                        "method": method,
                        "reference": "",
                        "metric": metric,
                        "value": value,
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "positive_means_method_better": "",
                    }
                )
        for method in METHOD_ORDER[1:]:
            for metric in METRICS:
                values = [
                    oriented_advantage(
                        float(methods[method][pert][metric]),
                        float(methods["target_control_copy"][pert][metric]),
                        metric,
                    )
                    for pert in members
                ]
                value, lo, hi = bootstrap_mean_ci(values, rng, args.bootstrap_reps)
                summary_rows.append(
                    {
                        "stratifier": stratum,
                        "stratum": label,
                        "n_perts": len(members),
                        "row_type": "oriented_advantage",
                        "method": method,
                        "reference": "target_control_copy",
                        "metric": metric,
                        "value": value,
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "positive_means_method_better": True,
                    }
                )
    summary_fields = [
        "stratifier", "stratum", "n_perts", "row_type", "method", "reference",
        "metric", "value", "ci_lo", "ci_hi", "positive_means_method_better",
    ]

    trend_rows = []
    descriptor_names = ("source_strength_rms", "cross_line_agreement_pearson")
    for descriptor_name in descriptor_names:
        x_values = [float(descriptor_by_pert[pert][descriptor_name]) for pert in perts]
        for method in METHOD_ORDER[1:]:
            for metric in METRICS:
                advantages = [
                    oriented_advantage(
                        float(methods[method][pert][metric]),
                        float(methods["target_control_copy"][pert][metric]),
                        metric,
                    )
                    for pert in perts
                ]
                constant_input = (
                    np.ptp(np.asarray(x_values, dtype=np.float64)) < 1e-15
                    or np.ptp(np.asarray(advantages, dtype=np.float64)) < 1e-15
                )
                if constant_input:
                    rho, p_value = 0.0, 1.0
                else:
                    test = spearmanr(x_values, advantages)
                    rho = float(test.statistic)
                    p_value = float(test.pvalue)
                    if not (math.isfinite(rho) and math.isfinite(p_value)):
                        raise ValueError(
                            f"Non-finite Spearman result for {descriptor_name}/{method}/{metric}"
                        )
                trend_rows.append(
                    {
                        "descriptor": descriptor_name,
                        "method": method,
                        "reference": "target_control_copy",
                        "metric": metric,
                        "n_perts": len(perts),
                        "spearman_rho": rho,
                        "p_value_advisory": p_value,
                        "constant_input": constant_input,
                        "positive_advantage_means_method_better": True,
                    }
                )
    trend_fields = list(trend_rows[0])

    bin_counts = {}
    for key in ("strength_bin", "agreement_bin", "joint_bin"):
        for row in descriptors:
            name = f"{key}:{row[key]}"
            bin_counts[name] = bin_counts.get(name, 0) + 1
    audit = {
        "protocol": {
            "strata_source_only": True,
            "source_lines": list(SOURCE_LINES),
            "target_line": "Jurkat",
            "strength_definition": "mean across source lines of sqrt(mean(delta_gene^2))",
            "agreement_definition": "Pearson across genes between K562 and RPE1 deltas",
            "binning": "deterministic rank-balanced tertiles; ties broken by gene symbol",
            "bootstrap_unit": "perturbation",
            "bootstrap_reps": args.bootstrap_reps,
            "bootstrap_seed": args.bootstrap_seed,
            "inference_boundary": "descriptive; perturbations are not claimed independent biological replicates",
        },
        "inputs": {
            "data": str(args.data),
            "data_size_bytes": args.data.stat().st_size,
            "test_slice": str(args.test_slice),
            "test_slice_sha256": file_sha256(args.test_slice),
            "method_csv": {
                name: {
                    "path": str(path),
                    "sha256": file_sha256(path),
                }
                for name, path in method_paths.items()
            },
        },
        "source_metadata": source_metadata,
        "bin_counts": dict(sorted(bin_counts.items())),
        "output_rows": {
            "source_descriptors": len(descriptors),
            "merged_per_pert": len(merged_rows),
            "stratum_summary": len(summary_rows),
            "trend_summary": len(trend_rows),
        },
    }
    report = [
        "# Replogle source-only effect stratification",
        "",
        f"- Perturbations: `{len(perts)}`",
        "- Strata use K562/RPE1 cells only; Jurkat outcomes are merged afterward.",
        "- Primary descriptors: source RMS effect strength and K562/RPE1 delta Pearson agreement.",
        "- Bins: deterministic rank-balanced tertiles plus their 3x3 joint grid.",
        "- Bootstrap unit: perturbation; intervals are descriptive, not biological-replicate inference.",
        "",
        "## Bin counts",
        "",
    ]
    report.extend(f"- {key}: `{value}`" for key, value in sorted(bin_counts.items()))
    report.extend(
        [
            "",
            "## Interpretation contract",
            "",
            "Report all bins and both shifted controls. Do not call the 577 genes essential/non-essential without an authoritative annotation. A result confined to strong/high-agreement strata supports the reviewer's selection-bias explanation; persistence in weak/low-agreement strata weakens, but does not eliminate, that explanation. CellFlow and control comparisons remain specific to the submitted Jurkat setting.",
        ]
    )
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{args.output_dir.name}.staging-", dir=args.output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        write_csv(staging / "source_descriptors.csv", descriptors, descriptor_fields)
        write_csv(staging / "merged_per_pert.csv", merged_rows, merged_fields)
        write_csv(staging / "stratum_summary.csv", summary_rows, summary_fields)
        write_csv(staging / "trend_summary.csv", trend_rows, trend_fields)
        (staging / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        (staging / "report.md").write_text("\n".join(report) + "\n")
        if args.output_dir.exists():
            raise FileExistsError(args.output_dir)
        staging.rename(args.output_dir)


if __name__ == "__main__":
    main()
