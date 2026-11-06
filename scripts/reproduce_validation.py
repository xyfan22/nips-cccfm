"""Rebuild the published factorial, PBMC, and stratification summaries on CPU.

This aggregates released predictions' metric rows; model fitting and metric
evaluation on expression clouds are separate experiment entry points.
"""

import argparse
import csv
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
FACTORIAL_METRICS = (
    "pearson_pert", "pearson20_pert", "centroid_accuracy", "energy", "w2_sinkhorn", "moment"
)
PBMC_METRICS = (
    "pearson_ctrl", "pearson_pert", "mse", "centroid_accuracy", "mmd", "energy", "centered_energy", "moment"
)
METHODS = ("target_control_copy", "shifted_control", "cellflow")


def read_csv(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(row, key):
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {key}")
    return value


def compare(actual, expected, keys, tolerance=1e-11):
    actual_by_key = {tuple(str(row[key]) for key in keys): row for row in actual}
    expected_by_key = {tuple(str(row[key]) for key in keys): row for row in expected}
    if len(actual_by_key) != len(actual) or len(expected_by_key) != len(expected):
        raise ValueError(f"Duplicate summary keys: {keys}")
    if actual_by_key.keys() != expected_by_key.keys():
        raise ValueError(f"Summary key mismatch: {keys}")
    largest_error = 0.0
    for identity, row in actual_by_key.items():
        reference = expected_by_key[identity]
        for field in reference:
            if field in keys:
                continue
            try:
                wanted = float(reference[field])
            except (TypeError, ValueError):
                if str(row.get(field, "")) != str(reference[field]):
                    raise ValueError(f"Metadata mismatch at {identity}: {field}")
                continue
            error = abs(float(row[field]) - wanted)
            if not math.isfinite(error) or error > tolerance:
                raise ValueError(f"Numeric mismatch at {identity}, {field}: {error}")
            largest_error = max(largest_error, error)
    return largest_error


def factorial(root, output):
    base = root / "results/rebuttal/A_fixed_width_rank"
    seed_rows, target_counts = [], None
    for rank in (256, 512):
        for hidden in (1024, 2048):
            for seed in (7, 42, 314):
                rows = read_csv(base / f"rank{rank}_hidden{hidden}_seed{seed}.csv")
                counts = {}
                for row in rows:
                    pert = row["pert"]
                    if pert in counts or int(row["n_pred_cells"]) != int(row["n_target_cells"]):
                        raise ValueError("Duplicate perturbation or non-uniform prediction count")
                    counts[pert] = int(row["n_target_cells"])
                if len(counts) != 577 or (target_counts is not None and counts != target_counts):
                    raise ValueError("Factorial perturbations/counts differ across cells")
                target_counts = counts
                seed_rows.append({"rank": rank, "hidden": hidden, "seed": seed,
                                  **{metric: mean(number(row, metric) for row in rows) for metric in FACTORIAL_METRICS}})
    cells = []
    for rank in (256, 512):
        for hidden in (1024, 2048):
            selected = [row for row in seed_rows if row["rank"] == rank and row["hidden"] == hidden]
            cell = {"rank": rank, "hidden": hidden, "n_seeds": len(selected)}
            for metric in FACTORIAL_METRICS:
                values = [row[metric] for row in selected]
                cell[metric + "_mean"], cell[metric + "_std"] = mean(values), stdev(values)
            cells.append(cell)
    error = compare(cells, read_csv(base / "audit/factorial_cell_summary.csv"), ("rank", "hidden"))
    write_csv(output / "factorial_seed_means.csv", seed_rows)
    write_csv(output / "factorial_summary.csv", cells)
    return {"runs": len(seed_rows), "perturbations_per_run": 577, "max_summary_error": error}


def pbmc(root, output):
    base = root / "results/rebuttal/B2_pbmc_cellflow/pilot/combined"
    groups, keys = defaultdict(list), set()
    for row in read_csv(base / "per_condition_combined.csv"):
        identity = (row["method"], row["target_donor"], row["cell_type"], row["cytokine"], int(row["sample_seed"]))
        if identity in keys or row["method"] not in METHODS:
            raise ValueError("Duplicate PBMC condition or unknown method")
        keys.add(identity)
        if int(row["n_cells"]) != 50 or int(row["n_source_donors"]) != 11:
            raise ValueError("PBMC protocol counts differ from 50 cells / 11 source donors")
        groups[identity[:4]].append(row)
    if len(keys) != 57024:
        raise ValueError(f"Expected 57024 PBMC rows, got {len(keys)}")
    expected_types = set(json.loads((root / "provenance/protocols.json").read_text())["pbmc"]["cell_types"])
    condition_sets = defaultdict(set)
    for method, donor, cell_type, cytokine in groups:
        condition_sets[method, donor].add((cell_type, cytokine))
    expected_conditions = next(iter(condition_sets.values()))
    if ({cell_type for cell_type, _ in expected_conditions} != expected_types
            or len({cytokine for _, cytokine in expected_conditions}) != 88
            or any(conditions != expected_conditions for conditions in condition_sets.values())):
        raise ValueError("PBMC methods/donors do not share the frozen six-type, 88-cytokine grid")
    donor_groups = defaultdict(list)
    for (method, donor, cell_type, cytokine), rows in groups.items():
        if {int(row["sample_seed"]) for row in rows} != {7, 42, 314} or len(rows) != 3:
            raise ValueError("Incomplete PBMC cell-sampling repeats")
        donor_groups[donor, method].append({metric: mean(number(row, metric) for row in rows) for metric in PBMC_METRICS})
    donor_rows = []
    for (donor, method), rows in sorted(donor_groups.items()):
        if len(rows) != 528:
            raise ValueError(f"Incomplete conditions for {donor}/{method}")
        donor_rows.append({"target_donor": donor, "method": method, "n_conditions": len(rows),
                           **{metric: mean(row[metric] for row in rows) for metric in PBMC_METRICS}})
    donors = {row["target_donor"] for row in donor_rows}
    if donors != {f"Donor{i}" for i in range(1, 13)} or len(donor_rows) != 36:
        raise ValueError("Incomplete donor/method grid")
    global_rows = []
    for method in METHODS:
        rows = [row for row in donor_rows if row["method"] == method]
        summary = {"method": method, "n_donors": len(rows)}
        for metric in PBMC_METRICS:
            values = [row[metric] for row in rows]
            summary[metric + "_mean"] = mean(values)
            summary[metric + "_sd_across_donors"] = stdev(values)
        global_rows.append(summary)
    error = max(compare(donor_rows, read_csv(base / "donor_summary.csv"), ("target_donor", "method")),
                compare(global_rows, read_csv(base / "global_summary.csv"), ("method",)))
    write_csv(output / "pbmc_donor_summary.csv", donor_rows)
    write_csv(output / "pbmc_global_summary.csv", global_rows)
    return {"metric_rows": len(keys), "donors": 12, "variability_unit": "donor", "max_summary_error": error}


def stratification(root, output):
    base = root / "results/rebuttal/C_effect_stratification"
    frozen = root / "results/rebuttal/C_frozen_inputs"
    spec = importlib.util.spec_from_file_location("source_stratification", root / "scripts/rebuttal/30_effect_stratification.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    descriptors = read_csv(base / "source_descriptors.csv")
    perts = [row["pert"] for row in descriptors]
    if len(perts) != 577 or len(set(perts)) != 577:
        raise ValueError("Source descriptors must contain 577 distinct perturbations")
    by_pert = {row["pert"]: row for row in descriptors}
    for field, column, labels in (("source_strength_rms", "strength_bin", ("weak", "middle", "strong")),
                                  ("cross_line_agreement_pearson", "agreement_bin", ("low", "middle", "high"))):
        bins = module.balanced_tertiles(perts, {p: float(by_pert[p][field]) for p in perts}, labels)
        if any(by_pert[p][column] != bins[p] for p in perts):
            raise ValueError("Published strata differ from deterministic source-only tertiles")
    paths = {
        "target_control_copy": "target_control_copy_2line.csv",
        "linear_shifted": "linear_ahlmann_eltze_2line_shifted_control.csv",
        "exact_shifted": "exact_gene_retrieval_2line_shifted_control.csv",
        "cellflow": "cellflow_2line_uniform_n.csv",
    }
    methods = {name: module.read_method_rows(name, frozen / filename, set(perts)) for name, filename in paths.items()}
    strata = [("all", "all", perts)]
    for field, labels in (("strength_bin", ("weak", "middle", "strong")), ("agreement_bin", ("low", "middle", "high"))):
        strata.extend((field, label, [p for p in perts if by_pert[p][field] == label]) for label in labels)
    strata.extend(("joint_bin", label, [p for p in perts if by_pert[p]["joint_bin"] == label])
                  for label in sorted({row["joint_bin"] for row in descriptors}))
    rng, summaries = np.random.default_rng(20260726), []
    for field, label, members in strata:
        for row_type, selected in (("method_mean", module.METHOD_ORDER), ("oriented_advantage", module.METHOD_ORDER[1:])):
            for method in selected:
                for metric in module.METRICS:
                    values = [float(methods[method][p][metric]) for p in members]
                    if row_type == "oriented_advantage":
                        values = [module.oriented_advantage(v, float(methods["target_control_copy"][p][metric]), metric) for p, v in zip(members, values)]
                    value, lo, hi = module.bootstrap_mean_ci(values, rng, 2000)
                    summaries.append({"stratifier": field, "stratum": label, "n_perts": len(members), "row_type": row_type,
                                      "method": method, "reference": "target_control_copy" if row_type == "oriented_advantage" else "",
                                      "metric": metric, "value": value, "ci_lo": lo, "ci_hi": hi,
                                      "positive_means_method_better": "True" if row_type == "oriented_advantage" else ""})
    error = compare(summaries, read_csv(base / "stratum_summary.csv"), ("stratifier", "stratum", "row_type", "method", "metric"))
    trend_rows = []
    for reference in read_csv(base / "trend_summary.csv"):
        x = [float(by_pert[p][reference["descriptor"]]) for p in perts]
        y = [module.oriented_advantage(float(methods[reference["method"]][p][reference["metric"]]),
                                      float(methods["target_control_copy"][p][reference["metric"]]), reference["metric"]) for p in perts]
        result = spearmanr(x, y)
        trend_rows.append({**reference, "spearman_rho": float(result.statistic), "p_value_advisory": float(result.pvalue)})
    error = max(error, compare(trend_rows, read_csv(base / "trend_summary.csv"), ("descriptor", "method", "reference", "metric")))
    write_csv(output / "source_strata_summary.csv", summaries)
    write_csv(output / "source_strata_trends.csv", trend_rows)
    return {"perturbations": 577, "strata": len(strata), "summary_rows": len(summaries), "max_summary_error": error,
            "descriptor_boundary": "Uses cached source descriptors; raw expression recomputation is a separate command"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/reproduced_validation")
    parser.add_argument("--section", choices=("all", "factorial", "pbmc", "stratification"), default="all")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    functions = {"factorial": factorial, "pbmc": pbmc, "stratification": stratification}
    report = {name: function(args.repo_root, args.output_dir) for name, function in functions.items() if args.section in ("all", name)}
    (args.output_dir / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
