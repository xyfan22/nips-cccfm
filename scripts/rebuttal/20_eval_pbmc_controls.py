"""Balanced leave-one-donor-out control audit on Parse-10M PBMC cytokines.

The script evaluates two diagnostics, not a new predictive model:

* target-control-copy: reuse the held-out donor's PBS cell cloud;
* shifted-control: add the mean cytokine effect estimated from all other donors
  to that same PBS cloud.

Target-donor cytokine cells are used only as evaluation targets.  Selection is
based solely on cell counts.  Raw counts are normalized per cell to 1e4 and
log1p transformed, matching the public CellFlow PBMC tutorial.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import h5py
import numpy as np


DEFAULT_CELL_TYPES = (
    "B Intermediate/Memory",
    "B Naive",
    "CD4 Memory",
    "CD8 Memory",
    "CD14 Mono",
    "NK",
)
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
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--cell_types", nargs="+", default=list(DEFAULT_CELL_TYPES))
    parser.add_argument("--n_cells", type=int, default=50)
    parser.add_argument("--min_cells", type=int, default=50)
    parser.add_argument("--expected_donors", type=int, default=12)
    parser.add_argument("--expected_cytokines", type=int, default=88)
    parser.add_argument("--sample_seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--obs_chunk_size", type=int, default=1_000_000)
    parser.add_argument(
        "--max_cytokines",
        type=int,
        default=0,
        help="Smoke-test only. Final evidence must leave this at 0.",
    )
    return parser.parse_args()


def decode(values: np.ndarray) -> list[str]:
    out = []
    for value in values.tolist():
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        out.append(str(value))
    return out


def categorical(handle: h5py.File, key: str) -> tuple[list[str], np.ndarray]:
    group = handle["obs"][key]
    if "categories" not in group or "codes" not in group:
        raise ValueError(f"obs/{key} is not stored as a categorical column")
    return decode(group["categories"][:]), np.asarray(group["codes"][:], dtype=np.int32)


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or b.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    value = float(np.corrcoef(a, b)[0, 1])
    return value if math.isfinite(value) else 0.0


def centroid_accuracy(pred: np.ndarray, centroids: np.ndarray, true_idx: int) -> float:
    distances = np.linalg.norm(centroids - pred[None, :], axis=1)
    return float(np.sum(distances > distances[true_idx]) / (len(distances) - 1))


def batch_sqdist(a, b):
    import torch

    a2 = (a * a).sum(-1, keepdim=True)
    b2 = (b * b).sum(-1, keepdim=True)
    return torch.clamp(a2 + b2.transpose(1, 2) - 2.0 * torch.bmm(a, b.transpose(1, 2)), min=0.0)


def batch_distribution_metrics(pred, target) -> dict[str, np.ndarray]:
    import torch

    pred = pred.float()
    target = target.float()
    dpp = batch_sqdist(pred, pred)
    dtt = batch_sqdist(target, target)
    dpt = batch_sqdist(pred, target)

    energy = (
        2.0 * torch.sqrt(dpt + 1e-12).mean((1, 2))
        - torch.sqrt(dpp + 1e-12).mean((1, 2))
        - torch.sqrt(dtt + 1e-12).mean((1, 2))
    )
    mmd = torch.zeros_like(energy)
    for bandwidth in (0.1, 1.0, 10.0):
        scale = 2.0 * bandwidth * bandwidth
        mmd += (
            torch.exp(-dpp / scale).mean((1, 2))
            + torch.exp(-dtt / scale).mean((1, 2))
            - 2.0 * torch.exp(-dpt / scale).mean((1, 2))
        )
    mmd /= 3.0

    pred_centered = pred - pred.mean(1, keepdim=True)
    target_centered = target - target.mean(1, keepdim=True)
    dpp_c = batch_sqdist(pred_centered, pred_centered)
    dtt_c = batch_sqdist(target_centered, target_centered)
    dpt_c = batch_sqdist(pred_centered, target_centered)
    centered_energy = (
        2.0 * torch.sqrt(dpt_c + 1e-12).mean((1, 2))
        - torch.sqrt(dpp_c + 1e-12).mean((1, 2))
        - torch.sqrt(dtt_c + 1e-12).mean((1, 2))
    )

    def moments(x):
        mu = x.mean(1)
        centered = x - mu[:, None, :]
        var = (centered * centered).mean(1)
        std = torch.sqrt(var)
        skew = torch.where(
            std > 1e-12,
            (centered * centered * centered).mean(1) / torch.clamp(std**3, min=1e-12),
            torch.zeros_like(std),
        )
        return mu, var, skew

    pred_mu, pred_var, pred_skew = moments(pred)
    true_mu, true_var, true_skew = moments(target)
    moment = (
        torch.abs(pred_mu - true_mu).mean(1)
        + torch.abs(pred_var - true_var).mean(1)
        + torch.abs(pred_skew - true_skew).mean(1)
    )
    return {
        "mmd": mmd.detach().cpu().numpy(),
        "energy": energy.detach().cpu().numpy(),
        "centered_energy": centered_energy.detach().cpu().numpy(),
        "moment": moment.detach().cpu().numpy(),
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.n_cells <= 1 or args.min_cells < args.n_cells:
        raise ValueError("Require 1 < n_cells <= min_cells")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if len(set(args.cell_types)) != len(args.cell_types):
        raise ValueError("Duplicate --cell_types")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    with h5py.File(args.input, "r") as handle:
        donors, donor_codes = categorical(handle, "donor")
        cell_types_all, cell_codes = categorical(handle, "cell_type")
        cytokines_all, cytokine_codes = categorical(handle, "cytokine")
        treatments, treatment_codes = categorical(handle, "treatment")
        n_obs = int(handle["X"].attrs["shape"][0])
        n_vars = int(handle["X"].attrs["shape"][1])

    if len(donors) != args.expected_donors:
        raise ValueError(f"Observed {len(donors)} donors, expected {args.expected_donors}")
    if any(len(codes) != n_obs for codes in (donor_codes, cell_codes, cytokine_codes, treatment_codes)):
        raise ValueError("obs code length does not match X")
    if "PBS" not in cytokines_all or "PBS" not in treatments:
        raise ValueError("PBS control missing")
    missing_cell_types = sorted(set(args.cell_types) - set(cell_types_all))
    if missing_cell_types:
        raise ValueError(f"Missing requested cell types: {missing_cell_types}")

    selected_cell_codes = [cell_types_all.index(name) for name in args.cell_types]
    cell_code_to_local = np.full(len(cell_types_all), -1, dtype=np.int16)
    cell_code_to_local[selected_cell_codes] = np.arange(len(selected_cell_codes), dtype=np.int16)

    valid = (
        (donor_codes >= 0)
        & (cell_codes >= 0)
        & (cytokine_codes >= 0)
        & (treatment_codes >= 0)
    )
    selected_cell_local = cell_code_to_local[np.clip(cell_codes, 0, len(cell_code_to_local) - 1)]
    selected_mask = valid & (selected_cell_local >= 0)
    full_flat = (
        (donor_codes[selected_mask].astype(np.int64) * len(args.cell_types)
         + selected_cell_local[selected_mask].astype(np.int64))
        * len(cytokines_all)
        + cytokine_codes[selected_mask].astype(np.int64)
    )
    counts = np.bincount(
        full_flat,
        minlength=len(donors) * len(args.cell_types) * len(cytokines_all),
    ).reshape(len(donors), len(args.cell_types), len(cytokines_all))
    pbs_full_idx = cytokines_all.index("PBS")
    if int(counts[:, :, pbs_full_idx].min()) < args.min_cells:
        raise ValueError("At least one selected donor/cell-type PBS group is under min_cells")

    eligible_full_idxs = [
        idx
        for idx, name in enumerate(cytokines_all)
        if name != "PBS" and int(counts[:, :, idx].min()) >= args.min_cells
    ]
    if len(eligible_full_idxs) != args.expected_cytokines:
        excluded = {
            cytokines_all[idx]: int(counts[:, :, idx].min())
            for idx in range(len(cytokines_all))
            if cytokines_all[idx] != "PBS" and idx not in eligible_full_idxs
        }
        raise ValueError(
            f"Observed {len(eligible_full_idxs)} eligible cytokines, expected "
            f"{args.expected_cytokines}; excluded={excluded}"
        )
    if args.max_cytokines > 0:
        eligible_full_idxs = eligible_full_idxs[: args.max_cytokines]
    cytokines = [cytokines_all[idx] for idx in eligible_full_idxs]
    selected_full_cytokine_idxs = [pbs_full_idx, *eligible_full_idxs]
    full_cyto_to_local = np.full(len(cytokines_all), -1, dtype=np.int16)
    full_cyto_to_local[selected_full_cytokine_idxs] = np.arange(
        len(selected_full_cytokine_idxs), dtype=np.int16
    )
    selected_cyto_local = full_cyto_to_local[
        np.clip(cytokine_codes, 0, len(full_cyto_to_local) - 1)
    ]
    final_mask = selected_mask & (selected_cyto_local >= 0)

    n_cell_types = len(args.cell_types)
    n_conditions = len(selected_full_cytokine_idxs)
    n_groups = len(donors) * n_cell_types * n_conditions
    group_ids = (
        (donor_codes[final_mask].astype(np.int64) * n_cell_types
         + selected_cell_local[final_mask].astype(np.int64))
        * n_conditions
        + selected_cyto_local[final_mask].astype(np.int64)
    )
    global_rows = np.flatnonzero(final_mask)
    order = np.argsort(group_ids, kind="stable")
    group_ids_sorted = group_ids[order]
    global_rows_sorted = global_rows[order]
    group_counts = np.bincount(group_ids_sorted, minlength=n_groups)
    if np.any(group_counts < args.n_cells):
        bad = np.flatnonzero(group_counts < args.n_cells).tolist()
        raise ValueError(f"Final groups below n_cells: {bad[:20]}")
    offsets = np.concatenate(([0], np.cumsum(group_counts)))

    sampled_rows = []
    sampled_group_ids = []
    for group_id in range(n_groups):
        candidates = global_rows_sorted[offsets[group_id] : offsets[group_id + 1]]
        group_rng = np.random.default_rng(
            np.random.SeedSequence([args.sample_seed, group_id])
        )
        chosen = group_rng.choice(candidates, args.n_cells, replace=False)
        sampled_rows.append(chosen)
        sampled_group_ids.append(np.full(args.n_cells, group_id, dtype=np.int32))
    sampled_rows_arr = np.concatenate(sampled_rows)
    sampled_group_arr = np.concatenate(sampled_group_ids)
    sample_order = np.argsort(sampled_rows_arr)
    sampled_rows_arr = sampled_rows_arr[sample_order]
    sampled_group_arr = sampled_group_arr[sample_order]

    import anndata as ad
    import scipy.sparse as sp
    import torch

    backed = ad.read_h5ad(args.input, backed="r")
    x_sample = backed.X[sampled_rows_arr]
    backed.file.close()
    if not sp.isspmatrix_csr(x_sample):
        x_sample = sp.csr_matrix(x_sample)
    x_sample = x_sample.astype(np.float32, copy=False)
    totals = np.asarray(x_sample.sum(axis=1)).ravel()
    if np.any(totals <= 0):
        raise ValueError("Sampled zero-count cells cannot be normalized")
    x_sample = x_sample.multiply((1e4 / totals)[:, None]).tocsr()
    np.log1p(x_sample.data, out=x_sample.data)

    group_order = np.argsort(sampled_group_arr, kind="stable")
    grouped_ids = sampled_group_arr[group_order]
    if not np.array_equal(
        grouped_ids, np.repeat(np.arange(n_groups, dtype=np.int32), args.n_cells)
    ):
        raise AssertionError("Sampled rows did not form exactly balanced groups")
    x_grouped = x_sample[group_order]
    assignment = sp.csr_matrix(
        (
            np.full(len(sampled_group_arr), 1.0 / args.n_cells, dtype=np.float32),
            (sampled_group_arr, np.arange(len(sampled_group_arr))),
        ),
        shape=(n_groups, len(sampled_group_arr)),
    )
    group_means = (assignment @ x_sample).toarray().astype(np.float32)

    def gid(donor_idx: int, cell_idx: int, condition_idx: int) -> int:
        return (donor_idx * n_cell_types + cell_idx) * n_conditions + condition_idx

    def cloud(group_id: int) -> np.ndarray:
        start = group_id * args.n_cells
        stop = start + args.n_cells
        return x_grouped[start:stop].toarray().astype(np.float32, copy=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    result_rows: list[dict[str, Any]] = []
    for target_donor_idx, target_donor in enumerate(donors):
        source_donor_idxs = [idx for idx in range(len(donors)) if idx != target_donor_idx]
        for cell_idx, cell_type in enumerate(args.cell_types):
            ctrl_group = gid(target_donor_idx, cell_idx, 0)
            ctrl_cloud = cloud(ctrl_group)
            ctrl_mean = group_means[ctrl_group]
            true_clouds = np.stack(
                [cloud(gid(target_donor_idx, cell_idx, cidx)) for cidx in range(1, n_conditions)]
            )
            true_centroids = np.stack(
                [group_means[gid(target_donor_idx, cell_idx, cidx)] for cidx in range(1, n_conditions)]
            )
            pert_reference = true_centroids.mean(axis=0)

            source_deltas = []
            for condition_idx in range(1, n_conditions):
                donor_deltas = [
                    group_means[gid(source_idx, cell_idx, condition_idx)]
                    - group_means[gid(source_idx, cell_idx, 0)]
                    for source_idx in source_donor_idxs
                ]
                source_deltas.append(np.stack(donor_deltas).mean(axis=0))
            source_deltas_arr = np.stack(source_deltas).astype(np.float32)

            ctrl_batch_np = np.broadcast_to(
                ctrl_cloud[None, :, :], true_clouds.shape
            ).copy()
            shifted_batch_np = ctrl_batch_np + source_deltas_arr[:, None, :]
            target_tensor = torch.from_numpy(true_clouds).to(device)
            prediction_batches = {
                "target_control_copy": torch.from_numpy(ctrl_batch_np).to(device),
                "shifted_control": torch.from_numpy(shifted_batch_np).to(device),
            }
            dist_by_method = {
                method: batch_distribution_metrics(pred_tensor, target_tensor)
                for method, pred_tensor in prediction_batches.items()
            }

            profiles = {
                "target_control_copy": np.broadcast_to(
                    ctrl_mean[None, :], true_centroids.shape
                ),
                "shifted_control": ctrl_mean[None, :] + source_deltas_arr,
            }
            for method in METHODS:
                for cyto_idx, cytokine in enumerate(cytokines):
                    pred_profile = profiles[method][cyto_idx]
                    true_profile = true_centroids[cyto_idx]
                    row = {
                        "method": method,
                        "sample_seed": args.sample_seed,
                        "target_donor": target_donor,
                        "cell_type": cell_type,
                        "cytokine": cytokine,
                        "n_cells": args.n_cells,
                        "n_source_donors": len(source_donor_idxs),
                        "pearson_ctrl": safe_pearson(
                            pred_profile - ctrl_mean, true_profile - ctrl_mean
                        ),
                        "pearson_pert": safe_pearson(
                            pred_profile - pert_reference, true_profile - pert_reference
                        ),
                        "mse": float(np.mean((pred_profile - true_profile) ** 2)),
                        "centroid_accuracy": centroid_accuracy(
                            pred_profile, true_centroids, cyto_idx
                        ),
                        "mmd": float(dist_by_method[method]["mmd"][cyto_idx]),
                        "energy": float(dist_by_method[method]["energy"][cyto_idx]),
                        "centered_energy": float(
                            dist_by_method[method]["centered_energy"][cyto_idx]
                        ),
                        "moment": float(dist_by_method[method]["moment"][cyto_idx]),
                        "source_target_delta_pearson": safe_pearson(
                            source_deltas_arr[cyto_idx], true_profile - ctrl_mean
                        ),
                        "source_delta_l2": float(np.linalg.norm(source_deltas_arr[cyto_idx])),
                        "target_delta_l2": float(np.linalg.norm(true_profile - ctrl_mean)),
                    }
                    if not all(math.isfinite(float(row[key])) for key in METRICS):
                        raise ValueError(f"Non-finite metric row: {row}")
                    result_rows.append(row)
            del target_tensor, prediction_batches
            if device.type == "cuda":
                torch.cuda.empty_cache()

    fields = [
        "method",
        "sample_seed",
        "target_donor",
        "cell_type",
        "cytokine",
        "n_cells",
        "n_source_donors",
        *METRICS,
        "source_target_delta_pearson",
        "source_delta_l2",
        "target_delta_l2",
    ]
    write_csv(args.output_dir / "per_condition_metrics.csv", fields, result_rows)

    summary_rows: list[dict[str, Any]] = []
    for method in METHODS:
        method_rows = [row for row in result_rows if row["method"] == method]
        for scope, scope_value, scoped in [
            ("overall", "all", method_rows),
            *[
                (
                    "cell_type",
                    cell_type,
                    [row for row in method_rows if row["cell_type"] == cell_type],
                )
                for cell_type in args.cell_types
            ],
            *[
                (
                    "target_donor",
                    donor,
                    [row for row in method_rows if row["target_donor"] == donor],
                )
                for donor in donors
            ],
        ]:
            summary = {
                "method": method,
                "sample_seed": args.sample_seed,
                "scope": scope,
                "scope_value": scope_value,
                "n_conditions": len(scoped),
            }
            summary.update({metric: mean(float(row[metric]) for row in scoped) for metric in METRICS})
            summary_rows.append(summary)
    write_csv(
        args.output_dir / "summary.csv",
        ["method", "sample_seed", "scope", "scope_value", "n_conditions", *METRICS],
        summary_rows,
    )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": os.uname().nodename,
        "input": str(args.input),
        "input_size_bytes": args.input.stat().st_size,
        "shape": [n_obs, n_vars],
        "normalization": "per-cell normalize_total(1e4) then log1p",
        "target_axis": "leave one donor out within fixed cell type",
        "donors": donors,
        "cell_types": list(args.cell_types),
        "cytokines": cytokines,
        "excluded_for_count_only": [
            name
            for name in cytokines_all
            if name != "PBS" and name not in cytokines
        ],
        "n_cells_per_condition": args.n_cells,
        "sample_seed": args.sample_seed,
        "smoke_only": args.max_cytokines > 0,
        "methods": list(METHODS),
        "metrics": list(METRICS),
        "leakage_rule": (
            "For each target donor, shifted-control uses cytokine-minus-PBS means "
            "from the other 11 donors only; target cytokine cells are evaluation-only."
        ),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    overall = {
        row["method"]: row
        for row in summary_rows
        if row["scope"] == "overall"
    }
    report = [
        "# Parse-10M PBMC balanced control audit",
        "",
        f"- Sample seed: `{args.sample_seed}`",
        f"- Target units: `{len(donors)} donors x {len(args.cell_types)} cell types x {len(cytokines)} cytokines`",
        f"- Cells per observed/predicted cloud: `{args.n_cells}`",
        f"- Smoke-only: `{args.max_cytokines > 0}`",
        "- Target-donor cytokine leakage check: `PASS by construction`",
        "",
        "| method | Pf^ctrl | Pf^pert | Energy | centered Energy | Moment |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = overall[method]
        report.append(
            f"| {method} | {row['pearson_ctrl']:.6f} | {row['pearson_pert']:.6f} | "
            f"{row['energy']:.6f} | {row['centered_energy']:.6f} | {row['moment']:.6f} |"
        )
    report.extend(
        [
            "",
            "This is an external control diagnostic, not a CRISPRi replication and not a model leaderboard.",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
