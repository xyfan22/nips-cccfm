"""Audit and summarize B2 CellFlow results across donor folds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

import numpy as np


METHODS = ("target_control_copy", "shifted_control", "cellflow")
CONTROL_METHODS = ("target_control_copy", "shifted_control")
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
HIGHER_IS_BETTER = {"pearson_ctrl", "pearson_pert", "centroid_accuracy"}
EXPECTED_SEEDS = (7, 42, 314)
EXPECTED_CELL_TYPES = 6
EXPECTED_CYTOKINES = 88
EXPECTED_N_CELLS = 50
HASH_CHUNK_BYTES = 8 * 1024 * 1024
CELLFLOW_COVARIATE_ADAPTER: dict[str, Any] = {
    "name": "cellflow-0.5.3-split-key-order",
    "donor_key": "donor",
    "original_cell_type_key": "cell_type",
    "aliased_cell_type_key": "z_b2_cell_type",
    "split_covariates": ["donor", "z_b2_cell_type"],
    "changes_values": False,
}
FROZEN_ARCHITECTURE: dict[str, Any] = {
    "solver": "otfm",
    "condition_mode": "deterministic",
    "regularization": 0.0,
    "pooling": "attention_token",
    "pooling_kwargs": {},
    "layers_before_pool": {
        "cytokine_treatment": {
            "layer_type": "mlp",
            "dims": [1024, 1024],
            "dropout_rate": 0.5,
        },
        "donor": {
            "layer_type": "mlp",
            "dims": [256, 256],
            "dropout_rate": 0.0,
        },
    },
    "layers_after_pool": {
        "layer_type": "mlp",
        "dims": [1024, 1024],
        "dropout_rate": 0.0,
    },
    "condition_embedding_dim": 256,
    "cond_output_dropout": 0.9,
    "condition_encoder_kwargs": {},
    "pool_sample_covariates": True,
    "time_freqs": 1024,
    "time_encoder_dims": [1024, 1024, 1024],
    "time_encoder_dropout": 0.0,
    "hidden_dims": [2048, 2048, 2048],
    "hidden_dropout": 0.0,
    "conditioning": "concatenation",
    "conditioning_kwargs": {},
    "decoder_dims": [4096, 4096, 4096],
    "decoder_dropout": 0.0,
    "activation": "silu",
    "vf_kwargs": None,
    "probability_path": {"constant_noise": 0.5},
    "match_epsilon": 0.5,
    "match_tau_a": 1.0,
    "match_tau_b": 1.0,
    "optimizer": "MultiSteps(Adam(5e-5), every_k=20)",
    "solver_kwargs": {},
    "layer_norm_before_concatenation": False,
    "linear_projection_before_concatenation": False,
    "iterations": 100_000,
    "model_seed": 42,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", type=Path, required=True)
    parser.add_argument("--b1_controls_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--donors", nargs="+", default=[f"Donor{index}" for index in range(1, 13)]
    )
    parser.add_argument("--control_tolerance", type=float, default=5e-4)
    parser.add_argument(
        "--allow_nonfrozen_fixture", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("--expected_cell_types", type=int, default=EXPECTED_CELL_TYPES)
    parser.add_argument("--expected_cytokines", type=int, default=EXPECTED_CYTOKINES)
    parser.add_argument("--expected_n_cells", type=int, default=EXPECTED_N_CELLS)
    parser.add_argument(
        "--expected_seeds", type=int, nargs="+", default=list(EXPECTED_SEEDS)
    )
    return parser.parse_args()


def hash_file(path: Path, algorithms: Iterable[str] = ("sha256",)) -> dict[str, str]:
    hashers = {name: hashlib.new(name) for name in algorithms}
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            for hasher in hashers.values():
                hasher.update(chunk)
    return {name: hasher.hexdigest() for name, hasher in hashers.items()}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def row_key(row: dict[str, Any]) -> tuple[int, str, str, str]:
    return (
        int(row["sample_seed"]),
        str(row["target_donor"]),
        str(row["cell_type"]),
        str(row["cytokine"]),
    )


def _validate_nonfrozen_args(args: argparse.Namespace) -> None:
    frozen = (
        tuple(args.expected_seeds) == EXPECTED_SEEDS
        and args.expected_cell_types == EXPECTED_CELL_TYPES
        and args.expected_cytokines == EXPECTED_CYTOKINES
        and args.expected_n_cells == EXPECTED_N_CELLS
    )
    if not frozen and not args.allow_nonfrozen_fixture:
        raise ValueError("Non-frozen expected counts require --allow_nonfrozen_fixture")
    if len(set(args.expected_seeds)) != len(args.expected_seeds):
        raise ValueError("Expected sampling seeds must be unique")
    if not args.expected_seeds:
        raise ValueError("At least one sampling seed is required")
    if args.control_tolerance < 0:
        raise ValueError("Control tolerance must be non-negative")


def load_and_audit_fold(
    fold_dir: Path,
    donor: str,
    expected_seeds: tuple[int, ...],
    expected_cell_types: int,
    expected_cytokines: int,
    expected_n_cells: int,
    allow_nonfrozen_fixture: bool = False,
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, str]]:
    required_paths = {
        "checkpoint": fold_dir / "CellFlow.pkl",
        "pca": fold_dir / "pca.npz",
        "config": fold_dir / "config.json",
        "pca_audit": fold_dir / "pca_audit.json",
        "training_summary": fold_dir / "training_summary.json",
        "training_loss": fold_dir / "training_loss.csv",
        "training_audit": fold_dir / "training_audit.json",
        "fold_audit": fold_dir / "fold_audit.json",
        "metrics": fold_dir / "evaluation" / "per_condition_metrics.csv",
        "summary": fold_dir / "evaluation" / "summary.csv",
        "evaluation_audit": fold_dir / "evaluation" / "evaluation_audit.json",
    }
    missing = [str(path) for path in required_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{donor}: missing fold artifacts: {missing}")

    config = json.loads(required_paths["config"].read_text())
    pca_audit = json.loads(required_paths["pca_audit"].read_text())
    training_summary = json.loads(required_paths["training_summary"].read_text())
    training_audit = json.loads(required_paths["training_audit"].read_text())
    fold_audit = json.loads(required_paths["fold_audit"].read_text())
    evaluation_audit = json.loads(required_paths["evaluation_audit"].read_text())

    if config.get("target_donor") != donor:
        raise ValueError(f"{donor}: config target mismatch")
    if tuple(config.get("sample_seeds", [])) != expected_seeds:
        raise ValueError(f"{donor}: sample seeds differ")
    cell_types = list(config.get("cell_types", []))
    cytokines = list(config.get("cytokines", []))
    if len(cell_types) != expected_cell_types or len(set(cell_types)) != len(
        cell_types
    ):
        raise ValueError(f"{donor}: cell-type set differs")
    if len(cytokines) != expected_cytokines or len(set(cytokines)) != len(cytokines):
        raise ValueError(f"{donor}: cytokine set differs")
    if int(config.get("n_cells_per_group", -1)) != expected_n_cells:
        raise ValueError(f"{donor}: n_cells differs")
    source_donors = list(config.get("source_donors", []))
    if (
        donor in source_donors
        or not source_donors
        or len(set(source_donors)) != len(source_donors)
    ):
        raise ValueError(f"{donor}: source-donor list is invalid")
    pca_config = config.get("pca", {})
    architecture = config.get("architecture", {})
    if config.get("cellflow_covariate_adapter") != CELLFLOW_COVARIATE_ADAPTER:
        raise ValueError(f"{donor}: CellFlow covariate adapter differs")
    if not allow_nonfrozen_fixture:
        if pca_config != {
            "n_components": 100,
            "solver": "randomized",
            "random_state": 42,
        }:
            raise ValueError(f"{donor}: PCA configuration differs")
        if architecture != FROZEN_ARCHITECTURE:
            raise ValueError(f"{donor}: architecture differs from the frozen protocol")
        if len(source_donors) != 11:
            raise ValueError(f"{donor}: expected 11 source donors")
    else:
        if (
            int(pca_config.get("n_components", -1)) < 1
            or pca_config.get("solver") != "randomized"
            or int(pca_config.get("random_state", -1)) < 0
        ):
            raise ValueError(f"{donor}: invalid fixture PCA configuration")
        if (
            int(architecture.get("iterations", -1)) < 1
            or int(architecture.get("model_seed", -1)) < 0
        ):
            raise ValueError(f"{donor}: invalid fixture training configuration")
    if int(config.get("gene_dim", -1)) < 1:
        raise ValueError(f"{donor}: invalid gene dimension")
    if int(config.get("batch_size", -1)) not in (1024, 512, 256, 128):
        raise ValueError(f"{donor}: invalid batch size")
    resource_probe = config.get("resource_probe", {})
    if (
        resource_probe.get("target_donor") != "Donor1"
        or int(resource_probe.get("selected_batch_size", -1))
        != int(config["batch_size"])
        or len(str(resource_probe.get("sha256", ""))) != 64
    ):
        raise ValueError(f"{donor}: resource-probe provenance differs")
    if not allow_nonfrozen_fixture:
        probe_path = Path(str(resource_probe.get("path", "")))
        if not probe_path.is_file():
            raise FileNotFoundError(f"{donor}: resource-probe result is unavailable")
        if hash_file(probe_path)["sha256"] != resource_probe["sha256"]:
            raise ValueError(f"{donor}: resource-probe result hash differs")
    if len(config.get("jax_devices", [])) != 1:
        raise ValueError(f"{donor}: training did not record exactly one JAX device")

    expected_fit_rows = (
        len(source_donors)
        * expected_cell_types
        * (expected_cytokines + 1)
        * expected_n_cells
    )
    expected_target_rows = (
        expected_cell_types * (expected_cytokines + 1) * expected_n_cells
    )
    expected_embedding_rows = expected_cell_types * expected_n_cells
    if (
        pca_audit.get("target_donor") != donor
        or pca_audit.get("source_donors") != source_donors
        or int(pca_audit.get("target_rows_in_fit", -1)) != 0
        or int(pca_audit.get("fit_row_count", -1)) != expected_fit_rows
        or int(pca_audit.get("target_rows_available", -1)) != expected_target_rows
        or int(pca_audit.get("n_components", -1)) != int(pca_config["n_components"])
        or pca_audit.get("solver") != "randomized"
        or int(pca_audit.get("random_state", -1)) != int(pca_config["random_state"])
    ):
        raise ValueError(f"{donor}: PCA leakage/configuration audit failed")
    target_embedding_digest = pca_audit.get(
        "target_donor_embedding_row_digest_sha256", ""
    )
    if (
        len(str(pca_audit.get("fit_source_row_digest_sha256", ""))) != 64
        or len(str(pca_audit.get("target_source_row_digest_sha256", ""))) != 64
    ):
        raise ValueError(f"{donor}: PCA row digests are missing")
    if (
        pca_audit.get("target_donor_embedding_source") != "seed-42 PBS controls only"
        or int(pca_audit.get("target_donor_embedding_row_count", -1))
        != expected_embedding_rows
        or len(target_embedding_digest) != 64
    ):
        raise ValueError(f"{donor}: target donor embedding audit failed")
    if (
        int(training_audit.get("target_rows_in_pca_fit", -1)) != 0
        or int(training_audit.get("target_rows_in_training", -1)) != 0
        or training_audit.get("target_donor_embedding_registered_in_training")
        is not False
        or int(training_audit.get("train_rows", -1)) != expected_fit_rows
        or int(training_audit.get("checkpoint_size_bytes", -1))
        != required_paths["checkpoint"].stat().st_size
        or int(training_audit.get("pca_size_bytes", -1))
        != required_paths["pca"].stat().st_size
        or training_audit.get("target_donor_embedding_row_digest_sha256")
        != target_embedding_digest
    ):
        raise ValueError(f"{donor}: training leakage audit failed")

    expected_iterations = int(architecture.get("iterations", -1))
    if (
        int(training_summary.get("iterations", -1)) != expected_iterations
        or int(training_summary.get("batch_size", -1)) != int(config["batch_size"])
        or int(training_summary.get("model_seed", -1))
        != int(architecture.get("model_seed", -1))
        or float(training_summary.get("wall_seconds", -1)) < 0
        or not training_summary.get("all_losses_finite", False)
        or not training_summary.get("loss_decreased", False)
    ):
        raise ValueError(f"{donor}: training summary gate failed")
    loss_rows = read_csv(required_paths["training_loss"])
    if len(loss_rows) != expected_iterations:
        raise ValueError(f"{donor}: training-loss row count differs")
    losses: list[float] = []
    for expected_iteration, row in enumerate(loss_rows, start=1):
        if int(row["iteration"]) != expected_iteration:
            raise ValueError(f"{donor}: training-loss iterations are not contiguous")
        value = float(row["loss"])
        if not math.isfinite(value):
            raise ValueError(f"{donor}: non-finite training loss")
        losses.append(value)
    window = max(1, expected_iterations // 10)
    first_loss = mean(losses[:window])
    last_loss = mean(losses[-window:])
    if (
        abs(first_loss - float(training_summary["first_10pct_mean_loss"])) > 1e-10
        or abs(last_loss - float(training_summary["last_10pct_mean_loss"])) > 1e-10
        or bool(last_loss < first_loss) != bool(training_summary["loss_decreased"])
    ):
        raise ValueError(f"{donor}: training-loss summary is inconsistent")

    artifact_names = {
        "checkpoint_sha256": "checkpoint",
        "pca_sha256": "pca",
        "pca_audit_sha256": "pca_audit",
        "config_sha256": "config",
        "training_summary_sha256": "training_summary",
        "training_loss_sha256": "training_loss",
    }
    artifact_hashes: dict[str, str] = {}
    for audit_key, path_key in artifact_names.items():
        observed = hash_file(required_paths[path_key])["sha256"]
        if observed != training_audit.get(audit_key):
            raise ValueError(f"{donor}: artifact hash mismatch for {path_key}")
        artifact_hashes[audit_key] = observed

    with np.load(required_paths["pca"]) as pca_values:
        required_arrays = {
            "mean",
            "components",
            "explained_variance",
            "explained_variance_ratio",
        }
        if set(pca_values.files) != required_arrays:
            raise ValueError(f"{donor}: PCA artifact fields differ")
        gene_dim = int(config.get("gene_dim", -1))
        n_components = int(pca_config["n_components"])
        expected_shapes = {
            "mean": (gene_dim,),
            "components": (n_components, gene_dim),
            "explained_variance": (n_components,),
            "explained_variance_ratio": (n_components,),
        }
        for name, shape in expected_shapes.items():
            values = pca_values[name]
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(f"{donor}: invalid PCA array {name}")

    rows = read_csv(required_paths["metrics"])
    expected_per_method = len(expected_seeds) * expected_cell_types * expected_cytokines
    expected_total = expected_per_method * len(METHODS)
    if len(rows) != expected_total:
        raise ValueError(f"{donor}: {len(rows)} rows, expected {expected_total}")
    expected_keys = {
        (seed, donor, cell_type, cytokine)
        for seed in expected_seeds
        for cell_type in cell_types
        for cytokine in cytokines
    }
    reference_keys: set[tuple[int, str, str, str]] | None = None
    for method in METHODS:
        method_rows = [row for row in rows if row.get("method") == method]
        keys = {row_key(row) for row in method_rows}
        if (
            len(method_rows) != expected_per_method
            or len(keys) != expected_per_method
            or keys != expected_keys
        ):
            raise ValueError(f"{donor}/{method}: incomplete, duplicate, or wrong keys")
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise ValueError(f"{donor}: method key sets differ")
        for row in method_rows:
            if int(row["n_cells"]) != expected_n_cells:
                raise ValueError(f"{donor}: row violates uniform n")
            if int(row["n_source_donors"]) != len(source_donors):
                raise ValueError(f"{donor}: row source-donor count differs")
            expected_condition_id = (
                f"seed{int(row['sample_seed'])}|{donor}|"
                f"{row['cell_type']}|{row['cytokine']}"
            )
            if row.get("condition_id") != expected_condition_id:
                raise ValueError(f"{donor}: invalid condition_id")
            for metric in METRICS:
                value = float(row[metric])
                if not math.isfinite(value):
                    raise ValueError(f"{donor}/{method}: non-finite {metric}")
    if set(row["method"] for row in rows) != set(METHODS):
        raise ValueError(f"{donor}: unexpected method set")

    summary_rows = read_csv(required_paths["summary"])
    if len(summary_rows) != len(METHODS) or {
        row["method"] for row in summary_rows
    } != set(METHODS):
        raise ValueError(f"{donor}: evaluation summary method set differs")
    for summary_row in summary_rows:
        method = summary_row["method"]
        method_rows = [row for row in rows if row["method"] == method]
        if int(summary_row["n_rows"]) != expected_per_method:
            raise ValueError(f"{donor}/{method}: summary row count differs")
        for metric in METRICS:
            recomputed = mean(float(row[metric]) for row in method_rows)
            if abs(recomputed - float(summary_row[metric])) > 1e-10:
                raise ValueError(f"{donor}/{method}: summary differs for {metric}")

    if (
        evaluation_audit.get("target_donor") != donor
        or evaluation_audit.get("methods") != list(METHODS)
        or int(evaluation_audit.get("n_rows", -1)) != expected_total
        or int(evaluation_audit.get("expected_rows", -1)) != expected_total
        or int(evaluation_audit.get("uniform_n", -1)) != expected_n_cells
        or not evaluation_audit.get("method_key_sets_equal", False)
        or not evaluation_audit.get("all_metrics_finite", False)
    ):
        raise ValueError(f"{donor}: evaluation audit failed")
    evaluation_hashes = {
        "metrics_sha256": hash_file(required_paths["metrics"])["sha256"],
        "evaluation_audit_sha256": hash_file(required_paths["evaluation_audit"])[
            "sha256"
        ],
        "evaluation_summary_sha256": hash_file(required_paths["summary"])["sha256"],
    }
    if not fold_audit.get("evaluation_complete", False):
        raise ValueError(f"{donor}: fold audit says evaluation incomplete")
    if float(fold_audit.get("evaluation_wall_seconds", -1)) < 0:
        raise ValueError(f"{donor}: evaluation wall time is missing")
    for key, observed in evaluation_hashes.items():
        if fold_audit.get(key) != observed:
            raise ValueError(f"{donor}: fold audit hash mismatch for {key}")
    if fold_audit.get("checkpoint_sha256") != artifact_hashes["checkpoint_sha256"]:
        raise ValueError(f"{donor}: fold/checkpoint audit mismatch")

    signature = {
        "sample_seeds": config["sample_seeds"],
        "cell_types": cell_types,
        "cytokines": cytokines,
        "n_cells_per_group": config["n_cells_per_group"],
        "gene_dim": config["gene_dim"],
        "pca": pca_config,
        "architecture": architecture,
        "cellflow_covariate_adapter": config["cellflow_covariate_adapter"],
        "batch_size": config["batch_size"],
        "resource_probe_sha256": resource_probe["sha256"],
        "resource_probe_target_donor": resource_probe["target_donor"],
        "balanced_manifest_sha256": fold_audit.get("balanced_manifest_sha256"),
    }
    if len(str(signature["balanced_manifest_sha256"])) != 64:
        raise ValueError(f"{donor}: balanced-manifest hash is missing")
    return rows, signature, {**artifact_hashes, **evaluation_hashes}


def load_b1_controls(
    root: Path,
    donors: set[str],
    expected_seeds: tuple[int, ...],
) -> dict[tuple[str, int, str, str, str], dict[str, str]]:
    mapping: dict[tuple[str, int, str, str, str], dict[str, str]] = {}
    for seed in expected_seeds:
        path = root / f"seed{seed}" / "per_condition_metrics.csv"
        for row in read_csv(path):
            if (
                row["method"] not in CONTROL_METHODS
                or row["target_donor"] not in donors
            ):
                continue
            key = (
                row["method"],
                int(row["sample_seed"]),
                row["target_donor"],
                row["cell_type"],
                row["cytokine"],
            )
            if key in mapping:
                raise ValueError(f"Duplicate B1 control key: {key}")
            mapping[key] = row
    return mapping


def audit_controls_against_b1(
    rows: list[dict[str, str]],
    b1: dict[tuple[str, int, str, str, str], dict[str, str]],
    tolerance: float,
) -> dict[str, Any]:
    expected_keys = {
        (
            row["method"],
            int(row["sample_seed"]),
            row["target_donor"],
            row["cell_type"],
            row["cytokine"],
        )
        for row in rows
        if row["method"] in CONTROL_METHODS
    }
    if expected_keys != set(b1):
        missing = sorted(expected_keys - set(b1))[:10]
        extra = sorted(set(b1) - expected_keys)[:10]
        raise ValueError(f"B1 control key mismatch; missing={missing}, extra={extra}")
    max_by_metric = {metric: 0.0 for metric in METRICS}
    for row in rows:
        if row["method"] not in CONTROL_METHODS:
            continue
        key = (
            row["method"],
            int(row["sample_seed"]),
            row["target_donor"],
            row["cell_type"],
            row["cytokine"],
        )
        reference = b1[key]
        for metric in METRICS:
            difference = abs(float(row[metric]) - float(reference[metric]))
            max_by_metric[metric] = max(max_by_metric[metric], difference)
            if difference > tolerance:
                raise ValueError(
                    f"Control mismatch {key}/{metric}: diff={difference:.8g} > {tolerance}"
                )
    return {
        "n_control_keys": len(expected_keys),
        "tolerance": tolerance,
        "max_absolute_difference_by_metric": max_by_metric,
    }


def aggregate_rows(
    rows: list[dict[str, str]],
    donors: list[str],
    expected_seeds: tuple[int, ...],
) -> dict[str, list[dict[str, Any]]]:
    condition_groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            row["target_donor"],
            row["cell_type"],
            row["cytokine"],
            row["method"],
        )
        condition_groups.setdefault(key, []).append(row)

    condition_means: list[dict[str, Any]] = []
    for (donor, cell_type, cytokine, method), group in sorted(condition_groups.items()):
        seeds = {int(row["sample_seed"]) for row in group}
        if seeds != set(expected_seeds) or len(group) != len(expected_seeds):
            raise ValueError("Condition group has missing or duplicate seed rows")
        output: dict[str, Any] = {
            "target_donor": donor,
            "cell_type": cell_type,
            "cytokine": cytokine,
            "method": method,
            "n_sampling_seeds": len(group),
        }
        for metric in METRICS:
            output[metric] = mean(float(row[metric]) for row in group)
        condition_means.append(output)

    donor_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in condition_means:
        donor_groups.setdefault((row["target_donor"], row["method"]), []).append(row)
    donor_summary: list[dict[str, Any]] = []
    for donor in donors:
        for method in METHODS:
            group = donor_groups.get((donor, method), [])
            if not group:
                raise ValueError(f"Missing donor aggregate for {donor}/{method}")
            output = {
                "target_donor": donor,
                "method": method,
                "n_conditions": len(group),
            }
            for metric in METRICS:
                output[metric] = mean(float(row[metric]) for row in group)
            donor_summary.append(output)

    global_summary: list[dict[str, Any]] = []
    for method in METHODS:
        group = [row for row in donor_summary if row["method"] == method]
        if len(group) != len(donors):
            raise ValueError(f"Global donor count differs for {method}")
        output: dict[str, Any] = {"method": method, "n_donors": len(group)}
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            output[f"{metric}_mean"] = mean(values)
            output[f"{metric}_sd_across_donors"] = (
                stdev(values) if len(values) > 1 else 0.0
            )
        global_summary.append(output)

    donor_lookup = {(row["target_donor"], row["method"]): row for row in donor_summary}
    paired_differences: list[dict[str, Any]] = []
    for donor in donors:
        cellflow = donor_lookup[(donor, "cellflow")]
        for comparator in CONTROL_METHODS:
            control = donor_lookup[(donor, comparator)]
            for metric in METRICS:
                raw = float(cellflow[metric]) - float(control[metric])
                oriented = raw if metric in HIGHER_IS_BETTER else -raw
                paired_differences.append(
                    {
                        "target_donor": donor,
                        "comparator": comparator,
                        "metric": metric,
                        "cellflow_value": cellflow[metric],
                        "control_value": control[metric],
                        "cellflow_minus_control": raw,
                        "oriented_cellflow_improvement": oriented,
                    }
                )

    paired_summary: list[dict[str, Any]] = []
    for comparator in CONTROL_METHODS:
        for metric in METRICS:
            group = [
                row
                for row in paired_differences
                if row["comparator"] == comparator and row["metric"] == metric
            ]
            values = [float(row["oriented_cellflow_improvement"]) for row in group]
            paired_summary.append(
                {
                    "comparator": comparator,
                    "metric": metric,
                    "n_donors": len(group),
                    "mean_oriented_cellflow_improvement": mean(values),
                    "sd_oriented_cellflow_improvement": (
                        stdev(values) if len(values) > 1 else 0.0
                    ),
                    "cellflow_wins": sum(value > 0 for value in values),
                    "control_wins": sum(value < 0 for value in values),
                    "ties": sum(value == 0 for value in values),
                }
            )
    return {
        "condition_means": condition_means,
        "donor_summary": donor_summary,
        "global_summary": global_summary,
        "paired_differences": paired_differences,
        "paired_summary": paired_summary,
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    _validate_nonfrozen_args(args)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if len(set(args.donors)) != len(args.donors):
        raise ValueError("Duplicate donor arguments")
    expected_seeds = tuple(args.expected_seeds)
    all_rows: list[dict[str, str]] = []
    reference_signature: dict[str, Any] | None = None
    fold_hashes: dict[str, dict[str, str]] = {}
    for donor in args.donors:
        fold_dir = args.result_root / donor
        rows, signature, hashes = load_and_audit_fold(
            fold_dir,
            donor,
            expected_seeds,
            args.expected_cell_types,
            args.expected_cytokines,
            args.expected_n_cells,
            args.allow_nonfrozen_fixture,
        )
        if reference_signature is None:
            reference_signature = signature
        elif signature != reference_signature:
            raise ValueError(f"{donor}: frozen fold configuration differs")
        all_rows.extend(rows)
        fold_hashes[donor] = hashes

    expected_total = (
        len(args.donors)
        * len(METHODS)
        * len(expected_seeds)
        * args.expected_cell_types
        * args.expected_cytokines
    )
    if len(all_rows) != expected_total:
        raise ValueError(f"Combined rows {len(all_rows)} != {expected_total}")
    b1 = load_b1_controls(args.b1_controls_dir, set(args.donors), expected_seeds)
    control_audit = audit_controls_against_b1(all_rows, b1, args.control_tolerance)
    aggregates = aggregate_rows(all_rows, args.donors, expected_seeds)

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{args.output_dir.name}.staging-", dir=args.output_dir.parent
        )
    )
    try:
        combined_fields = list(all_rows[0].keys())
        write_csv(stage / "per_condition_combined.csv", combined_fields, all_rows)
        condition_fields = [
            "target_donor",
            "cell_type",
            "cytokine",
            "method",
            "n_sampling_seeds",
            *METRICS,
        ]
        write_csv(
            stage / "condition_means.csv",
            condition_fields,
            aggregates["condition_means"],
        )
        donor_fields = ["target_donor", "method", "n_conditions", *METRICS]
        write_csv(
            stage / "donor_summary.csv", donor_fields, aggregates["donor_summary"]
        )
        global_fields = ["method", "n_donors"]
        for metric in METRICS:
            global_fields.extend((f"{metric}_mean", f"{metric}_sd_across_donors"))
        write_csv(
            stage / "global_summary.csv", global_fields, aggregates["global_summary"]
        )
        paired_fields = [
            "target_donor",
            "comparator",
            "metric",
            "cellflow_value",
            "control_value",
            "cellflow_minus_control",
            "oriented_cellflow_improvement",
        ]
        write_csv(
            stage / "paired_donor_differences.csv",
            paired_fields,
            aggregates["paired_differences"],
        )
        paired_summary_fields = [
            "comparator",
            "metric",
            "n_donors",
            "mean_oriented_cellflow_improvement",
            "sd_oriented_cellflow_improvement",
            "cellflow_wins",
            "control_wins",
            "ties",
        ]
        write_csv(
            stage / "paired_donor_summary.csv",
            paired_summary_fields,
            aggregates["paired_summary"],
        )
        audit = {
            "status": "AUDIT_PASS",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "donors": args.donors,
            "n_donors": len(args.donors),
            "methods": list(METHODS),
            "sample_seeds": list(expected_seeds),
            "n_rows_total": len(all_rows),
            "expected_rows_total": expected_total,
            "uniform_n": args.expected_n_cells,
            "configuration_identical_across_folds": True,
            "control_audit": control_audit,
            "fold_hashes": fold_hashes,
        }
        write_json(stage / "audit.json", audit)
        report = [
            "# B2 PBMC CellFlow donor-transfer summary",
            "",
            f"- Donor folds: `{len(args.donors)}`",
            f"- Per-condition rows: `{len(all_rows)}`",
            "- Unit for final variability: donor means",
            "- B1 control reproduction: `PASS`",
            "- Leakage/configuration/completeness audit: `PASS`",
            "",
            "| method | Pf-pert | Energy | centered Energy | Moment |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in aggregates["global_summary"]:
            report.append(
                f"| {row['method']} | "
                f"{row['pearson_pert_mean']:.6f} +/- {row['pearson_pert_sd_across_donors']:.6f} | "
                f"{row['energy_mean']:.6f} +/- {row['energy_sd_across_donors']:.6f} | "
                f"{row['centered_energy_mean']:.6f} +/- {row['centered_energy_sd_across_donors']:.6f} | "
                f"{row['moment_mean']:.6f} +/- {row['moment_sd_across_donors']:.6f} |"
            )
        report.extend(
            [
                "",
                "Values are equal-weight donor means +/- sample SD across donors. ",
                "This is a primary-PBMC cytokine donor-transfer result, not a CRISPRi replication.",
            ]
        )
        (stage / "report.md").write_text("\n".join(report) + "\n")
        stage.rename(args.output_dir)
        return audit
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    args = parse_args()
    audit = summarize(args)
    print("Status: AUDIT_PASS")
    print(f"Rows: {audit['n_rows_total']}")
    print(f"Donors: {audit['n_donors']}")


if __name__ == "__main__":
    main()
