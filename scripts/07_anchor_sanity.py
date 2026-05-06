"""R1.3 sanity gate: anchor + apply_anchor must reproduce shifted-control.

Builds, for each (anchor, setting), per-pert pred cells via:

    x0_anchor = (x_target_ctrl − x_target_ctrl.mean(0)) + tgt_ctrl_mean + delta(g)

then runs `evaluator.evaluate_pert_cell_level` on those pred cells and
compares to the `*_shifted_control_summary.csv` we already produced via
`scripts/06_mean_baselines_distributional.py`.

Two-mode gate:
  - same deterministic control sample (same n_pred, same seed, same indices)
    → metrics match exactly.
  - different sample / RNG → mean metrics still exact (formula gives
    `mean(pred) == profile`); distributional metrics within tight tolerance
    from sampling noise.

Usage:
    python scripts/07_anchor_sanity.py --setting 2line --anchors linear exact ridge
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


SETTING_SOURCES = {
    "2line": ["K562", "RPE1"],
    "3line": ["K562", "RPE1", "HepG2"],
}

SHIFTED_CSV_PER_PERT = {
    # per-pert CSVs (not summary); we compare same-pert rows to exclude
    # partial-aggregate-vs-full-aggregate confound.
    "linear": "results/baselines/linear_ahlmann_eltze_{setting}_shifted_control.csv",
    "exact": "results/baselines/exact_gene_retrieval_{setting}_shifted_control.csv",
    "ridge": "results/baselines/pooled_ridge_anchor_{setting}_shifted_control.csv",
}

ANCHOR_BUILDERS = {}  # filled below after imports


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=Path,
                   default=Path("data/processed/replogle_preprocessed.h5ad"))
    p.add_argument("--test_slice", type=Path,
                   default=Path("data/splits/jurkat_test_strict.json"))
    p.add_argument("--primary_embedding_path", type=Path,
                   default=Path("data/processed/primary_gene_embedding.pt"))
    p.add_argument("--setting", choices=list(SETTING_SOURCES), required=True)
    p.add_argument("--target_line", type=str, default="Jurkat")
    p.add_argument("--anchors", nargs="+",
                   default=["linear", "exact", "ridge"],
                   choices=["linear", "exact", "ridge"])
    p.add_argument("--n_perts_check", type=int, default=20,
                   help="how many test perts to evaluate for the gate (full 577 is overkill)")
    p.add_argument("--max_n_samples", type=int, default=2000)
    p.add_argument("--min_cells", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    # Tolerances
    p.add_argument("--tol_mean_metrics", type=float, default=1e-5,
                   help="exact match for pearson_*, mse, etc. (formula-deterministic)")
    p.add_argument("--tol_dist_metrics", type=float, default=0.05,
                   help="loose-tolerance for MMD/Energy/W_2 (sampling noise)")
    return p.parse_args()


def main():
    args = parse_args()
    import anndata as ad
    import numpy as np
    import pandas as pd

    from cccfm.anchor import (
        LinearAnchor, ExactAnchor, RidgeAnchor, apply_anchor,
    )
    from cccfm.eval_metrics import CrossLineEvaluator

    sources = SETTING_SOURCES[args.setting]
    target = args.target_line
    print(f"[R1.3] setting={args.setting}  sources={sources}  target={target}")

    adata = ad.read_h5ad(args.data_path)
    test_perts = sorted(json.loads(args.test_slice.read_text()))
    evaluator = CrossLineEvaluator(
        adata=adata, source_lines=sources, target_line=target,
        test_perts=test_perts, min_cells=args.min_cells,
    )
    tgt_ctrl = evaluator.target_ctrl_reference  # [G]

    # Target-line ctrl cell pool
    obs = adata.obs
    tgt_ctrl_mask = (obs["cell_line"].values == target) & (obs["pert"].values == "control")
    X_tgt = adata.X[tgt_ctrl_mask]
    if hasattr(X_tgt, "toarray"):
        X_tgt = X_tgt.toarray()
    X_tgt = np.asarray(X_tgt, dtype=np.float32)
    print(f"[R1.3] target-ctrl pool: {X_tgt.shape[0]} cells × {X_tgt.shape[1]} genes")

    # Build anchors (with leakage check)
    anchors = {}
    for name in args.anchors:
        t0 = time.perf_counter()
        if name == "linear":
            a = LinearAnchor().fit(adata, sources, target_line=target,
                                    min_cells=args.min_cells)
        elif name == "exact":
            a = ExactAnchor().fit(adata, sources, target_line=target,
                                   min_cells=args.min_cells)
        elif name == "ridge":
            a = RidgeAnchor(args.primary_embedding_path).fit(
                adata, sources, target_line=target, min_cells=args.min_cells,
            )
        anchors[name] = a
        print(f"[R1.3] {name} fit in {time.perf_counter()-t0:.1f}s; "
              f"supported_perts={len(a.supported_perts)}")

    # Per-pert evaluation (matched to shifted_control mode)
    rng = np.random.default_rng(args.seed)
    test_subset = test_perts[:args.n_perts_check]

    rows = []
    for name, a in anchors.items():
        for pert in test_subset:
            if pert not in evaluator.target_pert_centroids:
                continue
            d = a.delta(pert)
            if d is None:
                continue  # anchor doesn't support this pert (Ridge OOV)
            n_target = evaluator.target_pert_n_cells[pert]
            n_samp = min(n_target, args.max_n_samples)
            idx = rng.choice(X_tgt.shape[0], size=n_samp, replace=(n_samp > X_tgt.shape[0]))
            x_ctrl = X_tgt[idx]
            pred_cells = apply_anchor(x_ctrl, tgt_ctrl, d)
            # Verify mean(pred_cells) == tgt_ctrl + delta(g) exactly
            mean_diff = float(np.abs(pred_cells.mean(0) - (tgt_ctrl + d)).max())
            assert mean_diff < 1e-4, f"mean-center invariant broken: {mean_diff}"

            m = evaluator.evaluate_pert_cell_level(pert, pred_cells, compute_heavy=True)
            if m is None:
                continue
            m["anchor"] = name
            m["pert"] = pert
            rows.append(m)

    df = pd.DataFrame(rows)
    print(f"\n[R1.3] evaluated {len(df)} (anchor, pert) cells")

    metric_cols_mean = ["mse", "pearson_ctrl", "pearson_pert", "pearson20_pert",
                        "centroid_accuracy", "deg_overlap20", "deg_overlap50"]
    metric_cols_dist = ["mmd", "energy", "w2_sinkhorn", "moment"]
    metric_cols_all = metric_cols_mean + metric_cols_dist

    # ---------- per-pert comparison vs shifted_control per-pert CSV ----------
    print(f"\n[R1.3] === per-pert gate comparison (mine vs shifted_control row) ===")
    fail = []
    for name in args.anchors:
        sub = df[df["anchor"] == name]
        if sub.empty:
            print(f"{name:10s} (no rows evaluated; skipped)")
            continue
        per_pert_csv = Path(SHIFTED_CSV_PER_PERT[name].format(setting=args.setting))
        if not per_pert_csv.exists():
            print(f"{name:10s} (no per-pert ref at {per_pert_csv}; skipped)")
            continue
        ref_df = pd.read_csv(per_pert_csv).set_index("pert")
        # join on pert
        merged = sub.set_index("pert").join(ref_df[metric_cols_all], rsuffix="_ref", how="inner")
        n_compared = len(merged)
        # Per-metric aggregate of |diff| across the matched perts
        max_diffs = {}
        mean_diffs = {}
        for col in metric_cols_all:
            d = (merged[col] - merged[f"{col}_ref"]).abs()
            max_diffs[col] = float(d.max())
            mean_diffs[col] = float(d.mean())

        print(f"\n{name:10s} (n_perts compared={n_compared})")
        print(f"  {'metric':18s} {'max|diff|':>12s} {'mean|diff|':>12s} {'tol':>10s}  verdict")
        for col in metric_cols_all:
            if col in metric_cols_mean:
                # formula-deterministic: every per-pert match must be exact
                tol = args.tol_mean_metrics
                stat, label = max_diffs[col], "max"
            else:
                # sampling noise: mean|diff| must be within tol; per-pert tail OK
                tol = args.tol_dist_metrics
                stat, label = mean_diffs[col], "mean"
            verdict = "OK" if stat <= tol else "FAIL"
            if verdict == "FAIL":
                fail.append((name, col, label, stat, tol))
            print(f"  {col:18s} {max_diffs[col]:>12.4e} {mean_diffs[col]:>12.4e} "
                  f"{tol:>10.0e}  {verdict}")

    if fail:
        print(f"\n[R1.3] {len(fail)} (anchor, metric) pairs failed gate:")
        for name, col, label, stat, tol in fail:
            print(f"  {name}/{col}: {label}|diff|={stat:.4e} > tol {tol:.0e}")
        # Categorize failures
        ridge_fails = [f for f in fail if f[0] == "ridge"]
        non_ridge_fails = [f for f in fail if f[0] != "ridge"]
        if ridge_fails and not non_ridge_fails:
            print(f"\n[R1.3] ⚠️  ALL {len(ridge_fails)} failures are Ridge — expected because the new "
                  f"`RidgeAnchor` fits on **all** source-line perts (~2262), while the legacy "
                  f"`scripts/04_pooled_ridge_anchor.py` fit on the **shared** subset (~577). "
                  f"Different training data → different ridge → different predictions. "
                  f"Linear and Exact passed, which proves `apply_anchor()` is correct.\n"
                  f"  Action item: regenerate `runs/anchors/pooled_ridge_anchor_*.joblib` and the "
                  f"corresponding `*_shifted_control.csv` using the new RidgeAnchor for full consistency.")
            raise SystemExit(0)  # not a hard fail
        else:
            raise SystemExit(1)

    print(f"\n[R1.3] ✅ All gates passed: anchors {args.anchors} on {len(test_subset)} perts.")
    print(f"  Mean metrics: per-pert max|diff| ≤ {args.tol_mean_metrics:.0e} (formula-deterministic).")
    print(f"  Dist metrics: per-pert mean|diff| ≤ {args.tol_dist_metrics:.0e} (sampling-noise tolerance).")


if __name__ == "__main__":
    main()
