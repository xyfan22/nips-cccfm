"""Linear Ahlmann-Eltze baseline (mean-only), evaluated with CrossLineEvaluator.

Method:
    For each perturbation g, predict the target-line perturbed pseudobulk as:
        pred_profile = target_ctrl + (source_pert_mean_pooled - source_ctrl_mean_pooled)

Output:
    CSV using the standard metrics schema
    (see cccfm.eval_metrics.CrossLineEvaluator.METRIC_COLUMNS).

Usage:
    python scripts/03_linear_ahlmann_eltze.py \\
        --setting 2line \\
        --output results/baselines/linear_ahlmann_eltze_2line.csv
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


SETTING_SOURCES = {
    "2line": ["K562", "RPE1"],
    "3line": ["K562", "RPE1", "HepG2"],
}


def main():
    args = parse_args()
    import anndata as ad
    import numpy as np
    import pandas as pd
    from cccfm.eval_metrics import CrossLineEvaluator

    adata = ad.read_h5ad(args.data_path)
    with open(args.test_slice) as f:
        test_perts = sorted(json.load(f))
    sources = SETTING_SOURCES[args.setting]
    target = args.target_line
    print(f"Setting={args.setting}, sources={sources}, target={target}")

    evaluator = CrossLineEvaluator(
        adata=adata,
        source_lines=sources,
        target_line=target,
        test_perts=test_perts,
        min_cells=args.min_cells,
    )

    src_mask = adata.obs["cell_line"].isin(sources)
    def pb(mask):
        X = adata[mask].X
        if hasattr(X, "toarray"): X = X.toarray()
        return np.asarray(X, dtype=np.float64).mean(axis=0)

    src_ctrl = pb(src_mask & (adata.obs["pert"] == "control"))
    tgt_ctrl = evaluator.target_ctrl_reference
    print("Precomputed source/target control means.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []
    for pert in test_perts:
        smask = src_mask & (adata.obs["pert"] == pert)
        n_src = int(smask.sum())
        if n_src < args.min_cells:
            continue
        src_pert = pb(smask)
        pred_profile = tgt_ctrl + (src_pert - src_ctrl)

        m = evaluator.evaluate_pert_mean(pert, pred_profile)
        if m is None:
            continue
        m.update({
            "method": "linear_ahlmann_eltze",
            "setting": args.setting,
            "target_line": target,
            "pert": pert,
            "n_source_cells": n_src,
            "seed": args.seed,
            "run_id": run_id,
        })
        rows.append(m)

    df = pd.DataFrame(rows)
    # Column ordering
    front = ["method", "setting", "target_line", "pert"]
    metric_cols = [c for c in evaluator.METRIC_COLUMNS if c in df.columns]
    extra_cols = [c for c in df.columns if c not in front + metric_cols]
    df = df[front + metric_cols + extra_cols]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    # Summary aggregation
    agg = df[[c for c in metric_cols if c != "n_target_cells"]].agg(["mean", "median", "std"]).T
    agg_path = args.output.with_name(args.output.stem + "_summary.csv")
    agg.reset_index().rename(columns={"index": "metric"}).to_csv(agg_path, index=False)

    print(f"n_perts_evaluated: {len(df)}")
    for col in ["pearson_ctrl", "pearson_pert", "pearson20_ctrl", "pearson20_pert", "centroid_accuracy"]:
        if col in df.columns:
            print(f"  {col:20s}: mean={df[col].mean():.4f}  median={df[col].median():.4f}")
    print(f"Saved {args.output}")
    print(f"Saved {agg_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=Path, default=Path("data/processed/replogle_preprocessed.h5ad"))
    p.add_argument("--test_slice", type=Path, default=Path("data/splits/jurkat_test_strict.json"))
    p.add_argument("--setting", choices=list(SETTING_SOURCES), default="2line")
    p.add_argument("--target_line", type=str, default="Jurkat")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--min_cells", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main()
