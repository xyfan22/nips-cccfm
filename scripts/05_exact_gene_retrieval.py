"""Exact-gene retrieval baseline, evaluated with CrossLineEvaluator.

For every strict-slice target perturbation g:
    - If g is observed in >=1 source line with enough cells:
      pred_delta = mean of (pert_mean - ctrl_mean) across those source
      lines; the method field is 'exact_gene_retrieval'.
    - Otherwise (optional) fall back to the pooled_ridge anchor prediction;
      the method field is then 'exact_gene_retrieval_fallback'.

Usage:
    python scripts/05_exact_gene_retrieval.py \\
        --setting 2line \\
        --output results/baselines/exact_gene_retrieval_2line.csv \\
        --anchor_path runs/anchors/pooled_ridge_anchor_2line.joblib
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
        adata=adata, source_lines=sources, target_line=target,
        test_perts=test_perts, min_cells=args.min_cells,
    )
    tgt_ctrl = evaluator.target_ctrl_reference

    def pb(mask):
        X = adata[mask].X
        if hasattr(X, "toarray"): X = X.toarray()
        return np.asarray(X, dtype=np.float64).mean(axis=0)

    # Cache per-pert source-line deltas (pool over source lines)
    src_deltas: dict[str, list[np.ndarray]] = {}
    for cl in sources:
        lmask = adata.obs["cell_line"] == cl
        ctrl = pb(lmask & (adata.obs["pert"] == "control"))
        for pert in sorted(set(adata.obs[lmask]["pert"].unique()) - {"control"}):
            pmask = lmask & (adata.obs["pert"] == pert)
            if int(pmask.sum()) < args.min_cells:
                continue
            src_deltas.setdefault(pert, []).append(pb(pmask) - ctrl)
    print(f"Source deltas cached for {len(src_deltas)} unique perts")

    # Optional anchor fallback
    anchor, gene_embed = None, None
    if args.anchor_path and Path(args.anchor_path).exists():
        import torch
        from cccfm.anchor import PooledGeneEmbeddingRidgeAnchor
        anchor = PooledGeneEmbeddingRidgeAnchor().load(args.anchor_path)
        gene_embed = torch.load(args.primary_embedding_path, map_location="cpu", weights_only=False)
        print(f"Loaded anchor fallback: {args.anchor_path}")
    else:
        print("No anchor fallback; unseen perts skipped.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []
    for pert in test_perts:
        if pert in src_deltas:
            pred_delta = np.stack(src_deltas[pert]).mean(axis=0)
            method = "exact_gene_retrieval"
        elif anchor is not None and gene_embed is not None and pert in gene_embed:
            pred_delta = anchor.predict(
                gene_embed[pert].numpy().astype(np.float32).reshape(1, -1)
            ).flatten()
            method = "exact_gene_retrieval_fallback"
        else:
            continue
        pred_profile = tgt_ctrl + pred_delta

        m = evaluator.evaluate_pert_mean(pert, pred_profile)
        if m is None:
            continue
        m.update({
            "method": method,
            "setting": args.setting,
            "target_line": target,
            "pert": pert,
            "seed": args.seed,
            "run_id": run_id,
        })
        rows.append(m)

    df = pd.DataFrame(rows)
    front = ["method", "setting", "target_line", "pert"]
    metric_cols = [c for c in evaluator.METRIC_COLUMNS if c in df.columns]
    extra_cols = [c for c in df.columns if c not in front + metric_cols]
    df = df[front + metric_cols + extra_cols]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    # Summary per-method
    agg_rows = []
    for m in df["method"].unique():
        sub = df[df["method"] == m][[c for c in metric_cols if c != "n_target_cells"]]
        for stat_name, series in sub.agg(["mean", "median", "std"]).iterrows():
            agg_rows.append({"method": m, "stat": stat_name, **series.to_dict()})
    agg_path = args.output.with_name(args.output.stem + "_summary.csv")
    pd.DataFrame(agg_rows).to_csv(agg_path, index=False)

    print(f"n_perts_evaluated: {len(df)}")
    print(f"per-method counts:\n{df['method'].value_counts().to_string()}")
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
    p.add_argument("--anchor_path", type=Path, default=None)
    p.add_argument("--primary_embedding_path", type=Path, default=Path("data/processed/primary_gene_embedding.pt"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main()
