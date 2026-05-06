"""Pooled gene-embedding ridge anchor baseline, evaluated with CrossLineEvaluator.

Trains PooledGeneEmbeddingRidgeAnchor (see cccfm.anchor) on
(primary_gene_embedding, pseudobulk_delta) pairs pooled from source lines,
then evaluates on the strict target-line test slice.

Usage:
    python scripts/04_pooled_ridge_anchor.py \\
        --setting 2line \\
        --output results/baselines/pooled_ridge_anchor_2line.csv \\
        --save_anchor runs/anchors/pooled_ridge_anchor_2line.joblib
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
    import torch
    from cccfm.anchor import PooledGeneEmbeddingRidgeAnchor
    from cccfm.eval_metrics import CrossLineEvaluator

    adata = ad.read_h5ad(args.data_path)
    with open(args.test_slice) as f:
        test_perts = sorted(json.load(f))
    with open(args.shared_perts) as f:
        shared = set(json.load(f))
    gene_embed = torch.load(args.primary_embedding_path, map_location="cpu", weights_only=False)
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

    # Build training pairs
    X_list, y_list, meta = [], [], []
    for cl in sources:
        lmask = adata.obs["cell_line"] == cl
        ctrl = pb(lmask & (adata.obs["pert"] == "control"))
        for pert in shared:
            if pert not in gene_embed:
                continue
            pmask = lmask & (adata.obs["pert"] == pert)
            if int(pmask.sum()) < args.min_cells:
                continue
            delta = pb(pmask) - ctrl
            X_list.append(gene_embed[pert].numpy().astype(np.float32))
            y_list.append(delta.astype(np.float32))
            meta.append((cl, pert))

    if not X_list:
        raise RuntimeError("No training pairs found. Check primary_gene_embedding covers shared perts.")

    X_all = np.stack(X_list)
    y_all = np.stack(y_list)
    print(f"Training pairs: {X_all.shape[0]}  (X: {X_all.shape[1]}, y: {y_all.shape[1]})")

    # Pert-level train/val split (sorted for deterministic order across Python runs;
    # set iteration order depends on hash randomization otherwise).
    unique_perts = sorted({p for _, p in meta})
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique_perts)
    n_val_pert = max(1, int(args.val_frac * len(unique_perts)))
    val_perts = set(unique_perts[:n_val_pert])
    trn_idx = [i for i, (_, p) in enumerate(meta) if p not in val_perts]
    val_idx = [i for i, (_, p) in enumerate(meta) if p in val_perts]
    print(f"pert-level split: train {len(trn_idx)} samples, val {len(val_idx)} samples")

    anchor = PooledGeneEmbeddingRidgeAnchor()
    anchor.fit(
        X_all[trn_idx], y_all[trn_idx],
        X_all[val_idx] if val_idx else None,
        y_all[val_idx] if val_idx else None,
    )
    print(f"selected alpha={anchor.alpha}  val_pearson={anchor.val_score_}")

    if args.save_anchor:
        args.save_anchor.parent.mkdir(parents=True, exist_ok=True)
        anchor.save(args.save_anchor)
        print(f"Saved anchor -> {args.save_anchor}")

    # Evaluate
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []
    for pert in test_perts:
        if pert not in gene_embed:
            continue
        x_in = gene_embed[pert].numpy().astype(np.float32).reshape(1, -1)
        pred_delta = anchor.predict(x_in).flatten()
        pred_profile = tgt_ctrl + pred_delta

        m = evaluator.evaluate_pert_mean(pert, pred_profile)
        if m is None:
            continue
        m.update({
            "method": "pooled_ridge_anchor",
            "setting": args.setting,
            "target_line": target,
            "pert": pert,
            "alpha": anchor.alpha,
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
    p.add_argument("--shared_perts", type=Path, default=Path("data/splits/shared_perturbations.json"))
    p.add_argument("--primary_embedding_path", type=Path, default=Path("data/processed/primary_gene_embedding.pt"))
    p.add_argument("--setting", choices=list(SETTING_SOURCES), default="2line")
    p.add_argument("--target_line", type=str, default="Jurkat")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--save_anchor", type=Path, default=None)
    p.add_argument("--min_cells", type=int, default=50)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main()
