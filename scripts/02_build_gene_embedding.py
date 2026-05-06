"""Build a primary gene embedding via PCA of the gene-cell expression matrix.

Notes:
    - When no external GO embedding is available, use the simplest possible
      primary: PCA of gene patterns across cells.
    - Only source-line control cells are used (target line excluded) to
      avoid target leakage.
    - Input is the [N_cells, G] expression matrix; transposed to [G, N_cells]
      and PCA is run over genes to produce [G, D].
    - Output is a dict mapping gene_symbol -> torch.Tensor of shape [D].
    - Default D = 128.

Usage:
    python scripts/02_build_gene_embedding.py \\
        --data_path data/processed/replogle_preprocessed.h5ad \\
        --output data/processed/primary_gene_embedding.pt \\
        --n_components 128 \\
        --sources K562 RPE1 HepG2
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    args = parse_args()
    import anndata as ad
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.decomposition import PCA

    print(f"== loading {args.data_path}")
    adata = ad.read_h5ad(args.data_path)
    mask = (adata.obs["cell_line"].isin(args.sources)) & (adata.obs["pert"] == "control")
    print(f"Source control cells: {int(mask.sum())} (lines: {args.sources})")

    X = adata[mask].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)    # [N_cells, G]

    # transpose: each row is a gene's expression pattern across cells
    X_gene = X.T                           # [G, N_cells]
    mu = X_gene.mean(axis=1, keepdims=True)
    sd = X_gene.std(axis=1, keepdims=True) + 1e-8
    X_gene = (X_gene - mu) / sd

    n_comp = min(args.n_components, X_gene.shape[1])
    print(f"Running PCA n_components={n_comp}")
    pca = PCA(n_components=n_comp, random_state=args.seed).fit(X_gene)
    gene_embed = pca.transform(X_gene)      # [G, D]
    evr = float(pca.explained_variance_ratio_.sum())
    print(f"Explained variance ratio sum: {evr:.4f}")

    # save dict keyed by gene_symbol (matches obs.pert which uses gene symbols,
    # not Ensembl IDs). If var["gene_name"] present use it; else fall back to var_names.
    # On duplicate symbols, keep the first occurrence.
    if "gene_name" in adata.var.columns:
        symbols = adata.var["gene_name"].astype(str).values
        print(f"Using var['gene_name'] for keys")
    else:
        symbols = adata.var_names.astype(str).values
        print(f"var['gene_name'] missing; falling back to var_names")

    out = {}
    n_dup = 0
    for i, name in enumerate(symbols):
        if name in out:
            n_dup += 1
            continue
        out[name] = torch.tensor(gene_embed[i], dtype=torch.float32)
    if n_dup:
        print(f"Skipped {n_dup} duplicate gene_name rows (kept first occurrence)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)
    print(f"Saved {len(out)} genes × {n_comp}-d embedding to {args.output}")

    # stats csv
    stats = pd.DataFrame([
        {"name": "n_genes", "value": len(out)},
        {"name": "n_components", "value": n_comp},
        {"name": "explained_variance_ratio", "value": evr},
        {"name": "source_control_cells", "value": int(mask.sum())},
        {"name": "sources", "value": ",".join(args.sources)},
    ])
    stats_path = args.output.with_name(args.output.stem + "_stats.csv")
    stats.to_csv(stats_path, index=False)
    print(f"Saved {stats_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=Path, default=Path("data/processed/replogle_preprocessed.h5ad"))
    p.add_argument("--output", type=Path, default=Path("data/processed/primary_gene_embedding.pt"))
    p.add_argument("--n_components", type=int, default=128)
    p.add_argument("--sources", nargs="+", default=["K562", "RPE1", "HepG2"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main()
