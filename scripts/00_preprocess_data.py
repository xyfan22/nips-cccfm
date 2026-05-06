"""Preprocess 4 raw Replogle-family h5ads into one unified training file.

Inputs four h5ads (data/raw/) in two different normalization states:
    K562_essential_normalized_singlecell_01.h5ad     (already log1p)
    rpe1_normalized_singlecell_01.h5ad               (already log1p)
    GSE264667_hepg2_raw_singlecell_01.h5ad           (raw counts)
    GSE264667_jurkat_raw_singlecell_01.h5ad          (raw counts)
A heuristic auto-detects each one and harmonises to normalize_total(1e4) + log1p.
Genes are intersected across lines, then HVGs (top n_top_genes) are selected.
The output is data/processed/replogle_preprocessed.h5ad plus a statistics CSV.

Usage:
    python scripts/00_preprocess_data.py \\
        --raw_dir data/raw \\
        --output data/processed/replogle_preprocessed.h5ad \\
        --stats_csv data/processed/preprocess_stats.csv \\
        --hvg_count 6836
"""

from __future__ import annotations

import argparse
from pathlib import Path


CELL_LINE_MAP = {
    "K562_essential_normalized_singlecell_01.h5ad": "K562",
    "rpe1_normalized_singlecell_01.h5ad":           "RPE1",
    "GSE264667_hepg2_raw_singlecell_01.h5ad":       "HepG2",
    "GSE264667_jurkat_raw_singlecell_01.h5ad":      "Jurkat",
}


CONTROL_PATTERNS = r"^control$|^ctrl$|non[-_ ]?targeting|nontargeting|scramble|safe[-_ ]?harbor|^ntc$|^nt$"


def detect_normalization(X):
    import scipy.sparse as sp
    mx = X.max() if not sp.issparse(X) else float(X.max())
    if mx > 50:
        return "raw_counts"
    if mx < 15:
        return "log1p_normalized"
    return "unknown"


def find_pert_column(obs_cols):
    """Return the first column in obs that looks like a perturbation label."""
    for c in ["pert", "gene", "gene_symbol", "perturbation",
              "gene_target", "target_gene", "knockdown", "guide_identity"]:
        if c in obs_cols:
            return c
    return None


def load_and_harmonize(h5ad_path: Path, cell_line: str, normalize_target: float):
    import scanpy as sc
    a = sc.read_h5ad(h5ad_path)
    a.obs["cell_line"] = cell_line

    pcol = find_pert_column(a.obs.columns)
    if pcol is None:
        raise RuntimeError(f"No perturbation column in {h5ad_path}. "
                           f"obs columns: {list(a.obs.columns)}")
    if pcol != "pert":
        a.obs["pert"] = a.obs[pcol].astype(str)
    else:
        a.obs["pert"] = a.obs["pert"].astype(str)

    # Coalesce all control variants to a single "control" label.
    is_ctrl = a.obs["pert"].str.lower().str.contains(CONTROL_PATTERNS, regex=True, na=False)
    a.obs.loc[is_ctrl, "pert"] = "control"

    # normalize if still raw counts
    status = detect_normalization(a.X)
    if status == "raw_counts":
        sc.pp.normalize_total(a, target_sum=normalize_target)
        sc.pp.log1p(a)
        stage = "normalized_here"
    elif status == "log1p_normalized":
        stage = "already_log1p"
    else:
        # unknown; leave as-is, warn
        print(f"!! {h5ad_path.name}: normalization unknown (max={float(a.X.max()):.3f}); leaving as-is")
        stage = "unknown"

    return a, stage


def main():
    args = parse_args()
    import anndata as ad
    import pandas as pd
    import scanpy as sc

    stats = []
    adatas = []
    for fname, cl in CELL_LINE_MAP.items():
        fpath = args.raw_dir / fname
        if not fpath.exists():
            print(f"!! missing: {fpath}, skipping")
            continue
        print(f"== [{cl}] loading {fpath.name}")
        a, stage = load_and_harmonize(fpath, cl, args.normalize_target)
        stats.append({
            "cell_line": cl,
            "file": fname,
            "n_cells": int(a.n_obs),
            "n_genes": int(a.n_vars),
            "n_perts": int(a.obs["pert"].nunique()),
            "n_control": int((a.obs["pert"] == "control").sum()),
            "normalization": stage,
        })
        adatas.append(a)

    if len(adatas) < 2:
        raise RuntimeError("Need at least 2 cell lines; found fewer")

    # intersect genes
    common = set(adatas[0].var_names)
    for a in adatas[1:]:
        common &= set(a.var_names)
    common = sorted(common)
    print(f"== intersect genes across {len(adatas)} lines: {len(common)}")

    for i, a in enumerate(adatas):
        adatas[i] = a[:, common].copy()

    merged = ad.concat(adatas, join="inner", merge="unique", index_unique="-")
    print(f"== merged: {merged.n_obs} cells × {merged.n_vars} genes")

    # HVG selection. Two modes:
    #   (a) --hvg_gene_list <file>: read a 1-Ensembl-ID-per-line list and
    #       strictly subset + reorder by that list. Use this for
    #       bit-reproducible HVG selection across machines, since it avoids
    #       HVG flips caused by scanpy / BLAS version differences (recommended
    #       on a cluster).
    #   (b) without --hvg_gene_list: fall back to scanpy seurat-flavor HVG
    #       selection, which is non-deterministic across versions.
    if args.hvg_gene_list:
        with open(args.hvg_gene_list) as f:
            hvg_ids = [ln.strip() for ln in f if ln.strip()]
        print(f"== fixed HVG list provided: {len(hvg_ids)} genes")
        present = [g for g in hvg_ids if g in set(merged.var_names)]
        missing = [g for g in hvg_ids if g not in set(merged.var_names)]
        if missing:
            print(f"!! {len(missing)} HVG genes not in merged var_names; first few: {missing[:5]}")
            print(f"   The raw data may differ from the local copy; please reconcile first.")
        merged = merged[:, present].copy()
        # Reorder by the provided list (to stay byte-identical to the local build).
        merged = merged[:, [g for g in hvg_ids if g in present]].copy()
        merged.var["is_hvg"] = True
        merged.var["gene_symbol"] = merged.var_names.astype(str)
        print(f"== after fixed HVG: {merged.n_vars} genes")
    else:
        n_top = min(args.hvg_count, merged.n_vars)
        sc.pp.highly_variable_genes(merged, n_top_genes=n_top, flavor="seurat")
        merged = merged[:, merged.var["highly_variable"]].copy()
        merged.var["is_hvg"] = True
        merged.var["gene_symbol"] = merged.var_names.astype(str)
        print(f"== after HVG (scanpy seurat): {merged.n_vars} genes  (warn: non-deterministic across scanpy versions)")

    # Add per-(line, pert) cell count as an obs field for downstream filtering.
    grp = (merged.obs.groupby(["cell_line", "pert"], observed=True)
                   .size().rename("n_cells_in_line").reset_index())
    merged.obs = merged.obs.merge(grp, on=["cell_line", "pert"], how="left")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.write_h5ad(args.output)
    print(f"== wrote {args.output}")

    args.stats_csv.parent.mkdir(parents=True, exist_ok=True)
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(args.stats_csv, index=False)
    print(f"== wrote {args.stats_csv}")
    print(stats_df.to_string(index=False))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", type=Path, default=Path("data/raw"))
    p.add_argument("--output", type=Path, default=Path("data/processed/replogle_preprocessed.h5ad"))
    p.add_argument("--stats_csv", type=Path, default=Path("data/processed/preprocess_stats.csv"))
    p.add_argument("--hvg_count", type=int, default=6836,
                   help="Only used when --hvg_gene_list is not provided.")
    p.add_argument("--hvg_gene_list", type=Path, default=None,
                   help="1-Ensembl-ID-per-line list; if provided, the scanpy "
                        "HVG step is skipped and the merged adata is strictly "
                        "subset + reordered by this list (recommended on a "
                        "cluster, bit-reproducible).")
    p.add_argument("--normalize_target", type=float, default=1e4)
    return p.parse_args()


if __name__ == "__main__":
    main()
