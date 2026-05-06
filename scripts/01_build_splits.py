"""Build cross-line evaluation splits.

Outputs:
    data/splits/shared_perturbations.json   - perturbation symbols shared by all four lines
    data/splits/jurkat_test_strict.json     - strict Jurkat test slice
                                              (shared ∩ n_cells≥50 ∩ embedding-covered)
    data/splits/stats.csv                   - descriptive statistics

Notes:
    - No train/val/test columns are written: cross-line evaluation does not
      require a per-sample split (train data = cells from source lines;
      test data = cells from the target line × strict slice).
    - Strict-slice filter order:
        1) non-control perts in the Jurkat (target) line
        2) n_cells >= min_cells  (default 50)
        3) pert is in the shared set
        4) (optional) if primary_embedding_path exists, further intersect
           with the embedding's gene set.

Usage:
    python scripts/01_build_splits.py \\
        --data_path data/processed/replogle_preprocessed.h5ad \\
        --splits_dir data/splits \\
        --min_cells 50 \\
        --target_line Jurkat \\
        --primary_embedding_path data/processed/primary_gene_embedding.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    args = parse_args()
    import anndata as ad
    import pandas as pd

    print(f"== loading {args.data_path}")
    adata = ad.read_h5ad(args.data_path)
    cell_lines = sorted(adata.obs["cell_line"].unique())
    print(f"Cell lines: {cell_lines}")

    # 1) per-line pert set (n_cells >= min_cells, excluding control)
    pert_sets = {}
    for cl in cell_lines:
        sub = adata.obs[(adata.obs["cell_line"] == cl) & (adata.obs["pert"] != "control")]
        grp = sub.groupby("pert").size()
        pert_sets[cl] = set(grp[grp >= args.min_cells].index)
        print(f"  {cl}: {len(pert_sets[cl])} perts (n_cells >= {args.min_cells})")

    # 2) shared = present in all four lines
    shared = sorted(set.intersection(*pert_sets.values()))
    print(f"Shared perts across all lines: {len(shared)}")

    # 3) target test strict
    target_sub = adata.obs[(adata.obs["cell_line"] == args.target_line)
                           & (adata.obs["pert"] != "control")]
    target_counts = target_sub.groupby("pert").size()
    target_passing = set(target_counts[target_counts >= args.min_cells].index)
    strict = set(shared) & target_passing

    # 4) optional embedding filter
    embed_covered = None
    if args.primary_embedding_path and Path(args.primary_embedding_path).exists():
        import torch
        emb = torch.load(args.primary_embedding_path, map_location="cpu")
        embed_covered = len(strict & set(emb.keys()))
        strict &= set(emb.keys())
        print(f"After primary_gene_embedding filter: {len(strict)} perts")
    strict = sorted(strict)
    print(f"Strict {args.target_line} test slice: {len(strict)} perts")

    # save
    args.splits_dir.mkdir(parents=True, exist_ok=True)
    shared_p = args.splits_dir / "shared_perturbations.json"
    strict_p = args.splits_dir / f"{args.target_line.lower()}_test_strict.json"
    with open(shared_p, "w") as f:
        json.dump(shared, f, indent=0)
    with open(strict_p, "w") as f:
        json.dump(strict, f, indent=0)

    rows = [
        {"name": "total_unique_perts", "value": int(adata.obs["pert"].nunique() - 1)},
        {"name": "shared_perts", "value": len(shared)},
        {"name": f"{args.target_line.lower()}_test_strict", "value": len(strict)},
    ]
    for cl in cell_lines:
        rows.append({"name": f"line_perts_{cl}", "value": len(pert_sets[cl])})
    if embed_covered is not None:
        rows.append({"name": "strict_covered_by_embedding", "value": embed_covered})
    stats = pd.DataFrame(rows)
    stats_p = args.splits_dir / "stats.csv"
    stats.to_csv(stats_p, index=False)
    print(f"Wrote {shared_p}")
    print(f"Wrote {strict_p}")
    print(f"Wrote {stats_p}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=Path, default=Path("data/processed/replogle_preprocessed.h5ad"))
    p.add_argument("--splits_dir", type=Path, default=Path("data/splits"))
    p.add_argument("--min_cells", type=int, default=50)
    p.add_argument("--target_line", type=str, default="Jurkat")
    p.add_argument("--primary_embedding_path", type=Path, default=None)
    return p.parse_args()


if __name__ == "__main__":
    main()
