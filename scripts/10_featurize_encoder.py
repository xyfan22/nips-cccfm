"""Featurize all cells with a frozen context encoder, save to adata.obsm.

Unified CLI for Phase A. Router to `cccfm.encoders.ENCODER_REGISTRY`.

NOTE on external dependencies: every non-PCA encoder dynamically pulls
upstream code from `<repo-root>/baselines/<encoder>/` and requires the
upstream's pretrained checkpoint (NOT shipped in this artifact). See the
"External dependencies" block at the top of `code/cccfm/encoders.py` and
`data/README.md` Step 3 for the upstream URLs, licenses, and checkpoint
download instructions. The artifact's documented smoke-test
(`docs/reproducibility.md`) does NOT require this script to be re-run.

Inputs:
    - preprocessed .h5ad (N_cells × G_genes, log1p-normalized)
    - encoder name (one of: pca | state_se | geneformer | scgpt | uce | scfoundation)
    - optional checkpoint path (if encoder needs pretrained weights)

Output:
    - updated .h5ad with obsm[f"h_{encoder_name}"] populated ([N, D_enc])
    - results/featurization/<encoder>_stats.csv (shape, wall time, nan check)

Usage:
    # PCA (runs locally, no GPU required)
    python scripts/10_featurize_encoder.py \\
        --encoder pca \\
        --data_path data/processed/replogle_preprocessed.h5ad \\
        --fit_sources K562 RPE1 HepG2 \\
        --stats_csv results/featurization/pca_stats.csv \\
        --save_checkpoint runs/encoders/pca_128.joblib

    # STATE-SE (cluster, requires the upstream Arc weights)
    CUDA_VISIBLE_DEVICES=0 python scripts/10_featurize_encoder.py \\
        --encoder state_se \\
        --data_path data/processed/replogle_preprocessed.h5ad \\
        --checkpoint <FILL path to Arc SE-600M weights> \\
        --batch_size 512 \\
        --device cuda:0 \\
        --stats_csv results/featurization/state_se_stats.csv

Notes:
    - Each encoder writes one obsm key and updates the .h5ad in place.
    - PCA is a special case: it must first be fit (on source-line control
      cells); all other encoders are loaded via load_pretrained.
    - The stats_csv is intended to make cluster-side debugging easy:
      output shape, NaN count, wall time, and encoder version.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", type=str, required=True,
                   choices=["pca", "state_se", "geneformer", "scgpt", "uce", "scfoundation"])
    p.add_argument("--data_path", type=Path,
                   default=Path("data/processed/replogle_preprocessed.h5ad"))
    p.add_argument("--output_path", type=Path, default=None,
                   help="Output .h5ad (defaults to in-place overwrite).")
    p.add_argument("--obsm_key", type=str, default=None,
                   help="obsm key name (defaults to h_<encoder>).")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Encoder weight path (ignored for PCA, which uses --fit_sources).")
    p.add_argument("--save_checkpoint", type=Path, default=None,
                   help="After PCA fit, save the resulting joblib for later reuse.")
    p.add_argument("--fit_sources", nargs="+", default=["K562", "RPE1", "HepG2"],
                   help="Source cell lines used to fit PCA (control cells only).")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda:0",
                   help="Used by the foundation encoders; ignored for PCA.")
    p.add_argument("--n_components", type=int, default=128,
                   help="PCA only")
    p.add_argument("--stats_csv", type=Path, default=None)
    p.add_argument("--dry_run", action="store_true",
                   help="Only print the config and check whether data can be loaded; do not run the encoder.")
    return p.parse_args()


def main():
    args = parse_args()
    import anndata as ad
    import numpy as np
    import pandas as pd
    import torch

    from cccfm.encoders import build_encoder, PCANullEncoder

    obsm_key = args.obsm_key or f"h_{args.encoder}"
    output_path = args.output_path or args.data_path

    print(f"== Featurize encoder={args.encoder}  obsm_key={obsm_key}")
    print(f"   data: {args.data_path}  ->  {output_path}")

    t0 = time.perf_counter()
    adata = ad.read_h5ad(args.data_path)
    print(f"   loaded: {adata.n_obs} cells × {adata.n_vars} genes")

    # ---------------- Build encoder ----------------
    if args.encoder == "pca":
        # Special: fit from source-line control cells, not a pre-downloaded ckpt.
        enc = PCANullEncoder(n_components=args.n_components, batch_size=args.batch_size)
        if args.checkpoint and args.checkpoint.exists():
            print(f"   loading pre-fit PCA: {args.checkpoint}")
            enc.load_pretrained(args.checkpoint)
        else:
            mask = (
                adata.obs["cell_line"].isin(args.fit_sources)
                & (adata.obs["pert"] == "control")
            )
            n_fit = int(mask.sum())
            print(f"   fit PCA on {n_fit} source-line control cells "
                  f"(sources={args.fit_sources})")
            X_fit = adata[mask].X
            enc.fit(X_fit)
            if args.save_checkpoint:
                enc.save(args.save_checkpoint)
                print(f"   saved PCA -> {args.save_checkpoint}")
    else:
        # Foundation encoder: requires pretrained checkpoint.
        if args.checkpoint is None:
            raise ValueError(f"--checkpoint required for encoder='{args.encoder}'")
        enc = build_encoder(args.encoder, checkpoint_path=args.checkpoint)
        # Push to device for non-PCA encoders
        try:
            enc.to(args.device)
        except Exception as e:  # noqa: BLE001
            print(f"   warning: enc.to({args.device}) failed: {e}")

    if args.dry_run:
        print("   dry_run=True, not running encode; exiting.")
        return

    # ---------------- Featurize ----------------
    N = adata.n_obs

    if args.encoder == "uce":
        # UCE: per-batch encode with gene symbol prep
        print(f"   UCE: preparing gene-to-embedding mapping ...")
        enc.encode_adata_prep(adata.var["gene_name"].astype(str).values)
        print(f"   encoding in batches of {args.batch_size} ...")
        D = enc.output_dim
        h_all = np.empty((N, D), dtype=np.float32)
        n_nan = 0

        for start in range(0, N, args.batch_size):
            end = min(start + args.batch_size, N)
            x = adata.X[start:end]
            if hasattr(x, "toarray"):
                x = x.toarray()
            x = np.asarray(x, dtype=np.float32)
            with torch.no_grad():
                h = enc.encode(x).detach().cpu().numpy()
            if np.isnan(h).any():
                n_nan += int(np.isnan(h).sum())
            h_all[start:end] = h
            if start % (args.batch_size * 50) == 0:
                print(f"     [{start:>7d}/{N}]  wall={time.perf_counter()-t0:.1f}s")
    else:
        # Standard batched encode
        print(f"   encoding in batches of {args.batch_size} ...")
        D = enc.output_dim
        h_all = np.empty((N, enc.output_dim), dtype=np.float32)
        n_nan = 0

        for start in range(0, N, args.batch_size):
            end = min(start + args.batch_size, N)
            x = adata.X[start:end]
            if hasattr(x, "toarray"):
                x = x.toarray()
            x = np.asarray(x, dtype=np.float32)
            if args.encoder == "pca":
                h = enc.encode(x)
                h = h.cpu().numpy()
            else:
                xt = torch.as_tensor(x, dtype=torch.float32, device=args.device)
                with torch.no_grad():
                    h = enc.encode(xt).detach().cpu().numpy()
            if np.isnan(h).any():
                n_nan += int(np.isnan(h).sum())
            h_all[start:end] = h
            if start % (args.batch_size * 50) == 0:
                print(f"     [{start:>7d}/{N}]  wall={time.perf_counter()-t0:.1f}s")

    # ---------------- Write back ----------------
    adata.obsm[obsm_key] = h_all
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path)
    wall = time.perf_counter() - t0
    print(f"== done. obsm['{obsm_key}'].shape={h_all.shape}  wall={wall:.1f}s  NaN={n_nan}")

    # ---------------- Stats CSV ----------------
    if args.stats_csv:
        args.stats_csv.parent.mkdir(parents=True, exist_ok=True)
        stats = pd.DataFrame([{
            "encoder": args.encoder,
            "obsm_key": obsm_key,
            "data_path": str(args.data_path),
            "n_cells": N,
            "n_genes_in": adata.n_vars,
            "embed_dim": h_all.shape[1],
            "wall_seconds": round(wall, 2),
            "n_nan": n_nan,
            "mean_abs": float(np.abs(h_all).mean()),
            "std": float(h_all.std()),
            "checkpoint": str(args.checkpoint) if args.checkpoint else "",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }])
        stats.to_csv(args.stats_csv, index=False)
        print(f"   stats -> {args.stats_csv}")


if __name__ == "__main__":
    main()
