"""STATE zero-shot baseline (Arc Institute 2025) for ccCFM.

No training — loads Arc's pretrained `ST-HVG-Replogle` checkpoint and runs
`state tx infer` in zero-shot mode over target-line cells, then evaluates
each of the 577 test perts via `CrossLineEvaluator.evaluate_pert_cell_level`.

Paper caveat (**IMPORTANT**)
----------------------------
Arc released both `jurkat_zeroshot.toml` (trained on K562+RPE1+HepG2 with
Jurkat held out) AND `jurkat.toml` (trained on all 4 lines with a subset
of Jurkat perts held out). The public HF `ST-HVG-Replogle` checkpoint's
`config.yaml` must be inspected on cluster to confirm WHICH variant it is.

This script's output is the "zero-shot" STATE row in the paper. If the
checkpoint is the `jurkat.toml` (fewshot-with-holdout) variant, report it as
such rather than claiming a strict cross-line holdout.

Pipeline
--------
1. Build `state_input.h5ad`: target-line (Jurkat) cells (control + test perts),
   `obsm["X_hvg"]` aligned to STATE's trained gene vocab (zero-pad missing).
2. Shell out `state tx infer --all-perts` → `state_output.h5ad`.
3. Load output, for each of the 577 test perts:
     - extract predicted cells (those whose obs.pert == <pert>)
     - remap from STATE's gene vocab back to our 6,642 HVG (zero-pad missing)
     - feed to `CrossLineEvaluator.evaluate_pert_cell_level`
     - append CSV row (incremental save)

Cost: ~1-2 GPU-h for 577 perts on A100/H100.

Usage
-----
python scripts/13_eval_state_zeroshot.py \\
    --state_model_dir baselines/state/pretrained/ST-HVG-Replogle \\
    --output results/baselines/state_zeroshot.csv \\
    --workdir runs/state_zeroshot \\
    --setting 2line
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=Path,
                   default=Path("data/processed/replogle_preprocessed.h5ad"))
    p.add_argument("--test_slice", type=Path,
                   default=Path("data/splits/jurkat_test_strict.json"))
    p.add_argument("--target_line", type=str, default="Jurkat")
    p.add_argument("--setting", choices=["2line", "3line", "zeroshot"], default="zeroshot",
                   help=("Filled into CSV 'setting' column. STATE's own training "
                         "split is fixed, so the SAME STATE CSV serves both 2-line "
                         "and 3-line paper rows. Default 'zeroshot' = neutral label."))
    # STATE specifics
    p.add_argument("--state_model_dir", type=Path, required=True,
                   help="Path to Arc's checkpoint dir, e.g. "
                        "baselines/state/pretrained/ST-HVG-Replogle")
    p.add_argument("--state_checkpoint", type=Path, default=None,
                   help="defaults to <model_dir>/checkpoints/final.ckpt")
    p.add_argument("--min_cells", type=int, default=50)
    p.add_argument("--state_min_cells_flag", type=int, default=50,
                   help="passed to `state tx infer --min-cells`")
    p.add_argument("--state_max_cells_flag", type=int, default=64,
                   help="passed to `state tx infer --max-cells`; limits per-pert "
                        "cell count to control memory")
    # orchestration
    p.add_argument("--workdir", type=Path, required=True,
                   help="root for state_input.h5ad / state_output.h5ad + logs")
    p.add_argument("--output", type=Path, required=True,
                   help="per-pert CSV")
    p.add_argument("--skip_infer", action="store_true",
                   help="reuse existing state_output.h5ad (for eval-only reruns)")
    p.add_argument("--skip_build_input", action="store_true",
                   help="reuse existing state_input.h5ad")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compute_heavy", action="store_true", default=False,
                   help="pass compute_heavy=True to CrossLineEvaluator so "
                        "W_2 (sinkhorn) and per-gene moment metrics are filled")
    p.add_argument("--method_label", type=str, default="state_zeroshot",
                   help="string written into CSV 'method' column; use "
                        "'state_fewshot' when running the fewshot variant")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# STATE vocab helpers
# --------------------------------------------------------------------------- #

def load_state_gene_vocab(model_dir: Path) -> list[str]:
    """Load STATE's trained HVG gene list from var_dims.pkl.

    Schema assumption (from survey / _infer.py:404-417): var_dims is a dict
    with keys like 'gene_names' holding the ordered list. Cluster agent may
    need to patch the key if Arc uses a different label.
    """
    with open(model_dir / "var_dims.pkl", "rb") as f:
        var_dims = pickle.load(f)
    for key in ("gene_names", "genes", "hvg_genes", "var_names"):
        if key in var_dims:
            return list(var_dims[key])
    raise KeyError(
        f"Could not find gene list in var_dims.pkl; available keys: {list(var_dims.keys())}. "
        "Patch `load_state_gene_vocab` to use the correct key."
    )


def load_state_pert_vocab(model_dir: Path) -> set[str]:
    """Load STATE's known perturbation labels from pert_onehot_map.pt."""
    import torch
    pert_map = torch.load(model_dir / "pert_onehot_map.pt", map_location="cpu",
                          weights_only=False)
    # pert_map is expected to be dict[str -> tensor]
    if not isinstance(pert_map, dict):
        raise TypeError(f"pert_onehot_map.pt is not a dict: {type(pert_map)}")
    return set(pert_map.keys())


# --------------------------------------------------------------------------- #
# Gene-vocab alignment
# --------------------------------------------------------------------------- #

def align_matrix(
    X_ours,          # [N, n_our_genes]
    our_genes: list[str],
    state_genes: list[str],
):
    """Project our-gene-ordered matrix X to STATE-gene order.

    Returns ndarray [N, n_state_genes], zero-filled for genes STATE expects
    but we don't have.
    """
    import numpy as np
    our_gene_to_idx = {g: i for i, g in enumerate(our_genes)}
    n_state = len(state_genes)
    # Build column-pick indices (our_idx for each state gene, -1 if absent)
    pick = np.full(n_state, -1, dtype=np.int64)
    missing = 0
    for j, g in enumerate(state_genes):
        if g in our_gene_to_idx:
            pick[j] = our_gene_to_idx[g]
        else:
            missing += 1
    if hasattr(X_ours, "toarray"):
        X_ours = X_ours.toarray()
    X_ours = np.asarray(X_ours, dtype=np.float32)
    out = np.zeros((X_ours.shape[0], n_state), dtype=np.float32)
    mask = pick >= 0
    out[:, mask] = X_ours[:, pick[mask]]
    return out, {"state_dim": n_state, "ours_dim": X_ours.shape[1],
                 "state_genes_not_in_ours": int(missing),
                 "state_genes_covered_by_ours": int((pick >= 0).sum())}


def invert_alignment(
    X_state,         # [N, n_state_genes]
    state_genes: list[str],
    our_genes: list[str],
):
    """Inverse of align_matrix: STATE-gene-ordered predictions back to our-gene-order.

    Returns ndarray [N, n_our_genes], zero-filled for our genes that STATE
    didn't predict on.
    """
    import numpy as np
    state_gene_to_idx = {g: i for i, g in enumerate(state_genes)}
    n_ours = len(our_genes)
    pick = np.full(n_ours, -1, dtype=np.int64)
    for i, g in enumerate(our_genes):
        if g in state_gene_to_idx:
            pick[i] = state_gene_to_idx[g]
    X_state = np.asarray(X_state, dtype=np.float32)
    out = np.zeros((X_state.shape[0], n_ours), dtype=np.float32)
    mask = pick >= 0
    out[:, mask] = X_state[:, pick[mask]]
    return out, {"our_genes_covered_by_state": int(mask.sum()),
                 "our_genes_not_in_state": int((pick < 0).sum())}


# --------------------------------------------------------------------------- #
# Pipeline step 1: build state_input.h5ad
# --------------------------------------------------------------------------- #

def build_state_input(
    full_adata_path: Path,
    target_line: str,
    test_perts: list[str],
    state_model_dir: Path,
    output_path: Path,
    min_cells: int,
):
    """Filter to target-line (controls + test perts) and align obsm[X_hvg] to STATE vocab."""
    import anndata as ad
    import numpy as np

    print(f"Loading {full_adata_path}")
    full = ad.read_h5ad(full_adata_path)

    # Filter cells: target line, pert in {'control'} ∪ test_perts
    keep_perts = set(test_perts) | {"control"}
    mask = (full.obs["cell_line"] == target_line) & full.obs["pert"].isin(keep_perts)
    sub = full[mask].copy()

    # Drop perts with <min_cells within target line (should be rare given strict slice)
    counts = sub.obs.groupby("pert", observed=True).size()
    keep_pert_set = set(counts[counts >= min_cells].index) | {"control"}
    sub = sub[sub.obs["pert"].isin(keep_pert_set)].copy()
    sub.obs["pert"] = sub.obs["pert"].astype(str)
    sub.obs["cell_line"] = sub.obs["cell_line"].astype(str)
    print(f"After filter: {sub.n_obs} target-line cells, "
          f"{sub.obs['pert'].nunique()} unique perts "
          f"(incl. control)")

    # Build STATE-aligned obsm["X_hvg"]
    # var_names are Ensembl IDs; STATE uses gene symbols.
    # Use var["gene_name"] for alignment against the STATE gene vocabulary.
    state_genes = load_state_gene_vocab(state_model_dir)
    our_genes = list(sub.var["gene_name"].astype(str))
    X_state, align_stats = align_matrix(sub.X, our_genes, state_genes)
    sub.obsm["X_hvg"] = X_state
    print(f"Gene alignment: {align_stats}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub.write_h5ad(output_path)
    print(f"Wrote {output_path}")
    return align_stats


# --------------------------------------------------------------------------- #
# Pipeline step 2: shell out `state tx infer`
# --------------------------------------------------------------------------- #

def run_state_infer(
    state_model_dir: Path,
    checkpoint: Path | None,
    input_h5ad: Path,
    output_h5ad: Path,
    min_cells_flag: int,
    max_cells_flag: int = 64,
):
    """Run STATE inference via low-memory script (avoids OOM from CLI)."""
    ckpt = checkpoint or (state_model_dir / "checkpoints" / "final.ckpt")
    if not ckpt.exists():
        raise FileNotFoundError(f"STATE checkpoint not found: {ckpt}")
    cmd = [
        sys.executable, "scripts/14_state_infer_lowmem.py",
        "--state_model_dir", str(state_model_dir),
        "--state_checkpoint", str(ckpt),
        "--input_h5ad", str(input_h5ad),
        "--output_h5ad", str(output_h5ad),
    ]
    print(f"Running:\n  {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    if not output_h5ad.exists():
        raise RuntimeError(f"STATE infer did not produce {output_h5ad}")
    print(f"STATE output: {output_h5ad}")


# --------------------------------------------------------------------------- #
# Pipeline step 3: load output, remap to our vocab, evaluate per pert
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()

    # --- Paths ---
    args.workdir.mkdir(parents=True, exist_ok=True)
    state_input = args.workdir / "state_input.h5ad"
    state_output = args.workdir / "state_output.h5ad"

    # --- Load test perts ---
    with open(args.test_slice) as f:
        test_perts = sorted(json.load(f))
    print(f"Test perts: {len(test_perts)}")

    # --- STATE vocab pre-flight ---
    state_genes = load_state_gene_vocab(args.state_model_dir)
    state_pert_vocab = load_state_pert_vocab(args.state_model_dir)
    oov_perts = [p for p in test_perts if p not in state_pert_vocab]
    in_vocab_perts = [p for p in test_perts if p in state_pert_vocab]
    print(f"STATE gene vocab size: {len(state_genes)}")
    print(f"STATE pert vocab size: {len(state_pert_vocab)}")
    print(f"Test perts IN  STATE vocab: {len(in_vocab_perts)}")
    print(f"Test perts OOV STATE vocab: {len(oov_perts)}  "
          f"(first 5: {oov_perts[:5]})")

    # --- Step 1: build input ---
    if not args.skip_build_input:
        build_state_input(
            full_adata_path=args.data_path,
            target_line=args.target_line,
            test_perts=test_perts,
            state_model_dir=args.state_model_dir,
            output_path=state_input,
            min_cells=args.min_cells,
        )
    else:
        print(f"Skip build; reusing {state_input}")
        assert state_input.exists()

    # --- Step 2: STATE infer ---
    if not args.skip_infer:
        run_state_infer(
            state_model_dir=args.state_model_dir,
            checkpoint=args.state_checkpoint,
            input_h5ad=state_input,
            output_h5ad=state_output,
            min_cells_flag=args.state_min_cells_flag,
            max_cells_flag=args.state_max_cells_flag,
        )
    else:
        print(f"Skip infer; reusing {state_output}")
        assert state_output.exists()

    # --- Step 3: evaluate ---
    import anndata as ad
    import numpy as np
    import pandas as pd
    from cccfm.eval_metrics import CrossLineEvaluator

    print(f"Loading STATE output {state_output}")
    out_adata = ad.read_h5ad(state_output)
    if "X_hvg" not in out_adata.obsm:
        raise RuntimeError(
            f"STATE output h5ad is missing obsm['X_hvg']; got: {list(out_adata.obsm.keys())}")

    # Full adata for evaluator (target-line centroids / source-line DE)
    full_adata = ad.read_h5ad(args.data_path)
    sources_by_setting = {
        "2line": ["K562", "RPE1"],
        "3line": ["K562", "RPE1", "HepG2"],
        "zeroshot": ["K562", "RPE1", "HepG2"],  # for source-line DE computation
    }
    sources = sources_by_setting[args.setting]
    evaluator = CrossLineEvaluator(
        adata=full_adata,
        source_lines=sources,
        target_line=args.target_line,
        test_perts=test_perts,
        min_cells=args.min_cells,
    )

    # var_names are Ensembl IDs; STATE uses gene symbols.
    # Use var["gene_name"] so invert_alignment maps STATE→our gene symbols,
    # which correspond 1:1 to full_adata.X columns (same Ensembl ID ordering).
    our_genes = list(full_adata.var["gene_name"].astype(str))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, pert in enumerate(test_perts):
        mask = out_adata.obs["pert"] == pert
        n_pred = int(mask.sum())
        if n_pred == 0:
            print(f"[{i+1}/{len(test_perts)}] {pert}: no output rows; skip")
            continue
        pred_state = np.asarray(out_adata[mask].obsm["X_hvg"], dtype=np.float32)
        pred_cells, remap_stats = invert_alignment(pred_state, state_genes, our_genes)

        m = evaluator.evaluate_pert_cell_level(
            pert, pred_cells, compute_heavy=args.compute_heavy,
        )
        if m is None:
            print(f"[{i+1}/{len(test_perts)}] {pert}: evaluator None; skip")
            continue

        m.update({
            "method": args.method_label,
            "setting": args.setting,
            "target_line": args.target_line,
            "pert": pert,
            "pert_in_state_vocab": pert in state_pert_vocab,
            "n_state_pred_cells": n_pred,
            "our_genes_covered_by_state": remap_stats["our_genes_covered_by_state"],
            "seed": args.seed,
            "run_id": run_id,
        })
        rows.append(m)

        # Incremental CSV save
        pd.DataFrame(rows).to_csv(args.output, index=False)

        if (i + 1) % 50 == 0 or i < 5:
            print(f"[{i+1}/{len(test_perts)}] {pert}: "
                  f"Pf_ctrl={m['pearson_ctrl']:.3f}  Pf_pert={m['pearson_pert']:.3f}  "
                  f"MMD={m['mmd']:.4f}  OOV={not m['pert_in_state_vocab']}")

    if not rows:
        print("No perts evaluated.")
        return

    # --- Finalize + summary ---
    df = pd.DataFrame(rows)
    front = ["method", "setting", "target_line", "pert"]
    metric_cols = [c for c in evaluator.METRIC_COLUMNS if c in df.columns]
    extra_cols = [c for c in df.columns if c not in front + metric_cols]
    df = df[front + metric_cols + extra_cols]
    df.to_csv(args.output, index=False)

    # Two summaries: all rows + in-vocab-only rows
    out_base = args.output.with_suffix("")
    for tag, subset in [("summary_all", df),
                        ("summary_invocab", df[df["pert_in_state_vocab"]])]:
        if len(subset) == 0:
            continue
        agg = subset[[c for c in metric_cols if c != "n_target_cells"]
                     ].agg(["mean", "median", "std"]).T
        agg_path = Path(f"{out_base}_{tag}.csv")
        agg.reset_index().rename(columns={"index": "metric"}).to_csv(agg_path, index=False)
        print(f"Saved {agg_path} ({len(subset)} rows)")

    print(f"\nn_perts_evaluated: {len(df)} "
          f"(in-vocab: {int(df['pert_in_state_vocab'].sum())}, "
          f"OOV: {int((~df['pert_in_state_vocab']).sum())})")
    for col in ["pearson_ctrl", "pearson_pert", "pearson20_pert",
                "centroid_accuracy", "mmd", "energy"]:
        if col in df.columns:
            print(f"  {col:20s}: mean={df[col].mean():.4f}  median={df[col].median():.4f}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
