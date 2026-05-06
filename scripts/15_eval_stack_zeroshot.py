"""STACK zero-shot baseline (Arc Institute 2026) for ccCFM.

No training. Arc's `Stack-Large-Aligned` is an in-context foundation model:
  - base (context) = source-line cells carrying each test perturbation label
  - test (query)   = target-line (Jurkat) control cells
  - For each pert p in base[--split-column], STACK concatenates
    base[pert==p] + test_cells in a 256-cell attention window and iteratively
    denoises; output = counterfactual expression of test_cells under pert p.

Evaluation: per-pert cell-level predictions fed to
`CrossLineEvaluator.evaluate_pert_cell_level(pert, pred_cells)`.

Paper caveat
------------
STACK was pretrained on CellxGene + Parse_bio (no CRISPRi, no Replogle). This is
a genuine OOD zero-shot test for our cross-line CRISPRi benchmark.

Pipeline
--------
1. Build:
     runs/stack_zeroshot/stack_base.h5ad  — source-line cells where pert ∈ test_perts
     runs/stack_zeroshot/stack_test.h5ad  — target-line control cells
2. Shell out `stack-generation --split-column pert --concatenate` →
     runs/stack_zeroshot/stack_output/*.h5ad
3. Per-pert: remap STACK's 15,012-gene output back to our 6,642 HVG
   (zero-pad missing), call the evaluator, append CSV row.

Cost: ~10-30 GPU-hours for 577 perts on A100/H100.

Usage
-----
python scripts/15_eval_stack_zeroshot.py \\
    --stack_checkpoint baselines/stack/pretrained/Stack-Large-Aligned/bc_large_aligned.ckpt \\
    --stack_genelist   baselines/stack/pretrained/Stack-Large-Aligned/basecount_1000per_15000max.pkl \\
    --workdir runs/stack_zeroshot \\
    --output  results/baselines/stack_zeroshot.csv
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
from datetime import datetime
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=Path,
                   default=Path("data/processed/replogle_preprocessed.h5ad"))
    p.add_argument("--test_slice", type=Path,
                   default=Path("data/splits/jurkat_test_strict.json"))
    p.add_argument("--target_line", type=str, default="Jurkat")
    p.add_argument("--sources", nargs="+", default=["K562", "RPE1", "HepG2"],
                   help="Source lines providing base context cells (STACK is zero-shot; "
                        "we use all 3 non-target lines to maximise context pool).")
    p.add_argument("--setting", choices=["2line", "3line", "zeroshot"], default="zeroshot",
                   help="CSV label only — STACK's training split is fixed.")
    # STACK specifics
    p.add_argument("--stack_checkpoint", type=Path, required=True,
                   help="e.g. baselines/stack/pretrained/Stack-Large-Aligned/bc_large_aligned.ckpt")
    p.add_argument("--stack_genelist", type=Path, required=True,
                   help="e.g. baselines/stack/pretrained/Stack-Large-Aligned/basecount_1000per_15000max.pkl")
    p.add_argument("--batch_size", type=int, default=16,
                   help="stack-generation --batch-size (attention windows per call)")
    p.add_argument("--num_steps", type=int, default=5,
                   help="stack-generation --num-steps (iterative denoising)")
    p.add_argument("--prompt_ratio", type=float, default=0.25)
    p.add_argument("--context_ratio", type=float, default=0.4)
    p.add_argument("--max_base_cells_per_pert", type=int, default=500,
                   help="Cap per-pert context cells; 0 disables cap.")
    # orchestration
    p.add_argument("--workdir", type=Path, required=True,
                   help="root for stack_base.h5ad / stack_test.h5ad / stack_output/")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pert_subset", type=int, default=None,
                   help="Keep first N test perts (staged rollout).")
    p.add_argument("--min_cells", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_build_input", action="store_true")
    p.add_argument("--skip_infer", action="store_true")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# STACK vocab helper
# --------------------------------------------------------------------------- #

def load_stack_gene_list(pkl_path: Path) -> list[str]:
    """Load STACK's 15k gene list from `basecount_*.pkl`.

    The pkl is reported as either a list[str] or dict[str, int] or dict[int, str].
    Return an ordered list of gene symbols aligned to STACK's tensor columns.
    """
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, list):
        return [str(g) for g in obj]
    if isinstance(obj, dict):
        first_key = next(iter(obj.keys()))
        if isinstance(first_key, str):
            # {symbol: idx}
            return sorted(obj.keys(), key=lambda k: int(obj[k]))
        else:
            # {idx: symbol}
            return [str(obj[i]) for i in sorted(obj.keys())]
    raise TypeError(f"Unexpected genelist pkl format: {type(obj)}. "
                    "Patch `load_stack_gene_list` for the actual schema.")


def invert_gene_alignment(X_stack, stack_genes: list[str], our_genes: list[str]):
    """STACK output [N, len(stack_genes)] → [N, len(our_genes)], zero-fill missing."""
    import numpy as np
    s2i = {g: i for i, g in enumerate(stack_genes)}
    n_ours = len(our_genes)
    pick = np.full(n_ours, -1, dtype=np.int64)
    for i, g in enumerate(our_genes):
        if g in s2i:
            pick[i] = s2i[g]
    X_stack = np.asarray(X_stack, dtype=np.float32)
    out = np.zeros((X_stack.shape[0], n_ours), dtype=np.float32)
    mask = pick >= 0
    out[:, mask] = X_stack[:, pick[mask]]
    return out, {"our_genes_covered_by_stack": int(mask.sum()),
                 "our_genes_not_in_stack": int((pick < 0).sum())}


# --------------------------------------------------------------------------- #
# Step 1: build base / test h5ad
# --------------------------------------------------------------------------- #

def build_stack_inputs(full_adata_path: Path, sources: list[str], target_line: str,
                       test_perts: list[str], workdir: Path,
                       max_base_cells_per_pert: int, seed: int):
    import anndata as ad
    import numpy as np

    print(f"Loading {full_adata_path}")
    full = ad.read_h5ad(full_adata_path)
    obs = full.obs
    workdir.mkdir(parents=True, exist_ok=True)

    # --- Base: source-line perturbed cells for our test perts ---
    keep_perts = set(test_perts)
    base_mask = obs["cell_line"].isin(sources) & obs["pert"].isin(keep_perts)
    base = full[base_mask].copy()
    base.obs["pert"] = base.obs["pert"].astype(str)
    base.obs["cell_line"] = base.obs["cell_line"].astype(str)

    # Per-pert cap
    if max_base_cells_per_pert and max_base_cells_per_pert > 0:
        rng = np.random.default_rng(seed)
        keep_idx = []
        for pert in sorted(base.obs["pert"].unique()):
            idx = np.where(base.obs["pert"].values == pert)[0]
            if len(idx) > max_base_cells_per_pert:
                idx = rng.choice(idx, max_base_cells_per_pert, replace=False)
            keep_idx.extend(idx.tolist())
        base = base[sorted(keep_idx)].copy()

    # Ensure gene_symbol column holds HGNC symbols for STACK --gene-name-col
    if "gene_name" in base.var.columns:
        base.var["gene_symbol"] = base.var["gene_name"].astype(str)
    elif "gene_symbol" not in base.var.columns:
        base.var["gene_symbol"] = base.var_names.astype(str)

    # Deduplicate genes by HGNC symbol (keep first Ensembl ID for each)
    dup_mask = base.var["gene_symbol"].str.upper().duplicated(keep="first")
    if dup_mask.any():
        print(f"Dropping {dup_mask.sum()} duplicate gene_symbol entries")
        base = base[:, ~dup_mask.values].copy()

    base_path = workdir / "stack_base.h5ad"
    base.write_h5ad(base_path)
    print(f"base: {base.n_obs} cells × {base.n_vars} genes, "
          f"{base.obs['pert'].nunique()} unique perts")

    # --- Test: target-line control cells ---
    test_mask = (obs["cell_line"] == target_line) & (obs["pert"] == "control")
    test = full[test_mask].copy()
    test.obs["pert"] = "control"
    test.obs["cell_line"] = test.obs["cell_line"].astype(str)
    if "gene_name" in test.var.columns:
        test.var["gene_symbol"] = test.var["gene_name"].astype(str)
    elif "gene_symbol" not in test.var.columns:
        test.var["gene_symbol"] = test.var_names.astype(str)

    dup_mask_t = test.var["gene_symbol"].str.upper().duplicated(keep="first")
    if dup_mask_t.any():
        print(f"Dropping {dup_mask_t.sum()} duplicate gene_symbol entries in test")
        test = test[:, ~dup_mask_t.values].copy()

    test_path = workdir / "stack_test.h5ad"
    test.write_h5ad(test_path)
    print(f"test: {test.n_obs} cells × {test.n_vars} genes")

    return base_path, test_path


# --------------------------------------------------------------------------- #
# Step 2: shell `stack-generation`
# --------------------------------------------------------------------------- #

def run_stack_generation(checkpoint: Path, genelist: Path,
                         base_path: Path, test_path: Path, output_dir: Path,
                         batch_size: int, num_steps: int,
                         prompt_ratio: float, context_ratio: float):
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "stack-generation",
        "--checkpoint", str(checkpoint),
        "--base-adata", str(base_path),
        "--test-adata", str(test_path),
        "--genelist", str(genelist),
        "--split-column", "pert",
        "--gene-name-col", "gene_symbol",
        "--batch-size", str(batch_size),
        "--num-steps", str(num_steps),
        "--prompt-ratio", str(prompt_ratio),
        "--context-ratio", str(context_ratio),
        "--output-dir", str(output_dir),
        "--concatenate",
    ]
    print(f"Running:\n  {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()

    with open(args.test_slice) as f:
        test_perts = sorted(json.load(f))
    if args.pert_subset is not None and args.pert_subset < len(test_perts):
        test_perts = test_perts[: args.pert_subset]
    print(f"Test perts: {len(test_perts)}")

    args.workdir.mkdir(parents=True, exist_ok=True)
    base_path = args.workdir / "stack_base.h5ad"
    test_path = args.workdir / "stack_test.h5ad"
    output_dir = args.workdir / "stack_output"

    # --- Step 1 ---
    if not args.skip_build_input:
        base_path, test_path = build_stack_inputs(
            args.data_path, args.sources, args.target_line, test_perts,
            args.workdir, args.max_base_cells_per_pert, args.seed,
        )
    else:
        assert base_path.exists() and test_path.exists(), \
            f"skip_build_input set but inputs missing: {base_path} / {test_path}"

    # --- Step 2 ---
    if not args.skip_infer:
        run_stack_generation(
            args.stack_checkpoint, args.stack_genelist,
            base_path, test_path, output_dir,
            args.batch_size, args.num_steps,
            args.prompt_ratio, args.context_ratio,
        )
    else:
        assert output_dir.exists(), f"skip_infer set but {output_dir} missing"

    # --- Step 3: remap + evaluate ---
    import anndata as ad
    import numpy as np
    import pandas as pd
    from cccfm.eval_metrics import CrossLineEvaluator

    # STACK gene vocab
    stack_genes = load_stack_gene_list(args.stack_genelist)
    print(f"STACK gene list size: {len(stack_genes)}")

    # Locate STACK output. With --concatenate, a single h5ad. Without, per-pert files.
    concat_candidates = (
        list(output_dir.glob("*concatenated*.h5ad"))
        + list(output_dir.glob("*all*.h5ad"))
        + list(output_dir.glob("combined*.h5ad"))
    )
    per_pert_mode = not concat_candidates
    if not per_pert_mode:
        out_concat = ad.read_h5ad(concat_candidates[0])
        print(f"Loaded concatenated output: {concat_candidates[0]}  ({out_concat.n_obs} cells)")
    else:
        per_pert_files = sorted(output_dir.glob("*.h5ad"))
        print(f"Per-pert mode: {len(per_pert_files)} h5ad files in {output_dir}")

    # Full adata for evaluator (covers centroids / DE over source)
    full = ad.read_h5ad(args.data_path)
    if "gene_name" in full.var.columns:
        our_genes = list(full.var["gene_name"].astype(str))
    elif "gene_symbol" in full.var.columns:
        our_genes = list(full.var["gene_symbol"].astype(str))
    else:
        our_genes = list(full.var_names.astype(str))

    evaluator = CrossLineEvaluator(
        adata=full, source_lines=args.sources,
        target_line=args.target_line,
        test_perts=test_perts, min_cells=args.min_cells,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for i, pert in enumerate(test_perts):
        # Locate predictions for this pert
        if not per_pert_mode:
            cand_cols = [c for c in ("split_value", "pert", "split", "sm_name", "perturbation")
                         if c in out_concat.obs.columns]
            if not cand_cols:
                raise RuntimeError(f"STACK output obs has no split column; got {list(out_concat.obs.columns)}")
            col = cand_cols[0]
            mask = out_concat.obs[col].astype(str) == pert
            if mask.sum() == 0:
                print(f"[{i+1}/{len(test_perts)}] {pert}: no output rows (col={col}); skip")
                continue
            pred_X = out_concat[mask].X
        else:
            # filename convention: <pert>.h5ad or similar
            match = [p for p in per_pert_files if pert in p.stem]
            if not match:
                print(f"[{i+1}/{len(test_perts)}] {pert}: no output file; skip")
                continue
            pred_X = ad.read_h5ad(match[0]).X

        if hasattr(pred_X, "toarray"):
            pred_X = pred_X.toarray()
        pred_X = np.asarray(pred_X, dtype=np.float32)
        pred_cells, remap = invert_gene_alignment(pred_X, stack_genes, our_genes)

        m = evaluator.evaluate_pert_cell_level(pert, pred_cells)
        if m is None:
            print(f"[{i+1}/{len(test_perts)}] {pert}: evaluator None; skip")
            continue
        m.update({
            "method": "stack_zeroshot",
            "setting": args.setting,
            "target_line": args.target_line,
            "pert": pert,
            "n_pred_cells": int(pred_cells.shape[0]),
            "our_genes_covered_by_stack": remap["our_genes_covered_by_stack"],
            "seed": args.seed,
            "run_id": run_id,
        })
        rows.append(m)
        pd.DataFrame(rows).to_csv(args.output, index=False)

        if (i + 1) % 50 == 0 or i < 5:
            print(f"[{i+1}/{len(test_perts)}] {pert}: "
                  f"Pf_ctrl={m['pearson_ctrl']:.3f} Pf_pert={m['pearson_pert']:.3f} "
                  f"MMD={m['mmd']:.4f} C-Acc={m['centroid_accuracy']:.3f}")

    if not rows:
        print("No perts evaluated.")
        return

    df = pd.DataFrame(rows)
    front = ["method", "setting", "target_line", "pert"]
    metric_cols = [c for c in evaluator.METRIC_COLUMNS if c in df.columns]
    extra_cols = [c for c in df.columns if c not in front + metric_cols]
    df = df[front + metric_cols + extra_cols]
    df.to_csv(args.output, index=False)

    agg = df[[c for c in metric_cols if c != "n_target_cells"]].agg(["mean", "median", "std"]).T
    agg_path = args.output.with_name(args.output.stem + "_summary.csv")
    agg.reset_index().rename(columns={"index": "metric"}).to_csv(agg_path, index=False)

    print(f"\nn_perts_evaluated: {len(df)}")
    for col in ["pearson_ctrl", "pearson_pert", "pearson20_pert",
                "centroid_accuracy", "mmd", "energy"]:
        if col in df.columns:
            print(f"  {col:20s}: mean={df[col].mean():.4f}  median={df[col].median():.4f}")
    print(f"Saved {args.output} + {agg_path}")


if __name__ == "__main__":
    main()
