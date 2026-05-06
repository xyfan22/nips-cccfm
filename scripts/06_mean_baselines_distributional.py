"""Cell-level distributional re-evaluation for mean-only reference baselines.

Three modes:

  1. degenerate         (the original strawman; kept only for compatibility).
                        pred_cells = profile.repeat(N) — N identical copies of
                        the predicted pseudobulk.

  2. shifted_control    (the fair version). For each pert, sample N target-line
                        control cells, mean-center them so their sample mean
                        equals the global target_ctrl mean, then translate by
                        the predicted delta:

                          ctrl_centered = ctrl - ctrl.mean(0) + tgt_ctrl
                          delta         = profile - tgt_ctrl
                          pred_cells    = ctrl_centered + delta

                        Guarantees `pred_cells.mean(0) == profile` exactly,
                        isolates distributional gain to "adopting target-line
                        control variability" (rather than perturbation-specific
                        structure).

  3. control_copy       (absolute lower bound; method == 'target_control_copy'.)
                        Predicts no perturbation effect; pred_cells just is
                        the (mean-centered) target-line control cell cloud.

CSV schema additions per advisor:
  - distribution_mode      one of {degenerate, shifted_control, control_copy}
  - centered_controls      bool (True for shifted_control / control_copy, False for degenerate)
  - degenerate_pred        bool (True for degenerate)
  - n_pred_cells           int

Per-gene moment metric: with the recent eval_metrics.py fix the third moment
term collapses to 0 for any column with zero variance (degenerate) but
contributes properly for shifted_control / control_copy.

Usage:
    # main "fair" run for paper:
    python scripts/06_mean_baselines_distributional.py \\
        --setting 2line \\
        --modes shifted_control control_copy

    # full sweep (also re-emits degenerate, idempotent):
    python scripts/06_mean_baselines_distributional.py \\
        --setting 2line \\
        --modes degenerate shifted_control control_copy
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path


SETTING_SOURCES = {
    "2line": ["K562", "RPE1"],
    "3line": ["K562", "RPE1", "HepG2"],
    "target_K562_2src": ["RPE1", "HepG2"],
    "target_RPE1_2src": ["K562", "HepG2"],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=Path,
                   default=Path("data/processed/replogle_preprocessed.h5ad"))
    p.add_argument("--test_slice", type=Path,
                   default=Path("data/splits/jurkat_test_strict.json"))
    p.add_argument("--shared_perts", type=Path,
                   default=Path("data/splits/shared_perturbations.json"))
    p.add_argument("--primary_embedding_path", type=Path,
                   default=Path("data/processed/primary_gene_embedding.pt"))
    p.add_argument("--setting", choices=list(SETTING_SOURCES), required=True)
    p.add_argument("--target_line", type=str, default="Jurkat")
    p.add_argument("--anchor_path", type=Path, default=None,
                   help="joblib ridge anchor (defaults to runs/anchors/pooled_ridge_anchor_<setting>.joblib)")
    p.add_argument("--output_dir", type=Path, default=Path("results/baselines"))
    p.add_argument("--modes", nargs="+",
                   default=["shifted_control", "control_copy"],
                   choices=["degenerate", "shifted_control", "control_copy"],
                   help="which distributional modes to compute")
    p.add_argument("--min_cells", type=int, default=50)
    p.add_argument("--max_n_samples", type=int, default=2000,
                   help="cap on number of pred cells per pert")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    import anndata as ad
    import numpy as np
    import pandas as pd
    import torch

    from cccfm.anchor import PooledGeneEmbeddingRidgeAnchor
    from cccfm.eval_metrics import CrossLineEvaluator

    sources = SETTING_SOURCES[args.setting]
    target = args.target_line
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[{run_id}] setting={args.setting}  sources={sources}  target={target}  "
          f"modes={args.modes}")

    adata = ad.read_h5ad(args.data_path)
    test_perts = sorted(json.loads(args.test_slice.read_text()))
    evaluator = CrossLineEvaluator(
        adata=adata, source_lines=sources, target_line=target,
        test_perts=test_perts, min_cells=args.min_cells,
    )

    # -------------------- pseudobulk helpers + caches -------------------- #
    def _dense(X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.asarray(X, dtype=np.float64)

    def pb(mask) -> np.ndarray:
        return _dense(adata[mask].X).mean(axis=0)

    src_mask_all = adata.obs["cell_line"].isin(sources).values
    src_ctrl_mean_pool = pb(src_mask_all & (adata.obs["pert"].values == "control"))
    per_line_ctrl = {
        cl: pb((adata.obs["cell_line"].values == cl) & (adata.obs["pert"].values == "control"))
        for cl in sources
    }
    tgt_ctrl = evaluator.target_ctrl_reference  # [G]

    # full target-control cell matrix — used to draw shifted/copy samples
    tgt_ctrl_mask = (adata.obs["cell_line"].values == target) & (adata.obs["pert"].values == "control")
    tgt_ctrl_X = _dense(adata[tgt_ctrl_mask].X).astype(np.float32)  # [N_tgt_ctrl, G]
    print(f"   target_ctrl pool: {tgt_ctrl_X.shape[0]} cells × {tgt_ctrl_X.shape[1]} genes")

    src_pert_mean_pooled: dict[str, np.ndarray] = {}
    src_pert_delta_by_line: dict[str, list[np.ndarray]] = {}
    for cl in sources:
        lmask = adata.obs["cell_line"].values == cl
        for pert in sorted(set(adata.obs[lmask]["pert"].unique()) - {"control"}):
            pmask = lmask & (adata.obs["pert"].values == pert)
            if int(pmask.sum()) < args.min_cells:
                continue
            src_pert_delta_by_line.setdefault(pert, []).append(pb(pmask) - per_line_ctrl[cl])
    for pert in src_pert_delta_by_line:
        pmask = src_mask_all & (adata.obs["pert"].values == pert)
        if int(pmask.sum()) >= args.min_cells:
            src_pert_mean_pooled[pert] = pb(pmask)
    print(f"   source per-pert cache: {len(src_pert_mean_pooled)} perts pooled, "
          f"{len(src_pert_delta_by_line)} perts per-line")

    anchor_path = args.anchor_path or Path(f"runs/anchors/pooled_ridge_anchor_{args.setting}.joblib")
    anchor, gene_embed = None, None
    if anchor_path.exists():
        anchor = PooledGeneEmbeddingRidgeAnchor().load(anchor_path)
        gene_embed = torch.load(args.primary_embedding_path, map_location="cpu", weights_only=False)
        print(f"   loaded ridge anchor: {anchor_path}  alpha={anchor.alpha}")

    # -------------------- mode + method routing -------------------- #
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    profile_funcs = {
        "linear_ahlmann_eltze": lambda p: _linear_profile(
            p, tgt_ctrl, src_pert_mean_pooled, src_ctrl_mean_pool),
        "exact_gene_retrieval": lambda p: _exact_profile(
            p, tgt_ctrl, src_pert_delta_by_line),
    }
    if anchor is not None:
        profile_funcs["pooled_ridge_anchor"] = lambda p: _ridge_profile(
            p, tgt_ctrl, anchor, gene_embed)

    summary_rows = []   # for headline printout

    for mode in args.modes:
        print(f"\n=== mode = {mode} ===")
        if mode == "control_copy":
            # Single "method" — predicts no pert effect, just target-control cloud.
            method_name = "target_control_copy"
            csv_path = args.output_dir / f"{method_name}_{args.setting}.csv"
            agg = run_one(
                method_name=method_name, mode=mode,
                profile_fn=None, evaluator=evaluator, test_perts=test_perts,
                tgt_ctrl=tgt_ctrl, tgt_ctrl_X=tgt_ctrl_X,
                rng=rng, args=args, csv_path=csv_path, run_id=run_id,
            )
            if agg is not None:
                summary_rows.append((method_name, mode, agg))
        else:
            for base_method, profile_fn in profile_funcs.items():
                method_name = f"{base_method}_{mode}"
                csv_path = args.output_dir / f"{base_method}_{args.setting}_{mode}.csv"
                agg = run_one(
                    method_name=method_name, mode=mode,
                    profile_fn=profile_fn, evaluator=evaluator, test_perts=test_perts,
                    tgt_ctrl=tgt_ctrl, tgt_ctrl_X=tgt_ctrl_X,
                    rng=rng, args=args, csv_path=csv_path, run_id=run_id,
                )
                if agg is not None:
                    summary_rows.append((method_name, mode, agg))

    # headline printout
    print("\n=== distributional summary (mean across evaluated perts) ===")
    cols = ["mmd", "energy", "w2_sinkhorn", "moment",
            "pearson_ctrl", "pearson_pert", "centroid_accuracy"]
    print(f"{'method':40s} {'mode':18s} " + " ".join(f"{c:>13s}" for c in cols))
    for method, mode, agg in summary_rows:
        row = [f"{method:40s}", f"{mode:18s}"]
        for c in cols:
            if c in agg.index and not (agg.at[c, 'mean'] != agg.at[c, 'mean']):
                row.append(f"{agg.at[c,'mean']:13.4f}")
            else:
                row.append(f"{'—':>13s}")
        print(" ".join(row))


# --------------------------------------------------------------------------- #
# Per-method/mode loop
# --------------------------------------------------------------------------- #

def run_one(
    method_name: str,
    mode: str,
    profile_fn,
    evaluator,
    test_perts,
    tgt_ctrl,
    tgt_ctrl_X,
    rng,
    args,
    csv_path: Path,
    run_id: str,
):
    import numpy as np
    import pandas as pd

    target = args.target_line
    rows = []
    n_eval = n_skipped = 0
    t0 = time.perf_counter()

    centered_controls = mode in ("shifted_control", "control_copy")
    degenerate_pred = (mode == "degenerate")

    for k, pert in enumerate(test_perts):
        # for control_copy we don't need a profile — but we still skip if eval rejects
        if profile_fn is None:
            profile = tgt_ctrl  # placeholder; pred_cells uses tgt_ctrl_X centred
        else:
            profile = profile_fn(pert)
            if profile is None:
                n_skipped += 1
                continue

        n_target = evaluator.target_pert_n_cells.get(pert, 0)
        if n_target < args.min_cells:
            n_skipped += 1
            continue
        n_samp = min(n_target, args.max_n_samples)

        pred_cells = make_pred_cells(
            mode=mode, profile=profile,
            tgt_ctrl=tgt_ctrl, tgt_ctrl_X=tgt_ctrl_X,
            n_samp=n_samp, rng=rng,
        )

        m = evaluator.evaluate_pert_cell_level(pert, pred_cells, compute_heavy=True)
        if m is None:
            n_skipped += 1
            continue

        m.update({
            "method": method_name,
            "setting": args.setting,
            "target_line": target,
            "pert": pert,
            "distribution_mode": mode,
            "centered_controls": centered_controls,
            "degenerate_pred": degenerate_pred,
            "n_pred_cells": int(n_samp),
            "seed": args.seed,
            "run_id": run_id,
        })
        rows.append(m)
        n_eval += 1

        if (k + 1) % 50 == 0 or (k + 1) == len(test_perts):
            df_tmp = pd.DataFrame(rows)
            front = ["method", "setting", "target_line", "pert"]
            metric_cols = [c for c in evaluator.METRIC_COLUMNS if c in df_tmp.columns]
            extras = [c for c in df_tmp.columns if c not in front + metric_cols]
            df_tmp[front + metric_cols + extras].to_csv(csv_path, index=False)

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"   {method_name}: nothing evaluated; skipping.")
        return None

    front = ["method", "setting", "target_line", "pert"]
    metric_cols = [c for c in evaluator.METRIC_COLUMNS if c in df.columns]
    extras = [c for c in df.columns if c not in front + metric_cols]
    df = df[front + metric_cols + extras]
    df.to_csv(csv_path, index=False)

    agg = df[[c for c in metric_cols if c != "n_target_cells"]].agg(["mean", "median", "std"]).T
    agg_path = csv_path.with_name(csv_path.stem + "_summary.csv")
    agg.reset_index().rename(columns={"index": "metric"}).to_csv(agg_path, index=False)

    wall = time.perf_counter() - t0
    print(f"   {method_name:40s}  evaluated={n_eval}  skipped={n_skipped}  "
          f"wall={wall/60:.1f}min  -> {csv_path.name}")
    return agg


# --------------------------------------------------------------------------- #
# Pred-cells construction (the advisor-corrected version)
# --------------------------------------------------------------------------- #

def make_pred_cells(
    mode: str,
    profile,
    tgt_ctrl,
    tgt_ctrl_X,
    n_samp: int,
    rng,
):
    """Build [n_samp, G] predicted cells per advisor-spec.

    profile      [G]   method's predicted profile = tgt_ctrl + delta
                       (ignored for mode='control_copy')
    tgt_ctrl     [G]   target-line control mean (full population)
    tgt_ctrl_X   [N,G] target-line control cell matrix (CPU dense float32)
    n_samp       int   number of pred cells to generate (= match target_pert_n_cells)
    rng          numpy Generator
    """
    import numpy as np

    n_total = tgt_ctrl_X.shape[0]
    G = tgt_ctrl_X.shape[1]
    profile = np.asarray(profile, dtype=np.float32)

    if mode == "degenerate":
        return np.broadcast_to(profile, (n_samp, G)).copy().astype(np.float32)

    idx = rng.choice(n_total, size=n_samp, replace=(n_samp > n_total))
    ctrl = tgt_ctrl_X[idx].astype(np.float32)             # [n_samp, G]
    ctrl_centered = (ctrl - ctrl.mean(axis=0, keepdims=True)
                     + tgt_ctrl.astype(np.float32))        # mean(ctrl_centered) == tgt_ctrl

    if mode == "control_copy":
        return ctrl_centered

    if mode == "shifted_control":
        delta = (profile - tgt_ctrl).astype(np.float32)    # [G]
        return ctrl_centered + delta                        # mean(...) == profile

    raise ValueError(f"unknown mode '{mode}'")


# --------------------------------------------------------------------------- #
# Per-method profile helpers (unchanged)
# --------------------------------------------------------------------------- #

def _linear_profile(pert, tgt_ctrl, src_pert_mean_pooled, src_ctrl_mean_pool):
    pm = src_pert_mean_pooled.get(pert)
    if pm is None:
        return None
    return tgt_ctrl + (pm - src_ctrl_mean_pool)


def _exact_profile(pert, tgt_ctrl, src_pert_delta_by_line):
    deltas = src_pert_delta_by_line.get(pert)
    if not deltas:
        return None
    import numpy as np
    return tgt_ctrl + np.stack(deltas).mean(axis=0)


def _ridge_profile(pert, tgt_ctrl, anchor, gene_embed):
    import numpy as np
    if pert not in gene_embed:
        return None
    x_in = gene_embed[pert].numpy().astype(np.float32).reshape(1, -1)
    pred_delta = anchor.predict(x_in).flatten()
    return tgt_ctrl + pred_delta


if __name__ == "__main__":
    main()