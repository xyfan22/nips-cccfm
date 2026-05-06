"""R2 mean-preserving diagnostic (advisor 2026-04-26 round 5).

Two questions, both answered without retraining the velocity field:

  Q1 (recenter): Is the cell-level shape that R2 learned useful, only its
                 mean drifted? If so, post-hoc recentering each pert's pred
                 to anchor mean should recover the anchor's perturbation
                 metrics AND the residual flow's distributional structure.

  Q2 (lambda sweep): Does a SMALL residual strength help while a full
                 strength hurts? lambda=0 ≡ shifted_control (no flow at all);
                 lambda=1 ≡ raw R2 output. Sweep finds the sweet spot.

Implementation cost:
  - 1 ODE integration per pert (expensive part, same as 31_eval_cccfm).
  - 10 lightweight metric computations per pert (5 lambda × 2 recenter).
  Total ~ R2.4 eval wall × 1 ≈ 30 min per setting on 3090.

Output: `results/cccfm/r2_diagnostic_{setting}.csv` with one row per
(pert, lambda, recenter) tuple plus standard METRIC_COLUMNS.

Usage:
    python scripts/09_r2_recenter_lambda_sweep.py \
        --run_dir runs/cccfm/residual_linear_pca_2line \
        --output results/cccfm/r2_diagnostic_2line.csv
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
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=Path, required=True,
                   help="R2.4 run dir (must contain best.ckpt + config.json + anchor_artifacts.npz)")
    p.add_argument("--ckpt_name", type=str, default="best", choices=["best", "last"])
    p.add_argument("--data_path", type=Path,
                   default=Path("data/processed/replogle_preprocessed.h5ad"))
    p.add_argument("--primary_embed", type=Path,
                   default=Path("data/processed/primary_gene_embedding.pt"))
    p.add_argument("--test_slice", type=Path,
                   default=Path("data/splits/jurkat_test_strict.json"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--max_n_samples", type=int, default=2000)
    p.add_argument("--n_steps", type=int, default=50)
    p.add_argument("--ode_method", type=str, default="heun",
                   choices=["euler", "heun", "dopri5"])
    p.add_argument("--lambdas", nargs="+", type=float,
                   default=[0.0, 0.1, 0.25, 0.5, 1.0])
    p.add_argument("--min_cells", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pert_subset", nargs="*", default=None,
                   help="for smoke; restrict eval to these perts")
    return p.parse_args()


def main():
    args = parse_args()
    import anndata as ad
    import numpy as np
    import pandas as pd
    import torch

    from cccfm.anchor import apply_anchor_torch
    from cccfm.data import TargetControls
    from cccfm.eval_metrics import CrossLineEvaluator
    from cccfm.flow_module import CCCFMLitModule
    from cccfm.pert_tokens import PertTokenizer
    from cccfm.velocity_net import VelocityNet

    run_dir = args.run_dir
    cfg = json.loads((run_dir / "config.json").read_text())
    print(f"[diag] run_id={cfg['run_id']}  encoder={cfg['encoder']}  setting={cfg['setting']}  "
          f"anchor={cfg.get('anchor', 'none')}")
    assert cfg.get("anchor_mode") == "residual", \
        f"diagnostic only meaningful for residual runs; got {cfg.get('anchor_mode')}"

    ckpt_path = run_dir / f"{args.ckpt_name}.ckpt"
    assert ckpt_path.exists(), f"missing {ckpt_path}"

    sources = cfg.get("sources", SETTING_SOURCES[cfg["setting"]])
    target_line = cfg.get("target_line", "Jurkat")

    # ----- load adata + evaluator (full 577 test perts for correct centroids) ----- #
    t0 = time.perf_counter()
    adata = ad.read_h5ad(args.data_path)
    obsm_key = cfg["obsm_key"]
    test_perts_full = sorted(json.loads(args.test_slice.read_text()))
    if args.pert_subset:
        eval_perts = [p for p in test_perts_full if p in set(args.pert_subset)]
    else:
        eval_perts = test_perts_full
    evaluator = CrossLineEvaluator(
        adata=adata, source_lines=sources, target_line=target_line,
        test_perts=test_perts_full,
        min_cells=args.min_cells,
    )
    target_ctrls = TargetControls(adata, target_line=target_line, encoder_key=cfg["encoder"])
    print(f"[diag] target ctrl pool: {target_ctrls.n} cells")

    # ----- replay h alignment if any (parity with eval script) ----- #
    align_path = run_dir / "h_alignment.npz"
    if align_path.exists():
        npz = np.load(align_path, allow_pickle=True)
        mode = str(npz["mode"]) if "mode" in npz.files else "zscore"
        H = np.asarray(adata.obsm[obsm_key], dtype=np.float32)
        if mode == "zscore":
            adata.obsm[obsm_key] = ((H - npz["mean"]) / npz["std"]).astype(np.float32)
        elif mode == "zscore_per_line":
            cell_lines = [str(x) for x in npz["cell_lines"].tolist()]
            means, stds = npz["means"].astype(np.float32), npz["stds"].astype(np.float32)
            H_out = H.copy()
            line_arr = adata.obs["cell_line"].astype(str).values
            for i, cl in enumerate(cell_lines):
                m = (line_arr == cl)
                if m.any():
                    H_out[m] = (H[m] - means[i]) / stds[i]
            adata.obsm[obsm_key] = H_out.astype(np.float32)
        # rebuild target_ctrls so it picks up the aligned obsm
        target_ctrls = TargetControls(adata, target_line=target_line, encoder_key=cfg["encoder"])
        print(f"[diag] applied h_alignment mode={mode}")

    # ----- load anchor artifacts ----- #
    anc = np.load(run_dir / "anchor_artifacts.npz", allow_pickle=True)
    anchor_perts = [str(p) for p in anc["anchor_perts"].tolist()]
    anchor_deltas = anc["anchor_deltas"].astype(np.float32)
    line_names = [str(l) for l in anc["line_names"].tolist()]
    line_means = anc["line_ctrl_means"].astype(np.float32)
    anchor_delta_dict = {p: anchor_deltas[i] for i, p in enumerate(anchor_perts)}
    line_ctrl_means = {l: line_means[i] for i, l in enumerate(line_names)}
    target_ctrl_mean_np = line_ctrl_means[target_line]
    print(f"[diag] anchor: {len(anchor_delta_dict)} perts; lines={list(line_ctrl_means)}")

    # ----- rebuild + load lit module ----- #
    mcfg = cfg["model"]
    tokenizer = PertTokenizer(
        mode=mcfg.get("pert_mode", "gene_embedding"),
        primary_embedding_path=args.primary_embed,
        out_dim=mcfg["pert_dim"],
        freeze_embedding=True,
    )
    velocity = VelocityNet(
        gene_dim=cfg["gene_dim"], context_dim=cfg["context_dim"],
        pert_dim=mcfg["pert_dim"], hidden_dim=mcfg["hidden_dim"],
        n_blocks=mcfg["n_blocks"], time_dim=mcfg["time_dim"],
        dropout=mcfg.get("dropout", 0.0),
        zero_init_output=True,  # match training-time init (no effect on loaded weights)
    )
    lit = CCCFMLitModule.load_from_checkpoint(
        str(ckpt_path),
        velocity_net=velocity, pert_tokenizer=tokenizer,
        anchor_mode="residual",
        anchor_delta_dict=anchor_delta_dict,
        line_ctrl_means=line_ctrl_means,
        map_location="cpu",
    )
    lit.eval()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    lit.to(device)
    print(f"[diag] loaded {ckpt_path}  device={device}")

    # ----- per-pert: ODE once, blend × recenter cheaply ----- #
    lambdas = list(args.lambdas)
    rng = np.random.default_rng(args.seed)
    rows = []
    n_evaluated = n_skipped = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def flush():
        if not rows:
            return
        df = pd.DataFrame(rows)
        front = ["method", "setting", "target_line", "pert", "lambda", "recenter"]
        metric_cols = [c for c in evaluator.METRIC_COLUMNS if c in df.columns]
        extras = [c for c in df.columns if c not in front + metric_cols]
        df[front + metric_cols + extras].to_csv(args.output, index=False)

    for k, pert in enumerate(eval_perts):
        if pert not in evaluator.target_pert_centroids:
            n_skipped += 1
            continue
        if pert not in anchor_delta_dict:
            n_skipped += 1
            continue

        n_target = evaluator.target_pert_n_cells[pert]
        n_samp = min(n_target, args.max_n_samples)

        # Sample full ctrl cloud (matched to 12_eval_cccfm.py round-4 fix)
        x0_full_np, h_full_np = target_ctrls.sample_full(n_samp, rng)
        x0_full = torch.from_numpy(x0_full_np).to(device)
        h_full = torch.from_numpy(h_full_np).to(device)
        delta_g = lit.anchor_delta_matrix[lit.anchor_pert_to_idx[pert]]
        ctx_mean = lit.line_ctrl_mean_matrix[lit.line_to_idx[target_line]]
        x0_anchor_full = apply_anchor_torch(x0_full, ctx_mean, delta_g)

        # ODE integrate -> raw R2 output (lambda=1, no recenter)
        preds = []
        for start in range(0, x0_anchor_full.shape[0], args.batch_size):
            end = start + args.batch_size
            x0c = x0_anchor_full[start:end]
            hc = h_full[start:end]
            B = x0c.shape[0]
            z_pert = lit.pert_tokenizer([pert] * B)
            with torch.no_grad():
                x1c = lit.predict_perturbed(
                    x0c, hc, z_pert, n_steps=args.n_steps, method=args.ode_method,
                )
            preds.append(x1c.detach().cpu().numpy())
        pred_raw = np.concatenate(preds, axis=0)              # [N, G]  -- lambda=1, unrecentered
        x0_anchor_np = x0_anchor_full.cpu().numpy()           # [N, G]
        residual = pred_raw - x0_anchor_np                    # what the flow added
        target_anchor_profile = target_ctrl_mean_np + anchor_delta_dict[pert]   # [G]

        # 5 lambdas × 2 recenter modes = 10 lightweight evaluations
        for lam in lambdas:
            blend = x0_anchor_np + lam * residual
            for recenter in (False, True):
                if recenter:
                    pred_eval = blend - blend.mean(axis=0, keepdims=True) + target_anchor_profile
                else:
                    pred_eval = blend
                m = evaluator.evaluate_pert_cell_level(
                    pert, pred_eval.astype(np.float32), compute_heavy=True,
                )
                if m is None:
                    continue
                m.update({
                    "method": f"r2_lambda{lam}_{'recentered' if recenter else 'raw'}",
                    "setting": cfg["setting"],
                    "target_line": target_line,
                    "pert": pert,
                    "lambda": lam,
                    "recenter": bool(recenter),
                    "encoder": cfg["encoder"],
                    "anchor_type": cfg.get("anchor", "none"),
                    "n_pred_cells": int(pred_eval.shape[0]),
                    "n_steps": args.n_steps,
                    "ode_method": args.ode_method,
                    "seed": args.seed,
                    "run_id": cfg["run_id"],
                })
                rows.append(m)
        n_evaluated += 1

        if (k + 1) % 10 == 0 or (k + 1) == len(eval_perts):
            flush()
            print(f"  [{k+1:>4d}/{len(eval_perts)}] evaluated={n_evaluated} skipped={n_skipped}  "
                  f"wall={time.perf_counter()-t0:.0f}s")

    flush()

    # ----- per-(lambda, recenter) summary ----- #
    df = pd.DataFrame(rows)
    if df.empty:
        print("[diag] no rows evaluated.")
        return
    metric_cols = [c for c in evaluator.METRIC_COLUMNS if c in df.columns and c != "n_target_cells"]
    agg = df.groupby(["lambda", "recenter"])[metric_cols].mean().reset_index()
    agg_path = args.output.with_name(args.output.stem + "_summary.csv")
    agg.to_csv(agg_path, index=False)
    print(f"\n=== diagnostic summary (mean over {n_evaluated} perts, {cfg['setting']}) ===")
    print(f"{'lambda':>6s} {'rec':>4s} {'mse':>8s} {'mmd':>8s} {'energy':>8s} {'w2':>8s} {'mom':>8s} "
          f"{'Pf^p':>8s} {'Pf20p':>8s} {'C-Acc':>8s}")
    for _, r in agg.iterrows():
        print(f"{r['lambda']:>6.2f} {str(bool(r['recenter']))[0]:>4s} "
              f"{r.get('mse', 0):>8.4f} {r.get('mmd', 0):>8.4f} {r.get('energy', 0):>8.3f} "
              f"{r.get('w2_sinkhorn', 0):>8.4f} {r.get('moment', 0):>8.4f} "
              f"{r.get('pearson_pert', 0):>8.4f} {r.get('pearson20_pert', 0):>8.4f} "
              f"{r.get('centroid_accuracy', 0):>8.4f}")
    print(f"\n[diag] wrote {args.output}")
    print(f"[diag] wrote {agg_path}")


if __name__ == "__main__":
    main()
