"""Evaluate a trained ccCFM checkpoint on Jurkat cross-line holdout.

Loads a checkpoint from scripts/11_train_cccfm.py, then for each test pert:
    1. Sample n_samples Jurkat control cells (same line is the target, these are
       never seen in training).
    2. ODE-integrate via `lit.predict_for_pert(x_0, h, pert_symbol)` -> pred_cells [N, G].
    3. Feed pred_cells to CrossLineEvaluator.evaluate_pert_cell_level.
    4. Stream incremental CSV rows with the standard schema.

Output:
    results/cccfm/<tag>.csv           - per-pert rows (standard METRIC_COLUMNS + ccCFM meta)
    results/cccfm/<tag>_summary.csv   - mean / median / std aggregation

Usage:
    python scripts/12_eval_cccfm.py \
        --run_dir runs/cccfm/smoke_pca_2line \
        --output results/cccfm/smoke_pca_2line.csv
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
    p.add_argument("--run_dir", type=Path, required=True,
                   help="train out_dir; must contain config.json + best.ckpt")
    p.add_argument("--ckpt_name", type=str, default="best",
                   choices=["best", "last"])
    p.add_argument("--data_path", type=Path,
                   default=Path("data/processed/replogle_preprocessed.h5ad"))
    p.add_argument("--primary_embed", type=Path,
                   default=Path("data/processed/primary_gene_embedding.pt"))
    p.add_argument("--test_slice", type=Path,
                   default=Path("data/splits/jurkat_test_strict.json"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--n_samples", type=int, default=-1,
                   help=">0: fixed samples/pert; -1: match target_pert_n_cells")
    p.add_argument("--max_n_samples", type=int, default=2000,
                   help="cap when matching target counts (long-tail protection)")
    p.add_argument("--n_steps", type=int, default=50)
    p.add_argument("--ode_method", type=str, default="heun",
                   choices=["euler", "heun", "dopri5"])
    p.add_argument("--compute_heavy", action="store_true",
                   help="also compute w2_sinkhorn + moment (slower)")
    p.add_argument("--pert_subset", nargs="*", default=None,
                   help="If given, only evaluate these perturbations (smoke-test use).")
    p.add_argument("--min_cells", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    import anndata as ad
    import numpy as np
    import pandas as pd
    import torch

    from cccfm.data import TargetControls
    from cccfm.eval_metrics import CrossLineEvaluator
    from cccfm.flow_module import CCCFMLitModule
    from cccfm.pert_tokens import PertTokenizer
    from cccfm.velocity_net import VelocityNet

    run_dir = args.run_dir
    cfg_path = run_dir / "config.json"
    assert cfg_path.exists(), f"missing {cfg_path} (was 11_train_cccfm.py completed?)"
    cfg = json.loads(cfg_path.read_text())
    print(f"[eval] run_id={cfg['run_id']}  encoder={cfg['encoder']}  setting={cfg['setting']}")

    ckpt_path = run_dir / f"{args.ckpt_name}.ckpt"
    assert ckpt_path.exists(), f"missing {ckpt_path}"

    # ------------------------- load data + evaluator ------------------------- #
    t0 = time.perf_counter()
    adata = ad.read_h5ad(args.data_path)
    obsm_key = cfg["obsm_key"]
    if obsm_key not in adata.obsm:
        enc_name = obsm_key[2:] if obsm_key.startswith("h_") else obsm_key
        enc_h5ad = args.data_path.parent / f"replogle_{enc_name}.h5ad"
        if enc_h5ad.exists():
            adata_enc = ad.read_h5ad(enc_h5ad, backed="r")
            adata.obsm[obsm_key] = np.asarray(adata_enc.obsm[obsm_key])
            del adata_enc
            print(f"[eval] loaded {obsm_key} from {enc_h5ad}")
        else:
            raise RuntimeError(f"adata.obsm['{obsm_key}'] missing and no fallback at {enc_h5ad}; re-run featurization.")

    # ---- replay h alignment if training applied one ---- #
    align_path = run_dir / "h_alignment.npz"
    if align_path.exists():
        npz = np.load(align_path, allow_pickle=True)
        mode = str(npz["mode"]) if "mode" in npz.files else "zscore"
        H = np.asarray(adata.obsm[obsm_key], dtype=np.float32)
        if mode == "zscore":
            h_mean = npz["mean"].astype(np.float32)
            h_std = npz["std"].astype(np.float32)
            adata.obsm[obsm_key] = ((H - h_mean) / h_std).astype(np.float32)
            print(f"[eval] applied h_alignment mode=zscore  "
                  f"||mean||={float(np.linalg.norm(h_mean)):.3f} "
                  f"||std||={float(np.linalg.norm(h_std)):.3f}")
        elif mode == "zscore_per_line":
            cell_lines = [str(x) for x in npz["cell_lines"].tolist()]
            means = npz["means"].astype(np.float32)   # [n_lines, D]
            stds = npz["stds"].astype(np.float32)
            H_out = H.copy()
            line_arr = adata.obs["cell_line"].astype(str).values
            for i, cl in enumerate(cell_lines):
                line_mask = (line_arr == cl)
                if not line_mask.any():
                    continue
                H_out[line_mask] = (H[line_mask] - means[i]) / stds[i]
            adata.obsm[obsm_key] = H_out.astype(np.float32)
            print(f"[eval] applied h_alignment mode=zscore_per_line  lines={cell_lines}")
        else:
            raise ValueError(f"unknown h_alignment.npz mode={mode}")
    else:
        print(f"[eval] no h_alignment.npz found in {run_dir}; using raw obsm.")

    target_line = cfg.get("target_line", "Jurkat")
    sources = cfg.get("sources", SETTING_SOURCES[cfg["setting"]])
    test_perts = sorted(json.loads(args.test_slice.read_text()))
    if args.pert_subset:
        test_perts = [p for p in test_perts if p in set(args.pert_subset)]
        print(f"[eval] pert_subset applied: {len(test_perts)} perts")
    evaluator = CrossLineEvaluator(
        adata=adata,
        source_lines=sources,
        target_line=target_line,
        test_perts=test_perts,
        min_cells=args.min_cells,
    )

    target_ctrls = TargetControls(adata, target_line=target_line, encoder_key=cfg["encoder"])
    print(f"[eval] target control pool: {target_ctrls.n} cells")

    # ------------------------- rebuild module + load ckpt ------------------------- #
    mcfg = cfg["model"]
    tokenizer = PertTokenizer(
        mode=mcfg.get("pert_mode", "gene_embedding"),
        primary_embedding_path=args.primary_embed,
        secondary_embedding_path=None,  # D4 ablation will wire this
        out_dim=mcfg["pert_dim"],
        freeze_embedding=True,
    )
    # R3-LowRank: velocity net was trained on rank-d input (cfg['velocity_input_dim'])
    vel_input_dim = cfg.get("velocity_input_dim", cfg["gene_dim"])
    vel_hidden = cfg.get("velocity_hidden", mcfg["hidden_dim"])
    velocity = VelocityNet(
        gene_dim=vel_input_dim,
        context_dim=cfg["context_dim"],
        pert_dim=mcfg["pert_dim"],
        hidden_dim=vel_hidden,
        n_blocks=mcfg["n_blocks"],
        time_dim=mcfg["time_dim"],
        dropout=mcfg.get("dropout", 0.0),
    )
    # ---- residual mode: rebuild anchor_delta_dict + line_ctrl_means from artifacts ----
    anchor_mode = cfg.get("anchor_mode", "direct")
    anchor_target_mode = cfg.get("anchor_target_mode", "raw")
    flow_space = cfg.get("flow_space", "gene")
    anchor_delta_dict = None
    line_ctrl_means = None
    lowrank_basis_B = None
    if flow_space == "lowrank":
        from cccfm.basis import LowRankBasis
        basis_path = run_dir / "lowrank_basis.npz"
        assert basis_path.exists(), f"flow_space=lowrank but no basis at {basis_path}"
        basis = LowRankBasis.load(basis_path)
        lowrank_basis_B = basis.B
        print(f"[eval] flow_space=lowrank: rank={basis.rank} mode={basis.mode}")
    anc_path = run_dir / "anchor_artifacts.npz"
    if anchor_mode == "residual":
        if not anc_path.exists():
            raise RuntimeError(f"residual run is missing anchor_artifacts.npz at {anc_path}")
        anc = np.load(anc_path, allow_pickle=True)
        anchor_perts = [str(p) for p in anc["anchor_perts"].tolist()]
        anchor_deltas = anc["anchor_deltas"].astype(np.float32)
        line_names = [str(l) for l in anc["line_names"].tolist()]
        line_means = anc["line_ctrl_means"].astype(np.float32)
        anchor_delta_dict = {p: anchor_deltas[i] for i, p in enumerate(anchor_perts)}
        line_ctrl_means = {l: line_means[i] for i, l in enumerate(line_names)}
        print(f"[eval] residual mode: loaded anchor='{cfg.get('anchor', '?')}' with "
              f"{len(anchor_delta_dict)} pert deltas, {len(line_ctrl_means)} line means")
    else:
        print(f"[eval] direct mode (anchor='none')")

    lit = CCCFMLitModule.load_from_checkpoint(
        str(ckpt_path),
        velocity_net=velocity,
        pert_tokenizer=tokenizer,
        anchor_mode=anchor_mode,
        anchor_delta_dict=anchor_delta_dict,
        line_ctrl_means=line_ctrl_means,
        anchor_target_mode=anchor_target_mode,
        flow_space=flow_space,
        lowrank_basis_B=lowrank_basis_B,
        map_location="cpu",
    )
    if anchor_mode == "residual":
        print(f"[eval] anchor_target_mode='{anchor_target_mode}'  flow_space='{flow_space}'  "
              f"(eval-time projection: {anchor_target_mode == 'mean_preserved'})")
    lit.eval()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    lit.to(device)
    print(f"[eval] loaded {ckpt_path}  device={device}")

    # ------------------------- per-pert loop ------------------------- #
    rng = np.random.default_rng(args.seed)
    run_id = cfg["run_id"]
    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    metric_cols = list(CrossLineEvaluator.METRIC_COLUMNS)

    # incremental CSV helper
    def flush():
        if not rows:
            return
        df = pd.DataFrame(rows)
        front = ["method", "setting", "target_line", "pert"]
        present_metrics = [c for c in metric_cols if c in df.columns]
        extras = [c for c in df.columns if c not in front + present_metrics]
        df = df[front + present_metrics + extras]
        df.to_csv(out_path, index=False)

    n_evaluated = 0
    n_skipped = 0
    for k, pert in enumerate(test_perts):
        if pert not in evaluator.target_pert_centroids:
            n_skipped += 1
            continue

        # n_pred_cells policy: match the sample-size protocol of shifted_control
        # / target_control_copy baselines so distributional metrics are directly
        # comparable. Default = n_target_pert_cells (median ~145), capped by
        # --max_n_samples to protect long-tail high-count perts. NO floor at
        # batch_size — that would inflate n_pred for low-target perts and warp
        # MMD/Energy. (advisor 2026-04-26 round 4)
        n_target = evaluator.target_pert_n_cells[pert]
        if args.n_samples > 0:
            n_samp = args.n_samples
        else:
            n_samp = min(n_target, args.max_n_samples)

        preds = []
        if anchor_mode == "residual":
            if pert not in anchor_delta_dict:
                n_skipped += 1
                continue
            # 1. Sample full target-control cloud once
            x0_full_np, h_full_np = target_ctrls.sample_full(n_samp, rng)
            x0_full = torch.from_numpy(x0_full_np).to(device)
            h_full = torch.from_numpy(h_full_np).to(device)
            # 2. Build x0_anchor with single mean-centering over entire cloud
            from cccfm.anchor import apply_anchor_torch
            delta_g = lit.anchor_delta_matrix[lit.anchor_pert_to_idx[pert]]
            ctx_mean = lit.line_ctrl_mean_matrix[lit.line_to_idx[target_line]]
            x0_anchor_full = apply_anchor_torch(x0_full, ctx_mean, delta_g)
            anchor_profile_t = ctx_mean + delta_g                      # [G]

            if flow_space == "lowrank":
                # 3a. Project once: z0_full = (x0_anchor_full - anchor_profile) @ B
                z0_full = (x0_anchor_full - anchor_profile_t) @ lit.lowrank_B   # [N, rank]
                # 3b. Chunk z0 for ODE in rank space
                z_chunks = []
                for start in range(0, z0_full.shape[0], args.batch_size):
                    end = start + args.batch_size
                    z0c = z0_full[start:end]; hc = h_full[start:end]
                    Bsz = z0c.shape[0]
                    z_pert = lit.pert_tokenizer([pert] * Bsz)
                    with torch.no_grad():
                        z_pred_c = lit.predict_perturbed(
                            z0c, hc, z_pert,
                            n_steps=args.n_steps, method=args.ode_method,
                        )
                    z_chunks.append(z_pred_c)
                z_pred_full = torch.cat(z_chunks, dim=0)               # [N, rank]
                # 3c. Off-basis passthrough: pred = x0_anchor + (z_pred - z0) @ B.T
                #     Flow only edits low-rank component; off-basis variance preserved
                #     from input control cells. (See flow_module.predict_for_pert_anchored
                #     for rationale and the strict-§2.1 alternative.)
                delta_z = z_pred_full - z0_full
                pred_full = x0_anchor_full + delta_z @ lit.lowrank_B.T  # [N, G]
                pred_full = pred_full - pred_full.mean(dim=0, keepdim=True) + anchor_profile_t
                preds.append(pred_full.cpu().numpy())
            else:
                # gene-space residual flow (R2 / R3.1)
                for start in range(0, x0_anchor_full.shape[0], args.batch_size):
                    end = start + args.batch_size
                    x0c = x0_anchor_full[start:end]
                    hc = h_full[start:end]
                    Bsz = x0c.shape[0]
                    z_pert = lit.pert_tokenizer([pert] * Bsz)
                    with torch.no_grad():
                        x1c = lit.predict_perturbed(
                            x0c, hc, z_pert,
                            n_steps=args.n_steps, method=args.ode_method,
                        )
                    preds.append(x1c.detach().cpu().numpy())
                # R3.1 deterministic eval-time projection (advisor §6.4)
                if anchor_target_mode == "mean_preserved":
                    pred_concat = np.concatenate(preds, axis=0)
                    target_anchor_profile = (
                        line_ctrl_means[target_line] + anchor_delta_dict[pert]
                    ).astype(np.float32)
                    pred_concat = (pred_concat
                                   - pred_concat.mean(axis=0, keepdims=True)
                                   + target_anchor_profile)
                    preds = [pred_concat]
        else:
            for x0_np, h_np in target_ctrls.sample_batches(
                n_samples=n_samp, batch_size=args.batch_size, rng=rng,
            ):
                x0 = torch.from_numpy(x0_np).to(device)
                h = torch.from_numpy(h_np).to(device)
                with torch.no_grad():
                    x1 = lit.predict_for_pert(
                        x0, h, pert, n_steps=args.n_steps, method=args.ode_method,
                    )
                preds.append(x1.detach().cpu().numpy())
        pred_cells = np.concatenate(preds, axis=0)

        m = evaluator.evaluate_pert_cell_level(
            pert, pred_cells, compute_heavy=args.compute_heavy,
        )
        if m is None:
            n_skipped += 1
            continue

        m.update({
            "method": "cccfm" if anchor_mode == "direct" else f"cccfm_residual_{cfg.get('anchor', 'unknown')}",
            "setting": cfg["setting"],
            "target_line": target_line,
            "pert": pert,
            "encoder": cfg["encoder"],
            "anchor_mode": anchor_mode,
            "anchor_type": cfg.get("anchor", "none"),
            "n_pred_cells": int(pred_cells.shape[0]),
            "n_steps": args.n_steps,
            "ode_method": args.ode_method,
            "seed": args.seed,
            "run_id": run_id,
        })
        rows.append(m)
        n_evaluated += 1

        if (k + 1) % 10 == 0 or (k + 1) == len(test_perts):
            flush()
            pc_mean = np.mean([r.get("pearson_ctrl") or 0.0 for r in rows])
            pp_mean = np.mean([r.get("pearson_pert") or 0.0 for r in rows])
            print(f"  [{k+1:>4d}/{len(test_perts)}] evaluated={n_evaluated} "
                  f"skipped={n_skipped}  Pf^ctrl={pc_mean:.3f}  Pf^pert={pp_mean:.3f}  "
                  f"wall={time.perf_counter()-t0:.0f}s")

    flush()

    # ------------------------- summary ------------------------- #
    df = pd.DataFrame(rows)
    present_metrics = [c for c in metric_cols if c in df.columns and c != "n_target_cells"]
    if len(df):
        agg = df[present_metrics].agg(["mean", "median", "std"]).T
        summary_path = out_path.with_name(out_path.stem + "_summary.csv")
        agg.reset_index().rename(columns={"index": "metric"}).to_csv(summary_path, index=False)
        print(f"[eval] n_evaluated={n_evaluated}  skipped={n_skipped}")
        for col in ["pearson_ctrl", "pearson_pert", "pearson20_ctrl",
                    "pearson20_pert", "centroid_accuracy", "mmd", "energy"]:
            if col in df.columns:
                print(f"  {col:20s}: mean={df[col].mean():.4f}  median={df[col].median():.4f}")
        print(f"[eval] wrote {out_path}")
        print(f"[eval] wrote {summary_path}")
    else:
        print("[eval] WARN: no perts evaluated successfully.")


if __name__ == "__main__":
    main()
