"""R2 zero-init residual CFM sanity gate.

What this proves: with the velocity field's output projection zero-initialised
and **no training**, the residual CFM evaluation pipeline produces predictions
that match the corresponding shifted-control baseline per-pert, both at the
mean-metric level (formula-deterministic exact match) and at the distributional
level (within sampling tolerance).

How: build a CCCFMLitModule in residual mode, anchor=Linear, with
`zero_init_output=True`. Skip training entirely. Run the corrected eval path
(single mean-centering on the full cloud, then chunked ODE) on a subset of test
perts. Compare per-pert metrics to `linear_ahlmann_eltze_2line_shifted_control.csv`.

If this gate passes, the residual CFM eval path is faithful and any deviation
in a *trained* run reflects what the velocity field learned, not pipeline bugs.

Usage:
    python scripts/08_residual_zeroinit_gate.py --setting 2line --n_perts 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


SETTING_SOURCES = {
    "2line": ["K562", "RPE1"],
    "3line": ["K562", "RPE1", "HepG2"],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=Path,
                   default=Path("data/processed/replogle_preprocessed.h5ad"))
    p.add_argument("--test_slice", type=Path,
                   default=Path("data/splits/jurkat_test_strict.json"))
    p.add_argument("--primary_embedding_path", type=Path,
                   default=Path("data/processed/primary_gene_embedding.pt"))
    p.add_argument("--setting", choices=list(SETTING_SOURCES), default="2line")
    p.add_argument("--target_line", type=str, default="Jurkat")
    p.add_argument("--n_perts", type=int, default=20,
                   help="how many test perts to check")
    p.add_argument("--encoder", type=str, default="pca")
    p.add_argument("--n_steps", type=int, default=10,
                   help="ODE steps. Doesn't matter (v=0) but kept tiny.")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_n_samples", type=int, default=2000)
    p.add_argument("--min_cells", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    # Tolerances
    p.add_argument("--tol_mean", type=float, default=1e-4,
                   help="per-pert max|diff| on mean-level metrics. Should be "
                        "near-zero with same seed; bumped from 1e-6 because "
                        "ODE integration introduces tiny float accumulation.")
    p.add_argument("--tol_dist", type=float, default=1e-1,
                   help="per-pert mean|diff| on dist metrics. Different seeds "
                        "between us and the shifted_control reference produce "
                        "sampling noise; this loose tolerance covers it.")
    return p.parse_args()


def main():
    args = parse_args()
    import anndata as ad
    import numpy as np
    import pandas as pd
    import torch

    from cccfm.anchor import LinearAnchor, apply_anchor_torch
    from cccfm.data import compute_per_line_ctrl_means, TargetControls
    from cccfm.eval_metrics import CrossLineEvaluator
    from cccfm.flow_module import CCCFMLitModule
    from cccfm.pert_tokens import PertTokenizer
    from cccfm.velocity_net import VelocityNet

    sources = SETTING_SOURCES[args.setting]
    target = args.target_line
    print(f"[gate] setting={args.setting}  sources={sources}  target={target}  encoder={args.encoder}")

    adata = ad.read_h5ad(args.data_path)
    test_perts_full = sorted(json.loads(args.test_slice.read_text()))
    # Important: pass the FULL 577 test perts to the evaluator so its internal
    # `target_pert_reference` (Systema) and centroid matrix match the shifted-
    # control reference run. Only the per-pert *loop* uses the smaller subset.
    evaluator = CrossLineEvaluator(
        adata=adata, source_lines=sources, target_line=target,
        test_perts=test_perts_full, min_cells=args.min_cells,
    )
    test_perts = test_perts_full[:args.n_perts]
    target_ctrls = TargetControls(adata, target_line=target, encoder_key=args.encoder)
    print(f"[gate] target ctrl pool: {target_ctrls.n} cells")

    # Anchor + line means (matched to scripts/13 shifted_control)
    anchor = LinearAnchor().fit(adata, source_lines=sources, target_line=target,
                                min_cells=args.min_cells)
    relevant_perts = sorted(set(test_perts) & anchor.supported_perts)
    anchor_delta_dict = {p: anchor.delta(p) for p in relevant_perts}
    all_lines = sorted(set(sources) | {target})
    line_ctrl_means = compute_per_line_ctrl_means(adata, all_lines)
    print(f"[gate] anchor={len(anchor_delta_dict)} perts; lines={all_lines}")

    # Zero-init CCCFMLitModule (no training)
    G = adata.n_vars
    D_ctx = adata.obsm[f"h_{args.encoder}"].shape[1]
    D_pert = 128
    tokenizer = PertTokenizer(
        mode="gene_embedding",
        primary_embedding_path=args.primary_embedding_path,
        out_dim=D_pert, freeze_embedding=True,
    )
    velocity = VelocityNet(
        gene_dim=G, context_dim=D_ctx, pert_dim=D_pert,
        hidden_dim=256, n_blocks=2, time_dim=64,
        zero_init_output=True,    # output v_theta = 0 at init
    )
    lit = CCCFMLitModule(
        velocity_net=velocity, pert_tokenizer=tokenizer,
        use_ot_pairing=False,
        anchor_mode="residual",
        anchor_delta_dict=anchor_delta_dict,
        line_ctrl_means=line_ctrl_means,
    )
    lit.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lit.to(device)

    # Quick zero-velocity invariant check
    with torch.no_grad():
        x_t = torch.randn(8, G, device=device)
        t = torch.rand(8, device=device)
        h = torch.randn(8, D_ctx, device=device)
        z = torch.randn(8, D_pert, device=device)
        v = velocity(x_t, t, h, z)
        v_max = float(v.abs().max())
    assert v_max < 1e-6, f"zero_init failed: |v|_max = {v_max}"
    print(f"[gate] zero-init verified: |v_theta|_max = {v_max:.2e}")

    # Per-pert eval — mirror scripts/12_eval_cccfm.py residual path
    rng = np.random.default_rng(args.seed)
    rows = []
    for k, pert in enumerate(test_perts):
        if pert not in evaluator.target_pert_centroids:
            continue
        if pert not in anchor_delta_dict:
            continue
        n_target = evaluator.target_pert_n_cells[pert]
        n_samp = min(n_target, args.max_n_samples)

        x0_np, h_np = target_ctrls.sample_full(n_samp, rng)
        x0 = torch.from_numpy(x0_np).to(device)
        h = torch.from_numpy(h_np).to(device)
        delta_g = lit.anchor_delta_matrix[lit.anchor_pert_to_idx[pert]]
        ctx_mean = lit.line_ctrl_mean_matrix[lit.line_to_idx[target]]
        x0_anchor = apply_anchor_torch(x0, ctx_mean, delta_g)

        preds = []
        for start in range(0, x0_anchor.shape[0], args.batch_size):
            end = start + args.batch_size
            x0c = x0_anchor[start:end]; hc = h[start:end]
            B = x0c.shape[0]
            z_pert = lit.pert_tokenizer([pert] * B)
            with torch.no_grad():
                # With zero v_theta, ODE is identity → x1c == x0c.
                x1c = lit.predict_perturbed(x0c, hc, z_pert, n_steps=args.n_steps, method="heun")
            preds.append(x1c.cpu().numpy())
        pred_cells = np.concatenate(preds, axis=0)

        # mean(pred_cells) should equal anchored profile = tgt_ctrl + delta_g
        target_profile = evaluator.target_ctrl_reference + anchor_delta_dict[pert]
        mean_drift = float(np.abs(pred_cells.mean(0) - target_profile).max())
        if mean_drift > 1e-3:
            print(f"[gate] WARN  pert={pert}  mean(pred) drift = {mean_drift:.2e}")

        m = evaluator.evaluate_pert_cell_level(pert, pred_cells, compute_heavy=True)
        if m is None:
            continue
        m["pert"] = pert
        rows.append(m)

    df = pd.DataFrame(rows).set_index("pert")
    print(f"[gate] evaluated {len(df)} perts")

    # Compare to shifted-control reference
    ref_csv = Path(f"results/baselines/linear_ahlmann_eltze_{args.setting}_shifted_control.csv")
    if not ref_csv.exists():
        print(f"[gate] WARN: reference {ref_csv} missing. Run scripts/06_mean_baselines_distributional.py first.")
        raise SystemExit(2)
    ref_df = pd.read_csv(ref_csv).set_index("pert")
    metric_cols_mean = ["mse", "pearson_ctrl", "pearson_pert", "pearson20_pert",
                        "centroid_accuracy", "deg_overlap20", "deg_overlap50"]
    metric_cols_dist = ["mmd", "energy", "w2_sinkhorn", "moment"]

    merged = df.join(ref_df[metric_cols_mean + metric_cols_dist], rsuffix="_ref", how="inner")
    print(f"[gate] joined on {len(merged)} perts (where shifted_control reference exists)")

    print(f"\n{'metric':18s} {'max|diff|':>12s} {'mean|diff|':>12s} {'tol':>10s}  verdict")
    fail = []
    for col in metric_cols_mean + metric_cols_dist:
        d = (merged[col] - merged[f"{col}_ref"]).abs()
        max_d = float(d.max()); mean_d = float(d.mean())
        if col in metric_cols_mean:
            tol, label = args.tol_mean, "max"
            stat = max_d
        else:
            tol, label = args.tol_dist, "mean"
            stat = mean_d
        verdict = "OK" if stat <= tol else "FAIL"
        if verdict == "FAIL":
            fail.append((col, label, stat, tol))
        print(f"{col:18s} {max_d:>12.4e} {mean_d:>12.4e} {tol:>10.0e}  {verdict}")

    if fail:
        print(f"\n[gate] ❌ {len(fail)} metric(s) outside tolerance:")
        for col, label, stat, tol in fail:
            print(f"  {col}: {label}|diff|={stat:.4e} > tol {tol:.0e}")
        raise SystemExit(1)

    print(f"\n[gate] ✅ Zero-init residual CFM reproduces shifted-control on {len(merged)} perts.")
    print(f"  Mean metrics: max|diff| within tol={args.tol_mean:.0e}.")
    print(f"  Dist metrics: mean|diff| within tol={args.tol_dist:.0e}.")
    print(f"  Pipeline (anchor + apply_anchor + ODE w/ v=0) is faithful; any divergence in")
    print(f"  trained runs reflects what the velocity learned, not pipeline bugs.")


if __name__ == "__main__":
    main()
