"""Train one low-rank residual-flow cell for the rebuttal rank/width factorial.

This is an intentionally separate copy of the submitted-project trainer.  Its
only semantic change is that the velocity-network width is explicit and never
auto-scaled from rank.  Keeping this runner under ``scripts/rebuttal`` avoids
overwriting the divergent cluster copy of ``scripts/30_train_cccfm.py``.

Inputs:
    --encoder        one of {pca, state_se, geneformer, scgpt, uce, scfoundation}
                     (obsm key `h_<encoder>` must already be populated)
    --setting        '2line' (K562+RPE1 sources) or '3line' (+HepG2)
    --target_line    Jurkat (held out at training, used only by eval script)

Output:
    runs/cccfm/<tag>/
        best.ckpt
        last.ckpt
        config.json
        train_stats.csv
        tb/  (lightning TB logs)

Cross-line setup:
    - training cells = source-lines × (non-control perts with >= min_cells_per_pert cells in sources)
    - validation = 5% held-out perturbed cells (random, per-cell split)
    - Jurkat cells NEVER appear in training; target-line evaluation is done by
      scripts/rebuttal/11_eval_fixed_width_rank.py after training.

Usage (smoke, PCA encoder, 3090):
    python scripts/rebuttal/10_train_fixed_width_rank.py \
        --encoder pca --setting 2line --flow_space lowrank --rank 32 \
        --anchor linear --anchor_target_mode mean_preserved \
        --velocity_hidden_dim 128 --max_epochs 1 --batch_size 64 \
        --n_blocks 2 --num_workers 2 \
        --out_dir runs/rebuttal/A_fixed_width_rank/smoke
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import sys

RUNTIME_CODE = Path(__file__).resolve().parents[2] / "reproducibility" / "fixed_width_rank" / "code"
sys.path.insert(0, str(RUNTIME_CODE))


SETTING_SOURCES = {
    "2line": ["K562", "RPE1"],
    "3line": ["K562", "RPE1", "HepG2"],
    "target_K562_2src": ["RPE1", "HepG2"],
    "target_RPE1_2src": ["K562", "HepG2"],
}

ENCODER_DIMS_DEFAULT = {
    "pca": 128,
    "state_se": 2048,
    "geneformer": 768,
    "scgpt": 512,
    "uce": 1280,
    "scfoundation": 3072,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=Path,
                   default=Path("data/processed/replogle_preprocessed.h5ad"))
    p.add_argument("--primary_embed",
                   type=Path, default=Path("data/processed/primary_gene_embedding.pt"))
    p.add_argument("--secondary_embed", type=Path, default=None,
                   help="optional GeneCompass dump; enables concat_secondary mode")
    p.add_argument("--encoder", required=True,
                   choices=list(ENCODER_DIMS_DEFAULT))
    p.add_argument("--setting", choices=list(SETTING_SOURCES), default="2line")
    p.add_argument("--target_line", type=str, default="Jurkat")

    # model
    p.add_argument("--pert_dim", type=int, default=512)
    p.add_argument(
        "--velocity_hidden_dim",
        type=int,
        required=True,
        help="Exact VelocityNet hidden width. No rank-dependent autoscaling.",
    )
    p.add_argument("--n_blocks", type=int, default=4)
    p.add_argument("--time_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.0)

    # training
    p.add_argument("--max_epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--sigma", type=float, default=0.0)
    p.add_argument("--use_ot_pairing", action="store_true",
                   help="v1 默认关闭; D1 ablation 打开时记得配合 per-(line,pert) batching")
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--min_cells_per_pert", type=int, default=20,
                   help="training 过滤: 每个 (source_line 聚合) pert 至少多少细胞")
    p.add_argument("--align_h_mode", type=str, default="none",
                   choices=["none", "zscore", "zscore_per_line"],
                   help="cross-line h alignment. "
                        "'zscore' = subtract source-ctrl mean / divide by source-ctrl std "
                        "(weak: single normalizer, target may stay OOD). "
                        "'zscore_per_line' = per-cell-line normalizer using each line's "
                        "own ctrl stats (strong: each line N(0,1) in its own frame, "
                        "cross-line shift removed by construction).")

    # R2: anchored residual CFM
    p.add_argument("--anchor", type=str, default="none",
                   choices=["none", "linear", "ridge", "exact"],
                   help="R2: residual CFM mode. 'none' = direct (control->perturbed) flow. "
                        "linear/ridge/exact = anchored residual flow with the named delta provider.")
    p.add_argument("--n_holdout_perts", type=int, default=0,
                   help="R2.5: how many source-line perts to hold out as a per-pert val "
                        "set (disjoint from --test_slice). 0 disables (use per-cell val_frac).")
    p.add_argument("--test_slice", type=Path,
                   default=Path("data/splits/jurkat_test_strict.json"),
                   help="paper test perts (kept out of pert_holdout_split candidates).")
    p.add_argument("--anchor_target_mode", type=str, default="raw",
                   choices=["raw", "mean_preserved"],
                   help="R3.1 (residual mode only): training endpoint projection. "
                        "'raw' = use x_1 directly (R2 default). "
                        "'mean_preserved' = project x_1 to anchor mean per advisor 2026-04-27, "
                        "forcing the flow to learn cell-state shape, not mean correction. "
                        "Eval script auto-projects the ODE output back to anchor mean too.")

    # R3-LowRank (advisor §2.1): run CFM in a rank-D PCA subspace
    p.add_argument("--flow_space", type=str, default="gene",
                   choices=["gene", "lowrank"],
                   help="'gene' = full gene-space flow (R2/R3.1 default). "
                        "'lowrank' = project residuals to rank-D PCA subspace; "
                        "implies anchor + mean_preserved.")
    p.add_argument("--rank", type=int, default=64,
                   help="rank for --flow_space lowrank (sweep {32,64,128,256}).")
    p.add_argument("--basis_mode", type=str, default="both",
                   choices=["control", "perturbed", "both"],
                   help="data used for fitting the PCA basis.")

    # runtime
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--precision", type=str, default="bf16-mixed",
                   choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--accelerator", type=str, default="gpu")
    p.add_argument("--gradient_clip_val", type=float, default=1.0)
    p.add_argument("--log_every_n_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)

    # io
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--run_id", type=str, default=None,
                   help="默认用 timestamp")

    return p.parse_args()


def main():
    args = parse_args()

    if args.flow_space != "lowrank":
        raise ValueError("The fixed-width rebuttal runner requires --flow_space lowrank")
    if args.velocity_hidden_dim <= 0:
        raise ValueError("--velocity_hidden_dim must be positive")

    import anndata as ad
    import numpy as np
    import pandas as pd
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    from cccfm.anchor import LinearAnchor, ExactAnchor, RidgeAnchor
    from cccfm.basis import LowRankBasis
    from cccfm.data import (
        apply_source_alignment, build_train_dataloader,
        build_residual_dataloaders, compute_per_line_ctrl_means, pert_holdout_split,
    )
    from cccfm.flow_module import CCCFMLitModule
    from cccfm.pert_tokens import PertTokenizer
    from cccfm.velocity_net import VelocityNet

    pl.seed_everything(args.seed, workers=True)
    sources = SETTING_SOURCES[args.setting]
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------- load data ------------------------- #
    t0 = time.perf_counter()
    print(f"[{run_id}] loading {args.data_path} ...")
    adata = ad.read_h5ad(args.data_path)
    obsm_key = f"h_{args.encoder}"
    if obsm_key not in adata.obsm:
        enc_h5ad = args.data_path.parent / f"replogle_{args.encoder}.h5ad"
        if enc_h5ad.exists():
            adata_enc = ad.read_h5ad(enc_h5ad, backed="r")
            adata.obsm[obsm_key] = np.asarray(adata_enc.obsm[obsm_key])
            del adata_enc
            print(f"   loaded {obsm_key} from {enc_h5ad}")
        else:
            raise RuntimeError(
                f"adata.obsm['{obsm_key}'] missing. "
                f"Run scripts/20_featurize_encoder.py --encoder {args.encoder} first."
            )
    context_dim = int(adata.obsm[obsm_key].shape[1])
    gene_dim = int(adata.n_vars)
    print(f"   n_cells={adata.n_obs}  gene_dim={gene_dim}  context_dim={context_dim}")

    # ---- cross-line h alignment (optional) ---- #
    if args.align_h_mode != "none":
        out_dir.mkdir(parents=True, exist_ok=True)
        align_path = out_dir / "h_alignment.npz"
        if args.align_h_mode == "zscore":
            h_mean, h_std = apply_source_alignment(
                adata, encoder_key=args.encoder,
                source_lines=sources, mode="zscore",
            )
            np.savez(align_path,
                     mode=np.array("zscore"),
                     mean=h_mean, std=h_std,
                     encoder=np.array(args.encoder), sources=np.array(sources))
            print(f"   align_h='zscore' src_ctrl: "
                  f"||mean||={float(np.linalg.norm(h_mean)):.3f}  "
                  f"||std||={float(np.linalg.norm(h_std)):.3f}  saved {align_path}")
        else:  # zscore_per_line
            cell_lines, means, stds = apply_source_alignment(
                adata, encoder_key=args.encoder,
                source_lines=sources, mode="zscore_per_line",
            )
            np.savez(align_path,
                     mode=np.array("zscore_per_line"),
                     cell_lines=np.array(cell_lines),
                     means=means, stds=stds,
                     encoder=np.array(args.encoder), sources=np.array(sources))
            print(f"   align_h='zscore_per_line' lines={cell_lines}  "
                  f"||means||/line=" + " ".join(f"{float(np.linalg.norm(m)):.2f}" for m in means)
                  + f"  saved {align_path}")

    # Source-line perts with >= min_cells_per_pert cells pooled across sources.
    src_mask = adata.obs["cell_line"].isin(sources).values
    pert_arr = adata.obs["pert"].values
    perts_in_sources, counts = np.unique(pert_arr[src_mask], return_counts=True)
    pert_set = [
        p for p, c in zip(perts_in_sources, counts)
        if p != "control" and c >= args.min_cells_per_pert
    ]
    pert_set = sorted(pert_set)
    print(f"   training pert_set: {len(pert_set)} (min_cells_per_pert={args.min_cells_per_pert})")

    # ============================== R2 anchor branch ============================== #
    anchor = None
    anchor_delta_dict = None
    line_ctrl_means = None
    lowrank_basis_B = None

    if args.anchor != "none":
        # 1. Hold-out test perts (from --test_slice) so the anchor never sees them
        #    Wait: we DO want the anchor to predict for test perts; what we forbid is
        #    target-line CELLS in fit. Test perts come from source lines too.
        #    Actually, what we want: anchor.fit excludes nothing from source data;
        #    anchor predicts delta for any pert it has source data for, including
        #    the 577 test perts. R1.4 just says no target_line cells.
        if args.anchor == "linear":
            anchor = LinearAnchor()
        elif args.anchor == "exact":
            anchor = ExactAnchor()
        elif args.anchor == "ridge":
            anchor = RidgeAnchor(args.primary_embed)
        else:
            raise ValueError(args.anchor)

        anchor.fit(
            adata, source_lines=sources, target_line=args.target_line,
            min_cells=args.min_cells_per_pert,
        )
        print(f"   anchor='{args.anchor}': supported_perts={len(anchor.supported_perts)}")

        # 2. Filter pert_set to anchor.supported_perts (residual mode requires delta lookup)
        before = len(pert_set)
        pert_set = sorted(set(pert_set) & anchor.supported_perts)
        print(f"   after anchor coverage filter: {len(pert_set)}/{before} perts")

        # 3. Per-pert val holdout (R2.5)
        if args.n_holdout_perts > 0:
            test_perts = sorted(json.loads(args.test_slice.read_text()))
            train_perts, val_perts = pert_holdout_split(
                adata, sources,
                n_holdout_perts=args.n_holdout_perts,
                seed=args.seed,
                exclude_perts=test_perts,
                min_cells_per_pert=args.min_cells_per_pert,
            )
            # restrict to perts the anchor + dataset both support
            train_perts = sorted(set(train_perts) & set(pert_set))
            val_perts = sorted(set(val_perts) & set(pert_set))
            print(f"   per-pert holdout: train={len(train_perts)}  val={len(val_perts)}  "
                  f"(test_excluded={len(test_perts)})")
        else:
            # fall back to no per-pert val (val_loader will be empty / unused if val_perts is empty)
            train_perts = sorted(pert_set)
            val_perts = []

        # 4. Build anchor_delta_dict + line_ctrl_means
        #    Include train + val + test perts: train/val for residual training,
        #    test perts for downstream eval (they're source-line perts too, just
        #    held out of training; the anchor is allowed to predict them — its
        #    leakage rule is "no target-line cells", not "no test perts").
        eval_perts = sorted(json.loads(args.test_slice.read_text()))
        relevant_perts = sorted(set(train_perts) | set(val_perts) | set(eval_perts))
        anchor_delta_dict = {p: anchor.delta(p) for p in relevant_perts if anchor.delta(p) is not None}
        # Compute ctrl means for ALL lines (sources + target). Training uses source
        # line means as the per-batch context; eval uses target_line mean. R1.2.
        all_lines = sorted(set(sources) | {args.target_line})
        line_ctrl_means = compute_per_line_ctrl_means(adata, all_lines)
        print(f"   anchor_delta_dict: {len(anchor_delta_dict)} entries; "
              f"line_ctrl_means lines={list(line_ctrl_means)}")

        # 5a. R3-LowRank: fit + persist basis BEFORE the CCCFMLitModule init.
        lowrank_basis_B = None
        if args.flow_space == "lowrank":
            assert args.anchor_target_mode == "mean_preserved", (
                "--flow_space lowrank requires --anchor_target_mode mean_preserved"
            )
            basis = LowRankBasis().fit(
                adata, source_lines=sources, target_line=args.target_line,
                anchor_delta_dict=anchor_delta_dict, line_ctrl_means=line_ctrl_means,
                mode=args.basis_mode, rank=args.rank, seed=args.seed,
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            basis_path = out_dir / "lowrank_basis.npz"
            basis.save(basis_path)
            lowrank_basis_B = basis.B
            print(f"   saved {basis_path}  (rank={basis.rank}, mode={basis.mode}, "
                  f"cum_var={float(basis.explained_variance_ratio.sum()):.3f})")

        # 5b. Persist anchor artifacts for eval (R0.4 source-of-truth)
        out_dir.mkdir(parents=True, exist_ok=True)
        anc_path = out_dir / "anchor_artifacts.npz"
        anc_perts = sorted(anchor_delta_dict)
        anc_lines = sorted(line_ctrl_means)
        np.savez(
            anc_path,
            anchor_type=np.array(args.anchor),
            target_line=np.array(args.target_line),
            anchor_perts=np.array(anc_perts),
            anchor_deltas=np.stack([anchor_delta_dict[p] for p in anc_perts]),
            line_names=np.array(anc_lines),
            line_ctrl_means=np.stack([line_ctrl_means[l] for l in anc_lines]),
        )
        print(f"   saved {anc_path}")

        # 6. Build residual dataloaders (group-safe BatchSampler)
        if val_perts:
            train_loader, val_loader, _, _ = build_residual_dataloaders(
                adata=adata, source_lines=sources,
                train_perts=train_perts, val_perts=val_perts,
                encoder_key=args.encoder, batch_size=args.batch_size,
                num_workers=args.num_workers, seed=args.seed,
            )
        else:
            # no per-pert val: still need group-safe train; just no val dataloader
            from cccfm.data import CCCFMDataset, GroupBatchSampler, cccfm_collate, _worker_init_fn
            from torch.utils.data import DataLoader
            train_ds = CCCFMDataset(adata, sources, train_perts, args.encoder, seed=args.seed)
            train_sampler = GroupBatchSampler(train_ds, batch_size=args.batch_size, seed=args.seed)
            train_loader = DataLoader(
                train_ds, batch_sampler=train_sampler, num_workers=args.num_workers,
                pin_memory=True, collate_fn=cccfm_collate,
                worker_init_fn=_worker_init_fn, persistent_workers=(args.num_workers > 0),
            )
            val_loader = None
        print(f"   batches/epoch  train={len(train_loader)}  "
              f"val={len(val_loader) if val_loader else 0}")
    else:
        # direct mode (legacy): per-cell val_frac, random pairing
        train_loader, val_loader, _ = build_train_dataloader(
            adata=adata, source_lines=sources, pert_set=pert_set,
            encoder_key=args.encoder, batch_size=args.batch_size,
            num_workers=args.num_workers, val_frac=args.val_frac, seed=args.seed,
        )
        print(f"   batches/epoch  train={len(train_loader)}  val={len(val_loader)}")

    # ------------------------- build model ------------------------- #
    pert_mode = "concat_secondary" if args.secondary_embed is not None else "gene_embedding"
    tokenizer = PertTokenizer(
        mode=pert_mode,
        primary_embedding_path=args.primary_embed,
        secondary_embedding_path=args.secondary_embed,
        out_dim=args.pert_dim,
        freeze_embedding=True,
    )
    print(f"   PertTokenizer: mode={pert_mode}  V={len(tokenizer.gene_to_idx)}  pert_dim={args.pert_dim}")

    # Rebuttal factorial: rank and hidden width are independent, explicit axes.
    velocity_input_dim = args.rank if args.flow_space == "lowrank" else gene_dim
    velocity_hidden = args.velocity_hidden_dim
    print(
        "   [rebuttal fixed-width] "
        f"rank={args.rank} velocity_hidden={velocity_hidden} (no autoscaling)"
    )
    velocity = VelocityNet(
        gene_dim=velocity_input_dim,
        context_dim=context_dim,
        pert_dim=args.pert_dim,
        hidden_dim=velocity_hidden,
        n_blocks=args.n_blocks,
        time_dim=args.time_dim,
        dropout=args.dropout,
        zero_init_output=(args.anchor != "none"),
    )
    n_params = sum(p.numel() for p in velocity.parameters())
    print(f"   VelocityNet params: {n_params/1e6:.2f}M")

    lit = CCCFMLitModule(
        velocity_net=velocity,
        pert_tokenizer=tokenizer,
        use_ot_pairing=args.use_ot_pairing,
        sigma=args.sigma,
        lr=args.lr,
        weight_decay=args.weight_decay,
        anchor_mode="residual" if args.anchor != "none" else "direct",
        anchor_delta_dict=anchor_delta_dict,
        line_ctrl_means=line_ctrl_means,
        anchor_target_mode=args.anchor_target_mode,
        flow_space=args.flow_space,
        lowrank_basis_B=lowrank_basis_B,
    )

    # ------------------------- trainer ------------------------- #
    # If we don't have a val_loader (e.g. residual mode without --n_holdout_perts),
    # checkpoint on train/loss instead of val/loss to avoid Lightning errors.
    ckpt_monitor = "val/loss" if val_loader is not None else "train/loss"
    ckpt_cb = ModelCheckpoint(
        dirpath=out_dir,
        filename="best",
        monitor=ckpt_monitor,
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    logger = TensorBoardLogger(save_dir=str(out_dir), name="tb")
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        log_every_n_steps=args.log_every_n_steps,
        default_root_dir=str(out_dir),
        callbacks=[ckpt_cb],
        logger=logger,
        deterministic=False,  # OT / sparse sampling introduces nondeterminism
        enable_progress_bar=True,
    )

    # ------------------------- config dump ------------------------- #
    cfg = {
        "run_id": run_id,
        "encoder": args.encoder,
        "obsm_key": obsm_key,
        "context_dim": context_dim,
        "gene_dim": gene_dim,
        "setting": args.setting,
        "sources": sources,
        "target_line": args.target_line,
        "n_train_perts": len(pert_set),
        "model": {
            "pert_dim": args.pert_dim,
            "hidden_dim": args.velocity_hidden_dim,
            "n_blocks": args.n_blocks,
            "time_dim": args.time_dim,
            "dropout": args.dropout,
            "pert_mode": pert_mode,
        },
        "optim": {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "max_epochs": args.max_epochs,
            "batch_size": args.batch_size,
            "sigma": args.sigma,
            "use_ot_pairing": bool(args.use_ot_pairing),
        },
        "align_h_mode": args.align_h_mode,
        "anchor": args.anchor,
        "anchor_mode": "residual" if args.anchor != "none" else "direct",
        "anchor_target_mode": args.anchor_target_mode,
        "flow_space": args.flow_space,
        "rank": args.rank if args.flow_space == "lowrank" else None,
        "basis_mode": args.basis_mode if args.flow_space == "lowrank" else None,
        "velocity_input_dim": int(velocity_input_dim),
        "velocity_hidden": int(velocity_hidden),
        "velocity_hidden_policy": "explicit_rebuttal_factorial",
        "n_holdout_perts": args.n_holdout_perts,
        "n_params_velocity": int(n_params),
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"   wrote {out_dir / 'config.json'}")

    # ------------------------- fit ------------------------- #
    print(f"[{run_id}] starting trainer.fit ...")
    trainer.fit(
        lit,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader if val_loader is not None else None,
    )
    wall = time.perf_counter() - t0
    print(f"[{run_id}] done in {wall/60:.1f} min. "
          f"best_ckpt={ckpt_cb.best_model_path}  val/loss={ckpt_cb.best_model_score}")

    # ------------------------- minimal stats csv ------------------------- #
    stats_row = {
        "run_id": run_id,
        "encoder": args.encoder,
        "setting": args.setting,
        "best_ckpt": str(ckpt_cb.best_model_path),
        "best_val_loss": float(ckpt_cb.best_model_score) if ckpt_cb.best_model_score is not None else None,
        "wall_minutes": round(wall / 60, 2),
        "n_train_perts": len(pert_set),
        "n_params_velocity": int(n_params),
    }
    pd.DataFrame([stats_row]).to_csv(out_dir / "train_stats.csv", index=False)
    print(f"   wrote {out_dir / 'train_stats.csv'}")


if __name__ == "__main__":
    main()
