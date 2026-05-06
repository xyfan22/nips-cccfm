"""Low-memory STATE inference: loads control cells once, then runs per-pert
forward passes without constructing a giant padded AnnData.

This avoids the OOM that kills `state tx infer --all-perts` which clones
control cells for every missing perturbation and loads the whole thing.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--state_model_dir", type=Path, required=True)
    p.add_argument("--state_checkpoint", type=Path, default=None)
    p.add_argument("--input_h5ad", type=Path, required=True,
                   help="state_input.h5ad built by 13_eval_state_zeroshot.py")
    p.add_argument("--output_h5ad", type=Path, required=True)
    p.add_argument("--cell_set_len", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_model(model_dir: Path, checkpoint: Path | None, device: torch.device):
    from state.tx.models.state_transition import StateTransitionPerturbationModel

    ckpt = checkpoint or (model_dir / "checkpoints" / "final.ckpt")
    model = StateTransitionPerturbationModel.load_from_checkpoint(
        str(ckpt), map_location=device
    )
    model.eval()
    model.to(device)
    return model


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from {args.state_model_dir}")
    model = load_model(args.state_model_dir, args.state_checkpoint, device)

    # Load config
    with open(args.state_model_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cell_set_len = cfg.get("model", {}).get("kwargs", {}).get(
        "cell_set_len", args.cell_set_len
    )
    print(f"cell_set_len: {cell_set_len}")

    # Load pert onehot map
    pert_onehot_map = torch.load(
        args.state_model_dir / "pert_onehot_map.pt",
        map_location="cpu", weights_only=False,
    )
    pert_dim = next(iter(pert_onehot_map.values())).shape[0]
    print(f"pert vocab size: {len(pert_onehot_map)}, pert_dim: {pert_dim}")

    # Load batch onehot map (optional)
    batch_onehot_map_path = args.state_model_dir / "batch_onehot_map.pkl"
    batch_onehot_map = None
    if batch_onehot_map_path.exists():
        with open(batch_onehot_map_path, "rb") as f:
            batch_onehot_map = pickle.load(f)

    # Determine control pert name in adata vs pert_onehot_map
    # adata uses "control"; pert_onehot_map uses "non-targeting"
    control_pert_in_adata = "control"
    control_pert_in_vocab = "non-targeting" if "non-targeting" in pert_onehot_map else "control"

    # Load input data — only need X_hvg and obs
    import anndata as ad
    print(f"Loading {args.input_h5ad}")
    adata = ad.read_h5ad(args.input_h5ad)
    X_in = np.asarray(adata.obsm["X_hvg"], dtype=np.float32)
    pert_names = adata.obs["pert"].astype(str).values

    # Identify control cells
    ctrl_mask = pert_names == control_pert_in_adata
    ctrl_indices = np.where(ctrl_mask)[0]
    print(f"Total cells: {adata.n_obs}, control cells: {len(ctrl_indices)}")

    # Get unique non-control perturbations
    unique_perts = sorted(set(pert_names) - {control_pert_in_adata})
    print(f"Non-control perts: {len(unique_perts)}")

    # Prepare output storage — only for non-control cells
    # We'll collect predictions per perturbation
    rng = np.random.RandomState(args.seed)

    # Default pert vector (control fallback for OOV perts)
    control_key = control_pert_in_vocab
    if control_key in pert_onehot_map:
        default_pert_vec = pert_onehot_map[control_key].float().clone()
    else:
        default_pert_vec = torch.zeros(pert_dim, dtype=torch.float32)
        default_pert_vec[0] = 1.0

    # Check if model uses batch encoder
    uses_batch = batch_onehot_map is not None
    batch_indices_all = None
    if uses_batch and "gem_group" in adata.obs:
        raw_labels = adata.obs["gem_group"].astype(str).values
        label_to_idx = {}
        for k, v in batch_onehot_map.items():
            idx = int(torch.argmax(v).item()) if torch.is_tensor(v) else int(v)
            label_to_idx[str(k)] = idx
        batch_indices_all = np.array(
            [label_to_idx.get(l, 0) for l in raw_labels], dtype=np.int64
        )

    # Run inference per perturbation
    all_pred_rows = {}  # pert -> list of row indices in original adata
    all_pred_arrays = {}  # pert -> np.ndarray [n_cells, dim]

    model_device = next(model.parameters()).device
    with torch.no_grad():
        for pi, pert in enumerate(unique_perts):
            pert_mask = pert_names == pert
            pert_indices = np.where(pert_mask)[0]
            if len(pert_indices) == 0:
                continue

            # Get pert onehot vector
            map_key = pert
            vec = pert_onehot_map.get(map_key, None)
            if vec is None:
                vec = default_pert_vec

            # Process in windows of cell_set_len
            preds_list = []
            start = 0
            while start < len(pert_indices):
                end = min(start + cell_set_len, len(pert_indices))
                win_indices = pert_indices[start:end]
                win_size = len(win_indices)

                # Sample matched control basals
                sampled_ctrl_idx = rng.choice(ctrl_indices, size=win_size, replace=True)
                ctrl_basal = X_in[sampled_ctrl_idx, :]  # [win, E_in]

                # Build batch
                X_batch = torch.tensor(ctrl_basal, dtype=torch.float32, device=model_device)
                pert_oh = vec.float().unsqueeze(0).repeat(win_size, 1).to(model_device)

                batch = {
                    "ctrl_cell_emb": X_batch,
                    "pert_emb": pert_oh,
                    "pert_name": [pert] * win_size,
                }
                if uses_batch and batch_indices_all is not None:
                    batch["batch"] = torch.tensor(
                        batch_indices_all[win_indices], dtype=torch.long, device=model_device
                    )

                batch_out = model.predict_step(batch, batch_idx=0, padded=False)
                preds = batch_out["preds"].detach().cpu().numpy().astype(np.float32)
                preds_list.append(preds)
                start = end

            all_pred_rows[pert] = pert_indices
            all_pred_arrays[pert] = np.concatenate(preds_list, axis=0)

            if (pi + 1) % 50 == 0 or pi < 5:
                print(f"[{pi+1}/{len(unique_perts)}] {pert}: "
                      f"{len(pert_indices)} cells, pred shape {all_pred_arrays[pert].shape}")

    # Build output AnnData — keep same obs/var structure, replace obsm["X_hvg"]
    # For control cells, keep original X_hvg; for pert cells, use predictions
    out_hvg = X_in.copy()
    for pert, indices in all_pred_rows.items():
        out_hvg[indices] = all_pred_arrays[pert]

    adata.obsm["X_hvg"] = out_hvg
    args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output_h5ad)
    print(f"Wrote {args.output_h5ad} ({adata.n_obs} cells)")


if __name__ == "__main__":
    main()
