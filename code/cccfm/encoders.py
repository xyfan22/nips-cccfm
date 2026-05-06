"""Frozen scFM encoder registry (PCA null + 5 single-cell foundation models).

This module wraps each encoder with a uniform `FrozenEncoder` interface:
    load_pretrained(path): load weights, fix to eval mode + requires_grad=False.
    encode(x): [N, G_raw] -> [N, D_out].
    output_dim: int.

Offline-featurization workflow: pass every cell of an adata through `encode`,
store as `adata.obsm["h_<encoder>"]`. The training-time dataloader then reads
`obsm` directly; the encoder never participates in forward/backward and uses
no GPU memory during probe training.

==============================================================================
External dependencies (REQUIRED to run encoder featurization)
==============================================================================
The non-PCA encoders dynamically `sys.path.insert` into upstream repository
trees we vendored under `<repo-root>/baselines/<encoder>/`. The vendored
copies in this artifact contain ONLY the upstream Python sources, READMEs,
and configs (text files <=200 KB). They DO NOT include:
  - pretrained model checkpoints (multi-GB binaries),
  - large .npy / .pkl / .h5 / .h5ad data dumps,
  - example notebooks with output cells.

To actually run encoder featurization end-to-end, a reviewer must:
  1. Install each upstream's pip dependencies (per its `requirements.txt`
     under `baselines/<encoder>/`).
  2. Download the official pretrained checkpoint from the upstream and
     place it at the path each `load_pretrained` constructor expects.
  3. (For UCE) populate `baselines/UCE/model_files/` with the protein-
     embedding files distributed by the upstream.

Upstream URLs and licenses (all public; included only as third-party
references — none of these are author-identifying):
    PCA null     (no upstream): sklearn `IncrementalPCA`, fit on source-line
                                controls. D_out = 128.
    Geneformer:  https://huggingface.co/ctheodoris/Geneformer
                 Apache-2.0. D_out = 768 (V2-104M).
    scGPT:       https://github.com/bowang-lab/scGPT
                 MIT. D_out = 512 (whole-human checkpoint).
    UCE:         https://github.com/snap-stanford/UCE
                 MIT. D_out = 1280 (UCE-100M).
    scFoundation: https://github.com/biomap-research/scFoundation
                 Apache-2.0. D_out = 3072.
    STATE-SE:    https://github.com/ArcInstitute/state
                 Apache-2.0. D_out = 2048 (SE module).

For the artifact's documented evaluator-level smoke-test (see
`docs/reproducibility.md`) the encoder code below is NOT exercised: every
`_summary.csv` shipped under `results/` was produced from already-cached
`adata.obsm["h_<encoder>"]` vectors, and the smoke-test pipeline reads only
those CSVs and the master MD.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Type

import torch
import torch.nn as nn


class FrozenEncoder(ABC, nn.Module):
    """冻结 encoder 抽象基类."""

    @abstractmethod
    def load_pretrained(self, path: Path) -> None: ...

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, G_raw] -> h: [N, D_out]."""
        ...

    @property
    @abstractmethod
    def output_dim(self) -> int: ...

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()


# ---------------------------------------------------------------------------
# 1. PCA null encoder (ablation 对照)
# ---------------------------------------------------------------------------
class PCANullEncoder(FrozenEncoder):
    """PCA null encoder (encoder-ablation 对照组).

    中文:
        - 在 source-line control cells 上 fit sklearn.IncrementalPCA, 然后对所有 cells 做 transform.
        - 如果 ccCFM + PCA 已经接近最强 encoder, 说明 foundation-encoder 的贡献有限;
          如果差距大, 说明 pretrained scFM 确实在 cross-line 任务上有信号.
    """

    def __init__(self, n_components: int = 128, batch_size: int = 4096):
        super().__init__()
        self.n_components = n_components
        self.batch_size = batch_size
        self.pca = None  # sklearn.decomposition.IncrementalPCA 实例

    def fit(self, X_fit) -> "PCANullEncoder":
        """Fit IncrementalPCA on [N, G] expression array.

        中文: 典型调用是传入 source-line control cells 的 log1p 表达, X_fit.shape = [N, G].
              IncrementalPCA 比标准 PCA 更省内存, 适合几万~几十万 cell.
        """
        from sklearn.decomposition import IncrementalPCA
        import numpy as np
        Xn = X_fit.toarray() if hasattr(X_fit, "toarray") else X_fit
        Xn = np.asarray(Xn, dtype=np.float32)
        n = min(self.n_components, Xn.shape[0], Xn.shape[1])
        self.pca = IncrementalPCA(n_components=n, batch_size=self.batch_size)
        # chunk fit
        for start in range(0, Xn.shape[0], self.batch_size):
            self.pca.partial_fit(Xn[start : start + self.batch_size])
        self.n_components = n
        return self

    def load_pretrained(self, path: Path) -> None:
        import joblib
        self.pca = joblib.load(path)
        self.n_components = self.pca.n_components_

    def save(self, path: Path) -> None:
        import joblib
        from pathlib import Path as _P
        _P(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pca, path)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        assert self.pca is not None, "call fit() or load_pretrained() first"
        import numpy as np
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
            device = x.device
        else:
            x_np = np.asarray(x)
            device = torch.device("cpu")
        if hasattr(x_np, "toarray"):
            x_np = x_np.toarray()
        x_np = np.asarray(x_np, dtype=np.float32)
        h = self.pca.transform(x_np)
        return torch.from_numpy(h).to(device)

    @property
    def output_dim(self) -> int:
        return self.n_components


# ---------------------------------------------------------------------------
# 2. STATE-SE (Arc, perturbation-aware)
# ---------------------------------------------------------------------------
class STATESEEncoder(FrozenEncoder):
    """STATE-SE encoder (Arc Institute SE-600M, dim 2048).

    Uses Arc's frozen SE transformer to produce L2-normalized 2048-dim
    cell embeddings from log1p-normalized expression over our 6642 HVG.
    Gene mapping: Ensembl IDs → gene symbols → STATE-SE protein-embedding
    vocabulary (19790 genes, ESM2 5120-dim).
    """

    def __init__(self, checkpoint_path: Optional[Path] = None):
        super().__init__()
        self._out_dim = 2048
        self.backbone = None
        self._pad_length = 2048
        self._cls_token_idx = 3
        self.n_hvg_in_vocab: int = 0
        self.n_hvg_total: int = 0
        if checkpoint_path is not None:
            self.load_pretrained(checkpoint_path)

    def load_pretrained(self, path: Path) -> None:
        import sys
        import anndata as ad

        path = Path(path)
        proj_root = Path(__file__).resolve().parents[2]

        state_src = str(proj_root / "baselines" / "state" / "src")
        if state_src not in sys.path:
            sys.path.insert(0, state_src)

        from state.emb.inference import Inference

        pe_dict = torch.load(
            path / "protein_embeddings.pt", map_location="cpu", weights_only=False
        )

        inferer = Inference(cfg=None, protein_embeds=pe_dict)
        inferer.load_model(str(path / "se600m_epoch16.ckpt"))
        self.backbone = inferer.model

        cfg = inferer._vci_conf
        self._pad_length = cfg.dataset.pad_length
        self._cls_token_idx = cfg.dataset.cls_token_idx
        self._out_dim = cfg.model.output_dim

        pe_genes = list(pe_dict.keys())
        pe_gene2idx = {g: i for i, g in enumerate(pe_genes)}

        hvg_path = proj_root / "data" / "processed" / "hvg_genes.txt"
        with open(hvg_path) as f:
            hvg_ensembl = [line.strip() for line in f if line.strip()]

        adata = ad.read_h5ad(
            proj_root / "data" / "processed" / "replogle_preprocessed.h5ad",
            backed="r",
        )
        ens2sym = dict(zip(adata.var_names, adata.var["gene_name"].values))
        adata.file.close()

        indices, in_vocab = [], []
        for eid in hvg_ensembl:
            sym = ens2sym.get(eid)
            if sym and sym in pe_gene2idx:
                indices.append(pe_gene2idx[sym])
                in_vocab.append(True)
            else:
                indices.append(0)
                in_vocab.append(False)

        self.register_buffer(
            "_global_gene_indices", torch.tensor(indices, dtype=torch.long)
        )
        self.register_buffer(
            "_hvg_in_vocab", torch.tensor(in_vocab, dtype=torch.bool)
        )
        self.n_hvg_in_vocab = int(self._hvg_in_vocab.sum())
        self.n_hvg_total = len(hvg_ensembl)

        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        print(
            f"   STATESEEncoder: {self.n_hvg_in_vocab}/{self.n_hvg_total} HVG "
            f"in STATE-SE vocab ({self.n_hvg_in_vocab/self.n_hvg_total*100:.1f}%), "
            f"output_dim={self._out_dim}, pad_length={self._pad_length}"
        )

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 6642] log1p-normalized → [B, 2048] L2-normalized embeddings."""
        device = x.device
        B, G = x.shape
        pad_len = self._pad_length
        model = self.backbone

        x_v = x.clone()
        x_v[:, ~self._hvg_in_vocab] = 0.0

        x_sum = x_v.sum(dim=1, keepdim=True).clamp(min=1e-8)
        expr_w = x_v / x_sum

        x_rank = x_v + torch.rand_like(x_v) * 1e-6
        x_rank[:, ~self._hvg_in_vocab] = -1.0
        _, rank = x_rank.sort(dim=1, descending=True)
        rank = rank[:, : pad_len - 1]

        tokens = torch.zeros(B, pad_len, dtype=torch.long, device=device)
        tokens[:, 0] = self._cls_token_idx
        tokens[:, 1:] = self._global_gene_indices[rank]

        counts = torch.zeros(B, pad_len, device=device)
        counts[:, 1:] = 100.0 * torch.gather(expr_w, 1, rank)

        src = model.pe_embedding(tokens)
        src = nn.functional.normalize(src, dim=2)
        src[:, 0, :] = model.cls_token.expand(B, -1)

        if model.dataset_token is not None:
            ds_tok = model.dataset_token.expand(B, -1).unsqueeze(1)
            src = torch.cat([src, ds_tok], dim=1)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            _, embedding, _ = model.forward(
                src,
                mask=torch.zeros(B, pad_len, dtype=torch.bool, device=device),
                counts=counts,
            )

        return embedding.float()

    @property
    def output_dim(self) -> int:
        return self._out_dim


# ---------------------------------------------------------------------------
# 3. Geneformer
# ---------------------------------------------------------------------------
class GeneformerEncoder(FrozenEncoder):
    """Geneformer V2-104M encoder 封装.

    Tokenization: rank-order genes by (expm1(log1p_expr) / gene_median) descending,
    map Ensembl IDs → Geneformer token IDs, forward through frozen BERT, mean-pool
    over non-pad positions → [N, 768].
    """

    def __init__(self, checkpoint_path: Optional[Path] = None):
        super().__init__()
        self._out_dim = 768
        self.backbone = None
        self.token_dict: Dict[str, int] = {}
        self.gene_median_dict: Dict[str, float] = {}
        # Per-HVG-column arrays set in load_pretrained
        self.hvg_token_ids: Optional[torch.Tensor] = None  # [G] int
        self.hvg_medians: Optional[torch.Tensor] = None    # [G] float
        self.hvg_in_vocab: Optional[torch.Tensor] = None   # [G] bool
        self.pad_token_id = 0
        self.max_input_size = 4096
        if checkpoint_path is not None:
            self.load_pretrained(checkpoint_path)

    def load_pretrained(self, path: Path) -> None:
        import pickle
        from transformers import BertModel

        path = Path(path)
        # Load BERT backbone
        self.backbone = BertModel.from_pretrained(str(path))
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        cfg = self.backbone.config
        self._out_dim = cfg.hidden_size
        self.pad_token_id = cfg.pad_token_id
        self.max_input_size = cfg.max_position_embeddings

        # Dictionaries live under baselines/Geneformer/geneformer/
        pkg_dir = path.parent.parent / "geneformer"
        with open(pkg_dir / "token_dictionary_gc104M.pkl", "rb") as f:
            self.token_dict = pickle.load(f)
        with open(pkg_dir / "gene_median_dictionary_gc104M.pkl", "rb") as f:
            self.gene_median_dict = pickle.load(f)

        # Build per-HVG-column lookup from hvg_genes.txt
        hvg_path = Path("data/processed/hvg_genes.txt")
        with open(hvg_path) as f:
            hvg_genes = [line.strip() for line in f if line.strip()]

        token_ids = []
        medians = []
        in_vocab = []
        for g in hvg_genes:
            if g in self.token_dict:
                token_ids.append(self.token_dict[g])
                medians.append(self.gene_median_dict.get(g, 1.0))
                in_vocab.append(True)
            else:
                token_ids.append(self.pad_token_id)
                medians.append(1.0)
                in_vocab.append(False)

        self.hvg_token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.hvg_medians = torch.tensor(medians, dtype=torch.float32)
        self.hvg_in_vocab = torch.tensor(in_vocab, dtype=torch.bool)

        n_in = int(self.hvg_in_vocab.sum())
        print(f"   GeneformerEncoder: {n_in}/{len(hvg_genes)} HVG in vocab "
              f"({n_in/len(hvg_genes)*100:.1f}%), hidden={self._out_dim}, "
              f"max_pos={self.max_input_size}")

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, G] log1p-normalized expression → [B, D] embeddings."""
        device = x.device
        B, G = x.shape

        hvg_token_ids = self.hvg_token_ids.to(device)   # [G]
        hvg_medians = self.hvg_medians.to(device)        # [G]
        hvg_in_vocab = self.hvg_in_vocab.to(device)      # [G]

        # Recover linear-scale expression and normalize by gene median
        expr_linear = torch.expm1(x)  # [B, G]
        norm_expr = expr_linear / hvg_medians.unsqueeze(0)  # [B, G]

        # Mask: only keep genes that are expressed (>0) AND in vocab
        expressed = expr_linear > 0  # [B, G]
        valid = expressed & hvg_in_vocab.unsqueeze(0)  # [B, G]

        # Per-cell: rank valid genes by norm_expr descending, truncate to max_input_size
        norm_expr_masked = norm_expr.clone()
        norm_expr_masked[~valid] = -float("inf")

        # Cap at 2048 for efficiency (original Geneformer V1 uses 2048)
        max_len = min(self.max_input_size, 2048)
        _, rank_idx = norm_expr_masked.sort(dim=1, descending=True)  # [B, G]

        # Count valid genes per cell to determine actual sequence lengths
        n_valid = valid.sum(dim=1)  # [B]
        seq_len = n_valid.clamp(max=max_len)  # [B]
        actual_max = int(seq_len.max().item())

        # Truncate to actual_max (dynamic padding within batch)
        rank_idx = rank_idx[:, :actual_max]  # [B, actual_max]
        input_ids = hvg_token_ids[rank_idx]  # [B, actual_max]

        # Build attention mask: positions beyond each cell's seq_len get 0
        pos = torch.arange(actual_max, device=device).unsqueeze(0)  # [1, actual_max]
        attention_mask = (pos < seq_len.unsqueeze(1)).long()  # [B, actual_max]
        input_ids = input_ids * attention_mask

        # Forward through BERT with fp16
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        last_hidden = outputs.last_hidden_state  # [B, max_len, D]

        # Mean pool over non-pad positions
        mask_f = attention_mask.unsqueeze(-1).float()  # [B, max_len, 1]
        summed = (last_hidden * mask_f).sum(dim=1)     # [B, D]
        lengths = mask_f.sum(dim=1).clamp(min=1.0)     # [B, 1]
        pooled = summed / lengths                       # [B, D]

        return pooled

    @property
    def output_dim(self) -> int:
        return self._out_dim


# ---------------------------------------------------------------------------
# 4. scGPT
# ---------------------------------------------------------------------------
class SCGPTEncoder(FrozenEncoder):
    """scGPT whole-human encoder (Cui 2024, dim 512).

    Tokenization: per cell, select nonzero genes present in scGPT vocab,
    prepend <cls>, bin expression values into 51 bins, pad/sample to
    max_seq_len=1200, forward through frozen TransformerModel, take
    position-0 (CLS) embedding → [N, 512].
    """

    def __init__(self, checkpoint_path: Optional[Path] = None):
        super().__init__()
        self._out_dim = 512
        self.backbone = None
        self.vocab = None
        self.pad_id: int = 0
        self.cls_id: int = 0
        self.pad_value: float = -2.0
        self.max_seq_len: int = 1200
        self.n_bins: int = 51
        self._hvg_vocab_ids = None  # np int64 [G]
        self._hvg_in_vocab = None   # np bool  [G]
        self.n_hvg_in_vocab: int = 0
        self.n_hvg_total: int = 0
        if checkpoint_path is not None:
            self.load_pretrained(checkpoint_path)

    @staticmethod
    def _setup_scgpt_imports(scgpt_root: Path):
        """Register a minimal scgpt package so we can import submodules
        without torchtext (needed by scgpt.tokenizer but not by us)."""
        import sys
        import types
        import logging

        _p = str(scgpt_root)
        if _p not in sys.path:
            sys.path.insert(0, _p)
        if "scgpt" not in sys.modules:
            _pkg = types.ModuleType("scgpt")
            _pkg.__path__ = [str(scgpt_root / "scgpt")]
            _pkg.__package__ = "scgpt"
            _pkg.logger = logging.getLogger("scGPT")
            sys.modules["scgpt"] = _pkg

    def load_pretrained(self, path: Path) -> None:
        import json
        import numpy as np
        import pandas as pd

        path = Path(path)
        proj_root = Path(__file__).resolve().parents[2]
        scgpt_root = proj_root / "baselines" / "scGPT"
        self._setup_scgpt_imports(scgpt_root)

        from scgpt.model.model import TransformerModel

        vocab_file = scgpt_root / "scgpt" / "tokenizer" / "default_census_vocab.json"
        with open(vocab_file) as f:
            token2idx = json.load(f)
        next_id = max(token2idx.values()) + 1
        for s in ["<pad>", "<cls>", "<eoc>"]:
            if s not in token2idx:
                token2idx[s] = next_id
                next_id += 1
        self.vocab = token2idx
        self.pad_id = token2idx["<pad>"]
        self.cls_id = token2idx["<cls>"]

        with open(path / "args.json") as f:
            cfg = json.load(f)
        self._out_dim = cfg["embsize"]
        self.pad_value = float(cfg["pad_value"])
        self.max_seq_len = cfg.get("max_seq_len", 1200)

        self.backbone = TransformerModel(
            ntoken=len(token2idx),
            d_model=cfg["embsize"],
            nhead=cfg["nheads"],
            d_hid=cfg["d_hid"],
            nlayers=cfg["nlayers"],
            nlayers_cls=cfg.get("n_layers_cls", 3),
            n_cls=1,
            vocab=token2idx,
            dropout=cfg["dropout"],
            pad_token=cfg["pad_token"],
            pad_value=cfg["pad_value"],
            do_mvc=True,
            do_dab=False,
            use_batch_labels=False,
            domain_spec_batchnorm=False,
            input_emb_style="continuous",
            explicit_zero_prob=False,
            use_fast_transformer=False,
            pre_norm=False,
        )

        ckpt = torch.load(path / "best_model.pt", map_location="cpu")
        converted = {
            k.replace("Wqkv.", "in_proj_"): v for k, v in ckpt.items()
        }
        model_dict = self.backbone.state_dict()
        loaded = {
            k: v
            for k, v in converted.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(loaded)
        self.backbone.load_state_dict(model_dict)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        hvg_path = proj_root / "data" / "processed" / "hvg_genes.txt"
        with open(hvg_path) as f:
            hvg_ensembl = [line.strip() for line in f if line.strip()]

        gene_info = pd.read_csv(path / "gene_info.csv")
        ens2sym = dict(zip(gene_info["feature_id"], gene_info["feature_name"]))

        vocab_ids, in_vocab = [], []
        for eid in hvg_ensembl:
            sym = ens2sym.get(eid)
            if sym and sym in token2idx:
                vocab_ids.append(token2idx[sym])
                in_vocab.append(True)
            else:
                vocab_ids.append(self.pad_id)
                in_vocab.append(False)

        self._hvg_vocab_ids = np.array(vocab_ids, dtype=np.int64)
        self._hvg_in_vocab = np.array(in_vocab, dtype=bool)
        self.n_hvg_in_vocab = int(self._hvg_in_vocab.sum())
        self.n_hvg_total = len(hvg_ensembl)

        print(
            f"   SCGPTEncoder: {self.n_hvg_in_vocab}/{self.n_hvg_total} HVG in "
            f"scGPT vocab ({self.n_hvg_in_vocab / self.n_hvg_total * 100:.1f}%), "
            f"d_model={self._out_dim}, max_seq={self.max_seq_len}"
        )

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 6642] log1p-normalized → [B, 512] cell embeddings."""
        import numpy as np
        from scgpt.preprocess import binning

        device = x.device
        B, G = x.shape
        x_np = x.detach().cpu().numpy().astype(np.float32)

        all_genes, all_vals = [], []
        for i in range(B):
            row = x_np[i]
            mask = (row != 0) & self._hvg_in_vocab
            idx = np.where(mask)[0]
            vals = row[idx]
            gids = self._hvg_vocab_ids[idx]

            gids = np.insert(gids, 0, self.cls_id)
            vals = np.insert(vals, 0, self.pad_value)

            if len(vals) > 1:
                binned = binning(vals[1:].copy(), n_bins=self.n_bins)
                vals = np.concatenate(
                    [[self.pad_value], np.asarray(binned, dtype=np.float32)]
                )

            all_genes.append(torch.from_numpy(gids.copy()).long())
            all_vals.append(torch.from_numpy(vals.copy()).float())

        max_len = min(max(len(g) for g in all_genes), self.max_seq_len)
        p_genes, p_vals = [], []
        for g, v in zip(all_genes, all_vals):
            L = len(g)
            if L > max_len:
                sel = torch.randperm(L - 1)[: max_len - 1] + 1
                sel = torch.cat([torch.tensor([0]), sel])
                g, v = g[sel], v[sel]
            elif L < max_len:
                g = torch.cat([g, g.new_full((max_len - L,), self.pad_id)])
                v = torch.cat([v, v.new_full((max_len - L,), self.pad_value)])
            p_genes.append(g)
            p_vals.append(v)

        gene_ids = torch.stack(p_genes).to(device)
        values = torch.stack(p_vals).to(device)
        padding_mask = gene_ids.eq(self.pad_id)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out = self.backbone._encode(gene_ids, values, padding_mask)
        return out[:, 0, :].float()

    @property
    def output_dim(self) -> int:
        return self._out_dim


# ---------------------------------------------------------------------------
# 5. UCE — numba-accelerated per-batch encode
# ---------------------------------------------------------------------------
def _uce_build_tokens_batch(choice_idx, valid_pe, valid_chroms, valid_starts,
                            cls_idx, chrom_right_idx, chrom_offset, pad_length):
    """Build token sequences for a batch of cells. Returns (tokens, masks) arrays."""
    import numpy as np
    N = choice_idx.shape[0]
    all_tokens = np.full((N, pad_length), 0, dtype=np.int64)  # PAD=0
    all_masks = np.zeros((N, pad_length), dtype=np.float32)

    for c in range(N):
        ci = choice_idx[c]
        # Sort by chromosome
        chroms_c = valid_chroms[ci]
        order = np.argsort(chroms_c)
        ci = ci[order]
        chroms_sorted = valid_chroms[ci]
        starts_sorted = valid_starts[ci]

        all_tokens[c, 0] = cls_idx
        pos = 1
        i = 0
        while i < len(chroms_sorted) and pos < pad_length - 2:
            cur_chrom = chroms_sorted[i]
            all_tokens[c, pos] = cur_chrom + chrom_offset
            pos += 1
            j = i
            while j < len(chroms_sorted) and chroms_sorted[j] == cur_chrom:
                j += 1
            sub_starts = starts_sorted[i:j]
            sub_order = np.argsort(sub_starts)
            for k in sub_order:
                if pos >= pad_length - 1:
                    break
                all_tokens[c, pos] = valid_pe[ci[i + k]]
                pos += 1
            if pos < pad_length:
                all_tokens[c, pos] = chrom_right_idx
                pos += 1
            i = j
        all_masks[c, :pos] = 1.0

    return all_tokens, all_masks


# Try to use numba for speed
try:
    import numpy as _np
    from numba import njit as _njit

    @_njit(cache=True)
    def _uce_build_tokens_numba(choice_idx, valid_pe, valid_chroms, valid_starts,
                                cls_idx, chrom_right_idx, chrom_offset, pad_length,
                                all_tokens, all_masks):
        N = choice_idx.shape[0]
        for c in range(N):
            ci = choice_idx[c]
            chroms_c = valid_chroms[ci]
            order = _np.argsort(chroms_c)
            ci_sorted = ci[order]
            chroms_sorted = valid_chroms[ci_sorted]
            starts_sorted = valid_starts[ci_sorted]

            all_tokens[c, 0] = cls_idx
            pos = 1
            i = 0
            while i < len(chroms_sorted) and pos < pad_length - 2:
                cur_chrom = chroms_sorted[i]
                all_tokens[c, pos] = cur_chrom + chrom_offset
                pos += 1
                j = i
                while j < len(chroms_sorted) and chroms_sorted[j] == cur_chrom:
                    j += 1
                sub_starts = starts_sorted[i:j]
                sub_order = _np.argsort(sub_starts)
                for k in sub_order:
                    if pos >= pad_length - 1:
                        break
                    all_tokens[c, pos] = valid_pe[ci_sorted[i + int(k)]]
                    pos += 1
                if pos < pad_length:
                    all_tokens[c, pos] = chrom_right_idx
                    pos += 1
                i = j
            all_masks[c, :pos] = 1.0

    _UCE_USE_NUMBA = True
except ImportError:
    _UCE_USE_NUMBA = False


class UCEEncoder(FrozenEncoder):
    """UCE (Universal Cell Embeddings) per-batch encoder.

    Loads model weights + ESM2 protein embeddings + chromosome maps,
    encodes cells in batches by sampling gene sentences with numba JIT.
    """

    PAD_TOKEN_IDX = 0
    CHROM_TOKEN_RIGHT_IDX = 2
    CLS_TOKEN_IDX = 3
    CHROM_TOKEN_OFFSET = 143574
    PAD_LENGTH = 1536
    SAMPLE_SIZE = 1024

    def __init__(self, checkpoint_path: Optional[Path] = None):
        super().__init__()
        self._out_dim = 1280
        self.model_dir = None
        self.misc_dir = None
        self.model = None
        self.all_pe = None
        self._valid_pe = None
        self._valid_chroms = None
        self._valid_starts = None
        self._n_valid = None
        self.valid_gene_mask = None
        if checkpoint_path is not None:
            self.load_pretrained(checkpoint_path)

    def load_pretrained(self, path: Path) -> None:
        import pickle
        import sys
        import json
        import pandas as pd

        self.model_dir = Path(path)
        self.misc_dir = Path("baselines/UCE/model_files")

        if not self.model_dir.exists():
            raise FileNotFoundError(f"UCE model dir not found: {self.model_dir}")
        if not self.misc_dir.exists():
            raise FileNotFoundError(f"UCE misc dir not found: {self.misc_dir}")

        uce_root = str(Path("baselines/UCE").resolve())
        if uce_root not in sys.path:
            sys.path.insert(0, uce_root)

        from model import TransformerModel
        from utils import load_model_state_dict

        config_path = self.model_dir / "config.json"
        with open(config_path) as f:
            cfg = json.load(f)
        token_dim = cfg["token_dim"]
        self._out_dim = cfg["output_dim"]

        state_dict = load_model_state_dict(str(self.model_dir / "pytorch_model.bin"))
        model = TransformerModel(
            token_dim=token_dim, d_model=1280, nhead=20,
            d_hid=cfg["d_hid"], nlayers=cfg["nlayers"], dropout=0.05,
            output_dim=self._out_dim,
        )

        all_pe = torch.load(str(self.misc_dir / "all_tokens.torch"), map_location="cpu")
        if all_pe.shape[0] == 143574:
            torch.manual_seed(23)
            all_pe = torch.vstack((all_pe, torch.normal(0, 1, (1895, token_dim))))
        all_pe.requires_grad = False
        self.all_pe = all_pe

        model.pe_embedding = nn.Embedding.from_pretrained(all_pe)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        self.model = model

        # Human protein embeddings
        pe_path = (self.misc_dir / "protein_embeddings" /
                   "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt")
        gene_to_emb = torch.load(str(pe_path), map_location="cpu")
        self._gene_to_emb_upper = {k.upper(): v for k, v in gene_to_emb.items()}
        del gene_to_emb

        with open(self.misc_dir / "species_offsets.pkl", "rb") as f:
            self._human_offset = pickle.load(f)["human"]

        chrom_df = pd.read_csv(self.misc_dir / "species_chrom.csv")
        hc = chrom_df[chrom_df["species"] == "human"].copy()
        hc["gene_upper"] = hc["gene_symbol"].str.upper()
        hc = hc.set_index("gene_upper")
        hc["spec_chrom"] = pd.Categorical(hc["species"] + "_" + hc["chromosome"])
        self._human_chrom = hc
        self._chrom_code_map = {cat: code for code, cat in
                                enumerate(hc["spec_chrom"].cat.categories)}

    def encode_adata_prep(self, gene_names) -> None:
        """Build per-gene lookup arrays from gene names. Call once before encode()."""
        import numpy as np

        gene_list = [str(g).upper() for g in gene_names]
        emb_genes = set(self._gene_to_emb_upper.keys())
        spec_pe_genes = list(self._gene_to_emb_upper.keys())

        pe_idxs, chroms, starts, valid = [], [], [], []
        for g in gene_list:
            if g in emb_genes and g in self._human_chrom.index:
                pe_idxs.append(spec_pe_genes.index(g) + self._human_offset)
                row = self._human_chrom.loc[g]
                chroms.append(self._chrom_code_map.get(row["spec_chrom"], 0))
                starts.append(int(row["start"]))
                valid.append(True)
            else:
                valid.append(False)

        self._valid_pe = np.array(pe_idxs, dtype=np.int64)
        self._valid_chroms = np.array(chroms, dtype=np.int64)
        self._valid_starts = np.array(starts, dtype=np.float64)
        self.valid_gene_mask = np.array(valid, dtype=bool)
        self._n_valid = len(pe_idxs)
        print(f"   UCE: {len(pe_idxs)}/{len(gene_list)} genes matched to protein embeddings")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of cells: x is [N, G] expression matrix."""
        import numpy as np

        assert self.model is not None, "call load_pretrained() first"
        assert self._valid_pe is not None, "call encode_adata_prep() first"

        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)
        if hasattr(x_np, "toarray"):
            x_np = x_np.toarray()
        x_np = x_np.astype(np.float32)

        N = x_np.shape[0]
        device = next(self.model.parameters()).device

        # Vectorized sampling weights
        valid_x = x_np[:, self.valid_gene_mask]
        weights = np.log1p(valid_x.astype(np.float64))
        w_sums = weights.sum(axis=1, keepdims=True)
        w_sums = np.where(w_sums == 0, 1.0, w_sums)
        weights = weights / w_sums
        cum_w = np.cumsum(weights, axis=1)

        # Vectorized sampling
        n_sample = min(self.SAMPLE_SIZE, self._n_valid)
        rands = np.random.random((N, n_sample))
        choice_idx = np.empty((N, n_sample), dtype=np.int64)
        for c in range(N):
            choice_idx[c] = np.searchsorted(cum_w[c], rands[c])
        choice_idx = np.clip(choice_idx, 0, self._n_valid - 1)

        # Build tokens
        if _UCE_USE_NUMBA:
            all_tokens = np.full((N, self.PAD_LENGTH), self.PAD_TOKEN_IDX, dtype=np.int64)
            all_masks = np.zeros((N, self.PAD_LENGTH), dtype=np.float32)
            _uce_build_tokens_numba(
                choice_idx, self._valid_pe, self._valid_chroms, self._valid_starts,
                self.CLS_TOKEN_IDX, self.CHROM_TOKEN_RIGHT_IDX,
                self.CHROM_TOKEN_OFFSET, self.PAD_LENGTH,
                all_tokens, all_masks,
            )
        else:
            all_tokens, all_masks = _uce_build_tokens_batch(
                choice_idx, self._valid_pe, self._valid_chroms, self._valid_starts,
                self.CLS_TOKEN_IDX, self.CHROM_TOKEN_RIGHT_IDX,
                self.CHROM_TOKEN_OFFSET, self.PAD_LENGTH,
            )

        max_len = max(2, int(all_masks.sum(axis=1).max()))
        all_tokens_t = torch.from_numpy(all_tokens[:, :max_len]).to(device)
        all_masks_t = torch.from_numpy(all_masks[:, :max_len]).to(device)

        with torch.no_grad():
            batch_sentences = self.model.pe_embedding(all_tokens_t)
            batch_sentences = torch.nn.functional.normalize(batch_sentences, dim=2)
            batch_sentences = batch_sentences.permute(1, 0, 2)
            _, embedding = self.model.forward(batch_sentences, mask=all_masks_t)

        return embedding

# ---------------------------------------------------------------------------
# 6. scFoundation
# ---------------------------------------------------------------------------
class SCFoundationEncoder(FrozenEncoder):
    """scFoundation encoder (Hao 2024, 100M params, dim 3072).

    Uses raw counts from data/raw/*.h5ad when available (preferred);
    falls back to log1p from preprocessed adata.X.

    Cell embedding: encoder hidden_dim=768, pool_type='all' concat of
    [last_pos, second_last_pos, max_pool, mean_pool] → 768×4 = 3072.
    """

    # Cell-line to raw h5ad file mapping
    _RAW_FILES = {
        "K562": "K562_essential_normalized_singlecell_01.h5ad",
        "RPE1": "rpe1_normalized_singlecell_01.h5ad",
        "HepG2": "GSE264667_hepg2_raw_singlecell_01.h5ad",
        "Jurkat": "GSE264667_jurkat_raw_singlecell_01.h5ad",
    }
    _CELL_LINE_ORDER = ["K562", "RPE1", "HepG2", "Jurkat"]

    def __init__(self, checkpoint_path: Optional[Path] = None):
        super().__init__()
        self._out_dim = 3072
        self.model = None
        self.config = None
        self.pad_token_id: int = 103
        self.mask_token_id: int = 102
        # Gene mapping: for each of our 6642 HVG positions, the index in
        # scFoundation's 19264 vocabulary (or -1 if not in vocab).
        self._hvg_sf_idx: Optional[torch.Tensor] = None   # [6642] long
        self._hvg_in_vocab: Optional[torch.Tensor] = None  # [6642] bool
        self._n_hvg_in_vocab: int = 0
        # Raw counts alignment (populated by load_raw_counts)
        self._raw_X = None          # scipy sparse [N, 6642] or None
        self._input_mode: str = "log1p"  # "raw" or "log1p"
        if checkpoint_path is not None:
            self.load_pretrained(checkpoint_path)

    # ------------------------------------------------------------------
    # load_pretrained
    # ------------------------------------------------------------------
    def load_pretrained(self, path: Path) -> None:
        import sys
        import pandas as pd

        path = Path(path)
        proj_root = Path(__file__).resolve().parents[2]
        sf_model_dir = str(proj_root / "baselines" / "scFoundation" / "model")
        if sf_model_dir not in sys.path:
            sys.path.insert(0, sf_model_dir)

        from load import load_model_frommmf

        # Load model (key='cell' for cell embedding version)
        self.model, self.config = load_model_frommmf(str(path), key="cell")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.pad_token_id = self.config["pad_token_id"]
        self.mask_token_id = self.config["mask_token_id"]

        # ---- gene mapping ----
        gene_list_df = pd.read_csv(
            proj_root / "baselines" / "scFoundation" / "OS_scRNA_gene_index.19264.tsv",
            sep="\t",
        )
        sf_gene_list = list(gene_list_df["gene_name"])  # 19264 gene symbols
        sf_sym2idx = {s: i for i, s in enumerate(sf_gene_list)}

        # Load HVG Ensembl IDs and map to gene symbols via preprocessed var
        import anndata as ad

        hvg_path = proj_root / "data" / "processed" / "hvg_genes.txt"
        with open(hvg_path) as f:
            hvg_ensembl = [line.strip() for line in f if line.strip()]

        pp = ad.read_h5ad(
            proj_root / "data" / "processed" / "replogle_preprocessed.h5ad",
            backed="r",
        )
        ensembl2sym = dict(zip(pp.var_names, pp.var["gene_name"]))
        pp.file.close()

        sf_indices, in_vocab = [], []
        for eid in hvg_ensembl:
            sym = ensembl2sym.get(eid)
            if sym and sym in sf_sym2idx:
                sf_indices.append(sf_sym2idx[sym])
                in_vocab.append(True)
            else:
                sf_indices.append(-1)
                in_vocab.append(False)

        self._hvg_sf_idx = torch.tensor(sf_indices, dtype=torch.long)
        self._hvg_in_vocab = torch.tensor(in_vocab, dtype=torch.bool)
        self._n_hvg_in_vocab = int(self._hvg_in_vocab.sum())

        print(
            f"   SCFoundationEncoder: {self._n_hvg_in_vocab}/{len(hvg_ensembl)} HVG "
            f"in scFoundation vocab ({self._n_hvg_in_vocab/len(hvg_ensembl)*100:.1f}%), "
            f"hidden={self._out_dim}, pad_token={self.pad_token_id}"
        )

    # ------------------------------------------------------------------
    # load_raw_counts  (call after load_pretrained, before encode loop)
    # ------------------------------------------------------------------
    def load_raw_counts(self, adata, raw_dir: Path) -> None:
        """Load raw counts from data/raw/*.h5ad and align to preprocessed cells.

        After calling this, encode() will use raw counts internally and apply
        scFoundation's own normalisation (log1p per 10k).  If this method is
        not called, encode() falls back to the log1p-normalised input.
        """
        import anndata as ad
        import numpy as np
        from scipy import sparse

        proj_root = Path(__file__).resolve().parents[2]
        if raw_dir is None:
            raw_dir = proj_root / "data" / "raw"
        raw_dir = Path(raw_dir)

        hvg_path = proj_root / "data" / "processed" / "hvg_genes.txt"
        with open(hvg_path) as f:
            hvg_ensembl = [line.strip() for line in f if line.strip()]
        hvg_order = {eid: i for i, eid in enumerate(hvg_ensembl)}
        n_hvg = len(hvg_ensembl)

        # Load preprocessed cell-line ordering
        pp_cell_lines = adata.obs["cell_line"].values
        n_total = adata.n_obs

        raw_blocks = []
        for cl in self._CELL_LINE_ORDER:
            fname = self._RAW_FILES[cl]
            fpath = raw_dir / fname
            if not fpath.exists():
                print(f"   [SCFoundation] WARNING: raw file not found: {fpath}")
                self._raw_X = None
                return
            raw_ad = ad.read_h5ad(str(fpath), backed="r")
            n_cells = int((pp_cell_lines == cl).sum())
            if raw_ad.n_obs != n_cells:
                print(
                    f"   [SCFoundation] WARNING: {cl} cell count mismatch "
                    f"(raw={raw_ad.n_obs}, preprocessed={n_cells}). "
                    f"Falling back to log1p."
                )
                raw_ad.file.close()
                self._raw_X = None
                return

            # Map raw var_names (Ensembl IDs) to HVG positions
            raw_varnames = list(raw_ad.var_names)
            raw_to_hvg = {}  # raw col idx -> hvg col idx
            for ri, eid in enumerate(raw_varnames):
                if eid in hvg_order:
                    raw_to_hvg[ri] = hvg_order[eid]
            raw_ad.file.close()

            # Read full raw matrix
            raw_ad = ad.read_h5ad(str(fpath))
            X_raw = raw_ad.X
            if not hasattr(X_raw, "toarray"):
                X_raw = sparse.csr_matrix(X_raw)
            del raw_ad

            # Build column mapping: subset HVG columns and reorder
            # Create a sparse column selector matrix [n_raw_vars, n_hvg]
            # that maps raw columns -> HVG positions
            n_raw_vars = X_raw.shape[1]
            selector_rows = []
            selector_cols = []
            for raw_idx, hvg_idx in raw_to_hvg.items():
                selector_rows.append(raw_idx)
                selector_cols.append(hvg_idx)
            selector = sparse.csr_matrix(
                (np.ones(len(selector_rows), dtype=np.float32),
                 (selector_rows, selector_cols)),
                shape=(n_raw_vars, n_hvg),
            )
            X_hvg = X_raw @ selector  # [n_cells, n_hvg]
            del X_raw, selector

            raw_blocks.append(sparse.csr_matrix(X_hvg, dtype=np.float32))
            del X_hvg

        self._raw_X = sparse.vstack(raw_blocks, format="csr")
        assert self._raw_X.shape == (n_total, n_hvg), (
            f"Raw X shape mismatch: {self._raw_X.shape} vs ({n_total}, {n_hvg})"
        )
        self._input_mode = "raw"
        print(
            f"   [SCFoundation] Loaded raw counts: {self._raw_X.shape}, "
            f"nnz={self._raw_X.nnz}, density="
            f"{self._raw_X.nnz / (self._raw_X.shape[0]*self._raw_X.shape[1]):.4f}"
        )

    # ------------------------------------------------------------------
    # encode
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode(self, x: torch.Tensor, start_idx: int = 0) -> torch.Tensor:
        """x: [B, 6642] expression → [B, 3072] cell embeddings.

        If load_raw_counts() was called, raw counts for rows
        [start_idx : start_idx+B] are used instead of x.

        NOTE: scFoundation's transformer has O(N^2) attention where N is the
        number of non-zero genes per cell (~8000-9000 for Perturb-seq data).
        Batched processing causes huge memory spikes, so we process one cell
        at a time (matching the original get_embedding.py).
        """
        import numpy as np
        from load import gatherData

        device = x.device
        B = x.shape[0]
        G = x.shape[1]  # 6642

        hvg_sf_idx = self._hvg_sf_idx.to(device)    # [6642]
        hvg_in_vocab = self._hvg_in_vocab.to(device)  # [6642]
        in_vocab_sf = hvg_sf_idx[hvg_in_vocab]       # indices in sf vocab

        # Pre-compute per-cell input expression (19266-length)
        if self._raw_X is not None:
            end_idx = start_idx + B
            raw_batch = self._raw_X[start_idx:end_idx]
            if hasattr(raw_batch, "toarray"):
                raw_batch = raw_batch.toarray()
            raw_batch = np.asarray(raw_batch, dtype=np.float32)
            x_raw = torch.as_tensor(raw_batch, dtype=torch.float32, device=device)

            total_counts = x_raw.sum(dim=1, keepdim=True).clamp(min=1.0)
            x_norm = torch.log1p(x_raw / total_counts * 1e4)
            s_token = torch.log10(total_counts.squeeze(1).clamp(min=1.0))
            cell_expr = x_norm  # [B, 6642] normalised expression
        else:
            cell_expr = x.float()
            total_counts = cell_expr.sum(dim=1, keepdim=True).clamp(min=1e-10)
            s_token = torch.log10(total_counts.squeeze(1).clamp(min=1e-10))

        # Process one cell at a time to avoid O(N^2) attention OOM.
        # scFoundation's ~8800 non-zero genes/cell makes batched attention
        # prohibitively expensive (8800^2 * heads * B), so we loop per-cell
        # following the original get_embedding.py design.
        use_amp = device.type == "cuda"
        n_nonzero_log_interval = 200  # log summary every N cells

        results = []
        for i in range(B):
            # Build 19266-length vector for this cell
            sf_gene_x = torch.zeros(1, 19266, dtype=torch.float32, device=device)
            sf_gene_x[0, in_vocab_sf] = cell_expr[i, hvg_in_vocab]
            sf_gene_x[0, 19264] = 4.0    # T token (tgthighres='t4')
            sf_gene_x[0, 19265] = s_token[i]  # S token

            # Gather non-zero tokens
            value_labels = sf_gene_x > 0
            x_gathered, x_padding = gatherData(
                sf_gene_x, value_labels, self.pad_token_id
            )

            data_gene_ids = torch.arange(
                19266, device=device
            ).unsqueeze(0)
            position_gene_ids, _ = gatherData(
                data_gene_ids, value_labels, self.pad_token_id
            )

            # Forward through encoder with fp16 autocast for speed
            with torch.amp.autocast("cuda", enabled=use_amp):
                x_emb = self.model.token_emb(
                    torch.unsqueeze(x_gathered, 2).float(), output_weight=0
                )
                position_emb = self.model.pos_emb(position_gene_ids)
                x_emb = x_emb + position_emb
                geneemb = self.model.encoder(x_emb, x_padding)

            # Pool: concat(last, second_last, max, mean) → 3072
            geneemb = geneemb.float()  # back to fp32 for pooling
            geneemb1 = geneemb[:, -1, :]
            geneemb2 = geneemb[:, -2, :]
            geneemb3, _ = torch.max(geneemb[:, :-2, :], dim=1)
            geneemb4 = torch.mean(geneemb[:, :-2, :], dim=1)
            cell_emb = torch.cat(
                [geneemb1, geneemb2, geneemb3, geneemb4], dim=1
            )
            results.append(cell_emb)

        return torch.cat(results, dim=0).float()  # [B, 3072]

    @property
    def output_dim(self) -> int:
        return self._out_dim


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
ENCODER_REGISTRY: Dict[str, Type[FrozenEncoder]] = {
    "pca":           PCANullEncoder,
    "state_se":      STATESEEncoder,
    "geneformer":    GeneformerEncoder,
    "scgpt":         SCGPTEncoder,
    "uce":           UCEEncoder,
    "scfoundation":  SCFoundationEncoder,
}


def build_encoder(name: str, **kwargs) -> FrozenEncoder:
    """根据 name 构造 encoder 实例.

    中文: 名字必须在 ENCODER_REGISTRY 里; 其它 kwargs 传给对应类的 __init__.
    """
    if name not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder '{name}'. Available: {list(ENCODER_REGISTRY)}"
        )
    enc = ENCODER_REGISTRY[name](**kwargs)
    enc.freeze()
    return enc
