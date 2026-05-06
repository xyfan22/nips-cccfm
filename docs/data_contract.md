# Data Contract

This document specifies the structure required by the ccCFM training and
evaluation pipeline. All upstream preprocessing (scanpy, normalization,
HVG selection, shared-perturbation filtering) must produce data conforming
to this schema before `scripts/featurize_encoders.py` can run.

## File layout

```
data/
├── raw/                        # 原始 Replogle Perturb-seq h5ad (per cell line)
│   ├── K562_essential_normalized_singlecell_01.h5ad
│   ├── rpe1_normalized_singlecell_01.h5ad
│   ├── GSE264667_jurkat_raw_singlecell_01.h5ad
│   └── GSE264667_hepg2_raw_singlecell_01.h5ad
├── processed/                  # 下游训练用的 h5ad + embeddings
│   ├── replogle_preprocessed.h5ad
│   ├── primary_gene_embedding.pt
│   └── secondary_gene_embedding.pt   # optional
└── splits/
    ├── jurkat_test_strict.json
    └── shared_perturbations.json
```

## `processed/replogle_preprocessed.h5ad`

The single source of truth for training and evaluation.

### `X`

- dtype: `float32` (sparse `scipy.sparse.csr_matrix` or dense).
- content: normalized + log1p expression over the shared HVG set
  (default 6,836 genes).

### `obs` (required columns)

| Column | Type | Example | Meaning |
|---|---|---|---|
| `cell_line` | str | `K562` | Source line identifier. |
| `pert` | str | `GATA1` or `control` | Perturbation gene symbol; `control` for unperturbed. |
| `split` | str | `train` \| `val` \| `test` | Dataset split. |
| `n_cells` | int | `73` | Cells recorded for this perturbation within the line (used by filters). |

### `var` (required columns)

| Column | Type | Meaning |
|---|---|---|
| `gene_symbol` | str | HVG identifier matching `obs.pert` values. |
| `is_hvg` | bool | `True` for the selected HVG set. |

### `obsm` (added by `scripts/featurize_encoders.py`)

Each frozen context encoder caches its per-cell embedding under a key:

| Key | Shape | Encoder |
|---|---|---|
| `h_state_se` | `[N, 2048]` | STATE single-cell embedding (Arc 2025). |
| `h_geneformer` | `[N, 512]` | Geneformer (Theodoris 2023). |
| `h_scgpt` | `[N, 512]` | scGPT (Cui 2024). |
| `h_uce` | `[N, 1280]` | UCE (Rosen 2024). |
| `h_scfoundation` | `[N, 3072]` | scFoundation (Hao 2024). |
| `h_pca` | `[N, 128]` | PCA null encoder. |

Dimensions are the expected defaults; each encoder wrapper reports its
actual `output_dim` via `cccfm.encoders.build_encoder(...).output_dim`.

## Gene embedding files

### `processed/primary_gene_embedding.pt`

- Format: `torch.save(dict[gene_symbol -> torch.Tensor])`
- Tensor shape: `[D_embed]` (typically 512 or 1024).
- Coverage: at least every gene in `splits/shared_perturbations.json`.
- Origin: any consistent gene embedding that encodes gene-function signal
  (e.g. a GO-feature-based embedding). The specific choice is a
  benchmark-level decision and does not affect the schema.

### `processed/secondary_gene_embedding.pt` (optional)

- Same format.
- Required only when `pert_token.mode = concat_secondary` is selected.
- Used for the secondary-gene-embedding ablation (e.g. GeneCompass-style).

## Split files

### `splits/jurkat_test_strict.json`

- List of perturbation symbols in the strict Jurkat test slice.
- Definition: see `docs/problem_setup.md` (Strict Jurkat test slice).

### `splits/shared_perturbations.json`

- List of perturbation symbols present in all four cell lines.

## Invariants the loader enforces

- All `pert` values referenced in split files either appear in
  `var.gene_symbol` or equal `control`.
- Every `(cell_line, pert)` pair with `n_cells >= 50` appears in exactly
  one of `train` / `val` / `test`.
- The `obsm` key required by the active encoder configuration must be
  populated before training starts. The loader raises an explicit error
  otherwise.
