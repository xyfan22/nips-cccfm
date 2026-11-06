# Data contract

Preparation commands and official inputs are listed in
[data/README.md](../data/README.md). This document describes the actual
Replogle input consumed by the released probe scripts.

## Expression matrix

`data/processed/replogle_preprocessed.h5ad` contains normalized, log1p
expression as float32. The frozen snapshot has 966,728 cells and 6,642
features, ordered as `data/processed/hvg_genes.txt`. Loaders accept dense
arrays or scipy sparse matrices; the fingerprinted snapshot is dense.

Required observation columns are `cell_line` (K562, RPE1, Jurkat, HepG2)
and `pert` (gene symbol or `control`). Train/validation membership is
constructed by each runner; `obs.split` and `obs.n_cells` are not required.

Feature identifiers are Ensembl IDs in `var_names` / `var.gene_id`;
`var.gene_name` provides gene symbols for the perturbation-token mapping.
The preprocessing script standardizes these fields. `var.is_hvg` and
`var.gene_symbol` are not loader requirements.

## Frozen context

`scripts/10_featurize_encoder.py` writes per-cell context to
`obsm["h_<encoder>"]`. A selected key must be present before training;
encoder weights are not loaded by the flow training loop. For the PCA
context, `h_pca` has 128 columns. Other context dimensions depend on the
selected encoder checkpoint and extraction wrapper.

## Perturbation embedding

`data/processed/primary_gene_embedding.pt` is a `torch.save` dictionary
mapping gene symbols to float tensors. The historical file has 6,641
keys, each with 128 values. It is built by
`scripts/02_build_gene_embedding.py` from standardized gene-expression
patterns across K562/RPE1/HepG2 control cells using PCA. Duplicate gene
symbols are collapsed by the builder. The tokenizer has an UNK row for
missing symbols.

An optional secondary embedding uses the same dictionary format; it is
needed only when explicitly selected by a perturbation-token mode.

## Splits and fingerprints

`data/splits/jurkat_test_strict.json` contains the fixed 577-perturbation
evaluation set. The K562/RPE1 cross-target lists and shared-perturbation
list are also distributed. Training uses source-line perturbation cells;
the target's perturbation cells are excluded from fitting.

`provenance/protocols.json` records the split, ordered-feature, primary
embedding, and core expression fingerprints. Fresh numerical
preprocessing or PCA fits may differ across dependency versions; the CPU
reproduction path uses the released frozen metric rows and descriptors.

## PBMC

The PBMC pipeline has a separate preparation contract enforced by
`scripts/rebuttal/50_prepare_pbmc_cellflow_balanced.py`: 12 donors, six
shared cell types, 88 retained cytokines plus PBS, and 50 cells per group.
The official processed release supplies its 2,000-gene feature space and
cytokine embeddings. Its values are normalized/log-transformed by the
preparation runner, not by the Replogle preprocessing script.
