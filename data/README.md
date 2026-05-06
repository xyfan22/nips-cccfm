# Data — fetch and rebuild instructions

The artifact ships **no raw or processed single-cell expression files**.
Reviewers can complete every documented smoke-test step (`docs/reproducibility.md`)
without downloading any h5ad: every numeric cell in the paper traces to a
`_summary.csv` already present under `results/`.

The instructions below are for reviewers who additionally want to (a) verify
the 577-perturbation × 6{,}642-HVG slice from scratch, or (b) re-run an
encoder featurization end-to-end.

## 1. Source dataset

We use the four-cell-line Replogle CRISPRi Perturb-seq corpus
(K562, RPE1, HepG2, Jurkat). The raw counts are publicly available from
the Replogle 2022 release; please consult the original paper for the
authoritative download instructions:

> Replogle, J. M., Saunders, R. A., Pogson, A. N., et al.
> *Mapping information-rich genotype–phenotype landscapes with genome-scale
> Perturb-seq.* Cell, 2022.

Download into `data/raw/` so the layout mirrors the paths used by the
preprocessing script.

## 2. Preprocessing → `data/processed/replogle_pca.h5ad`

```bash
python scripts/00_preprocess_data.py
python scripts/01_build_splits.py
```

These two scripts together produce:

- `data/processed/replogle_pca.h5ad` — log-normalized counts on the shared
  6{,}642-HVG common gene space, with the 577-perturbation slice flagged.
- `data/splits/holdouts.json` — the 2-line and 3-line holdout definitions
  used by every downstream evaluator call.

The slice construction is described in §3.3 of the paper. Briefly:
(i) per-line drop perturbations whose target gene is absent from that
line's expression matrix or that have fewer than 50 cells in that line,
then (ii) intersect across all four lines (-> 577 perturbations);
(iii) union of per-line top-2{,}000 HVGs intersected with genes present in
every line (-> 6{,}642 HVGs).

## 3. Encoder featurization → `data/processed/replogle_<encoder>.h5ad`

```bash
python scripts/10_featurize_encoder.py --encoder pca
python scripts/10_featurize_encoder.py --encoder geneformer
python scripts/10_featurize_encoder.py --encoder scgpt
python scripts/10_featurize_encoder.py --encoder uce
python scripts/10_featurize_encoder.py --encoder scfoundation
python scripts/10_featurize_encoder.py --encoder state_se
```

Each run writes a per-cell context vector into `adata.obsm["h_<encoder>"]`
and saves to a dedicated h5ad. The PCA branch fits an `IncrementalPCA(D=128)`
on source-line controls and is fully self-contained.

The five scFM branches dynamically `sys.path.insert` into the upstream code
that we vendored under `baselines/<encoder>/`. The vendored copies in this
artifact include only the upstream Python sources, READMEs, and small
configs; they DO NOT include the pretrained checkpoints (which are
distributed separately by each upstream and total tens of GB across all
five encoders). To run an scFM encoder end-to-end you must:

  1. Install that upstream's pip dependencies, listed in
     `baselines/<encoder>/requirements.txt` (or equivalent).
  2. Download the official pretrained checkpoint from the upstream URL
     below and place it where each `load_pretrained` constructor expects it
     (path conventions are documented in `code/cccfm/encoders.py`).
  3. (UCE only) Populate `baselines/UCE/model_files/` with the
     protein-embedding files distributed by the UCE upstream.

| encoder | upstream | output dim | license |
|---|---|---|---|
| Geneformer  | https://huggingface.co/ctheodoris/Geneformer | 768  | Apache-2.0 |
| scGPT       | https://github.com/bowang-lab/scGPT          | 512  | MIT |
| UCE         | https://github.com/snap-stanford/UCE         | 1280 | MIT |
| scFoundation| https://github.com/biomap-research/scFoundation | 3072 | Apache-2.0 |
| STATE-SE    | https://github.com/ArcInstitute/state        | 2048 | Apache-2.0 |

This step is **GPU-only and slow** for the scFM encoders. The artifact's
documented smoke-test (`docs/reproducibility.md`) does **not** require it:
every `_summary.csv` shipped under `results/` was already produced from
cached `obsm["h_<encoder>"]` vectors, and the smoke-test pipeline only
re-aggregates from those CSVs.

## 4. SHA-256 of the canonical featurized h5ads

> The exact SHA-256 hashes will be added here once the artifact is
> finalized for OpenReview submission.

| file | size | sha256 |
|---|---|---|
| `data/processed/replogle_pca.h5ad`         | -- | -- |
| `data/processed/replogle_geneformer.h5ad`  | -- | -- |
| `data/processed/replogle_scgpt.h5ad`       | -- | -- |
| `data/processed/replogle_uce.h5ad`         | -- | -- |
| `data/processed/replogle_scfoundation.h5ad`| -- | -- |
| `data/processed/replogle_state_se.h5ad`    | -- | -- |
