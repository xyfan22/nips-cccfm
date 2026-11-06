# Data acquisition and preparation

The biological inputs are public upstream datasets. No single-cell expression
matrix is included. CPU table reproduction uses released metric CSVs and
cached source descriptors.

## Official inputs

| ID | Filename | Official download | Input state |
|---|---|---|---|
| K562 | `K562_essential_normalized_singlecell_01.h5ad` | [Figshare 35773219](https://plus.figshare.com/ndownloader/files/35773219) | Already log1p-normalized |
| RPE1 | `rpe1_normalized_singlecell_01.h5ad` | [Figshare 35775606](https://plus.figshare.com/ndownloader/files/35775606) | Already log1p-normalized |
| Jurkat | `GSE264667_jurkat_raw_singlecell_01.h5ad` | [GEO Jurkat file](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE264667&format=file&file=GSE264667%5Fjurkat%5Fraw%5Fsinglecell%5F01%2Eh5ad) | Raw counts |
| HepG2 | `GSE264667_hepg2_raw_singlecell_01.h5ad` | [GEO HepG2 file](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE264667&format=file&file=GSE264667%5Fhepg2%5Fraw%5Fsinglecell%5F01%2Eh5ad) | Raw counts |
| PBMC | `adata_for_cellflow_datasets_with_embeddings.h5ad` | [Figshare 53372768](https://ndownloader.figshare.com/files/53372768) | Public 2,000-gene derivative containing raw counts and cytokine embeddings |

Source records: [Replogle](https://doi.org/10.25452/figshare.plus.20022944),
[GSE264667](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667),
[Nadig et al. (HepG2/Jurkat)](https://doi.org/10.1038/s41588-025-02169-3),
[processed Parse data](https://figshare.com/articles/dataset/pbmc_parse/28589774?file=53372768),
and [Human Cytokine Dictionary](https://doi.org/10.64898/2025.12.12.693897).
Original provider terms apply; the repository software license does not replace them.

## Download and verify

`sources.json` records byte sizes and MD5 hashes of the files used.
The downloader resumes partial files and verifies completed/existing files.

```bash
python scripts/download_data.py --list
python scripts/download_data.py --dataset K562 RPE1 Jurkat HepG2
python scripts/download_data.py --dataset PBMC
```

Files go to `data/raw/`. If a provider rejects automated access, download
the named file through its official record and place it there; the same command
verifies an existing file. A different processed gene space is not interchangeable.

## Frozen Replogle definitions

The release includes the ordered 6,642 Ensembl IDs in
`data/processed/hvg_genes.txt`, the 577-perturbation Jurkat split in
`data/splits/jurkat_test_strict.json`, and the cross-target/shared lists.

The primary perturbation embedding used by the scripts is **128-dimensional
PCA of gene-expression patterns across K562/RPE1/HepG2 control cells**.
It is frozen and projected to a 512-dimensional perturbation token. This is
not a supplied Gene Ontology embedding. The recorded matrix has 6,641 gene-symbol
keys; duplicate symbols keep their first feature occurrence. Construction
statistics are in `data/processed/primary_gene_embedding_stats.csv`.

## Rebuild expression and PCA features

Use the PyTorch environment from the experiment guide:

```bash
python scripts/00_preprocess_data.py --raw_dir data/raw \
  --hvg_gene_list data/processed/hvg_genes.txt \
  --output data/processed/replogle_preprocessed.h5ad \
  --stats_csv data/processed/preprocess_stats.csv
python scripts/02_build_gene_embedding.py \
  --data_path data/processed/replogle_preprocessed.h5ad \
  --sources K562 RPE1 HepG2 --n_components 128 --seed 42 \
  --output data/processed/primary_gene_embedding.pt
python scripts/10_featurize_encoder.py --encoder pca \
  --data_path data/processed/replogle_preprocessed.h5ad \
  --fit_sources K562 RPE1 HepG2 --n_components 128
```

Raw HepG2/Jurkat inputs receive total-count normalization to 10,000 and log1p;
already normalized K562/RPE1 inputs are retained. The fixed feature list
bypasses fresh HVG selection. The PCA command appends `obsm["h_pca"]`
in place and fits controls only. Use the shipped splits for the reported runs.

`provenance/protocols.json` records the historical embedding-file hash
and the expression-core fingerprint used for stratification. A serialization
hash identifies the historical file; regenerated PCA/serialization can depend
on library versions. Preserve input order, versions, and seeds.

## Foundation encoder weights

Weights are not redistributed. Adapters and upstream references are in
`code/cccfm/encoders.py`, `baselines/`, and `download_list.md`.
Check checkpoint-specific upstream terms: code and weight licenses can differ.
