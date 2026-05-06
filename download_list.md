# Baseline weights / data download checklist

This document collects the publicly-distributed pretrained weights and
auxiliary data for the eleven baseline repositories vendored under
`baselines/`.

Conventions used throughout:
- Large checkpoints are placed under `baselines/<name>/pretrained/` by
  default, unless the upstream code hard-codes a different path.
- When upstream code hard-codes a path, the upstream layout is preserved
  rather than wrapping it.
- `n/a` indicates that the baseline must be trained from scratch, that no
  public checkpoint could be located, or that the precise filename / size
  is not exposed by the discoverable upstream sources.

### Geneformer
- **Type**: encoder
- **Requires weights**: yes
- **Primary weights**:
  - URL: https://huggingface.co/ctheodoris/Geneformer/tree/main/Geneformer-V2-104M
  - Size: ~418 MB checkpoint + ~2.4 MB of required V2 dictionaries
  - License: Apache-2.0
  - Location: place the checkpoint at `baselines/Geneformer/pretrained/Geneformer-V2-104M/`; place the required dictionaries at `baselines/Geneformer/geneformer/*.pkl`, because `geneformer/__init__.py` resolves them through package-relative paths.
  - Download command: `mkdir -p baselines/Geneformer/pretrained && huggingface-cli download ctheodoris/Geneformer --local-dir baselines/Geneformer --include "geneformer/gene_median_dictionary_gc104M.pkl" --include "geneformer/token_dictionary_gc104M.pkl" --include "geneformer/gene_name_id_dict_gc104M.pkl" --include "geneformer/ensembl_mapping_dict_gc104M.pkl" && huggingface-cli download ctheodoris/Geneformer --local-dir baselines/Geneformer/pretrained --include "Geneformer-V2-104M/*"`
- **Secondary weights (variants)**:
  - `Geneformer-V2-316M`
    - URL: https://huggingface.co/ctheodoris/Geneformer/tree/main/Geneformer-V2-316M
    - Size: ~1.27 GB
    - License: Apache-2.0
    - Location: `baselines/Geneformer/pretrained/Geneformer-V2-316M/`
    - Download command: `mkdir -p baselines/Geneformer/pretrained && huggingface-cli download ctheodoris/Geneformer --local-dir baselines/Geneformer/pretrained --include "Geneformer-V2-316M/*"`
  - `Geneformer-V1-10M`
    - URL: https://huggingface.co/ctheodoris/Geneformer/tree/main/Geneformer-V1-10M
    - Size: ~82.4 MB checkpoint + ~3.4 MB of required 30M dictionaries
    - License: Apache-2.0
    - Location: `baselines/Geneformer/pretrained/Geneformer-V1-10M/`, plus `baselines/Geneformer/geneformer/gene_dictionaries_30m/*.pkl`
    - Download command: `mkdir -p baselines/Geneformer/pretrained && huggingface-cli download ctheodoris/Geneformer --local-dir baselines/Geneformer --include "geneformer/gene_dictionaries_30m/gene_median_dictionary_gc30M.pkl" --include "geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl" --include "geneformer/gene_dictionaries_30m/gene_name_id_dict_gc30M.pkl" --include "geneformer/gene_dictionaries_30m/ensembl_mapping_dict_gc30M.pkl" && huggingface-cli download ctheodoris/Geneformer --local-dir baselines/Geneformer/pretrained --include "Geneformer-V1-10M/*"`
  - `Geneformer-V2-104M_CLcancer`
    - URL: https://huggingface.co/ctheodoris/Geneformer/tree/main/Geneformer-V2-104M_CLcancer
    - Size: ~418 MB
    - License: Apache-2.0
    - Location: `baselines/Geneformer/pretrained/Geneformer-V2-104M_CLcancer/`
    - Download command: `mkdir -p baselines/Geneformer/pretrained && huggingface-cli download ctheodoris/Geneformer --local-dir baselines/Geneformer/pretrained --include "Geneformer-V2-104M_CLcancer/*"`
- **Auxiliary data**:
  - Demo / fine-tuning example inputs at `ctheodoris/Genecorpus-30M/example_input_files`: https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/example_input_files
  - For the full pretraining corpus, the public HF datasets `ctheodoris/Genecorpus-30M` and `theodoris-lab/Genecorpus-104M` are available.
- **Access restrictions**: none
- **Information source**: `baselines/Geneformer/README.md L16-L34, L63-L86`; `baselines/Geneformer/geneformer/__init__.py L7-L15`; HF model tree `ctheodoris/Geneformer`; HF dataset trees `ctheodoris/Genecorpus-30M`, `theodoris-lab/Genecorpus-104M`
- **Open items**: the upstream examples default to `V2-316M`, but `V2-104M` is the more recent and lighter V2 checkpoint and is more practical as a benchmark default.

### scGPT
- **Type**: encoder
- **Requires weights**: yes
- **Primary weights**:
  - URL: https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y?usp=sharing
  - Size: n/a (Google Drive folder; the README does not list a total size)
  - License: MIT
  - Location: `baselines/scGPT/pretrained/whole-human/`, because the code reads the checkpoint as a directory containing `vocab.json`, `args.json`, and `best_model.pt`.
  - Download command: `mkdir -p baselines/scGPT/pretrained/whole-human && gdown --folder "https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y?usp=sharing" -O baselines/scGPT/pretrained/whole-human`
- **Secondary weights (variants)**:
  - `continual pretrained`
    - URL: https://drive.google.com/drive/folders/1_GROJTzXiAV8HB4imruOTk6PEGuNOcgB?usp=sharing
    - Size: n/a
    - License: MIT
    - Location: `baselines/scGPT/pretrained/continual-pretrained/`
    - Download command: `mkdir -p baselines/scGPT/pretrained/continual-pretrained && gdown --folder "https://drive.google.com/drive/folders/1_GROJTzXiAV8HB4imruOTk6PEGuNOcgB?usp=sharing" -O baselines/scGPT/pretrained/continual-pretrained`
  - `brain`, `blood`, `heart`, `lung`, `kidney`, `pan-cancer`
    - URL: see the model-zoo table in the upstream README
    - Size: n/a
    - License: MIT
    - Location: `baselines/scGPT/pretrained/<variant>/`
    - Download command: `gdown --folder "<drive-folder-url>" -O baselines/scGPT/pretrained/<variant>`
- **Auxiliary data**:
  - `gene_info.csv` ENSEMBL conversion table: https://github.com/bowang-lab/scGPT/files/13243634/gene_info.csv
  - Each checkpoint directory normally bundles its matching vocabulary file.
- **Access restrictions**: only Google Drive accessibility; no additional form or license gate observed.
- **Information source**: `baselines/scGPT/README.md L56-L73`; `baselines/scGPT/scgpt/tasks/cell_emb.py L199-L203`; `baselines/scGPT/tutorials/zero-shot/README.md L16-L18`
- **Open items**: the link to `continual pretrained` in the main README and the `scGPT_CP` link in the zero-shot README do not match exactly; the upstream link organization is inconsistent.

### UCE
- **Type**: encoder
- **Requires weights**: yes
- **Primary weights**:
  - URL: https://huggingface.co/minwoosun/uce-100m
  - Size: total snapshot size is not directly exposed by the page; the repo name confirms a 100M-parameter version.
  - License: the Hugging Face top banner shows `MIT`, but the model card body states `CC-BY-NC-ND 4.0`; the license description is inconsistent and the latest upstream documentation should be checked before use.
  - Location: place the raw snapshot at `baselines/UCE/hf/uce-100m/`; to remain compatible with the hard-coded default path in the current `baselines/UCE/evaluate.py`, the final model weights also need to be manually copied / renamed to `baselines/UCE/model_files/4layer_model.torch`.
  - Download command: `mkdir -p baselines/UCE/model_files baselines/UCE/hf/uce-100m && huggingface-cli download minwoosun/uce-misc --local-dir baselines/UCE/model_files && huggingface-cli download minwoosun/uce-100m --local-dir baselines/UCE/hf/uce-100m`
- **Secondary weights (variants)**:
  - `minwoosun/uce-650m`
    - URL: https://huggingface.co/minwoosun/uce-650m
    - Size: total snapshot size is not directly exposed by the page; the repo name confirms a 650M-parameter version (the 600M-class large model).
    - License: HF banner shows `MIT`, model card body states `CC-BY-NC-ND 4.0`; the same inconsistency as for `uce-100m`.
    - Location: place the raw snapshot at `baselines/UCE/hf/uce-650m/`; to remain compatible with the legacy code, copy / rename the actual checkpoint to `baselines/UCE/model_files/33layer_model.torch`, or pass `--model_loc` explicitly at runtime.
    - Download command: `mkdir -p baselines/UCE/hf/uce-650m && huggingface-cli download minwoosun/uce-650m --local-dir baselines/UCE/hf/uce-650m`
- **Auxiliary data**:
  - The current misc-resources repo is `minwoosun/uce-misc`: https://huggingface.co/minwoosun/uce-misc
  - Files verified visible include:
    - `all_tokens.torch`, ~2.98 GB
    - `species_chrom.csv`, ~4.1 MB
    - `protein_embeddings.tar.gz`, ~2.74 GB
    - `protein_embeddings/` directory, ~2.99 GB total
    - `10k_pbmcs_proc.h5ad`, ~85.6 MB
  - `species_offsets.pkl` is most likely also in `uce-misc`, since the local code requires it and the repo's initial commit history shows "Upload 3 files"; this is inferred from local code requirements and commit history rather than directly observed on the discovery page.
  - Suggested layout: `baselines/UCE/model_files/` and `baselines/UCE/data/`.
- **Access restrictions**: none
- **Information source**: `baselines/UCE/README.md L19-L45`; `baselines/UCE/evaluate.py L64-L88, L197-L212`; HF pages `minwoosun/uce-misc`, `minwoosun/uce-100m`, `minwoosun/uce-650m`
- **Open items**: the current canonical UCE distribution channel has migrated to the new HF repos, but the vendored local code still uses the legacy `model_files/*.torch` directory convention. The new repos are confirmed to exist, but the discovery pages do not directly expose the internal checkpoint filenames inside `uce-100m` / `uce-650m`. The most reliable workflow is therefore to do a full snapshot download and then either pass `--model_loc` explicitly or add a renaming step.

### scFoundation
- **Type**: encoder
- **Requires weights**: yes
- **Primary weights**:
  - URL: SharePoint folder pointed to by the upstream repo: https://hopebio2020.sharepoint.com/:f:/s/PublicSharedfiles/IgBlEJ72TBE5Q76AmgXbgjXiAR69fzcrgzqgUYdSThPLrqk
  - Size: n/a (the upstream repo does not publish exact filenames / sizes)
  - License: scFoundation License; non-commercial research only; the code is Apache-2.0.
  - Location: `baselines/scFoundation/model/models/models.ckpt`, because `get_embedding.py` hard-codes `./models/models.ckpt`.
  - Download command: `n/a; manual download from SharePoint to baselines/scFoundation/model/models/models.ckpt`.
- **Secondary weights (variants)**:
  - n/a
- **Auxiliary data**:
  - Inference example data: Figshare bundle, https://doi.org/10.6084/m9.figshare.24049200, unpack into `baselines/scFoundation/model/examples/`.
  - GEARS-experiment `h5ad` files: same Figshare collection.
  - The integrated GEARS workflow additionally requires `go.csv.zip`: https://www.dropbox.com/s/wl0uxiz5z9dbliv/go.csv.zip?dl=0 -> `baselines/scFoundation/GEARS/data/<dataset>/go.csv`
  - The hosted-inference API is an alternative path: https://aigp.biomap.com/; the upstream documentation also references the older apply page https://api.biomap.com/xTrimoGene/apply.
- **Access restrictions**: model weights require a manual SharePoint download; the hosted API requires a token / platform access; the model license is non-commercial.
- **Information source**: `baselines/scFoundation/README.md L12-L13, L46-L69`; `baselines/scFoundation/model/README.md L34-L41`; `baselines/scFoundation/model/models/download.txt L1`; `baselines/scFoundation/model/get_embedding.py L111-L116`; `baselines/scFoundation/GEARS/README.md L1-L6, L24-L42`; the BioMap new-platform announcement.
- **Open items**: the upstream repo does not publish a directly-downloadable checkpoint filename, only a SharePoint folder. Whether the API request entry and local-weight download share the same access control is not documented.

### state
- **Type**: encoder + foundation perturbation model
- **Requires weights**: yes
- **Primary weights**:
  - URL: https://huggingface.co/arcinstitute/ST-Tahoe/tree/main
  - Size: full repo snapshot ~6.09 GB; the primary inference checkpoint `final.ckpt` is ~3.01 GB.
  - License: Arc Research Institute State Model Non-Commercial License; code under CC BY-NC-SA 4.0; ⚠️ non-commercial.
  - Location: `baselines/state/pretrained/ST-Tahoe/`, because `state tx infer` expects a model directory containing both the checkpoint and the config / map files.
  - Download command: `mkdir -p baselines/state/pretrained/ST-Tahoe && huggingface-cli download arcinstitute/ST-Tahoe --local-dir baselines/state/pretrained/ST-Tahoe`
- **Secondary weights (variants)**:
  - `SE-600M` embedding model
    - URL: https://huggingface.co/arcinstitute/SE-600M
    - Size: `se600m_epoch15.ckpt` ~11.5 GB, `protein_embeddings.pt` ~411 MB, plus a few small config files.
    - License: Arc Research Institute State Model Non-Commercial License; ⚠️ non-commercial.
    - Location: `baselines/state/pretrained/SE-600M/`
    - Download command: `mkdir -p baselines/state/pretrained/SE-600M && huggingface-cli download arcinstitute/SE-600M --local-dir baselines/state/pretrained/SE-600M --include "config.yaml" --include "protein_embeddings.pt" --include "se600m_epoch15.ckpt"`
  - `ST-SE-Replogle`
    - URL: https://huggingface.co/arcinstitute/ST-SE-Replogle/tree/main
    - Size: full repo ~95.1 GB.
    - License: Arc Research Institute State Model Non-Commercial License; ⚠️ non-commercial.
    - Location: `baselines/state/pretrained/ST-SE-Replogle/`
    - Download command: `mkdir -p baselines/state/pretrained/ST-SE-Replogle && huggingface-cli download arcinstitute/ST-SE-Replogle --local-dir baselines/state/pretrained/ST-SE-Replogle`
- **Auxiliary data**:
  - The `ST-Tahoe` and `SE-600M` repos already bundle the required config / map / embedding support files; no additional repo-local downloads are observed to be required.
- **Access restrictions**: no transport-layer gate, but the model license is non-commercial.
- **Information source**: `baselines/state/README.md L12-L17, L153-L218, L286-L288`; `baselines/state/LICENSE`; `baselines/state/MODEL_LICENSE.md`; HF model cards / trees `arcinstitute/ST-Tahoe`, `arcinstitute/SE-600M`, `arcinstitute/ST-SE-Replogle`.
- **Open items**: for cross-cell-line perturbation prediction, `ST-Tahoe` is the easiest starting point, but if storage permits, `ST-SE-Replogle` may be a better fit for CRISPR-screen scenarios.

### stack
- **Type**: foundation perturbation model
- **Requires weights**: yes
- **Primary weights**:
  - URL: https://huggingface.co/arcinstitute/Stack-Large-Aligned
  - Size: n/a (the discoverable model card explains the snapshot-download pattern but does not directly expose the aligned-weight tree size).
  - License: Arc Research Institute Stack Model Non-Commercial License; code under CC BY-NC-SA 4.0; ⚠️ non-commercial.
  - Location: `baselines/stack/pretrained/Stack-Large-Aligned/`
  - Download command: `mkdir -p baselines/stack/pretrained/Stack-Large-Aligned && huggingface-cli download arcinstitute/Stack-Large-Aligned --local-dir baselines/stack/pretrained/Stack-Large-Aligned`
- **Secondary weights (variants)**:
  - `Stack-Large`
    - URL: https://huggingface.co/arcinstitute/Stack-Large/tree/main
    - Size: `bc_large.ckpt` ~2.61 GB, `basecount_1000per_15000max.pkl` ~925 kB.
    - License: Arc Research Institute Stack Model Non-Commercial License; ⚠️ non-commercial.
    - Location: `baselines/stack/pretrained/Stack-Large/`
    - Download command: `mkdir -p baselines/stack/pretrained/Stack-Large && huggingface-cli download arcinstitute/Stack-Large --local-dir baselines/stack/pretrained/Stack-Large`
- **Auxiliary data**:
  - The model snapshot already bundles support files such as `basecount_1000per_15000max.pkl`; no additional repo-local downloads are observed to be required.
  - `Perturb-Sapiens` is a prediction *dataset* generated with `Stack-Large-Aligned`, not a weight bundle: https://huggingface.co/datasets/arcinstitute/Perturb-Sapiens
- **Access restrictions**: no transport-layer gate, but the model license is non-commercial.
- **Information source**: `baselines/stack/README.md L31-L73, L171-L174`; `baselines/stack/MODEL_LICENSE.md`; HF model cards `arcinstitute/Stack-Large`, `arcinstitute/Stack-Large-Aligned`.
- **Open items**: `Stack-Large-Aligned` matches the prediction workflow most closely, but the discoverable pages do not directly expose the aligned-checkpoint filename, so a full snapshot download remains the most reliable command.

### CellFlow
- **Type**: foundation perturbation model
- **Requires weights**: no
- **Primary weights**:
  - URL: n/a
  - Size: n/a
  - License: n/a
  - Location: n/a
  - Download command: n/a
- **Secondary weights (variants)**:
  - n/a
- **Auxiliary data**:
  - Bundled tutorial data from HF dataset `theislab/cellflow-datasets`:
    - `zesta.h5ad` -> `baselines/CellFlow/data/zesta.h5ad`
    - `ineurons.h5ad` -> `baselines/CellFlow/data/ineurons.h5ad`
    - `pbmc_parse.h5ad` -> `baselines/CellFlow/data/pbmc_parse.h5ad`
  - Download command: `mkdir -p baselines/CellFlow/data && huggingface-cli download theislab/cellflow-datasets --repo-type dataset --local-dir baselines/CellFlow/data --include "zesta.h5ad" --include "ineurons.h5ad" --include "pbmc_parse.h5ad"`
- **Access restrictions**: none
- **Information source**: `baselines/CellFlow/README.md L10-L24`; `baselines/CellFlow/src/cellflow/datasets.py L11-L110`.
- **Open items**: no official pretrained CellFlow checkpoint was located in the upstream repo or documentation; this baseline is therefore best treated as a from-scratch training framework + tutorial datasets.

### Perturbgen
- **Type**: foundation perturbation model
- **Requires weights**: yes
- **Primary weights**:
  - URL: https://huggingface.co/lotfollahi-lab/PerturbGen/tree/main
  - Size: ~574 MB
  - License: MIT (code-repo license; no separate weights license observed in the discoverable sources).
  - Location: `baselines/Perturbgen/pretrained/`
  - Download command: `mkdir -p baselines/Perturbgen/pretrained && huggingface-cli download lotfollahi-lab/PerturbGen --local-dir baselines/Perturbgen/pretrained`
- **Secondary weights (variants)**:
  - n/a (the README does not link further public checkpoints).
- **Auxiliary data**:
  - The `token_id_to_genename_*.pkl`, `tokenid_to_rowid_*.pkl`, source / target `.dataset` files, and the matching `.h5ad` directories that appear in configs and notebooks look like user-side preprocessing outputs, not directly-downloadable public data bundles.
- **Access restrictions**: none
- **Information source**: `baselines/Perturbgen/README.md L67-L74`; HF model tree `lotfollahi-lab/PerturbGen`.
- **Open items**: the public HF repo currently exposes only a masking-model checkpoint; the count-model checkpoint referenced by configs and batch scripts points to internal cluster paths, with no public mirror observed.

### cpa
- **Type**: classical (trained from scratch)
- **Requires weights**: no
- **Primary weights**:
  - URL: n/a
  - Size: n/a
  - License: n/a
  - Location: n/a
  - Download command: n/a
- **Secondary weights (variants)**:
  - Optional bundled datasets / pretrained models
    - URL: https://drive.google.com/drive/folders/1yFB0gBr72_KLLp1asojxTgTqgz6cwpju?usp=drive_link
    - Size: n/a
    - License: code under BSD-3-Clause; the upstream README does not separately specify a license for this Drive bundle.
    - Location: `baselines/cpa/assets/`
    - Download command: `mkdir -p baselines/cpa/assets && gdown --folder "https://drive.google.com/drive/folders/1yFB0gBr72_KLLp1asojxTgTqgz6cwpju?usp=drive_link" -O baselines/cpa/assets`
- **Auxiliary data**:
  - The same Drive folder is the only dataset / pretrained-model entry referenced by the upstream repo.
- **Access restrictions**: only Google Drive accessibility; no additional repo-level gate observed.
- **Information source**: `baselines/cpa/README.md L57-L59`.
- **Open items**: the upstream README does not enumerate the contents of the Drive folder, so exact filenames and sizes remain unverified until the folder is opened directly.

### GEARS
- **Type**: classical (trained from scratch)
- **Requires weights**: no
- **Primary weights**:
  - URL: n/a
  - Size: n/a
  - License: n/a
  - Location: n/a
  - Download command: n/a
- **Secondary weights (variants)**:
  - n/a (the `load_pretrained('gears')` example in the README refers to a user-saved local checkpoint, not an upstream public download).
- **Auxiliary data**:
  - GO-graph support files:
    - `gene2go_all.pkl`: `https://dataverse.harvard.edu/api/access/datafile/6153417` -> `baselines/GEARS/data/gene2go_all.pkl`
    - `essential_all_data_pert_genes.pkl`: `https://dataverse.harvard.edu/api/access/datafile/6934320` -> `baselines/GEARS/data/essential_all_data_pert_genes.pkl`
  - Bundled benchmark datasets:
    - `norman`: `https://dataverse.harvard.edu/api/access/datafile/6154020`
    - `adamson`: `https://dataverse.harvard.edu/api/access/datafile/6154417`
    - `dixit`: `https://dataverse.harvard.edu/api/access/datafile/6154416`
    - `replogle_k562_essential`: `https://dataverse.harvard.edu/api/access/datafile/7458695`
    - `replogle_rpe1_essential`: `https://dataverse.harvard.edu/api/access/datafile/7458694`
  - Suggested download commands:
    - `mkdir -p baselines/GEARS/data && wget -O baselines/GEARS/data/gene2go_all.pkl https://dataverse.harvard.edu/api/access/datafile/6153417`
    - `mkdir -p baselines/GEARS/data && wget -O baselines/GEARS/data/essential_all_data_pert_genes.pkl https://dataverse.harvard.edu/api/access/datafile/6934320`
    - `mkdir -p baselines/GEARS/data/norman && wget -O /tmp/gears_norman.zip https://dataverse.harvard.edu/api/access/datafile/6154020 && unzip -o /tmp/gears_norman.zip -d baselines/GEARS/data/norman`
    - `mkdir -p baselines/GEARS/data/adamson && wget -O /tmp/gears_adamson.zip https://dataverse.harvard.edu/api/access/datafile/6154417 && unzip -o /tmp/gears_adamson.zip -d baselines/GEARS/data/adamson`
    - `mkdir -p baselines/GEARS/data/dixit && wget -O /tmp/gears_dixit.zip https://dataverse.harvard.edu/api/access/datafile/6154416 && unzip -o /tmp/gears_dixit.zip -d baselines/GEARS/data/dixit`
    - `mkdir -p baselines/GEARS/data/replogle_k562_essential && wget -O /tmp/gears_replogle_k562.zip https://dataverse.harvard.edu/api/access/datafile/7458695 && unzip -o /tmp/gears_replogle_k562.zip -d baselines/GEARS/data/replogle_k562_essential`
    - `mkdir -p baselines/GEARS/data/replogle_rpe1_essential && wget -O /tmp/gears_replogle_rpe1.zip https://dataverse.harvard.edu/api/access/datafile/7458694 && unzip -o /tmp/gears_replogle_rpe1.zip -d baselines/GEARS/data/replogle_rpe1_essential`
- **Access restrictions**: none
- **Information source**: `baselines/GEARS/README.md L28-L49`; `baselines/GEARS/gears/pertdata.py L90-L167`.
- **Open items**: no official downloadable pretrained GEARS checkpoint was located in the upstream README, so this baseline is most likely trained per perturbation dataset to produce a local `model.pt`.

### cellot
- **Type**: classical (trained from scratch)
- **Requires weights**: no
- **Primary weights**:
  - URL: n/a
  - Size: n/a
  - License: n/a
  - Location: n/a
  - Download command: n/a
- **Secondary weights (variants)**:
  - n/a
- **Auxiliary data**:
  - Processed dataset package: https://www.research-collection.ethz.ch/handle/20.500.11850/609681
  - Size: ~398.87 MB (`All Processed Datasets (ZIP, 398.87 MB)`)
  - License: In Copyright - Non-Commercial Use Permitted
  - Location: `baselines/cellot/datasets/`
  - Download command: `n/a; the most reliable path is currently a manual download from the ETH Research Collection page into baselines/cellot/datasets/`
- **Access restrictions**: the page itself is public, but the data permission is non-commercial.
- **Information source**: `baselines/cellot/README.md L33-L40`; ETH Research Collection data page (DOI `10.3929/ethz-b-000609681`).
- **Open items**: the README provides only the ETH landing page, not a stable direct ZIP link; manual click-through download remains the most reliable route.

## Cluster download script

```bash
#!/usr/bin/env bash
set -eux

# Expected to be installed in the environment:
#   huggingface-cli
#   gdown
#   wget
#   unzip
#   tar

echo "[encoder] Geneformer"
mkdir -p baselines/Geneformer/pretrained
huggingface-cli download ctheodoris/Geneformer --local-dir baselines/Geneformer \
  --include "geneformer/gene_median_dictionary_gc104M.pkl" \
  --include "geneformer/token_dictionary_gc104M.pkl" \
  --include "geneformer/gene_name_id_dict_gc104M.pkl" \
  --include "geneformer/ensembl_mapping_dict_gc104M.pkl"
huggingface-cli download ctheodoris/Geneformer --local-dir baselines/Geneformer/pretrained \
  --include "Geneformer-V2-104M/*"
# Optional larger version:
# huggingface-cli download ctheodoris/Geneformer --local-dir baselines/Geneformer/pretrained \
#   --include "Geneformer-V2-316M/*"

echo "[encoder] scGPT"
mkdir -p baselines/scGPT/pretrained/whole-human
gdown --folder "https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y?usp=sharing" \
  -O baselines/scGPT/pretrained/whole-human
# Optional continual-pretrained version:
# mkdir -p baselines/scGPT/pretrained/continual-pretrained
# gdown --folder "https://drive.google.com/drive/folders/1_GROJTzXiAV8HB4imruOTk6PEGuNOcgB?usp=sharing" \
#   -O baselines/scGPT/pretrained/continual-pretrained

echo "[encoder] UCE"
mkdir -p baselines/UCE/model_files baselines/UCE/data baselines/UCE/hf/uce-100m
huggingface-cli download minwoosun/uce-misc --local-dir baselines/UCE/model_files
huggingface-cli download minwoosun/uce-100m --local-dir baselines/UCE/hf/uce-100m
echo "UCE note: the local code defaults to baselines/UCE/model_files/4layer_model.torch"
echo "If the 100M checkpoint is not manually copied/renamed there, pass --model_loc explicitly at runtime"
# Optional 650M version:
# mkdir -p baselines/UCE/hf/uce-650m
# huggingface-cli download minwoosun/uce-650m --local-dir baselines/UCE/hf/uce-650m
# echo "650M is analogous: if not renamed to model_files/33layer_model.torch, pass --model_loc explicitly at runtime"

echo "[encoder] scFoundation"
echo "Manual step: download the SharePoint model into baselines/scFoundation/model/models/models.ckpt"
echo "SharePoint URL: https://hopebio2020.sharepoint.com/:f:/s/PublicSharedfiles/IgBlEJ72TBE5Q76AmgXbgjXiAR69fzcrgzqgUYdSThPLrqk"
echo "Optional example data: manually download https://doi.org/10.6084/m9.figshare.24049200 into baselines/scFoundation/model/examples/"

echo "[foundation model] state"
mkdir -p baselines/state/pretrained/ST-Tahoe
huggingface-cli download arcinstitute/ST-Tahoe --local-dir baselines/state/pretrained/ST-Tahoe
mkdir -p baselines/state/pretrained/SE-600M
huggingface-cli download arcinstitute/SE-600M --local-dir baselines/state/pretrained/SE-600M \
  --include "config.yaml" \
  --include "protein_embeddings.pt" \
  --include "se600m_epoch15.ckpt"
# Optional Replogle-specialised snapshot:
# mkdir -p baselines/state/pretrained/ST-SE-Replogle
# huggingface-cli download arcinstitute/ST-SE-Replogle --local-dir baselines/state/pretrained/ST-SE-Replogle

echo "[foundation model] stack"
mkdir -p baselines/stack/pretrained/Stack-Large-Aligned
huggingface-cli download arcinstitute/Stack-Large-Aligned --local-dir baselines/stack/pretrained/Stack-Large-Aligned
# Optional base version:
# mkdir -p baselines/stack/pretrained/Stack-Large
# huggingface-cli download arcinstitute/Stack-Large --local-dir baselines/stack/pretrained/Stack-Large

echo "[foundation model] CellFlow"
mkdir -p baselines/CellFlow/data
huggingface-cli download theislab/cellflow-datasets --repo-type dataset --local-dir baselines/CellFlow/data \
  --include "zesta.h5ad" \
  --include "ineurons.h5ad" \
  --include "pbmc_parse.h5ad"

echo "[foundation model] Perturbgen"
mkdir -p baselines/Perturbgen/pretrained
huggingface-cli download lotfollahi-lab/PerturbGen --local-dir baselines/Perturbgen/pretrained

echo "[classical model] cpa"
# Optional upstream Drive bundle:
# mkdir -p baselines/cpa/assets
# gdown --folder "https://drive.google.com/drive/folders/1yFB0gBr72_KLLp1asojxTgTqgz6cwpju?usp=drive_link" \
#   -O baselines/cpa/assets

echo "[classical model] GEARS"
mkdir -p baselines/GEARS/data
wget -O baselines/GEARS/data/gene2go_all.pkl https://dataverse.harvard.edu/api/access/datafile/6153417
wget -O baselines/GEARS/data/essential_all_data_pert_genes.pkl https://dataverse.harvard.edu/api/access/datafile/6934320
mkdir -p baselines/GEARS/data/norman
wget -O /tmp/gears_norman.zip https://dataverse.harvard.edu/api/access/datafile/6154020
unzip -o /tmp/gears_norman.zip -d baselines/GEARS/data/norman
mkdir -p baselines/GEARS/data/adamson
wget -O /tmp/gears_adamson.zip https://dataverse.harvard.edu/api/access/datafile/6154417
unzip -o /tmp/gears_adamson.zip -d baselines/GEARS/data/adamson
mkdir -p baselines/GEARS/data/dixit
wget -O /tmp/gears_dixit.zip https://dataverse.harvard.edu/api/access/datafile/6154416
unzip -o /tmp/gears_dixit.zip -d baselines/GEARS/data/dixit
mkdir -p baselines/GEARS/data/replogle_k562_essential
wget -O /tmp/gears_replogle_k562.zip https://dataverse.harvard.edu/api/access/datafile/7458695
unzip -o /tmp/gears_replogle_k562.zip -d baselines/GEARS/data/replogle_k562_essential
mkdir -p baselines/GEARS/data/replogle_rpe1_essential
wget -O /tmp/gears_replogle_rpe1.zip https://dataverse.harvard.edu/api/access/datafile/7458694
unzip -o /tmp/gears_replogle_rpe1.zip -d baselines/GEARS/data/replogle_rpe1_essential

echo "[classical model] cellot"
echo "Manual step: download 'All Processed Datasets (ZIP, 398.87 MB)' from the page below"
echo "https://www.research-collection.ethz.ch/handle/20.500.11850/609681"
echo "Then unpack into baselines/cellot/datasets/"
```
