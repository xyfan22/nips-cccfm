# Zero-Shot Cross-Cell-Line Perturbation Prediction Needs Controls

Code and numerical analysis outputs for our NeurIPS 2026 Evaluations and Datasets Track paper.

**Authors:** Xingyu Fan, Jinghao Wang, Chang-Yu Hsieh, Chunbin Gu, and Pheng-Ann Heng.
Affiliations and citation metadata are in [CITATION.cff](CITATION.cff).

This repository provides target-control-copy and shifted-control, the shared
evaluator, residual-flow probes, and the validation experiments in the final paper.

## Rebuild the validation tables on CPU

Run from the repository root with Python 3.11:

```bash
python3 -m venv .venv-eval
source .venv-eval/bin/activate
python -m pip install -r requirements/evaluation.txt
python scripts/reproduce_validation.py
python scripts/16_metric_decoupling_rankcorr.py
```

The first script rebuilds the factorial, PBMC donor summaries, and source
stratification statistics from released CSV rows, checks the recorded reference
summaries, and writes `results/reproduced_validation/`. The second regenerates
the original metric-ranking analysis.

This path needs no GPU, checkpoint, or expression-data download. It checks
aggregation, statistical summaries, and numerical provenance. Cell-cloud
metric recomputation and training use the separate experiment commands below.

## Released experiments

| Experiment | Code | Numerical evidence |
|---|---|---|
| Rank x hidden width | `scripts/rebuttal/10_*`, `11_*`, `12_*`; frozen runtime in `reproducibility/fixed_width_rank/` | 12 runs x 577 perturbations and training-seed summaries |
| Primary-PBMC donor transfer | `scripts/rebuttal/20_*`, `21_*`, `50_*`, `51_*`, `52_*` | Controls and CellFlow, 12 donors, six shared cell types, 88 cytokines, three cell-sampling repeats |
| Source-defined stratification | `scripts/rebuttal/29_*`, `30_*` | Frozen metric inputs, source descriptors, marginal/joint strata and descriptive intervals |
| Original Replogle comparisons | `scripts/00_*` through `16_*`, `code/cccfm/` | Master table and per-method metric summaries |

The historical `scripts/rebuttal/` name is retained for experiment-record
compatibility. It contains executable code, not reviews or author discussions.

## Data and environments

- [data/README.md](data/README.md): official downloads, checksums, frozen split
  and feature lists, and preprocessing.
- [docs/reproducibility.md](docs/reproducibility.md): CPU checks and tests.
- [docs/validation_experiments.md](docs/validation_experiments.md): full reruns.
- [docs/compute.md](docs/compute.md): original and additional compute records.
- [provenance/protocols.json](provenance/protocols.json): settings and fingerprints.

Use separate evaluation, PyTorch, and CellFlow/JAX environments. Raw/processed
expression matrices, checkpoints, training logs, cluster instructions, and
author discussions are not distributed. The original data remain with their
official providers.

## Figures and licenses

`figures/` contains the paper figures. Rebuild Figure 3 with
`python figures/plot_rank_curve.py`; its sidecar CSV records inputs,
rank, width, and plotted values.

Our code, documentation, and numerical analysis outputs use the MIT license. Third-party sources, upstream
data, and model weights retain their own terms; see [THIRD_PARTY.md](THIRD_PARTY.md).
Please cite the paper when using its controls, evaluation protocol, or outputs.
