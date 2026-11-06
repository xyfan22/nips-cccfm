# Zero-Shot Cross-Cell-Line Perturbation Prediction Needs Controls

**Official code and results | NeurIPS 2026 Evaluations and Datasets Track**

Xingyu Fan<sup>1</sup>, Jinghao Wang<sup>1</sup>, Chang-Yu Hsieh<sup>2</sup>, Chunbin Gu<sup>1</sup>, Pheng-Ann Heng<sup>1</sup>

<sup>1</sup> Department of Computer Science and Engineering, The Chinese University of Hong Kong<br>
<sup>2</sup> College of Pharmaceutical Sciences, Zhejiang University

[Camera-ready release](https://github.com/xyfan22/crossline-controls/releases/tag/v0.1.0) | [Data and preprocessing](data/README.md) | [Full experiment guide](docs/validation_experiments.md)

Predicting a target cell line's background distribution and predicting its
perturbation response are different tasks. This work introduces two diagnostic
controls, **target-control-copy** and **shifted-control**, and evaluates models
with a shared mean-level and distributional evaluator.

The main benchmark studies strict zero-shot cross-cell-line CRISPRi transfer:
target-line controls are available, while target-line perturbation responses are
held out. Additional experiments cover frozen scFM contexts, anchored residual
flows, and primary-PBMC cytokine donor transfer within shared cell types.

## Quick start: reproduce the numerical summaries

The CPU reproduction path uses the released metric CSVs and cached source
descriptors. It requires Python 3.11, with no GPU, checkpoint, or expression-data
download.

```bash
git clone https://github.com/xyfan22/crossline-controls.git
cd crossline-controls

python3.11 -m venv .venv-eval
source .venv-eval/bin/activate
python -m pip install -r requirements/evaluation-tested.txt

python scripts/verify_release.py
python scripts/reproduce_validation.py
python scripts/16_metric_decoupling_rankcorr.py
```

- `verify_release.py` checks the frozen numerical inputs, definitions, and runtime files.
- `reproduce_validation.py` rebuilds the rank-by-width, PBMC donor, and stratification summaries, compares them with the recorded results, and writes `results/reproduced_validation/verification.json`.
- `16_metric_decoupling_rankcorr.py` reproduces the metric-ranking analysis in `results/metric_decoupling_rankcorr.md`.

This checks aggregation and statistical summaries from released metric rows.
For model training and metric evaluation on expression clouds, follow the
[full experiment guide](docs/validation_experiments.md).

## Experiments and results

| Study | Main entry points | Released results |
|---|---|---|
| Cross-cell-line CRISPRi | [Evaluator](code/cccfm/eval_metrics.py), [baseline and probe scripts](scripts/) | [Uniform-n method table](results/MASTER_baseline_table_uniform_n.md) |
| Rank-by-width factorial | [Training](scripts/rebuttal/10_train_fixed_width_rank.py), [evaluation](scripts/rebuttal/11_eval_fixed_width_rank.py) | [Three-seed summary](results/rebuttal/A_fixed_width_rank/audit/factorial_cell_summary.csv) |
| Primary-PBMC donor transfer | [Controls](scripts/rebuttal/20_eval_pbmc_controls.py), [preparation](scripts/rebuttal/50_prepare_pbmc_cellflow_balanced.py), [CellFlow](scripts/rebuttal/51_run_pbmc_cellflow_fold.py) | [Donor-level results](results/rebuttal/B2_pbmc_cellflow/pilot/combined/donor_summary.csv), [global summary](results/rebuttal/B2_pbmc_cellflow/pilot/combined/global_summary.csv) |
| Source-defined stratification | [Analysis](scripts/rebuttal/30_effect_stratification.py) | [Stratum summaries](results/rebuttal/C_effect_stratification/stratum_summary.csv), [continuous trends](results/rebuttal/C_effect_stratification/trend_summary.csv) |

The factorial tests two ranks and two hidden widths with three training seeds.
The PBMC audit covers 12 donors, six cell types, and 88 cytokines; its three
sampling seeds are cell-sampling repeats, and variability is reported across
donors. Experiment scripts retain their original paths under `scripts/rebuttal/`.

## Data and full reruns

[data/README.md](data/README.md) maps each input filename to its official
download, checksum, and preprocessing steps. The repository includes the frozen
Replogle feature and split lists, alongside numerical experimental outputs.
Expression matrices and trained checkpoints are not bundled.

List the official input downloads without starting a download:

```bash
python scripts/download_data.py --list
```

Use separate environments for CPU evaluation, PyTorch training, and CellFlow/JAX:

- [Reproducibility guide](docs/reproducibility.md): environment setup, table checks, and tests.
- [Full experiment guide](docs/validation_experiments.md): data preparation, training, and evaluation commands.
- [Compute records](docs/compute.md): resources used for the original and additional experiments.
- [Protocol metadata](provenance/protocols.json): settings, feature/split fingerprints, and sampling conventions.

## Tests and figures

Run the CPU tests in the evaluation environment:

```bash
python -m pytest tests/test_release.py scripts/rebuttal/tests/test_b2_pbmc_cellflow.py -q
```

The GPU integration test is opt-in; its environment and command are documented
in the [reproducibility guide](docs/reproducibility.md).

Paper figures are under `figures/`. Rebuild Figure 3 and its input-provenance CSV:

```bash
python figures/plot_rank_curve.py
```

## Citation

```bibtex
@inproceedings{fan2026controls,
  title={Zero-Shot Cross-Cell-Line Perturbation Prediction Needs Controls},
  author={Fan, Xingyu and Wang, Jinghao and Hsieh, Chang-Yu and Gu, Chunbin and Heng, Pheng-Ann},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026},
  note={Evaluations and Datasets Track}
}
```

Software citation metadata and author contact information are in
[CITATION.cff](CITATION.cff). The [v0.1.0 release](https://github.com/xyfan22/crossline-controls/releases/tag/v0.1.0)
provides a fixed snapshot of the camera-ready code and results.

## License

Our code, documentation, and numerical analysis outputs use the [MIT license](LICENSE).
Third-party code, data, and model weights retain their upstream terms;
see [THIRD_PARTY.md](THIRD_PARTY.md) for attribution and license details.
