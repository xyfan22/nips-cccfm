# Reproducing numerical results

## CPU table checks

Use Python 3.11 and `requirements/evaluation.txt` in a fresh venv.
`requirements/evaluation-tested.txt` records the complete environment
used for the release checks, including transitive dependency versions.

```bash
python scripts/reproduce_validation.py
python scripts/16_metric_decoupling_rankcorr.py
```

The first command checks the 12 factorial CSVs and matched cell counts;
all 57,024 PBMC rows and donor aggregation; deterministic source tertiles,
448 paired-bootstrap summaries, and 24 continuous trends. Generated files
and `verification.json` go to `results/reproduced_validation/`.
All summaries are compared numerically with the released references.

PBMC variability is across 12 donors; its three seeds are cell-sampling
repeats. Stratification intervals are descriptive paired perturbation
percentile intervals. This path uses frozen metric rows and cached source
descriptors. Full expression-metric recomputation/training is documented in
[validation_experiments.md](validation_experiments.md).

## Tests

CPU preparation/evaluation/summarizer fixtures:

```bash
python scripts/verify_release.py
pytest tests/test_release.py -q
pytest scripts/rebuttal/tests/test_b2_pbmc_cellflow.py -q
```

One final test is an opt-in JAX/GPU integration test, run in the separate
CellFlow environment:

```bash
CELLFLOW_SCHOOL_TEST=1 PYTHONPATH=baselines/CellFlow/src \
  pytest scripts/rebuttal/tests/test_b2_pbmc_cellflow.py -q
```

The variable name is retained from the validated experiment. The test
predicts for a donor absent from training.

Original probe/control tests require the PyTorch environment:

```bash
python -m pip install -e . --no-deps
pytest code/tests/test_smoke.py -q
```

## Figure and integrity

```bash
python figures/plot_rank_curve.py
```

Figure 3 and `rank_curve.csv` are rebuilt from uniform-n summaries,
retaining the original four-decimal plotted inputs. Each point displays
rank and hidden width. Compare numerical values rather than PDF bytes.

`provenance/frozen_runtime.sha256` identifies the ten modules actually
used for the factorial. The release input manifest records checksums of the
numeric sources and frozen definitions.
