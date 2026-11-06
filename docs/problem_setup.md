# Evaluation setup

## Zero-shot cross-cell-line CRISPRi

Source lines provide control and perturbed expression profiles. The
held-out target line provides controls but no perturbation responses for
fitting. The target perturbation distributions are evaluation data.

The primary target is Jurkat. Sources are K562/RPE1 (2-line) or
K562/RPE1/HepG2 (3-line). K562 and RPE1 cross-target tests are separate
robustness arms. They do not include CellFlow.

The fixed Jurkat slice contains 577 perturbations, with at least 50
observed cells in the required groups. Expression uses the same ordered
6,642 features for all methods. The lists in `data/splits/` and
`data/processed/hvg_genes.txt` are the released definitions, rather than
an additional random test split.

## Necessary controls

- **Target-control-copy** predicts samples from the target's control
  distribution without a perturbation effect.
- **Shifted-control** centers the sampled target controls and reanchors
  them at a source-derived perturbation mean. Linear and exact-gene
  variants differ in how that mean is estimated.

These controls expose differences between perturbation-mean recovery and
matching an aggregate cell cloud. Neither requires learning a target
perturbation response.

## Metrics

The original evaluator reports control-referenced Pearson,
perturbation-referenced Pearson, top-20 perturbation-referenced Pearson,
centroid accuracy, MMD, Energy, normalized Sinkhorn transport cost, and
per-gene Moment. Higher is better for Pearson/centroid accuracy; lower is
better for MMD/Energy/transport cost/Moment. Definitions are implemented
in `code/cccfm/eval_metrics.py` and described in the paper.

Replogle predictions use the observed target-cell count for each
perturbation. Comparisons use one shared evaluator and feature space.
Different metrics answer different questions; a single cloud-distance
ranking is not a substitute for perturbation-specific evaluation.

## Frozen contexts and probes

The context diagnostic compares five frozen scFM encoders (Geneformer,
scGPT, UCE, scFoundation, STATE-SE) with a PCA/null context through the
same downstream flow head. The rank-by-width factorial separately tests
ranks 256/512 and hidden widths 1024/2048 with three training seeds. Its
immutable runtime is under `reproducibility/fixed_width_rank/`.

## External primary-PBMC validation

This is cytokine **donor transfer**, not held-out-cell-type prediction:
12 donors, six shared cell types, 88 cytokines, and three cell-sampling
repeats. In each fold, no target-donor cell enters PCA fitting or CellFlow
training; its PBS controls supply prediction-time context and the control
cloud. Reported variability is across donors, not sampling seeds.

See [validation_experiments.md](validation_experiments.md) for complete
rerun commands and [reproducibility.md](reproducibility.md) for CPU table
reconstruction from released metric rows.
