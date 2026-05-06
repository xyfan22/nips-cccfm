# Reviewer reproducibility guide

This guide takes a reviewer from a fresh clone to verifying the paper's
central evaluation claims in **5–20 minutes on a CPU-only machine**.
It does not require a GPU and does not require any h5ad to be downloaded.

## Step 0 — Environment (≤3 min)

```bash
conda env create -f environment.yml -n cccfm-review
conda activate cccfm-review
python -c "import cccfm; from cccfm.eval_metrics import CrossLineEvaluator, sinkhorn_w2; print('imports ok')"
```

Expected: clean import with no GPU touched.

## Step 1 — Re-derive the rank-correlation table (≤1 min)

```bash
python scripts/16_metric_decoupling_rankcorr.py
```

This script reads `results/MASTER_baseline_table_uniform_n.md` and
re-emits `results/metric_decoupling_rankcorr.md`. The shipped MD already
contains the expected output; the regenerated MD should match it byte-for-byte.

```bash
diff results/metric_decoupling_rankcorr.md results/metric_decoupling_rankcorr.md.golden
```

Expected key numbers (Spearman ρ on the main 9-row 2-line block, also
in `tab:rankcorr` of the paper Appendix):
- Pf$^{\mathrm{pert}}$ vs $-$Energy: **+0.68** (p=0.045)
- Pf$^{\mathrm{pert}}$ vs $-W_2$: **+0.49** (p=0.179)
- Pf$^{\mathrm{pert}}$ vs $-$Moment: **+0.59** (p=0.092)

## Step 2 — Inspect the master table source-of-truth

```bash
head -45 results/MASTER_baseline_table_uniform_n.md
```

Every row in `tab:main-2line`, `tab:main-3line`, `tab:appendix-fullmethod`,
`tab:cross-target`, `tab:appendix-rank`, and `tab:encoder-ablation` of
the paper traces back to a `_summary.csv` under `results/`. The MD is
the human-readable aggregation; the CSVs are the raw aggregations
emitted by the evaluator.

To independently re-aggregate from the shipped CSVs, see
`results/baselines/`, `results/cccfm/`, `results/cross_target/{C1,C2}/`,
and `results/per_pert_holdout/jurkat_2line_h100/`.

## Step 3 — Inspect the shipped Figure 2 and Figure 3 PDFs (≤1 min)

```bash
ls -lh figures/metric_decoupling.pdf figures/rank_curve.pdf
```

The artifact ships these paper figure PDFs as reference outputs. Figure-generation
scripts are not part of the reviewer smoke-test path; the numeric provenance for
their plotted values is provided by `results/MASTER_baseline_table_uniform_n.md`,
`results/metric_decoupling_rankcorr.md`, and the rank-curve `_summary.csv` files
under `results/cccfm/`.

## Step 4 — Run the package smoke test (≤2 min, CPU)

```bash
pytest code/tests/test_smoke.py -v
```

This exercises:
- `CrossLineEvaluator` on a tiny synthetic cell matrix,
- target-control-copy and at least one shifted-control,
- the documented anchor-mean invariant of Eq. (shifted-control) in §4
  of the paper (predicted batch mean equals anchor mean to numerical
  tolerance).

## Pass criteria

- All four steps exit zero.
- Step 1's regenerated MD matches the shipped reference.
- Step 3 confirms the shipped Figure 2 and Figure 3 PDFs are present.
- Step 4's pytest output is all green.

If all four pass, you have independently verified that:
1. The necessary controls (target-control-copy, shifted-control) behave as
   described in §4 of the paper.
2. Every main-table number traces deterministically to a shipped
   `_summary.csv`.
3. The metric-decoupling rank-correlation finding is reproducible from
   the master MD with no hidden state.
4. The two main figure PDFs are present as shipped reference outputs, and
   their numeric provenance is available from the master MD and summary CSVs.

## What is NOT exercised by this smoke-test

The artifact intentionally scopes reproduction to *evaluator-level* claims.
The following are documented but not re-run:
- Training of the residual-flow probes (`scripts/11_train_cccfm.py`):
  reviewers can read the code but reproducing it requires a GPU and the
  per-encoder featurized h5ads (see `data/README.md`).
- Inference of released zero-shot baselines (STATE, STACK, CellFlow,
  GEARS, CPA): the per-row `_summary.csv` files are shipped, but
  re-running them requires the upstream model checkpoints.
- Encoder featurization (`scripts/10_featurize_encoder.py`): GPU-only
  and depends on each scFM's official checkpoint.

These are intentional scope boundaries, not gaps. The paper's core
evaluation claims do not depend on retraining any of the above.
