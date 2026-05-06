# Problem Setup

This document defines the task, splits, and evaluation framing for the
ccCFM project. It is meant to be read by collaborators who have **no**
prior context from other internal projects.

## Task

Cross-cell-line genetic perturbation response prediction.

Given (i) control and perturbed single-cell RNA-seq observations from a
set of *source* cell lines, and (ii) a perturbation gene `g`, predict the
perturbed-cell-state distribution for a *held-out target* cell line.

The perturbation modality here is CRISPRi knockdown (Replogle Perturb-seq
family of datasets).

## Notation

- `L = {c_1, ..., c_L}` — cell lines.
- `G` — the perturbation gene set.
- Each cell `x ∈ R^H` where `H` is the shared HVG count (default 6,836).
- Each cell has labels `(c, g)` where `c ∈ L` and
  `g ∈ G ∪ {control}`; `control` denotes an unperturbed cell.
- Population `P^{c,g}(x) := Pr(x | line=c, pert=g)`.

## Source / target split

- Source set `S ⊂ L`; held-out target `c*`.
- Two fixed configurations, both with `c* = Jurkat`:
  - **2-line**: `S = {K562, RPE1}`.
  - **3-line**: `S = {K562, RPE1, HepG2}`.
- Training observes `{P^{c,g}}_{c ∈ S, g ∈ G}` at the cell level.
- Evaluation targets `{P^{c*, g}}_{g ∈ G}`.

## Shared perturbation space

To enable clean cross-line transfer, `G` is restricted to perturbations
that appear in *all four* cell lines. This is enforced once upstream and
frozen as the canonical perturbation set.

## Strict Jurkat test slice

A fixed subset of the Jurkat test split, selected by:

1. `obs.split == test`
2. `obs.n_cells >= 50` (sufficient for distributional metrics)
3. Perturbation is in the 3-way shared set
4. A primary gene embedding exists for this perturbation

The resulting ~197 perturbations form the `jurkat_test_strict` slice. All
headline numbers are reported on this slice.

## Two-level evaluation

Every method is evaluated at two levels in parallel.

### Mean-level (pseudobulk-delta)
- **Pearson (Pf)** of predicted vs. observed pseudobulk delta.
- **MSE** of the delta vector.
- **DEG top-20 / top-50 recall** (overlap between top-|Δ| genes).

### Distribution-level (cell-level sample sets)
- **MMD** with multi-scale RBF kernel.
- **Energy distance**.
- **Sinkhorn-W2** in a 50-d PCA subspace.
- **Per-gene moment match** (mean + variance + skew).

Both levels are reported in every main table. The headline claim is on
distributional metrics; the linear-baseline critique of this field
(Ahlmann-Eltze et al., 2025, *Nat. Methods*) concerns mean-level metrics.

## Cross-line holdout integrity

The target cell line `c*` is never seen at train time. Held-out-perturbation
regimes (where `g` is unseen) are a separate axis of evaluation and are
reported separately when relevant.

## Method categories benchmarked

- Classical reference baselines: linear Ahlmann-Eltze, pooled ridge anchor,
  exact-gene retrieval.
- Classical deep: CPA, GEARS, CellOT.
- 2025-2026 generative: STATE, STACK, X-Cell, CellFlow, PerturbGen,
  Lingshu-Cell (zero-shot).
- Our method: ccCFM with each of six frozen scFM context encoders
  (PCA, STATE-SE, Geneformer, scGPT, UCE, scFoundation).

See `docs/reference_baselines.md` for the meaning and independent
implementation of each reference baseline.
