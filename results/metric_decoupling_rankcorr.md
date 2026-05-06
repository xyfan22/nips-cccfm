# Metric decoupling: Spearman rank-correlation quantification
**Generated**: 2026-05-05
**Source**: `results/MASTER_baseline_table_uniform_n.md` (uniform-n source-of-truth).
**Method**: For each row group, compute Spearman rank correlation between perturbation-specific Pf^pert (higher better) and three distributional metrics with sign flipped so higher is better (-Energy, -W_2, -Moment). p-values from scipy.stats.spearmanr two-sided test; n is small (9--22) so p-values are advisory only.

## Spearman rank correlations
| Setting | n | Spearman(Pf, -Energy) | Spearman(Pf, -W_2) | Spearman(Pf, -Moment) |
|---|---|---|---|---|
| A. Main 9-row, 2-line holdout | 9 | +0.678 (p=0.045) | +0.492 (p=0.179) | +0.593 (p=0.092) |
| B. Main 9-row, 3-line holdout | 9 | +0.576 (p=0.104) | +0.448 (p=0.226) | +0.509 (p=0.162) |
| C1. Full 2-line, all reported rows | 22 | +0.779 (p<0.001) | +0.675 (p<0.001) | +0.664 (p<0.001) |
| C2. Full 2-line, no rank-curve duplicates | 18 | +0.781 (p<0.001) | +0.631 (p=0.005) | +0.549 (p=0.018) |

## Rows used per block

### A. Main 9-row, 2-line holdout  (n=9)
- target_control_copy
- Linear (shifted)
- Exact (shifted)
- GEARS
- STATE zero-shot
- CellFlow
- ccCFM-PCA direct
- MP full-space (r31_mp_concat)
- MP r=256 (low-rank)

### B. Main 9-row, 3-line holdout  (n=9)
- target_control_copy
- Linear (shifted)
- Exact (shifted)
- GEARS
- STATE zero-shot
- CellFlow
- ccCFM-PCA direct
- MP full-space (r31_mp_concat)
- MP r=256 (low-rank)

### C1. Full 2-line, all reported rows  (n=22)
- target_control_copy
- Linear (shifted)
- Ridge (shifted)
- Exact (shifted)
- CellFlow
- CPA
- GEARS
- STACK zero-shot
- STATE zero-shot
- STATE fewshot
- ccCFM-PCA direct
- ccCFM-state_se (zscore_per_line)
- ccCFM-geneformer (zscore_per_line)
- ccCFM-scgpt (zscore_per_line)
- ccCFM-uce (zscore_per_line)
- ccCFM-scfoundation (zscore_per_line)
- MP full-space (r31_mp_concat)
- MP r=256 (low-rank)
- MP r=512 (low-rank)
- MP r=128
- MP r=64
- MP r=32

### C2. Full 2-line, no rank-curve duplicates  (n=18)
- target_control_copy
- Linear (shifted)
- Ridge (shifted)
- Exact (shifted)
- CellFlow
- CPA
- GEARS
- STACK zero-shot
- STATE zero-shot
- STATE fewshot
- ccCFM-PCA direct
- ccCFM-state_se (zscore_per_line)
- ccCFM-geneformer (zscore_per_line)
- ccCFM-scgpt (zscore_per_line)
- ccCFM-uce (zscore_per_line)
- ccCFM-scfoundation (zscore_per_line)
- MP full-space (r31_mp_concat)
- MP r=256 (low-rank)

## Rank disagreement summary (Main 9-row, 2-line)
- Top-3 by Pf^pert (high better): Exact (shifted) (0.331); Linear (shifted) (0.271); MP full-space (r31_mp_concat) (0.271)
- Top-3 by Energy (low better): target_control_copy (1.040); Exact (shifted) (1.060); Linear (shifted) (1.280)
- Top-3 by W_2 (low better): ccCFM-PCA direct (0.115); MP full-space (r31_mp_concat) (0.132); MP r=256 (low-rank) (0.157)
- Top-3 by Moment (low better): target_control_copy (0.400); Exact (shifted) (0.403); Linear (shifted) (0.410)

## Honest reading of the numbers
The Spearman correlations are **positive but incomplete**, not low or negative. They are inflated when the full method set is included (C1, C2: ~+0.55 to +0.78) because rows like STATE/STACK/scFoundation score badly on every metric simultaneously, so any "good methods are also less bad on dist metrics" signal pushes Spearman up. The cleaner 9-row blocks (A, B) give ~+0.45 to +0.68. The decoupling evidence therefore is **not** "the metrics are uncorrelated overall". It is that **the top-ranked row is metric-dependent**: the top-ranked method differs across metrics (e.g.\ target-control-copy ranks #1 on Energy and Moment but middle on Pf^pert; ccCFM-PCA direct ranks #1 on W_2 but near bottom on Pf^pert). A reader who picks the "best" method by any single distributional metric in the main 2-line table picks a row that is not best on Pf^pert.

## Suggested wording (locked to paper.tex)
**Main-text (one-sentence, near Figure 2 lead-in)**: Quantitatively, rank correlations are positive but incomplete across the nine main 2-line rows (Spearman $\rho{=}0.68$ for Pf$^{\mathrm{pert}}$ vs.\ $-$Energy, $0.49$ for Pf$^{\mathrm{pert}}$ vs.\ $-W_2$, and $0.59$ for Pf$^{\mathrm{pert}}$ vs.\ $-$Moment), and the top-ranked row is metric-dependent: Exact shifted-control is best on Pf$^{\mathrm{pert}}$, target-control-copy is best on Energy and Moment, and ccCFM-PCA direct is best on $W_2$ (Appendix~\ref{sec:appendix-rankcorr}).

**Appendix subsection sentence**: Table~\ref{tab:rankcorr} reports Spearman rank correlations between Pf$^{\mathrm{pert}}$ and sign-flipped distributional metrics across the main 2-line rows, the main 3-line rows, and the full 2-line method set (with and without rank-curve duplicates). The full-set correlations are inflated by uniformly-bad rows (STATE, STACK, scFoundation-conditioned probes) that fail simultaneously on every metric. The 9-row blocks give the cleaner read: Spearman is positive but incomplete, and the top-ranked row is metric-dependent in the main 2-line table --- target-control-copy is first on Energy ($1.04$) and Moment ($0.400$); ccCFM-PCA direct is first on $W_2$ ($0.115$); Exact shifted-control is first on Pf$^{\mathrm{pert}}$ ($0.331$). Picking the best method by any single distributional metric in Table~\ref{tab:main-2line} therefore selects a different row than picking it by Pf$^{\mathrm{pert}}$.
