# Cross-Cell-Line Perturbation Prediction Needs Controls — Reviewer Artifact

Anonymous reproducibility artifact for the NeurIPS 2026 **Evaluations &
Datasets** track submission *Cross-Cell-Line Perturbation Prediction
Needs Controls*.

## What this artifact is for

This artifact lets a reviewer **independently verify the paper's central
evaluation claims in 5–20 minutes on a CPU-only machine**:

1. The two necessary controls (target-control-copy, shifted-control)
   behave as described in §4 of the paper.
2. Every main-table number traces deterministically to a shipped
   `_summary.csv` produced by the unified `CrossLineEvaluator`.
3. The metric-decoupling rank-correlation finding is reproducible from
   the shipped master Markdown with no hidden state (Spearman ρ =
   +0.68 / +0.49 / +0.59 against $-$Energy / $-W_2$ / $-$Moment on the
   main 9-row 2-line block).
4. The shipped Figure 2 and Figure 3 PDFs are included as reference outputs.

## Quickstart

```bash
conda env create -f environment.yml -n cccfm-review
conda activate cccfm-review
pytest code/tests/test_smoke.py -v
python scripts/16_metric_decoupling_rankcorr.py
```

For the full smoke-test recipe see [`docs/reproducibility.md`](docs/reproducibility.md).

## Repository layout

```
nips-cccfm/
├─ README.md                          this file
├─ LICENSE                            MIT
├─ environment.yml                    pinned conda env (CPU-only review)
├─ pyproject.toml                     installs the cccfm package
├─ code/
│  ├─ cccfm/                          Python package
│  │   ├─ eval_metrics.py             CrossLineEvaluator, sinkhorn_w2, MMD/Energy/Moment
│  │   ├─ anchor.py                   target-control-copy + Linear/Ridge/Exact shifted-control
│  │   ├─ basis.py                    source-fit PCA basis (low-rank passthrough probe)
│  │   ├─ flow_module.py              residual-flow probe (Lightning module)
│  │   ├─ velocity_net.py             velocity-network architecture
│  │   ├─ pert_tokens.py              gene-embedding pert token
│  │   ├─ data.py                     dataset / dataloader
│  │   ├─ encoders.py                 frozen scFM encoder adapters
│  │   ├─ sampling.py                 ODE integration helpers
│  │   └─ utils.py
│  ├─ configs/default.yaml
│  └─ tests/test_smoke.py             CPU-only invariants on the controls
├─ scripts/                            (numbered 00-16 in execution order)
│  ├─ 00_preprocess_data.py            raw Replogle -> log-normalized adata
│  ├─ 01_build_splits.py               577-pert × 6,642-HVG slice + holdouts
│  ├─ 02_build_gene_embedding.py       gene-embedding for the pert token
│  ├─ 03_linear_ahlmann_eltze.py       Linear shifted-control anchor fit
│  ├─ 04_pooled_ridge_anchor.py        Ridge shifted-control anchor
│  ├─ 05_exact_gene_retrieval.py       Exact-gene shifted-control anchor
│  ├─ 06_mean_baselines_distributional.py   target-control-copy + shifted-controls eval
│  ├─ 07_anchor_sanity.py              anchor-mean invariants (§4 of paper)
│  ├─ 08_residual_zeroinit_gate.py     probe gating diagnostic
│  ├─ 09_r2_recenter_lambda_sweep.py   probe re-center sweep (ablation)
│  ├─ 10_featurize_encoder.py          per-cell context featurization (GPU; see data/README.md)
│  ├─ 11_train_cccfm.py                residual-flow probe training (GPU; documentation)
│  ├─ 12_eval_cccfm.py                 evaluator entry point (produces _summary.csv)
│  ├─ 13_eval_state_zeroshot.py        STATE released-checkpoint inference
│  ├─ 14_state_infer_lowmem.py         STATE low-memory inference helper
│  ├─ 15_eval_stack_zeroshot.py        STACK released-checkpoint inference
│  └─ 16_metric_decoupling_rankcorr.py rank-correlation analysis (Appendix tab:rankcorr)
├─ results/
│  ├─ MASTER_baseline_table_uniform_n.md   single source-of-truth
│  ├─ metric_decoupling_rankcorr.md        re-derived deterministically
│  ├─ baselines/*_summary.csv              36 files
│  ├─ cccfm/*_summary.csv                  72 files
│  ├─ cccfm/seed_variance/*_summary.csv    8 files
│  ├─ cross_target/{C1,C2}/*_summary.csv   10 files
│  └─ per_pert_holdout/jurkat_2line_h100/*_summary.csv   4 files
├─ figures/                           expected Figure 1/2/3 PDF outputs
├─ docs/
│  ├─ data_contract.md                schema reviewers need to interpret the evaluator
│  ├─ problem_setup.md                strict cross-cell-line setup
│  └─ reproducibility.md              full smoke-test recipe (≤20 min, CPU)
├─ data/
│  └─ README.md                       fetch + rebuild instructions for raw / featurized data
└─ baselines/                         vendored upstream code for the 5 scFM encoders
   ├─ Geneformer/    scGPT/    UCE/   text-only (Python sources, READMEs, configs ≤200 KB)
   ├─ scFoundation/  state/           pretrained checkpoints NOT included; see data/README.md
   └─ ...                             also covers CellFlow / GEARS / CPA / cellot / stack /
                                      Perturbgen / systema as third-party references
```

## Scope boundaries

The artifact intentionally scopes reproduction to the *evaluator level*:

- **Included**: the unified `CrossLineEvaluator`, all metric implementations
  (MMD, Energy, Sinkhorn-$W_2$, per-gene Moment), the necessary controls
  (target-control-copy, Linear/Ridge/Exact shifted-control), the
  residual-flow probe definition, every `_summary.csv` underlying the
  paper's tables, the master MD, the rank-correlation analysis.
- **Not included**: raw and processed h5ads (~57 GB on disk; see
  `data/README.md` for the public download), any model checkpoints or
  run dirs (~39 GB), and per-cell prediction dumps. Vendored third-party
  baseline source trees are included only as public-code references; full
  reruns require each upstream's official dependencies and checkpoints.

This artifact does **not** ship the paper PDF or LaTeX sources; those are
uploaded separately through the OpenReview paper-upload channel.

## Anonymity statement

The artifact does not contain *our* author names, GitHub usernames,
hostnames, absolute home-directory paths, or wandb/project/team
identifiers. The vendored `baselines/<encoder>/` trees retain the
**upstream maintainers'** names, license files, and repository URLs —
these are public third-party references for the encoders / baselines we
benchmark, not the authors of this submission.
