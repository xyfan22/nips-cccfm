# Full validation experiments

Run from the repository root. [data/README.md](../data/README.md) gives
downloads and preparation. Shipped CSVs are the historical evidence used by
the paper; new training runs produce new result sets.

## PyTorch rank-by-width factorial

Use a separate Python 3.10/3.11 venv:

```bash
python3 -m venv .venv-torch
source .venv-torch/bin/activate
python -m pip install -r requirements/torch-training.txt
python -m pip install -e . --no-deps
```

The dedicated training/evaluation scripts select the frozen package in
`reproducibility/fixed_width_rank/code/`. Its ten modules match the
validated runtime, including the tokenizer fix. The public entry scripts
add a portable import path.

Prepare `replogle_preprocessed.h5ad`, `h_pca`, and the primary gene
embedding. Then run the independent cells, assigning a distinct GPU to
each concurrent process if parallelizing:

```bash
for rank in 256 512; do
  for width in 1024 2048; do
    for seed in 7 42 314; do
      printf -v tag 'rank%s_hidden%s_seed%s' "$rank" "$width" "$seed"
      python scripts/rebuttal/10_train_fixed_width_rank.py \
        --encoder pca --setting 2line --target_line Jurkat \
        --anchor linear --anchor_target_mode mean_preserved \
        --flow_space lowrank --rank "$rank" --basis_mode both \
        --velocity_hidden_dim "$width" --n_holdout_perts 80 \
        --align_h_mode none --max_epochs 30 --batch_size 256 \
        --n_blocks 4 --lr 1e-4 --weight_decay 1e-5 --sigma 0 \
        --num_workers 4 --precision bf16-mixed --seed "$seed" \
        --run_id "$tag" --out_dir "runs/rebuttal/A_fixed_width_rank/$tag"
      python scripts/rebuttal/11_eval_fixed_width_rank.py \
        --run_dir "runs/rebuttal/A_fixed_width_rank/$tag" \
        --output "results/new_factorial/$tag.csv" \
        --n_samples -1 --max_n_samples 0 --compute_heavy
    done
  done
done
```

The recorded runs used bf16 mixed-precision training on NVIDIA H20-3e.
Evaluation matches target cell counts for all 577 perturbations. Use
`scripts/reproduce_validation.py` to summarize shipped rows without
historical checkpoint directories. The original `12_summarize_*`
script also audits newly created full run directories after a rerun.

## Primary-PBMC CellFlow

The standalone control audit can be rerun in the PyTorch environment:

```bash
for seed in 7 42 314; do
  python scripts/rebuttal/20_eval_pbmc_controls.py \
    --input data/raw/adata_for_cellflow_datasets_with_embeddings.h5ad \
    --sample_seed "$seed" --device cuda:0 \
    --output_dir "results/new_pbmc_controls/seed$seed"
done
python scripts/rebuttal/21_summarize_pbmc_controls.py \
  --result_root results/new_pbmc_controls \
  --output_dir results/new_pbmc_controls/combined
```

The successful GPU environment used Python 3.11.15 and JAX/JAXlib 0.5.3.
Use a separate environment. Observed pins are recorded in
`requirements/cellflow-observed.txt`:

```bash
python3.11 -m venv .venv-cellflow
source .venv-cellflow/bin/activate
python -m pip install "jax[cuda12]==0.5.3" -r requirements/cellflow-observed.txt
python -m pip install -e ./baselines/CellFlow -c requirements/cellflow-observed.txt
PYTHONPATH=baselines/CellFlow/src python -c \
  "import cellflow, jax; print(cellflow.__file__); print(jax.devices())"
```

Install `pytest` in this environment to run the integration test. The
CUDA installation must match the host driver; see the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
Run the opt-in GPU integration test from the reproducibility guide before
biological training.

Prepare the balanced snapshots:

```bash
python scripts/rebuttal/50_prepare_pbmc_cellflow_balanced.py \
  --input data/raw/adata_for_cellflow_datasets_with_embeddings.h5ad \
  --output_dir data/rebuttal/B2_pbmc_balanced
```

Run a resource-only probe:

```bash
PYTHONPATH=baselines/CellFlow/src python scripts/rebuttal/51_run_pbmc_cellflow_fold.py \
  --balanced_dir data/rebuttal/B2_pbmc_balanced --target_donor Donor1 \
  --out_dir results/new_pbmc_resource_probe --resource_probe
```

The paper used batch size 1024, the first successful candidate. Preserve it
for the reported configuration; a smaller batch is a different configuration.
For each held-out donor:

```bash
PYTHONPATH=baselines/CellFlow/src python scripts/rebuttal/51_run_pbmc_cellflow_fold.py \
  --balanced_dir data/rebuttal/B2_pbmc_balanced --target_donor Donor1 \
  --out_dir runs/rebuttal/B2_pbmc_cellflow/Donor1 \
  --batch_size 1024 \
  --resource_probe_result results/new_pbmc_resource_probe/resource_probe.json
```

Repeat for `Donor2` through `Donor12`. Each fold trains one model
for 100,000 iterations with model/training-cell seed 42, then evaluates
sampling seeds 7/42/314. Source-only PCA has 100 dimensions. Held-out PBS
controls supply prediction-time context; no held-out donor enters PCA or training.

The full-fold summarizer checks checkpoint/PCA/training hashes and the
released B1 control references:

```bash
python scripts/rebuttal/52_summarize_pbmc_cellflow.py \
  --result_root runs/rebuttal/B2_pbmc_cellflow \
  --b1_controls_dir results/rebuttal/B1_pbmc_controls \
  --output_dir results/new_pbmc_summary --control_tolerance 0.01
```

The cross-backend control consistency tolerance was increased from 0.0005
to 0.01 for the PyTorch/GPU-versus-NumPy/CPU comparison. The three-method
paper table itself uses one NumPy/CPU evaluation backend. Lightweight table
reproduction uses the shipped CSVs without the checkpoint files.

## Source-defined effect stratification

Four frozen 577-row inputs are under `results/rebuttal/C_frozen_inputs/`.
The expression-core fingerprint and feature/split hashes are in
`provenance/protocols.json`. Full descriptor recomputation uses:

```bash
python scripts/rebuttal/30_effect_stratification.py \
  --data data/processed/replogle_preprocessed.h5ad \
  --test_slice data/splits/jurkat_test_strict.json \
  --method_csv target_control_copy results/rebuttal/C_frozen_inputs/target_control_copy_2line.csv \
  --method_csv linear_shifted results/rebuttal/C_frozen_inputs/linear_ahlmann_eltze_2line_shifted_control.csv \
  --method_csv exact_shifted results/rebuttal/C_frozen_inputs/exact_gene_retrieval_2line_shifted_control.csv \
  --method_csv cellflow results/rebuttal/C_frozen_inputs/cellflow_2line_uniform_n.csv \
  --output_dir results/new_source_stratification
```

Descriptors use K562/RPE1 cells only. All marginal/joint strata are retained.
The full-577 evaluation reference stays fixed. Intervals use 2,000 paired
perturbation resamples and seed 20260726.
