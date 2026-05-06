# Systema - Reference Notes

- **Paper:** "Selection bias and systematic variation in perturbation experiments" (Nature Biotechnology, 2025)
- **Repo:** https://github.com/mlbio-epfl/systema
- **Commit SHA:** aaf5b5353993b48b78543f2f93b3e18ca65df515
- **License:** Not explicitly stated in repo

## Why shortlisted

Systema addresses the fundamental evaluation problem in perturbation prediction: standard metrics (Pearson on delta vs control) can be inflated by systematic variation - the shared signal across all perturbations that any mean predictor already captures. Their proposed reference correction (using the average of perturbation centroids rather than control mean as reference) is directly relevant to interpreting our B1-B4 results.

## Reusable components

1. **evaluation/pearson_delta_reference_metrics.py**: PearsonDelta and PearsonDelta20 with customizable reference (control mean or perturbation centroid mean). Small, self-contained. Can be adapted to our pseudobulk delta evaluation.
2. **evaluation/centroid_accuracy.py**: Centroid accuracy metric - measures if predicted profile is closer to the correct perturbation centroid than others. Novel metric not in our current pipeline.
3. **evaluation/eval_utils.py**: Top-k DE gene identification, shift similarities, cosine metrics.
4. **src/run_nonctl-mean.py** and **src/run_matching-mean.py**: Non-parametric baselines. Non-control mean is similar to our B2 but uses all perturbed cells (not just training). Matching mean is a kNN-style baseline that could inform B5 design.

## Adaptation risk

- Code is cell-level (single-cell AnnData). Our setup is pseudobulk delta. Metric functions need adaptation from cell-level centroids to pseudobulk vectors.
- Data loading depends on GEARS PertData class. We use our own manifest/npz format.
- The metrics are small enough to port directly rather than use as a library dependency.

## What to try first

1. Port PearsonDelta metrics (with perturbation centroid reference) to our eval pipeline.
2. Implement Systema-style evaluation on our B1-B4 results to check if the ranking changes.
3. Consider matching-mean as a new baseline candidate (kNN over training perturbation profiles).
