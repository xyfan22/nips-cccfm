# Compute records

Workloads are reported by accelerator type without an unmeasured conversion
to A100-equivalent time.

| Workload | Recorded resources |
|---|---|
| Original Replogle suite | Approximately 620 A100 GPU-hours, or 26 A100 GPU-days; excludes later validation |
| Rank-by-width factorial | 12 single-GPU training runs on NVIDIA H20-3e; sum of recorded training times is 540.35 minutes, approximately 9.01 H20 GPU-hours |
| Factorial evaluation | 12 evaluations over 577 perturbations; no complete separate timing record was synchronized |
| PBMC CellFlow | 12 donor models x 100,000 iterations; one GPU per fold; reported hardware RTX 3090, RTX A6000, and A40 |
| Source stratification | CPU analysis; recorded wall time about 18 seconds, peak RSS about 0.29 GB; core fingerprinting is separate |

The Donor1 PBMC pilot took approximately 54 minutes. Complete synchronized
per-fold timings for the remaining donors were not returned, so no precise
aggregate PBMC GPU-hour total is asserted.

The CPU table path reads metric CSVs and cached descriptors. It performs no
training or expression-data download. The factorial used bf16 mixed-precision
training; the CellFlow environment and the original benchmark are documented
separately in the paper and `provenance/protocols.json`.
