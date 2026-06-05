# FROZEN — Do Not Modify

This benchmark folder contains **final, frozen results** from the ViroMiR v1 tool comparison.

**Date frozen:** 2026-06-04

## Contents

- `tools/` — Benchmark runner scripts (IntaRNA, miRanda, RNA22, ViroMiR)
- `data/` — Cross-virus test pair CSVs
- `logs/` — Execution logs from benchmark runs
- `results/` — Final benchmark summary tables
- `remoteRNA22v2/` — RNA22 Java remote client

## Benchmark Design

- **Negative strategy:** Cross-virus decoys (same miRNA tested against mRNA from a *different* virus)
- **Test set:** 120 pairs (60 positive, 60 negative), sampled from 31 viruses
- **Tools compared:** IntaRNA 3.4.1, miRanda 3.3a, RNA22 v2, ViroMiR v1

## Final Results (AUC on cross-virus negatives)

| Tool | AUC | AUC-PR | P@10 | P@20 |
|---|---|---|---|---|
| IntaRNA (delta_G) | 0.7547 | 0.7022 | 0.6000 | 0.7000 |
| miRanda | 0.6587 | 0.6673 | 0.8000 | 0.8000 |
| RNA22 | 0.5075 | 0.5021 | 0.6000 | 0.6500 |
| **ViroMiR v1** | **0.7206** | **0.6259** | 0.5000 | 0.6000 |

## Rules

- **Do NOT rerun** benchmarks unless the benchmark design itself changes.
- **Do NOT overwrite** logs or results.
- Production predictions go under `results/`, not here.
