# ViroMiR

**ViroMiR** is a high-precision bioinformatics command-line framework designed to discover and predict interactions between Human microRNAs (miRNAs) and Viral mRNA target genomes. 

By combining rigorous thermodynamic binding sequence analysis with advanced Machine Learning, ViroMiR drastically reduces the false-positive inflation commonly seen in generic miRNA targeting tools, yielding biologically sound, publication-ready predictions.

## 🧬 Pipeline Overview
ViroMiR utilizes a three-stage processing architecture:

1. **IntaRNA Scanning:** Parallelized thermodynamic evaluation of sequences to isolate stable RNA-RNA interaction candidates, automatically fetching up to 5 sub-optimal binding configurations per miRNA.
2. **XGBoost Rescoring:** Candidates are evaluated across a 16-feature matrix by an embedded XGBoost model, deriving a unified composite `viromir_score`.
3. **Deterministic Genomic Clustering:** Nearby overlapping binding sites on the viral genome are dynamically merged into deterministic intervals. Within each cluster, the single most biologically stable hit for a specific miRNA is retained via strict tie-breaking (`viromir_score` → `delta_G` → `n_base_pairs`). 

## 💻 Installation & Requirements

### Dependencies
ViroMiR relies on the following Python stack:
- `pandas >= 1.3.0`
- `numpy >= 1.21.0`
- `xgboost >= 1.5.0`
- `scikit-learn >= 1.0.2`

**CRITICAL REQUIREMENT:** ViroMiR requires the **IntaRNA** binary to be installed and available in your system `$PATH` for the Stage 1 thermodynamic scanning.
- Install via Conda: `conda install -c bioconda intarna`

### Setup
```bash
git clone https://github.com/somenath/ViroMiR.git
cd ViroMiR
pip install -e .
```

## 🚀 CLI Usage

Scan a viral genome against the bundled full human miRNA library (2,656 miRNAs):
```bash
viromir-scan --input virus.fa --mode strict --out results.csv --threads 8
```

Scan a viral genome against a specific subset of miRNAs (e.g., our bundled test file):
```bash
viromir-scan --input examples/hiv_1.fa --mirna-file examples/hiv_500_test.fa --mode balanced --out results.csv
```

### Filtering Modes
The pipeline outputs are controlled by three stringency presets:
- `--mode discovery` **(Score ≥ 0.50):** Highly permissive. Ideal for exploratory full-genome scans where maximum recall is required.
- `--mode balanced` **(Score ≥ 0.70):** The default setting. Filters out high-noise false positives while retaining robust sensitivity.
- `--mode strict` **(Score ≥ 0.75):** Stringent filtering for publication-quality output. Only the most thermodynamically stable and ML-confident predictions survive.

### Outputs
ViroMiR generates dual outputs for every run:
1. **`results.csv`**: The clean, clustered, and deduplicated main dataset. 
2. **`results_full.csv`**: The supplementary raw data containing **all** ML-scored candidates before clustering and strict mode thresholds were applied.

## 📊 Benchmark Summary

ViroMiR has been comprehensively benchmarked against complete viral genomes using the full human miRNA library (2,656 miRNAs). 
*(Note: The experimental validation sets discussed below are derived from manuscript benchmark datasets).*

### SARS-CoV-2 (NC_045512.2)
- **Genome Size:** ~30 kb
- **Raw IntaRNA Hits:** 12,775
- **Balanced Mode Predictions (Score ≥ 0.70):** 4,929
- **Strict Mode Predictions (Score ≥ 0.75):** 4,339
- **Experimental Validation:** Recovered 15 / 16 (93.8%) known curated experimental targets in `strict` mode. 

### HIV-1 (NC_001802.1)
- **Genome Size:** ~9.7 kb
- **Raw IntaRNA Hits:** 13,272
- **Balanced Mode Predictions:** 4,040
- **Experimental Validation:** Recovered 13 / 13 (100%) known experimental targets.

## 📖 Citation & Contact
If you utilize ViroMiR in your research pipeline, please refer to the `CITATION.cff` file in this repository.

*For bug reports, please open a GitHub Issue containing the exact command used and a snippet of the triggering FASTA sequence.*
