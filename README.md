# ViroMiR

**ViroMiR** is a bioinformatics command-line framework designed to predict interactions between Human microRNAs (miRNAs) and Viral mRNA target genomes. 

ViroMiR combines thermodynamic scanning and machine learning to reduce false-positive inflation and produce ranked, publication-oriented predictions.

---

## ⚡ Quick Start

Get up and running with ViroMiR in under a minute:

**1. Install**
We highly recommend creating a dedicated Conda environment to avoid conflicts with your system's Python/pip paths.

```bash
git clone https://github.com/somenath-combio/ViroMiR.git
cd ViroMiR

# Create and activate a clean environment
conda create -n viromir_env python=3.10 -y
conda activate viromir_env

# Install IntaRNA and package dependencies
conda install -c bioconda intarna -y
pip install -e .
```

**2. Run a Bundled Example**
Scan the HIV-1 genome against a subset of miRNAs:
```bash
viromir-scan --input examples/hiv_1.fa --mirna-file examples/hiv_500_test.fa --mode balanced --out results.csv --threads 4
```

**3. Inspect Outputs**
- **`results.csv`**: The clean, deduplicated, ranked final predictions.
- **`results_full.csv`**: The supplementary table of all raw scored candidates.

---

## 🧬 Pipeline Overview
ViroMiR utilizes a three-stage processing architecture:

1. **IntaRNA Scanning:** Parallelized thermodynamic evaluation of sequences to isolate stable RNA-RNA interaction candidates, automatically fetching up to 5 sub-optimal binding configurations per miRNA.
2. **XGBoost Rescoring:** Candidates are evaluated across a 16-feature matrix by an embedded XGBoost model, deriving a unified composite `viromir_score`.
3. **Deterministic Genomic Clustering:** Nearby overlapping binding sites on the viral genome are dynamically merged into continuous intervals. Within each cluster, the single most biologically stable hit for a specific miRNA is retained via strict tie-breaking (`viromir_score` → `delta_G` → `n_base_pairs`). 

---

## 🚀 CLI Usage

ViroMiR is run from the command line with a viral FASTA input and optional miRNA FASTA input. If no miRNA file is provided, ViroMiR automatically scans against the bundled full human miRNA library (2,656 miRNAs).

```bash
# Default scan against the bundled full human miRNA library
viromir-scan --input virus.fa --mode balanced --out results.csv --threads 8

# Scan against a custom miRNA subset
viromir-scan --input examples/hiv_1.fa --mirna-file custom_mirnas.fa --mode strict --out results.csv
```

### Options

| Option | Meaning | Default |
|---|---|---|
| `--input`, `-i` | Viral FASTA or multi-FASTA | **required** |
| `--mirna-file` | Custom miRNA FASTA | *Bundled human library* |
| `--mode` | Filtering strictness: `discovery`, `balanced`, or `strict` | `balanced` |
| `--out`, `-o` | Output path for the main CSV | `viromir_predictions.csv` |
| `--threads` | Parallel workers for IntaRNA scanning | *System-dependent* |

### Filtering Modes
The pipeline outputs are controlled by three stringency presets:
- `--mode discovery` **(Score ≥ 0.50):** Highly permissive. Ideal for exploratory full-genome scans where maximum recall is required.
- `--mode balanced` **(Score ≥ 0.70):** The default setting. Filters out high-noise false positives while retaining robust sensitivity.
- `--mode strict` **(Score ≥ 0.75):** Stringent filtering for publication-quality output. Only the most thermodynamically stable and ML-confident predictions survive.

---

## 📄 Understanding the Outputs

ViroMiR generates dual outputs for every run:
1. **`results.csv`**: The clean, clustered, and deduplicated main dataset. **This is the file you should cite and use for downstream analysis.**
2. **`results_full.csv`**: The supplementary raw data containing **all** ML-scored candidates before clustering and strict mode thresholds were applied.

### How to Read the Results

Rows in the final output are sorted descending by `viromir_score`, with `delta_G` (thermodynamic stability) acting as a tie-breaker.

* **`viromir_score`**: A probability score (0.0 to 1.0) assigned by the XGBoost model based on 16 structural and sequence features. Higher is better.
* **`confidence`**: A human-readable label (`High`, `Medium`, `Low`) based strictly on the `viromir_score`. 
* **`delta_G`**: The raw thermodynamic binding energy (kcal/mol) calculated by IntaRNA. While the ML score dictates ranking, `delta_G` remains a vital sanity check for biological feasibility.
* **`start` / `end`**: The exact coordinate mapping of the binding site on the viral genome. Because ViroMiR uses interval clustering, you will only see the single best hit for a given miRNA within a specific genomic window.

*(Example Interpretation: A `viromir_score` of `0.88` with a `delta_G` of `-20.2` indicates a highly confident prediction with a very stable physical interaction)*



## 📁 Repository Layout
```
ViroMiR/
├── benchmark/       # Reference genomes and data for manuscript figures
├── examples/        # Bundled FASTA files for quick testing
├── tests/           # Automated pytest suite
├── viromir/         # Core Python package & XGBoost model
├── CITATION.cff
├── README.md
├── LICENSE
└── requirements.txt
```

---

## 🛠️ Troubleshooting

- **`FileNotFoundError` or IntaRNA crash:** Ensure IntaRNA is installed and accessible in your system `$PATH` (e.g., `conda install -c bioconda intarna`).
- **Missing modules (xgboost, pandas):** Make sure you ran `pip install -e .` in your active environment.
- **Empty Output CSV:** Check your input FASTA files for formatting issues. The viral sequence must be standard nucleotide characters.
- **Slow runs on large genomes:** Scanning 2,656 human miRNAs against a 30kb+ genome takes significant compute. Use the `--threads` argument to maximize your CPU cores.

---

## 📖 Citation & Contact
If you utilize ViroMiR in your research pipeline, please refer to the `CITATION.cff` file in this repository.

*For bug reports, please open a GitHub Issue containing the exact command used and a snippet of the triggering FASTA sequence.*
