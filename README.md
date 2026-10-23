# Client-Level Differential Privacy and Quantization in Cross-Silo Federated Clinical Risk Prediction

Code for the paper **"How Much Does Privacy Cost? An Empirical Study of Client-Level
Differential Privacy and Quantization in Cross-Silo Federated Clinical Risk Prediction"**
(Muslims in ML Workshop, NeurIPS 2026).

The repository contains the full evaluation pipeline in a single notebook,
`qdp_fed_pipeline.ipynb`. It trains and compares FedAvg, FedProx, DP-FedProx
(client-level DP) and Q-DP-FedProx (DP plus 8-bit update quantization) on three
tabular risk-prediction datasets, with Rényi-DP accounting, and reproduces every
table and figure in the paper. It is an evaluation pipeline, not a new DP algorithm.

| Dataset | Source | Clients |
|---|---|---|
| Fed-Heart-Disease | FLamby (UCI Heart Disease) | 4 real clinical sites |
| Fed-TCGA-BRCA | FLamby (TCGA breast cancer clinical records), 5-year event label | 6 real clinical sites |
| Wisconsin Diagnostic Breast Cancer | scikit-learn (bundled) | 5 synthetic clients; random, sorted and Dirichlet partitions |

## Repository layout

```
.
├── qdp_fed_pipeline.ipynb   # the full pipeline (setup, models, DP accounting, evaluation, controls, audit)
├── qdp_fed_results/         # outputs of the run reported in the paper (tables, tests, controls, audit, figures)
├── requirements.txt
├── README.md
└── LICENSE
```

`qdp_fed_results/` holds the summary tables, per-seed results, Wilcoxon tests, controls, audit files
and figures from the run reported in the paper, so the numbers can be checked without re-running
anything. Re-running the notebook writes to the same folder and overwrites these files.

Running the notebook also creates `heart_disease_dataset/` and a cloned `FLamby/` folder.

## Requirements

- Python 3.9 or newer is recommended
- CPU only. No GPU is needed; all models are logistic regression or a Random Forest on tabular data.
- Internet access for the first run (to clone FLamby and download Fed-Heart-Disease).
- The pipeline has been run on Kaggle Notebooks (CPU) and on a MacBook Air (M1, 2020).
  The full run takes a few minutes.

## Setup

```bash
git clone https://github.com/suyaibalsifat/dp-fed-clinical-risk.git
cd dp-fed-clinical-risk
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook qdp_fed_pipeline.ipynb
```

The Wisconsin dataset needs nothing else. For the two FLamby datasets, the first
notebook cells do the following for you (or you can run the commands by hand):

```bash
git clone https://github.com/owkin/FLamby.git
pip install -e "FLamby[heart,tcga]"
python FLamby/flamby/datasets/fed_heart_disease/dataset_creation_scripts/download.py \
       --output-folder ./heart_disease_dataset
```

The Heart download asks you to accept a data-licence prompt once (answer `y`). Run that
notebook cell on its own, in the kernel, so the prompt can reach you. Fed-TCGA-BRCA ships
inside the FLamby repository and prompts the same way the first time it is loaded.

If FLamby is not installed, the pipeline skips the Heart and BRCA experiments and runs
Wisconsin only.

## Reproducing the paper

Run the notebook top to bottom. The cells appear in this order:

| Step | Notebook section | What it does | Paper |
|---|---|---|---|
| 1 | Setup | Installs FLamby and downloads Fed-Heart-Disease | n/a |
| 2 | Smoke test | `run_pipeline(seeds=1, rounds=3)`, a quick end-to-end check written to `./_smoke_test` | n/a |
| 3 | Full run | `run_pipeline(seeds=5, rounds=30)` on all three datasets, written to `./qdp_fed_results` | Table 1; Figs 2 to 4 |
| 4 | Wilcoxon results | Prints the paired Wilcoxon tests (`*_significance.txt`) | Appendix Table 3 |
| 5 | Matched-partition control | Gives Wisconsin the client sizes and label skew of Fed-Heart-Disease | Table 2 (variants A to C) |
| 6 | IID re-partition control | Pools Heart and BRCA and re-splits them into IID clients | Table 2 (IID rows) |
| 7 | Covariate audit | Population-level attribution and per-race metrics for the BRCA FedProx model | Section 5; Appendix Table 4 |
| 8 | Zip and preview | Archives `qdp_fed_results/` | n/a |

Figure 1 of the paper is a hand-drawn schematic of the pipeline. The notebook also writes an
automatically generated schematic, `fig1_system_threat_model.png`; Figures 2 to 4 are produced by the
full run (`*_fig_privacy_utility_comm.png`, `wisconsin_fig_noniid_robustness.png`,
`brca_FedProx_fig_local_explanations.png`).

All experiments use 5 seeds (seeds 0 to 4), 30 communication rounds, 5 local epochs, learning
rate 0.1, FedProx weight 0.05, clipping norm 1.0, noise multiplier 1.5 and 8-bit quantization.
The privacy budget is computed at delta = 1e-5 and comes out at epsilon of about 24.2 for all
three datasets. These values are set in the `CONFIGS` object and `run_pipeline` arguments, and
the run is deterministic for a given seed. Library versions can change the last printed digit.

To run a single dataset or change the settings:

```python
run_pipeline(datasets=("wisconsin",), seeds=5, rounds=30, output_dir="./qdp_fed_results")
```

## Outputs

Everything is written to `qdp_fed_results/`.

| File | Contents |
|---|---|
| `{heart,brca,wisconsin}_summary.csv` | Per-model mean and standard deviation across seeds of every metric, plus epsilon and uplink MB per round (Table 1) |
| `{heart,brca,wisconsin}_raw_results.csv` | One row per (model, seed) |
| `{heart,brca,wisconsin}_significance.txt` | Paired Wilcoxon tests down the chain FedAvg, FedProx, DP-FedProx, Q-DP-FedProx |
| `{heart,brca}_privacy_utility_sweep.csv` | Noise-multiplier sweep (Fig 2) |
| `wisconsin_noniid_robustness.csv` | Random, sorted and Dirichlet partitions (Fig 3) |
| `wisconsin_matched_partition.csv` | Matched-partition control (Table 2) |
| `iid_repartition_control.csv` | IID re-partition control (Table 2) |
| `brca_global_attribution.csv`, `brca_group_audit.csv` | Covariate audit (Section 5, Appendix Table 4) |
| `*_fig_*.png` | Per-dataset figures: client prevalence, main comparison, convergence, ROC, calibration, confusion matrices, local explanations |
| `manifest.json` | Seeds, rounds and the full model configuration used for the run |

## Method in brief

- **Local training.** Class-weighted, L2- and FedProx-regularized logistic regression, 5 local epochs per round.
- **Client-level DP.** Each client's update is clipped to L2 norm `C` and perturbed with independent
  Gaussian noise of standard deviation `sigma * C / sqrt(K)`. After averaging over the `K` clients the
  aggregate noise has standard deviation `sigma * C / K`, so `sigma` is the effective noise multiplier
  of the aggregated update.
- **Accounting.** Rényi-DP of the Gaussian mechanism, composed over rounds and converted to
  (epsilon, delta)-DP by minimizing over the order. No subsampling amplification is claimed.
- **Quantization.** Per-array uniform 8-bit quantization of the noised update, with the scale and
  minimum sent alongside the payload.
- **Privacy unit and threat model.** One hospital (client). The guarantee holds for the aggregate
  update and assumes the server sees only the sum. Secure aggregation is not implemented.

## Troubleshooting

- `FLamby not installed` is printed and Heart and BRCA are skipped: FLamby is not on PyPI, so
  `pip install flamby` does not work. Install it from GitHub as shown above.
- `EOFError` during the Heart download: the download script was run through `!` or a subprocess.
  Run it as a normal notebook cell (`%run ...`) so the licence prompt can be answered.
- `np.NaN` errors with NumPy 2: the notebook restores the alias before importing FLamby.
  If you use the code outside the notebook, add `np.NaN = np.nan` first.
- The Centralized-RF explanation figure is skipped when `shap` is not installed.

## Data and licences

No new patient data was collected. The datasets are used under their original licences and access
terms: UCI Heart Disease and UCI Breast Cancer Wisconsin (Diagnostic), both CC BY 4.0, and
TCGA-BRCA clinical fields from the GDC Open Access tier. Controlled-access TCGA data is not used.
Please follow the FLamby and GDC terms when downloading the data.

## Citation

```bibtex
@inproceedings{sifat2026privacycost,
  title     = {How Much Does Privacy Cost? An Empirical Study of Client-Level Differential Privacy
               and Quantization in Cross-Silo Federated Clinical Risk Prediction},
  author    = {Al Sifat, Suyaib and Khan, Noor Ahmad Faiz and Uddin, Khandaker Mohammad Mohi},
  booktitle = {Muslims in ML Workshop at NeurIPS},
  year      = {2026}
}
```

## License

Released under the MIT License (see `LICENSE`).
