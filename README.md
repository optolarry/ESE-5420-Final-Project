# ESE 5420 Final Project

Code and summary results for the course paper *Neighborhood Information and Regularization in New York City House Price Prediction* (ESE 5420 Data Science, University of Pennsylvania, Fall 2026).

The paper asks how much neighborhood indicators improve prediction of log sale price beyond building attributes, and when ridge or lasso helps, on one year of one- to three-family house sales in Brooklyn and Queens.

## Contents

| Path | What it is |
|---|---|
| `baseline_checkpoint_v2.py`, `baseline_v2.txt`, `baseline_v2.png` | Baseline checkpoint (9 November): mean, gross-area, OLS and ridge, scored by 5-fold cross-validation on sales before July 2026. |
| `analysis/run_analysis.py` | Main analysis. Loads the raw spreadsheets, applies the filters, fits all nine models on sales before July 1, 2026, then scores them once on July and August 2026. Also produces the learning curves and robustness checks. |
| `analysis/protocol_frozen.json` | Feature sets, penalty grids, training sizes and seeds, saved before the test set was scored. |
| `analysis/model_results.csv` | Cross-validation and test RMSE with bootstrap intervals for all nine models (Table 1 of the paper). |
| `analysis/paired_comparisons.csv` | Paired bootstrap differences between models (Table 2). |
| `analysis/learning_runs.csv`, `learning_intervals.csv`, `learning_paired_comparisons.csv` | Learning-curve results for every training size and seed (Figure 2 and Appendix A). |
| `analysis/development_folds.csv`, `time_ordered_sensitivity.csv`, `diagnostics.json`, `pretest_fits.json`, `data_audit.json` | Per-fold scores, robustness checks, fitted-model summaries, and the filter counts with SHA-256 digests of the raw files. |
| `nonlinear/run_nonlinear.py`, `summarize.py` | Nonlinear extension: random forest, gradient boosting and a spline-plus-ridge model against OLS, by nested cross-validation on sales before July 2026 only. These models were not scored on the test set. |
| `nonlinear/protocol.json`, `model_summary.csv`, `paired_vs_OLS.csv`, `fold_results.csv` | Search grids and results for the extension. |

## Data

The data are the New York City Department of Finance Rolling Sales files for Brooklyn and Queens:
https://www.nyc.gov/site/finance/property/property-rolling-sales-data.page

The raw files are not included here. The site publishes a moving twelve-month window, so a download today covers different months from the snapshot used in the paper (September 2, 2025 to August 31, 2026, downloaded October 2026). `analysis/data_audit.json` records the SHA-256 digests of the snapshot files.

To rerun, create a `datasets/` folder at the top of the repository containing `rollingsales_brooklyn.xlsx` and `rollingsales_queens.xlsx` (and `houses_clean.csv` for the extension), then run the scripts with Python 3 with numpy, scipy, pandas, scikit-learn, threadpoolctl, matplotlib and openpyxl installed. The only change from the scripts as originally run is that the data paths are now relative to the repository.

Per-sale prediction files are not included because they carry property identifiers; the repository holds aggregate results only.

## Seeds

Outer 5-fold cross-validation uses seed 42; inner tuning folds use seeds 43 to 47; learning-curve subsets use seeds 42 to 46; bootstrap intervals use 2,000 resamples.
