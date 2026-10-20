# Federated Learning: Depression <-> Diabetes (NHANES) - now with DP-SGD

Each NHANES cycle file is one hospital (Hospital A 2015-16, B 2017-20, C 2021-23). Raw rows never leave a hospital.

## Setup (once)
1. Put the three Excel files in `input/`:  Merged_2015-2016.xlsx, Merged_2017-2020.xlsx, Merged_2021-2023.xlsx
2. `pip install pandas numpy scipy scikit-learn matplotlib openpyxl statsmodels`
   (DP-SGD needs NOTHING extra: numpy + scipy only. Opacus/PyTorch are NOT required.)
3. `npm install docx`  (only for the optional Word report, step build_report.js)

## Run in order (from inside this folder)
    python 01_prepare_data.py            # per-hospital cleaning                      (unchanged)
    python 02_federated_learning.py      # FedAvg, NO privacy  -> original results    (unchanged)
    python 03_association_analysis.py    # federated associations (meta-analysis)     (unchanged)
    python 04_charts.py                  #                                            (unchanged)
    python 05_adjusted_association.py    # adjusted associations                      (unchanged)
    python 06_adjusted_charts.py         #                                            (unchanged)
    python 07_prepare_report_data.py     #                                            (unchanged)
    python 02b_dp_federated_learning.py  # NEW: FedAvg + DP-SGD (about 5 min on one CPU core)
    python 08_dp_charts.py               # NEW: DP charts + results/dp_summary.csv
    python tests_dp.py                   # NEW: 13 correctness tests for the DP code
    node build_report.js                 # optional Word report (does not include the DP section yet)

## New files
- `dp_accountant.py`            privacy accounting (Renyi DP -> epsilon, delta); finds the noise level for a target epsilon
- `02b_dp_federated_learning.py` DP-SGD local training + FedAvg; the ORIGINAL 02 file is not modified
- `08_dp_charts.py`, `tests_dp.py`

## Does DP-SGD change the earlier results?
- Steps 01, 03-07 and the original 02 outputs: NO. Verified byte-for-byte / cell-for-cell.
- DP is an extra, separate set of results (results/dp_fedavg_results.json). With DP switched off, 02b reproduces 02 exactly.
- Association results (03, 05) do not use gradient descent, so DP-SGD cannot apply to them. Each hospital still shares an
  un-noised coefficient + standard error for those. Say this explicitly in the thesis: DP protects the MODEL TRAINING only.

## Privacy guarantee - exactly what it means
- Example-level (epsilon, delta)-DP per hospital, delta = 1e-5, for everything a hospital sends over all 15 rounds.
- epsilon is the TOTAL for the depression and diabetes models together (they use the same patients, so costs add).
- Not covered: the association statistics above; hyper-parameter tuning (done on validation data, not charged to epsilon);
  no secure aggregation is included in this version.
- NOTE: this pipeline uses the PHQ9_Total column exactly as supplied, so numbers match your thesis (N = 19,097).
  That column is defined inconsistently across the three files (see earlier audit); this has not been changed here.
