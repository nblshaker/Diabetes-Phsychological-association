# Federated Learning: Depression <-> Diabetes (NHANES) -  with DP-SGD

Each NHANES cycle file is one hospital (Hospital A 2015-16, B 2017-20, C 2021-23). Raw rows never leave a hospital.

## Setup (once)
1. Put the three Excel files in `input/`:  Merged_2015-2016.xlsx, Merged_2017-2020.xlsx, Merged_2021-2023.xlsx
2. `pip install pandas numpy scipy scikit-learn matplotlib openpyxl statsmodels`
3. `npm install docx`  (only for the optional Word report, step build_report.js)

## Run in order (from inside this folder)
    python 01_prepare_data.py            # per-hospital cleaning                      
    python 02_federated_learning.py      # FedAvg, NO privacy  -> original results    
    python 03_association_analysis.py    # federated associations (meta-analysis)    
    python 04_charts.py                  #                                            
    python 05_adjusted_association.py    # adjusted associations                      
    python 06_adjusted_charts.py         #                                            
    python 07_prepare_report_data.py     #                                            
    python 02b_dp_federated_learning.py  #  FedAvg + DP-SGD (about 5 min on one CPU core)
    python 08_dp_charts.py               #  DP charts + results/dp_summary.csv
    python tests_dp.py                   #  13 correctness tests for the DP code
  
