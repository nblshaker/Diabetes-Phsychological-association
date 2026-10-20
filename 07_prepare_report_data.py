"""
STEP 7 - Consolidate results into the JSON/CSV files build_report.js needs.
Run this AFTER scripts 01-06 have produced everything in /results and /charts.
"""
import json
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

with open(os.path.join(RESULTS_DIR, "client_summaries.json")) as f:
    cs = json.load(f)
with open(os.path.join(RESULTS_DIR, "fedavg_depression.json")) as f:
    dep = json.load(f)
with open(os.path.join(RESULTS_DIR, "fedavg_diabetes.json")) as f:
    dia = json.load(f)
with open(os.path.join(RESULTS_DIR, "depression_diabetes_link.json")) as f:
    link = json.load(f)

out = {
    "clients": cs,
    "dep_eval": dep["eval_federated"],
    "dep_local": dep["eval_local_only"],
    "dia_eval": dia["eval_federated"],
    "dia_local": dia["eval_local_only"],
    "link": link,
}
with open(os.path.join(RESULTS_DIR, "report_data.json"), "w") as f:
    json.dump(out, f, indent=2)

pd.read_csv(os.path.join(RESULTS_DIR, "association_diabetes.csv")).round(3).to_json(
    os.path.join(RESULTS_DIR, "assoc_diabetes.json"), orient="records")
pd.read_csv(os.path.join(RESULTS_DIR, "association_depression.csv")).round(3).to_json(
    os.path.join(RESULTS_DIR, "assoc_depression.json"), orient="records")

pd.read_csv(os.path.join(RESULTS_DIR, "adjusted_diabetes_model.csv")).round(4).to_json(
    os.path.join(RESULTS_DIR, "adjusted_diabetes_model.csv.json"), orient="records")
pd.read_csv(os.path.join(RESULTS_DIR, "adjusted_depression_model.csv")).round(4).to_json(
    os.path.join(RESULTS_DIR, "adjusted_depression_model.csv.json"), orient="records")

print("Report data files written to", RESULTS_DIR)
