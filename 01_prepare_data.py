"""
STEP 1 — Federated data preparation.

Federated rule enforced here: each client (survey era) cleans and prepares
ITS OWN data independently. No raw records are pooled or written to a shared
file. Only summary statistics (needed for federated normalization) will later
be exchanged as aggregates (sums / counts), never as individual rows.
"""
import pandas as pd
import numpy as np
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input")
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

FILES = {
    "Client_A_2015-2016": os.path.join(INPUT_DIR, "Merged_2015-2016.xlsx"),
    "Client_B_2017-2020": os.path.join(INPUT_DIR, "Merged_2017-2020.xlsx"),
    "Client_C_2021-2023": os.path.join(INPUT_DIR, "Merged_2021-2023.xlsx"),
}

for cid, path in FILES.items():
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}\n"
            f"-> Put your 3 Excel files inside the 'input' folder next to this script, "
            f"named exactly: Merged_2015-2016.xlsx, Merged_2017-2020.xlsx, Merged_2021-2023.xlsx"
        )

DPQ_CODES = ["DPQ010","DPQ020","DPQ030","DPQ040","DPQ050","DPQ060","DPQ070","DPQ080","DPQ090"]
DEMO_CODES = ["RIAGENDR","RIDAGEYR","DMDEDUC2","INDFMPIR","BMXBMI"]

client_summaries = {}

for client_id, path in FILES.items():
    df = pd.read_excel(path)
    df = df.dropna(axis=1, how="all")
    df.columns = [c.strip() for c in df.columns]

    # Harmonize column name differences across NHANES cycles
    if "PHQ9_TOTAL" in df.columns and "PHQ9_Total" not in df.columns:
        df = df.rename(columns={"PHQ9_TOTAL": "PHQ9_Total"})

    keep_cols = ["SEQN"] + DEMO_CODES + DPQ_CODES + ["PHQ9_Total", "LBXGH"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # Only adults with BOTH a valid PHQ-9 total and a valid HbA1c measurement
    # are eligible for this analysis (both modules are adult-only in NHANES,
    # which is why pediatric rows are ~40-55% missing).
    df = df.dropna(subset=["PHQ9_Total", "LBXGH"]).copy()

    # Local (client-side) imputation for remaining covariates - each client
    # imputes using ONLY its own statistics. This is what keeps the
    # preparation "federated": no client statistic is influenced by another
    # client's raw records.
    for col in ["DMDEDUC2", "INDFMPIR", "BMXBMI"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Labels
    df["depression_binary"] = (df["PHQ9_Total"] >= 10).astype(int)   # PHQ-9 >=10: moderate+ depression (standard clinical cutoff)
    df["diabetes_binary"] = (df["LBXGH"] >= 6.5).astype(int)         # HbA1c >=6.5%: ADA diabetes threshold

    df.to_csv(os.path.join(DATA_DIR, f"{client_id}.csv"), index=False)

    n = len(df)
    summary = {
        "client_id": client_id,
        "n_records": int(n),
        "n_depression_pos": int(df["depression_binary"].sum()),
        "n_diabetes_pos": int(df["diabetes_binary"].sum()),
        "depression_prevalence": float(df["depression_binary"].mean()),
        "diabetes_prevalence": float(df["diabetes_binary"].mean()),
        # local feature means/stds -> will be combined into a federated
        # (pooled) mean/std WITHOUT sharing any raw row, via the standard
        # parallel-variance ("Chan's") aggregation formula.
        "feature_stats": {}
    }
    feat_cols = DEMO_CODES + ["PHQ9_Total", "LBXGH"]
    for c in feat_cols:
        summary["feature_stats"][c] = {
            "n": int(df[c].notna().sum()),
            "mean": float(df[c].mean()),
            "var": float(df[c].var(ddof=0)),
        }
    client_summaries[client_id] = summary
    print(client_id, "-> rows kept:", n)

with open(os.path.join(RESULTS_DIR, "client_summaries.json"), "w") as f:
    json.dump(client_summaries, f, indent=2)

print("\nDone. Per-hospital CSV files written locally; only aggregate summaries saved centrally.")
