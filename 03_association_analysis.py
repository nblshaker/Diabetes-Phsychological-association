"""
STEP 3 — Federated association analysis ("show association with all the codes").

Federated rule enforced: each client fits its OWN univariate logistic
regression locally (coefficient + standard error only). The coordinator
combines the three local estimates with a fixed-effect inverse-variance
meta-analysis. This is the standard way to obtain a single, dataset-wide
association estimate from horizontally-partitioned data WITHOUT ever
pooling raw records - the statistical analogue of FedAvg for classic
(non-gradient) statistics.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CLIENTS = ["Client_A_2015-2016", "Client_B_2017-2020", "Client_C_2021-2023"]

CODES = {
    "DPQ010": "Little interest or pleasure in doing things",
    "DPQ020": "Feeling down, depressed, or hopeless",
    "DPQ030": "Trouble falling/staying asleep, or sleeping too much",
    "DPQ040": "Feeling tired or having little energy",
    "DPQ050": "Poor appetite or overeating",
    "DPQ060": "Feeling bad about yourself / failure",
    "DPQ070": "Trouble concentrating",
    "DPQ080": "Moving/speaking slowly or being fidgety/restless",
    "DPQ090": "Thoughts of being better off dead / self-harm",
    "PHQ9_Total": "Total PHQ-9 depression score",
    "RIAGENDR": "Gender (1=Male, 2=Female)",
    "RIDAGEYR": "Age in years",
    "DMDEDUC2": "Education level",
    "INDFMPIR": "Income-to-poverty ratio",
    "BMXBMI": "Body Mass Index",
    "LBXGH": "Glycohemoglobin / HbA1c (%)",
}

def load_client(cid):
    return pd.read_csv(os.path.join(DATA_DIR, f"{cid}.csv"))

def local_univariate_logit(df, xcol, ycol):
    d = df[[xcol, ycol]].dropna()
    X = sm.add_constant(d[xcol].astype(float))
    y = d[ycol].astype(float)
    try:
        model = sm.Logit(y, X).fit(disp=0)
        b = model.params[xcol]
        se = model.bse[xcol]
        return b, se, len(d)
    except Exception:
        return None, None, len(d)

def meta_combine(estimates):
    """Fixed-effect inverse-variance meta-analysis across clients."""
    betas, ses = [], []
    for b, se, n in estimates:
        if b is not None and se is not None and se > 0 and np.isfinite(b) and np.isfinite(se):
            betas.append(b); ses.append(se)
    if not betas:
        return None
    weights = [1 / (se ** 2) for se in ses]
    b_pool = sum(w * b for w, b in zip(weights, betas)) / sum(weights)
    se_pool = np.sqrt(1 / sum(weights))
    z = b_pool / se_pool
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(z)))
    or_pool = np.exp(b_pool)
    ci_lo = np.exp(b_pool - 1.96 * se_pool)
    ci_hi = np.exp(b_pool + 1.96 * se_pool)
    return {"beta": b_pool, "se": se_pool, "OR": or_pool, "CI_low": ci_lo, "CI_high": ci_hi,
            "z": z, "p_value": p}

def run_association(target_col, exclude_codes=()):
    dfs = {c: load_client(c) for c in CLIENTS}
    rows = []
    for code, label in CODES.items():
        if code == target_col or code in exclude_codes:
            continue
        if code not in dfs[CLIENTS[0]].columns:
            continue
        per_client = []
        for cid in CLIENTS:
            b, se, n = local_univariate_logit(dfs[cid], code, target_col)
            per_client.append((b, se, n))
        pooled = meta_combine(per_client)
        if pooled is None:
            continue
        row = {"code": code, "label": label, **pooled}
        for cid, (b, se, n) in zip(CLIENTS, per_client):
            row[f"{cid}_beta"] = b
            row[f"{cid}_n"] = n
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("p_value")
    return out

if __name__ == "__main__":
    assoc_diabetes = run_association("diabetes_binary", exclude_codes=("LBXGH",))
    assoc_depression = run_association("depression_binary", exclude_codes=("PHQ9_Total",))

    assoc_diabetes.to_csv(os.path.join(RESULTS_DIR, "association_diabetes.csv"), index=False)
    assoc_depression.to_csv(os.path.join(RESULTS_DIR, "association_depression.csv"), index=False)

    print("=== Association with DIABETES (HbA1c >=6.5%) ===")
    print(assoc_diabetes[["code","label","OR","CI_low","CI_high","p_value"]].to_string(index=False))
    print("\n=== Association with DEPRESSION (PHQ-9 >=10) ===")
    print(assoc_depression[["code","label","OR","CI_low","CI_high","p_value"]].to_string(index=False))

    # ---- Core depression <-> diabetes cross-association (federated meta-analysis) ----
    dfs = {c: load_client(c) for c in CLIENTS}
    per_client = []
    for cid in CLIENTS:
        b, se, n = local_univariate_logit(dfs[cid], "PHQ9_Total", "diabetes_binary")
        per_client.append((b, se, n))
    dep_dia_phq_on_dia = meta_combine(per_client)

    per_client2 = []
    for cid in CLIENTS:
        b, se, n = local_univariate_logit(dfs[cid], "LBXGH", "depression_binary")
        per_client2.append((b, se, n))
    dia_on_dep = meta_combine(per_client2)

    with open(os.path.join(RESULTS_DIR, "depression_diabetes_link.json"), "w") as f:
        json.dump({
            "PHQ9_Total_predicting_diabetes_binary": dep_dia_phq_on_dia,
            "LBXGH_predicting_depression_binary": dia_on_dep,
        }, f, indent=2)

    print("\n=== Depression <-> Diabetes cross-link (federated meta-analysis) ===")
    print("PHQ-9 total -> diabetes (per 1-point PHQ-9 increase):", dep_dia_phq_on_dia)
    print("HbA1c -> depression (per 1% HbA1c increase):", dia_on_dep)
