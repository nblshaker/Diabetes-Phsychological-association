"""
STEP 5 - Adjusted (multivariable) federated association analysis.

Same federated rule as the univariate version: each client fits its OWN
multivariable logistic regression locally (never sees another client's
rows). Only the resulting coefficients + standard errors are sent to the
coordinator, which combines them per-covariate via fixed-effect
inverse-variance meta-analysis.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CLIENTS = ["Client_A_2015-2016", "Client_B_2017-2020", "Client_C_2021-2023"]

COVARS = ["RIAGENDR", "RIDAGEYR", "DMDEDUC2", "INDFMPIR", "BMXBMI"]

LABELS = {
    "PHQ9_Total": "Total PHQ-9 depression score",
    "LBXGH": "Glycohemoglobin / HbA1c (%)",
    "RIAGENDR": "Gender (1=Male, 2=Female)",
    "RIDAGEYR": "Age in years",
    "DMDEDUC2": "Education level",
    "INDFMPIR": "Income-to-poverty ratio",
    "BMXBMI": "Body Mass Index",
    "const": "Intercept",
}

def load_client(cid):
    return pd.read_csv(os.path.join(DATA_DIR, f"{cid}.csv"))

def local_multivariable_logit(df, xcols, ycol):
    d = df[xcols + [ycol]].dropna()
    X = sm.add_constant(d[xcols].astype(float))
    y = d[ycol].astype(float)
    model = sm.Logit(y, X).fit(disp=0)
    return model.params, model.bse, len(d)

def meta_combine(betas, ses):
    betas = [b for b in betas if np.isfinite(b)]
    ses = [s for s in ses if np.isfinite(s) and s > 0]
    weights = [1/(s**2) for s in ses]
    b_pool = sum(w*b for w,b in zip(weights, betas)) / sum(weights)
    se_pool = np.sqrt(1/sum(weights))
    z = b_pool / se_pool
    p = 2*(1-norm.cdf(abs(z)))
    return {
        "beta": b_pool, "se": se_pool, "OR": np.exp(b_pool),
        "CI_low": np.exp(b_pool-1.96*se_pool), "CI_high": np.exp(b_pool+1.96*se_pool),
        "z": z, "p_value": p
    }

def run_adjusted(target_col, key_predictor, exclude=()):
    xcols = [key_predictor] + [c for c in COVARS if c not in exclude]
    dfs = {c: load_client(c) for c in CLIENTS}

    per_client_params = {c: None for c in ["const", key_predictor] + [c for c in COVARS if c not in exclude]}
    client_fits = {}
    for cid in CLIENTS:
        params, bse, n = local_multivariable_logit(dfs[cid], xcols, target_col)
        client_fits[cid] = {"params": params, "bse": bse, "n": n}

    rows = []
    for var in ["const", key_predictor] + [c for c in COVARS if c not in exclude]:
        betas = [client_fits[c]["params"][var] for c in CLIENTS]
        ses   = [client_fits[c]["bse"][var] for c in CLIENTS]
        pooled = meta_combine(betas, ses)
        row = {"code": var, "label": LABELS.get(var, var), **pooled}
        for c in CLIENTS:
            row[f"{c}_beta"] = client_fits[c]["params"][var]
            row[f"{c}_n"] = client_fits[c]["n"]
        rows.append(row)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    adj_dia = run_adjusted("diabetes_binary", "PHQ9_Total")
    adj_dep = run_adjusted("depression_binary", "LBXGH")

    adj_dia.to_csv(os.path.join(RESULTS_DIR, "adjusted_diabetes_model.csv"), index=False)
    adj_dep.to_csv(os.path.join(RESULTS_DIR, "adjusted_depression_model.csv"), index=False)

    pd.set_option("display.width", 160)
    print("=== ADJUSTED: outcome = diabetes_binary, key predictor = PHQ9_Total ===")
    print(adj_dia[["code","label","OR","CI_low","CI_high","p_value"]].round(4).to_string(index=False))
    print("\n=== ADJUSTED: outcome = depression_binary, key predictor = LBXGH ===")
    print(adj_dep[["code","label","OR","CI_low","CI_high","p_value"]].round(4).to_string(index=False))
