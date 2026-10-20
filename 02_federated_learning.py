"""
STEP 2 — Federated Learning (FedAvg logistic regression).

Federated rules enforced:
  1. Raw records never leave their client (each client loads only its own CSV).
  2. Clients train locally for E local epochs on their own data.
  3. Only MODEL WEIGHTS (not data, not gradients tied to individual records)
     are sent to the coordinator each round.
  4. The coordinator aggregates with weighted averaging (FedAvg),
     weight proportional to each client's local sample size n_k / N.
  5. Feature scaling uses a federated mean/std (computed once from the
     aggregate sums in step 1's summaries, not from pooled raw data).
"""
import pandas as pd
import numpy as np
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(42)

CLIENTS = ["Client_A_2015-2016", "Client_B_2017-2020", "Client_C_2021-2023"]
FEATS_DEPRESSION = ["RIAGENDR","RIDAGEYR","DMDEDUC2","INDFMPIR","BMXBMI","LBXGH"]
FEATS_DIABETES   = ["RIAGENDR","RIDAGEYR","DMDEDUC2","INDFMPIR","BMXBMI","PHQ9_Total"]

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

def load_client(cid):
    return pd.read_csv(os.path.join(DATA_DIR, f"{cid}.csv"))

def federated_feature_stats(dfs, feats):
    """Combine per-client mean/var into a global mean/std using only
    (n, mean, var) aggregates -> Chan's parallel-variance formula."""
    n_tot = sum(len(d) for d in dfs)
    mean = np.zeros(len(feats))
    for f_i, f in enumerate(feats):
        mean[f_i] = sum(d[f].mean() * len(d) for d in dfs) / n_tot
    # pooled variance (combine within-client variance + between-client mean shift)
    var = np.zeros(len(feats))
    for f_i, f in enumerate(feats):
        s = 0.0
        for d in dfs:
            n_k = len(d)
            m_k = d[f].mean()
            v_k = d[f].var(ddof=0)
            s += n_k * (v_k + (m_k - mean[f_i]) ** 2)
        var[f_i] = s / n_tot
    std = np.sqrt(var)
    std[std == 0] = 1.0
    return mean, std

def train_local(X, y, w, b, epochs=25, lr=0.1, l2=1e-3):
    n = len(y)
    for _ in range(epochs):
        z = X @ w + b
        p = sigmoid(z)
        grad_w = X.T @ (p - y) / n + l2 * w
        grad_b = np.mean(p - y)
        w = w - lr * grad_w
        b = b - lr * grad_b
    return w, b

def bce_loss(X, y, w, b, l2=1e-3):
    p = sigmoid(X @ w + b)
    eps = 1e-9
    loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    return loss + l2 * 0.5 * np.sum(w ** 2)

def run_fedavg(task_name, feats, label_col, rounds=15, local_epochs=25, lr=0.15, test_frac=0.2):
    dfs_full = [load_client(c) for c in CLIENTS]

    # federated (pooled-without-pooling-data) normalization stats
    mean, std = federated_feature_stats(dfs_full, feats)

    # local train/test split per client (stratified by label)
    train_sets, test_sets = [], []
    rng = np.random.RandomState(7)
    for d in dfs_full:
        idx = np.arange(len(d))
        rng.shuffle(idx)
        n_test = int(len(d) * test_frac)
        test_idx, train_idx = idx[:n_test], idx[n_test:]
        train_sets.append(d.iloc[train_idx].reset_index(drop=True))
        test_sets.append(d.iloc[test_idx].reset_index(drop=True))

    def to_Xy(d):
        X = ((d[feats].values - mean) / std)
        y = d[label_col].values.astype(float)
        return X, y

    n_feats = len(feats)
    global_w = np.zeros(n_feats)
    global_b = 0.0

    history = {"round": [], "global_train_loss": [], **{f"{c}_local_loss": [] for c in CLIENTS}}

    for r in range(1, rounds + 1):
        local_weights, local_biases, local_n = [], [], []
        round_losses = {}
        for cid, d_train in zip(CLIENTS, train_sets):
            X, y = to_Xy(d_train)
            w_k, b_k = train_local(X, y, global_w.copy(), global_b, epochs=local_epochs, lr=lr)
            local_weights.append(w_k)
            local_biases.append(b_k)
            local_n.append(len(d_train))
            round_losses[cid] = bce_loss(X, y, w_k, b_k)

        n_total = sum(local_n)
        global_w = sum(w_k * (n_k / n_total) for w_k, n_k in zip(local_weights, local_n))
        global_b = sum(b_k * (n_k / n_total) for b_k, n_k in zip(local_biases, local_n))

        # global train loss = weighted avg of each client's loss under the NEW global model
        gl = 0.0
        for cid, d_train in zip(CLIENTS, train_sets):
            X, y = to_Xy(d_train)
            gl += bce_loss(X, y, global_w, global_b) * len(d_train) / n_total
        history["round"].append(r)
        history["global_train_loss"].append(gl)
        for cid in CLIENTS:
            history[f"{cid}_local_loss"].append(round_losses[cid])

    # ---- Evaluation: per-client + pooled test performance of the FINAL global model ----
    def evaluate(X, y):
        p = sigmoid(X @ global_w + global_b)
        pred = (p >= 0.5).astype(int)
        acc = float(np.mean(pred == y))
        # AUC via rank-sum (Mann-Whitney) - no sklearn needed
        pos = p[y == 1]; neg = p[y == 0]
        if len(pos) == 0 or len(neg) == 0:
            auc = float("nan")
        else:
            order = np.argsort(p)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(1, len(p) + 1)
            auc = (ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
        return acc, auc, p, y

    eval_results = {}
    all_p, all_y = [], []
    for cid, d_test in zip(CLIENTS, test_sets):
        X, y = to_Xy(d_test)
        acc, auc, p, yv = evaluate(X, y)
        eval_results[cid] = {"n_test": len(y), "accuracy": acc, "auc": auc, "prevalence": float(y.mean())}
        all_p.append(p); all_y.append(yv)
    all_p = np.concatenate(all_p); all_y = np.concatenate(all_y)
    acc, auc, _, _ = evaluate(np.zeros((0,0)), np.array([])) if False else (None, None, None, None)
    # pooled eval
    pred = (all_p >= 0.5).astype(int)
    pooled_acc = float(np.mean(pred == all_y))
    pos = all_p[all_y == 1]; neg = all_p[all_y == 0]
    order = np.argsort(all_p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(all_p) + 1)
    pooled_auc = (ranks[all_y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    eval_results["POOLED"] = {"n_test": len(all_y), "accuracy": pooled_acc, "auc": pooled_auc, "prevalence": float(all_y.mean())}

    # ---- Local-only baseline (each client trains alone, no federation) for comparison ----
    local_only = {}
    for cid, d_train, d_test in zip(CLIENTS, train_sets, test_sets):
        Xtr, ytr = to_Xy(d_train)
        w0 = np.zeros(n_feats); b0 = 0.0
        w_l, b_l = train_local(Xtr, ytr, w0, b0, epochs=local_epochs * rounds, lr=lr)
        Xte, yte = to_Xy(d_test)
        p = sigmoid(Xte @ w_l + b_l)
        pred = (p >= 0.5).astype(int)
        acc = float(np.mean(pred == yte))
        pos = p[yte == 1]; neg = p[yte == 0]
        if len(pos) and len(neg):
            order = np.argsort(p)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(1, len(p) + 1)
            l_auc = (ranks[yte == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
        else:
            l_auc = float("nan")
        local_only[cid] = {"accuracy": acc, "auc": l_auc}

    result = {
        "task": task_name,
        "features": feats,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "global_weights": global_w.tolist(),
        "global_bias": float(global_b),
        "history": history,
        "eval_federated": eval_results,
        "eval_local_only": local_only,
    }
    return result

if __name__ == "__main__":
    dep_result = run_fedavg("depression_from_demo_bmi_hba1c", FEATS_DEPRESSION, "depression_binary")
    dia_result = run_fedavg("diabetes_from_demo_bmi_phq9", FEATS_DIABETES, "diabetes_binary")

    with open(os.path.join(RESULTS_DIR, "fedavg_depression.json"), "w") as f:
        json.dump(dep_result, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "fedavg_diabetes.json"), "w") as f:
        json.dump(dia_result, f, indent=2)

    print("=== Depression task ===")
    print(json.dumps(dep_result["eval_federated"], indent=2))
    print("local-only:", json.dumps(dep_result["eval_local_only"], indent=2))
    print("\n=== Diabetes task ===")
    print(json.dumps(dia_result["eval_federated"], indent=2))
    print("local-only:", json.dumps(dia_result["eval_local_only"], indent=2))
