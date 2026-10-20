"""
STEP 2b - DP-SGD added to the existing federated learning step.   (NEW file: 02_federated_learning.py is untouched.)

What changes relative to step 02 (FedAvg logistic regression, 3 hospitals)
  * Local training at every hospital is DP-SGD instead of plain full-batch gradient descent:
        - Poisson-sample a mini-batch (each patient included independently with probability q)
        - compute each patient's own gradient, clip it to L2 norm <= C       (bounds one patient's influence)
        - add Gaussian noise N(0, sigma^2 C^2) to the summed gradient        (hides one patient's presence)
        - divide by the expected batch size and take the gradient step
  * sigma is chosen per hospital so that ALL of that hospital's local steps together satisfy (epsilon, delta)-DP,
    using a Renyi-DP accountant (dp_accountant.py).
  * The server side is unchanged: weighted FedAvg of the returned weights.

What does NOT change
  * data preparation (step 01), the train/test split, the features, the normalisation, FedAvg itself;
  * the association / meta-analysis steps (03, 05): they never use SGD, so DP-SGD does not touch them (see README).
  * the non-private numbers: setting q=1, sigma=0, C=inf makes this code reproduce step 02 exactly (checked in tests_dp.py).

Privacy accounting convention (conservative): the depression model and the diabetes model are trained on the same
patients, so their privacy costs add up. sigma is calibrated so that the TWO models TOGETHER meet the stated epsilon.
"""
import importlib.util
import json
import os
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

import dp_accountant as acct

# reuse the exact data handling, features and metrics of step 02
_spec = importlib.util.spec_from_file_location("fl02", os.path.join(BASE, "02_federated_learning.py"))
fl02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fl02)
CLIENTS, FEATS = fl02.CLIENTS, {"depression": fl02.FEATS_DEPRESSION, "diabetes": fl02.FEATS_DIABETES}
LABEL = {"depression": "depression_binary", "diabetes": "diabetes_binary"}
ROUNDS, STEPS_PER_ROUND, L2, DELTA = 15, 25, 1e-3, 1e-5     # same schedule as step 02 (15 rounds x 25 local steps)
sigmoid = fl02.sigmoid


def auc_rank(p, y):
    """Same rank-based AUC as step 02."""
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(p); ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(p) + 1)
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ------------------------------------------------------------------ data (identical split to step 02)
def load_task(task, split_seed=7, test_frac=0.2):
    feats, label = FEATS[task], LABEL[task]
    dfs = [fl02.load_client(c) for c in CLIENTS]
    mean, std = fl02.federated_feature_stats(dfs, feats)       # federated normalisation, as in step 02
    rng = np.random.RandomState(split_seed)
    out = []
    for d in dfs:
        idx = np.arange(len(d)); rng.shuffle(idx)
        n_test = int(len(d) * test_frac)
        te, tr = d.iloc[idx[:n_test]], d.iloc[idx[n_test:]]
        f = lambda s: (((s[feats].values - mean) / std), s[label].values.astype(float))
        out.append({"train": f(tr), "test": f(te)})
    return out


def validation_split(clients, seed=100, frac=0.25):
    """Hyper-parameter tuning must not touch the test set: carve a validation part out of each TRAIN set."""
    rng = np.random.RandomState(seed); out = []
    for c in clients:
        X, y = c["train"]; idx = rng.permutation(len(y)); nv = int(len(y) * frac)
        out.append({"train": (X[idx[nv:]], y[idx[nv:]]), "test": (X[idx[:nv]], y[idx[:nv]])})
    return out


# ------------------------------------------------------------------ DP-SGD local training
def local_train(X, y, w, b, steps, lr, q, sigma, C, rng):
    """One hospital's local training.  q=1,sigma=0,C=inf  ->  plain full-batch gradient descent (== step 02)."""
    n, d = X.shape
    w, b = w.copy(), float(b)
    for _ in range(steps):
        if q >= 1.0:
            Xb, yb = X, y
        else:
            m = rng.random(n) < q                                   # Poisson sampling
            Xb, yb = X[m], y[m]
        r = sigmoid(Xb @ w + b) - yb                                # d loss_i / d logit_i
        G = np.hstack([Xb * r[:, None], r[:, None]])                # exact per-patient gradients, shape (batch, d+1)
        if np.isfinite(C):
            norms = np.linalg.norm(G, axis=1)
            G = G * np.minimum(1.0, C / (norms + 1e-12))[:, None]   # clip each patient's gradient
        g = G.sum(0)
        if sigma > 0:
            g = g + rng.normal(0.0, sigma * C, size=d + 1)          # Gaussian mechanism
        g = g / (q * n)                                             # expected batch size
        w = w - lr * (g[:d] + L2 * w)                               # L2 term is data-independent
        b = b - lr * g[d]
    return w, b


def fedavg(clients, lr, B=None, C=np.inf, eps=None, comp=2, seed=0, rounds=ROUNDS, steps=STEPS_PER_ROUND, delta=DELTA):
    """FedAvg across hospitals.  B=None -> full batch.  eps=None -> no noise.  comp = number of models sharing the budget."""
    d = clients[0]["train"][0].shape[1]
    w, b = np.zeros(d), 0.0
    rng = np.random.default_rng(seed)
    info = {"sigma": [], "q": [], "eps": []}
    qs, sigmas = [], []
    for c in clients:
        n = len(c["train"][1]); q = 1.0 if B is None else min(1.0, B / n)
        sg = 0.0
        if eps is not None:
            sg = acct.noise_for_target(eps, delta, q, rounds * steps * comp)
            info["eps"].append(acct.epsilon(q, sg, rounds * steps * comp, delta))
        qs.append(q); sigmas.append(sg)
    info["sigma"], info["q"] = sigmas, qs
    for _ in range(rounds):
        ws, bs, ns = [], [], []
        for c, q, sg in zip(clients, qs, sigmas):
            X, y = c["train"]
            wk, bk = local_train(X, y, w, b, steps, lr, q, sg, C, rng)
            ws.append(wk); bs.append(bk); ns.append(len(y))
        tot = sum(ns)
        w = sum(wk * (n / tot) for wk, n in zip(ws, ns)); b = sum(bk * (n / tot) for bk, n in zip(bs, ns))
    return w, b, info


def evaluate(clients, w, b):
    per, P, Y = {}, [], []
    for name, c in zip(CLIENTS, clients):
        X, y = c["test"]; p = sigmoid(X @ w + b)
        per[name] = {"auc": auc_rank(p, y), "accuracy": float(np.mean((p >= 0.5) == y))}
        P.append(p); Y.append(y)
    P, Y = np.concatenate(P), np.concatenate(Y)
    return {"per_hospital": per, "pooled_auc": auc_rank(P, Y), "pooled_accuracy": float(np.mean((P >= 0.5) == Y))}


# ------------------------------------------------------------------ tuning on validation data only
def tune(task_clients_val, eps_list=(1.0, 4.0)):
    """Pick (lr, C, B) by mean validation AUC over both tasks and eps in eps_list. NOTE: tuning on data is not charged to
    the privacy budget (standard caveat) - stated in the README."""
    best, grid_scores = None, {}
    for lr in (0.05, 0.15, 0.5):
        for C in (0.5, 1.0, 2.0, 5.0):
            for B in (256, 512, 1024):
                s = []
                for task, cl in task_clients_val.items():
                    for eps in eps_list:
                        for seed in (0, 1, 2):
                            w, b, _ = fedavg(cl, lr, B, C, eps, seed=100 + seed)
                            s.append(evaluate(cl, w, b)["pooled_auc"])
                score = float(np.mean(s)); grid_scores[f"lr={lr},C={C},B={B}"] = score
                if best is None or score > best[0]:
                    best = (score, dict(lr=lr, C=C, B=B))
    return best[1], grid_scores


def main(n_seeds=20):
    eps_grid = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    tasks = {t: load_task(t) for t in FEATS}
    val = {t: validation_split(c) for t, c in tasks.items()}
    hp, grid = tune(val)
    print("selected on validation data:", hp, flush=True)
    lr, C, B = hp["lr"], hp["C"], hp["B"]

    results = {"hyperparameters": hp, "delta": DELTA, "rounds": ROUNDS, "steps_per_round": STEPS_PER_ROUND,
               "composition": "epsilon is the TOTAL for the depression + diabetes models together (sigma calibrated on 2x steps)",
               "tuning_grid": grid, "tasks": {}}
    for task, cl in tasks.items():
        R = {}
        # (1) original non-private FedAvg (step 02) reproduced by the DP code path: full batch, no clipping, no noise
        w, b, _ = fedavg(cl, 0.15, None, np.inf, None)
        R["original_nonprivate"] = {"runs": [evaluate(cl, w, b)]}
        # (2) isolate the effect of mini-batch sampling, then of clipping, before any noise
        w, b, _ = fedavg(cl, lr, B, np.inf, None, seed=0); R["minibatch_no_clip_no_noise"] = {"runs": [evaluate(cl, w, b)]}
        runs = []
        for s in range(n_seeds):
            w, b, _ = fedavg(cl, lr, B, C, None, seed=s); runs.append(evaluate(cl, w, b))
        R["minibatch_clip_no_noise"] = {"runs": runs}
        # (3) full DP-SGD at each epsilon
        for eps in eps_grid:
            runs = []
            for s in range(n_seeds):
                w, b, info = fedavg(cl, lr, B, C, eps, seed=s); runs.append(evaluate(cl, w, b))
            R[f"dp_eps{eps:g}"] = {"runs": runs, "sigma_per_hospital": info["sigma"], "q_per_hospital": info["q"],
                                   "eps_total_achieved": info["eps"], "eps_per_model": [
                                       acct.epsilon(q, sg, ROUNDS * STEPS_PER_ROUND, DELTA) for q, sg in zip(info["q"], info["sigma"])]}
        results["tasks"][task] = R
        print(f"[{task}] done", flush=True)
    json.dump(results, open(os.path.join(RESULTS_DIR, "dp_fedavg_results.json"), "w"), indent=1)
    return results


if __name__ == "__main__":
    main()
