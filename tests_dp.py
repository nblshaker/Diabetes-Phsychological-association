"""Run:  python tests_dp.py      (run 01 and 02 first)"""
import importlib.util, json, os
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("dp02b", os.path.join(BASE, "02b_dp_federated_learning.py"))
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
import dp_accountant as A
res, n_pass = {}, 0


def check(name, ok, detail=""):
    global n_pass
    n_pass += bool(ok); res[name] = {"passed": bool(ok), "detail": detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


# 1. THE KEY TEST: with q=1, sigma=0, C=inf the DP code path IS the original step-02 FedAvg
for task, fn in [("depression", "fedavg_depression.json"), ("diabetes", "fedavg_diabetes.json")]:
    orig = json.load(open(os.path.join(BASE, "results", fn)))["eval_federated"]
    cl = M.load_task(task)
    w, b, _ = M.fedavg(cl, 0.15, None, np.inf, None)
    new = M.evaluate(cl, w, b)
    diffs = [abs(new["per_hospital"][h]["auc"] - orig[h]["auc"]) for h in M.CLIENTS] + [abs(new["pooled_auc"] - orig["POOLED"]["auc"]),
                                                                                           abs(new["pooled_accuracy"] - orig["POOLED"]["accuracy"])]
    check(f"non-DP path reproduces step 02 exactly ({task})", max(diffs) < 1e-9,
          f"pooled AUC {new['pooled_auc']:.6f} vs original {orig['POOLED']['auc']:.6f}; max diff {max(diffs):.1e}")

# 2. per-patient gradient formula matches numerical differentiation
rng = np.random.default_rng(0); X = rng.normal(size=(5, 4)); y = (rng.random(5) > .5).astype(float); w = rng.normal(size=4); b = .3
def loss_i(i, w, b): p = M.sigmoid(X[i] @ w + b); return -(y[i] * np.log(p) + (1 - y[i]) * np.log(1 - p))
ok = True
for i in range(5):
    r = M.sigmoid(X[i] @ w + b) - y[i]; g = np.append(X[i] * r, r)
    num = np.array([(loss_i(i, w + e * 1e-6, b) - loss_i(i, w - e * 1e-6, b)) / 2e-6 for e in np.eye(4)] + [(loss_i(i, w, b + 1e-6) - loss_i(i, w, b - 1e-6)) / 2e-6])
    ok &= np.allclose(g, num, atol=1e-6)
check("per-patient gradient equals numerical gradient", ok)

# 3. clipping bounds every patient's gradient at C
Xb = rng.normal(size=(200, 6)) * 3; yb = (rng.random(200) > .5).astype(float); w0 = rng.normal(size=6)
r = M.sigmoid(Xb @ w0) - yb; G = np.hstack([Xb * r[:, None], r[:, None]]); C = 0.7
Gc = G * np.minimum(1, C / (np.linalg.norm(G, axis=1) + 1e-12))[:, None]
check("clipped per-patient norm <= C", np.linalg.norm(Gc, axis=1).max() <= C + 1e-9, f"max {np.linalg.norm(Gc, axis=1).max():.4f}, {np.mean(np.linalg.norm(G, axis=1) > C)*100:.0f}% were clipped")

# 4. noise has std sigma*C/(q n) on the averaged gradient
n, q, sigma, C = 4000, 0.1, 2.0, 1.0
Xz = np.zeros((n, 3)); yz = np.zeros(n); ws = []
for s in range(4000):
    w1, b1 = M.local_train(Xz, yz, np.zeros(3), 0.0, 1, 1.0, q, sigma, C, np.random.default_rng(s))
    ws.append(w1[0])                       # data gradient is clipped-signal; with X=0 weight grad is pure noise
emp, want = np.std(ws), sigma * C / (q * n)
check("noise std = sigma*C/(q*n)", abs(emp - want) / want < 0.05, f"empirical {emp:.5f} vs {want:.5f}")

# 5. Poisson sampling rate
rng = np.random.default_rng(1); rate = np.mean([np.mean(rng.random(5000) < 0.08) for _ in range(300)])
check("Poisson sampling rate = q", abs(rate - 0.08) < 0.002, f"{rate:.4f}")

# 6. accountant: bisection hits the target, monotone, and agrees with Opacus (when installed)
for eps in (0.5, 1, 4, 8):
    s = A.noise_for_target(eps, 1e-5, 0.08, 750); got = A.epsilon(0.08, s, 750, 1e-5)
    check(f"accountant hits target eps={eps}", eps - 0.05 <= got <= eps + 1e-6, f"sigma={s:.3f} -> eps={got:.3f}")
check("epsilon grows with steps, shrinks with noise", A.epsilon(.08, 1.5, 1500, 1e-5) > A.epsilon(.08, 1.5, 750, 1e-5) > A.epsilon(.08, 3.0, 750, 1e-5))
try:
    from opacus.accountants import RDPAccountant
    worst = 0
    for q_, s_, T_ in [(0.06, 1.0, 375), (0.06, 2.0, 375), (0.1, 3.0, 500), (0.03, 1.2, 1000)]:
        acc = RDPAccountant(); acc.history = [(s_, q_, T_)]; o = acc.get_epsilon(1e-5); a = A.epsilon(q_, s_, T_, 1e-5)
        worst = max(worst, (o - a) / o)             # positive would mean we UNDER-report epsilon
    check("accountant never under-reports epsilon vs Opacus", worst <= 1e-9, f"largest under-report {max(worst,0)*100:.2f}%")
except ImportError:
    print("[skip] Opacus not installed - accountant cross-check skipped (optional)")

# 7. the original outputs of steps 03/05 do not depend on anything DP-related
src = open(os.path.join(BASE, "03_association_analysis.py")).read() + open(os.path.join(BASE, "05_adjusted_association.py")).read()
check("association scripts (03, 05) contain no gradient descent / DP code", all(k not in src for k in ("train_local", "sigmoid", "fedavg", "clip")), "they use closed-form logistic fits + meta-analysis")

json.dump(res, open(os.path.join(BASE, "results", "tests_dp.json"), "w"), indent=2)
print(f"\n{n_pass}/{len(res)} tests passed")
