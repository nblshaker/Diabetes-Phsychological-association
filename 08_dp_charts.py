"""STEP 8 - charts + summary table for the DP-SGD results (run after 02b)."""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
RES, CH = os.path.join(BASE, "results"), os.path.join(BASE, "charts")
os.makedirs(CH, exist_ok=True)
R = json.load(open(os.path.join(RES, "dp_fedavg_results.json")))
H = ["Client_A_2015-2016", "Client_B_2017-2020", "Client_C_2021-2023"]
HL = ["Hospital A\n(2015-2016)", "Hospital B\n(2017-2020)", "Hospital C\n(2021-2023)"]
EPS = [0.5, 1, 2, 4, 8, 16]
TITLE = {"depression": "Depression model (PHQ-9 \u2265 10)", "diabetes": "Diabetes model (HbA1c \u2265 6.5%)"}
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Serif"})


def stat(task, key, f=lambda r: r["pooled_auc"]):
    v = np.array([f(r) for r in R["tasks"][task][key]["runs"]]); return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


# ---- Figure 1: privacy-utility curve
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, task in zip(axes, ["depression", "diabetes"]):
    m = np.array([stat(task, f"dp_eps{e:g}")[0] for e in EPS]); s = np.array([stat(task, f"dp_eps{e:g}")[1] for e in EPS])
    base = stat(task, "original_nonprivate")[0]
    ax.plot(range(len(EPS)), m, "o-", color="#2E5C8A", lw=2, label="Federated + DP-SGD")
    ax.fill_between(range(len(EPS)), m - s, m + s, color="#2E5C8A", alpha=.2, label="\u00b1 1 SD (20 noise seeds)")
    ax.axhline(base, color="#C0724A", ls="--", lw=1.8, label="Original, no privacy")
    ax.set_xticks(range(len(EPS))); ax.set_xticklabels([f"{e:g}" for e in EPS])
    ax.set_xlabel("Total privacy budget \u03b5 (smaller = stronger privacy)"); ax.set_ylabel("Pooled test AUC")
    ax.set_title(TITLE[task], fontsize=10.5); ax.grid(alpha=.25)
    pad = max(0.01, (base - m.min()) * 1.8); ax.set_ylim(base - pad - .005, base + .006)
axes[0].legend(fontsize=8, loc="lower right")
plt.suptitle("Privacy\u2013utility trade-off of DP-SGD in the federated pipeline (\u03b4 = 1e-5)", fontsize=11, y=1.02)
plt.tight_layout(); plt.savefig(os.path.join(CH, "08_dp_privacy_utility.png"), dpi=200, bbox_inches="tight"); plt.close()

# ---- Figure 2: per-hospital AUC, original vs DP
fig, axes = plt.subplots(1, 2, figsize=(11, 4)); keys = [("original_nonprivate", "Original (no privacy)", "#C0724A"), ("dp_eps8", "DP \u03b5 = 8", "#7FA7C9"),
                                                         ("dp_eps1", "DP \u03b5 = 1", "#2E5C8A"), ("dp_eps0.5", "DP \u03b5 = 0.5", "#1B3350")]
for ax, task in zip(axes, ["depression", "diabetes"]):
    w = 0.2
    for i, (k, lab, col) in enumerate(keys):
        vals = [stat(task, k, lambda r, h=h: r["per_hospital"][h]["auc"])[0] for h in H]
        errs = [stat(task, k, lambda r, h=h: r["per_hospital"][h]["auc"])[1] for h in H]
        ax.bar(np.arange(3) + (i - 1.5) * w, vals, w, yerr=errs, color=col, label=lab, capsize=2)
    ax.set_xticks(range(3)); ax.set_xticklabels(HL, fontsize=8); ax.set_ylabel("Test AUC"); ax.set_title(TITLE[task], fontsize=10.5)
    lo = min(stat(task, "dp_eps0.5", lambda r, h=h: r["per_hospital"][h]["auc"])[0] for h in H); ax.set_ylim(lo - .05, None); ax.grid(axis="y", alpha=.25)
axes[0].legend(fontsize=8, ncol=2)
plt.tight_layout(); plt.savefig(os.path.join(CH, "09_dp_per_hospital.png"), dpi=200); plt.close()

# ---- Summary table
rows = []
for task in ["depression", "diabetes"]:
    for key, v in R["tasks"][task].items():
        a = [r["pooled_auc"] for r in v["runs"]]
        rows.append({"task": task, "setting": key, "n_runs": len(a), "pooled_auc_mean": np.mean(a), "pooled_auc_sd": np.std(a, ddof=1) if len(a) > 1 else 0.0,
                     "sigma_A": v["sigma_per_hospital"][0] if "sigma_per_hospital" in v else None,
                     "sigma_B": v["sigma_per_hospital"][1] if "sigma_per_hospital" in v else None,
                     "sigma_C": v["sigma_per_hospital"][2] if "sigma_per_hospital" in v else None,
                     "eps_total_both_models": max(v["eps_total_achieved"]) if "eps_total_achieved" in v else None,
                     "eps_per_single_model": max(v["eps_per_model"]) if "eps_per_model" in v else None})
pd.DataFrame(rows).round(4).to_csv(os.path.join(RES, "dp_summary.csv"), index=False)
print("wrote charts/08_dp_privacy_utility.png, charts/09_dp_per_hospital.png, results/dp_summary.csv")
