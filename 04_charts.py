import json
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHARTS_DIR = os.path.join(BASE_DIR, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

CLIENTS = ["Client_A_2015-2016", "Client_B_2017-2020", "Client_C_2021-2023"]
LEGEND_LABELS = ["Hospital A (2015-2016)", "Hospital B (2017-2020)", "Hospital C (2021-2023)"]
LABELS = ["Hospital A\n(2015-2016 cohort)", "Hospital B\n(2017-2020 cohort)", "Hospital C\n(2021-2023 cohort)"]
COLORS = ["#2E5C8A", "#4E8098", "#7EB09B"]

with open(os.path.join(RESULTS_DIR, "client_summaries.json")) as f:
    summaries = json.load(f)
with open(os.path.join(RESULTS_DIR, "fedavg_depression.json")) as f:
    dep = json.load(f)
with open(os.path.join(RESULTS_DIR, "fedavg_diabetes.json")) as f:
    dia = json.load(f)

plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "axes.facecolor": "white"})

# ---- Chart 1: FedAvg convergence (both tasks) ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, result, title in zip(axes, [dep, dia], ["Depression model (PHQ-9 \u2265 10)", "Diabetes model (HbA1c \u2265 6.5%)"]):
    h = result["history"]
    ax.plot(h["round"], h["global_train_loss"], color="black", linewidth=2.5, label="Global (FedAvg)")
    for cid, lab, col in zip(CLIENTS, LEGEND_LABELS, COLORS):
        ax.plot(h["round"], h[f"{cid}_local_loss"], color=col, linewidth=1.3, linestyle="--", label=lab)
    ax.set_title(title)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Binary cross-entropy loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "01_fedavg_convergence.png"), dpi=170)
plt.close()

# ---- Chart 2: Prevalence by era ----
dep_prev = [summaries[c]["depression_prevalence"] * 100 for c in CLIENTS]
dia_prev = [summaries[c]["diabetes_prevalence"] * 100 for c in CLIENTS]
x = np.arange(len(CLIENTS))
w = 0.35
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.bar(x - w/2, dep_prev, w, label="Depression (PHQ-9 \u2265 10)", color="#2E5C8A")
ax.bar(x + w/2, dia_prev, w, label="Diabetes (HbA1c \u2265 6.5%)", color="#C0724A")
for i, v in enumerate(dep_prev):
    ax.text(i - w/2, v + 0.2, f"{v:.1f}%", ha="center", fontsize=9)
for i, v in enumerate(dia_prev):
    ax.text(i + w/2, v + 0.2, f"{v:.1f}%", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(LABELS)
ax.set_ylabel("Prevalence (%)")
ax.set_title("Depression & Diabetes Prevalence by Hospital (Federated Client)")
ax.legend()
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "02_prevalence_by_era.png"), dpi=170)
plt.close()

# ---- Chart 3 & 4: Forest plots of associations ----
def forest_plot(csv_path, title, fname, top_n=16):
    df = pd.read_csv(csv_path)
    df = df.sort_values("OR")
    fig, ax = plt.subplots(figsize=(10.5, 0.42 * len(df) + 1.4))
    y = np.arange(len(df))
    lo = (df["OR"] - df["CI_low"]).clip(lower=0)
    hi = (df["CI_high"] - df["OR"]).clip(lower=0)
    ax.errorbar(df["OR"], y, xerr=[lo, hi],
                fmt="o", color="#2E5C8A", ecolor="#8fa9c2", capsize=3, markersize=6)
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"] + "  (" + df["code"] + ")")
    ax.set_xlabel("Odds Ratio (95% CI) \u2014 federated meta-analysis across 3 hospitals")
    ax.set_title(title, fontsize=13)
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(fname, dpi=170)
    plt.close()

forest_plot(os.path.join(RESULTS_DIR, "association_diabetes.csv"),
            "Association of All Codes with Diabetes (HbA1c \u2265 6.5%)",
            os.path.join(CHARTS_DIR, "03_forest_diabetes.png"))

forest_plot(os.path.join(RESULTS_DIR, "association_depression.csv"),
            "Association of All Codes with Depression (PHQ-9 \u2265 10)",
            os.path.join(CHARTS_DIR, "04_forest_depression.png"))

# ---- Chart 5: Federated vs Local-only model comparison ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, result, title in zip(axes, [dep, dia], ["Depression model", "Diabetes model"]):
    fed_auc = [result["eval_federated"][c]["auc"] for c in CLIENTS]
    local_auc = [result["eval_local_only"][c]["auc"] for c in CLIENTS]
    x = np.arange(len(CLIENTS))
    w = 0.35
    ax.bar(x - w/2, local_auc, w, label="Local-only (no federation)", color="#B0B0B0")
    ax.bar(x + w/2, fed_auc, w, label="Federated (FedAvg)", color="#2E5C8A")
    ax.set_xticks(x); ax.set_xticklabels(LABELS)
    ax.set_ylabel("AUC (test set)")
    ax.set_ylim(0.5, max(max(fed_auc), max(local_auc)) + 0.08)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "05_federated_vs_local.png"), dpi=170)
plt.close()

# ---- Chart 6: Depression-Diabetes cross link ----
with open(os.path.join(RESULTS_DIR, "depression_diabetes_link.json")) as f:
    link = json.load(f)

fig, ax = plt.subplots(figsize=(6.5, 4))
labels = ["PHQ-9 total score\n\u2192 odds of diabetes\n(per +1 point)", "HbA1c\n\u2192 odds of depression\n(per +1%)"]
ors = [link["PHQ9_Total_predicting_diabetes_binary"]["OR"], link["LBXGH_predicting_depression_binary"]["OR"]]
los = [link["PHQ9_Total_predicting_diabetes_binary"]["CI_low"], link["LBXGH_predicting_depression_binary"]["CI_low"]]
his = [link["PHQ9_Total_predicting_diabetes_binary"]["CI_high"], link["LBXGH_predicting_depression_binary"]["CI_high"]]
y = np.arange(2)
xerr_lo = [max(0, o - l) for o, l in zip(ors, los)]
xerr_hi = [max(0, h - o) for o, h in zip(his, ors)]
ax.errorbar(ors, y, xerr=[xerr_lo, xerr_hi],
            fmt="o", color="#8A3B2E", ecolor="#d3a595", capsize=4, markersize=9)
ax.axvline(1.0, color="gray", linestyle="--")
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.set_xlabel("Odds Ratio (95% CI)")
ax.set_title("Federated Depression \u2194 Diabetes Association")
ax.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "06_depression_diabetes_link.png"), dpi=170)
plt.close()

print("Charts written to", CHARTS_DIR)
print(os.listdir(CHARTS_DIR))
