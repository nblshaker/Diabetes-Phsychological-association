import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHARTS_DIR = os.path.join(BASE_DIR, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

def forest_plot(csv_path, title, fname, exclude_const=True):
    df = pd.read_csv(csv_path)
    if exclude_const:
        df = df[df.code != "const"]
    df = df.sort_values("OR")
    fig, ax = plt.subplots(figsize=(10.5, 0.5 * len(df) + 1.5))
    y = np.arange(len(df))
    lo = (df["OR"] - df["CI_low"]).clip(lower=0)
    hi = (df["CI_high"] - df["OR"]).clip(lower=0)
    colors = ["#C0724A" if c in ("PHQ9_Total", "LBXGH") else "#2E5C8A" for c in df["code"]]
    ax.errorbar(df["OR"], y, xerr=[lo, hi], fmt="none", ecolor="#aaaaaa", capsize=3, zorder=1)
    ax.scatter(df["OR"], y, c=colors, s=70, zorder=2)
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"] + "  (" + df["code"] + ")")
    ax.set_xlabel("Adjusted Odds Ratio (95% CI) \u2014 federated meta-analysis across 3 hospitals")
    ax.set_title(title, fontsize=12)
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(fname, dpi=170)
    plt.close()

forest_plot(os.path.join(RESULTS_DIR, "adjusted_diabetes_model.csv"),
            "Adjusted Predictors of Diabetes (HbA1c \u2265 6.5%)\nKey predictor (orange) = PHQ-9 total, adjusted for demographics + BMI",
            os.path.join(CHARTS_DIR, "07_forest_adjusted_diabetes.png"))

forest_plot(os.path.join(RESULTS_DIR, "adjusted_depression_model.csv"),
            "Adjusted Predictors of Depression (PHQ-9 \u2265 10)\nKey predictor (orange) = HbA1c, adjusted for demographics + BMI",
            os.path.join(CHARTS_DIR, "08_forest_adjusted_depression.png"))

# ---- Unadjusted vs Adjusted comparison for the two key cross-links ----
dia_u = pd.read_csv(os.path.join(RESULTS_DIR, "association_diabetes.csv"))
dia_a = pd.read_csv(os.path.join(RESULTS_DIR, "adjusted_diabetes_model.csv"))
dep_u = pd.read_csv(os.path.join(RESULTS_DIR, "association_depression.csv"))
dep_a = pd.read_csv(os.path.join(RESULTS_DIR, "adjusted_depression_model.csv"))

phq_u = dia_u[dia_u.code=="PHQ9_Total"].iloc[0]
phq_a = dia_a[dia_a.code=="PHQ9_Total"].iloc[0]
hba_u = dep_u[dep_u.code=="LBXGH"].iloc[0]
hba_a = dep_a[dep_a.code=="LBXGH"].iloc[0]

fig, ax = plt.subplots(figsize=(7.5, 4.2))
labels = ["PHQ-9 total \u2192 diabetes\n(unadjusted)", "PHQ-9 total \u2192 diabetes\n(adjusted)",
          "HbA1c \u2192 depression\n(unadjusted)", "HbA1c \u2192 depression\n(adjusted)"]
ors = [phq_u.OR, phq_a.OR, hba_u.OR, hba_a.OR]
los = [phq_u.CI_low, phq_a.CI_low, hba_u.CI_low, hba_a.CI_low]
his = [phq_u.CI_high, phq_a.CI_high, hba_u.CI_high, hba_a.CI_high]
colors = ["#9fb8d1", "#2E5C8A", "#e0b199", "#C0724A"]
y = np.arange(4)
lo_e = [max(0, o-l) for o,l in zip(ors, los)]
hi_e = [max(0, h-o) for o,h in zip(his, ors)]
ax.errorbar(ors, y, xerr=[lo_e, hi_e], fmt="none", ecolor="#999999", capsize=4, zorder=1)
ax.scatter(ors, y, c=colors, s=90, zorder=2)
ax.axvline(1.0, color="gray", linestyle="--")
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.set_xlabel("Odds Ratio (95% CI)")
ax.set_title("Depression \u2194 Diabetes Link: Unadjusted vs. Adjusted")
ax.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "09_unadjusted_vs_adjusted.png"), dpi=170)
plt.close()

print("done")
