"""
02_dual_signature_stratify.py — 2x2 patient classification and survival analysis

Creates 4 patient groups based on median split of both risk scores:
  A: Concordant High (high ROS + high altitude)
  B: Ferroptosis-dominant (high ROS + low altitude)
  C: Hypoxia-dominant (low ROS + high altitude)
  D: Concordant Low (low ROS + low altitude)

Tests whether discordant groups have intermediate survival.
"""
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
FIGS = os.path.join(BASE, "results", "figures", "main")
SFIGS = os.path.join(BASE, "results", "figures", "supplementary")
TABLES = os.path.join(BASE, "results", "tables")

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA & CLASSIFY
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("1. LOADING DATA & CREATING 2x2 CLASSIFICATION")
print("=" * 70)

df = pd.read_csv(os.path.join(DATA, "tcga_convergence_master.csv"))
df = df.dropna(subset=["OS_months", "OS_event"]).copy()
print(f"Patients with survival data: {len(df)}")

ros_med = df["ros_risk_score"].median()
alt_med = df["alt_risk_score"].median()
print(f"ROS risk median: {ros_med:.4f}")
print(f"Altitude risk median: {alt_med:.4f}")

# 2x2 classification
conditions = [
    (df["ros_risk_score"] >= ros_med) & (df["alt_risk_score"] >= alt_med),  # A
    (df["ros_risk_score"] >= ros_med) & (df["alt_risk_score"] < alt_med),   # B
    (df["ros_risk_score"] < ros_med) & (df["alt_risk_score"] >= alt_med),   # C
    (df["ros_risk_score"] < ros_med) & (df["alt_risk_score"] < alt_med),    # D
]
labels = [
    "A: Concordant High",
    "B: Ferroptosis-dominant",
    "C: Hypoxia-dominant",
    "D: Concordant Low",
]
df["dual_group"] = np.select(conditions, labels, default="Unknown")

print("\nGroup distribution:")
for label in labels:
    grp = df[df["dual_group"] == label]
    n_events = int(grp["OS_event"].sum())
    print(f"  {label}: n={len(grp)}, events={n_events} ({100*n_events/len(grp):.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# 2. KAPLAN-MEIER SURVIVAL BY 4 GROUPS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. KAPLAN-MEIER SURVIVAL BY DUAL GROUP")
print("=" * 70)

colors = {
    "A: Concordant High": "#d62728",
    "B: Ferroptosis-dominant": "#ff7f0e",
    "C: Hypoxia-dominant": "#9467bd",
    "D: Concordant Low": "#2ca02c",
}

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
kmf = KaplanMeierFitter()

for label in labels:
    grp = df[df["dual_group"] == label]
    kmf.fit(grp["OS_months"], event_observed=grp["OS_event"], label=label)
    kmf.plot_survival_function(ax=ax, ci_show=True, color=colors[label], linewidth=2)

# Log-rank test (4-group comparison)
result = multivariate_logrank_test(df["OS_months"], df["dual_group"], df["OS_event"])
p_overall = result.p_value

ax.set_title(f"Overall Survival by Dual-Signature Classification\n(log-rank p = {p_overall:.2e})",
             fontsize=13, fontweight='bold')
ax.set_xlabel("Time (months)", fontsize=12)
ax.set_ylabel("Overall Survival Probability", fontsize=12)
ax.legend(fontsize=10, loc="lower left")
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig2a_km_dual_groups.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"Overall 4-group log-rank p = {p_overall:.2e}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. PAIRWISE LOG-RANK TESTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. PAIRWISE SURVIVAL COMPARISONS")
print("=" * 70)

pairwise_results = []
for i, l1 in enumerate(labels):
    for l2 in labels[i+1:]:
        g1 = df[df["dual_group"] == l1]
        g2 = df[df["dual_group"] == l2]
        lr = logrank_test(g1["OS_months"], g2["OS_months"],
                          event_observed_A=g1["OS_event"],
                          event_observed_B=g2["OS_event"])
        pairwise_results.append({
            "group_1": l1,
            "group_2": l2,
            "n_1": len(g1),
            "n_2": len(g2),
            "events_1": int(g1["OS_event"].sum()),
            "events_2": int(g2["OS_event"].sum()),
            "chi2": lr.test_statistic,
            "p_value": lr.p_value,
        })
        print(f"  {l1} vs {l2}: chi2={lr.test_statistic:.2f}, p={lr.p_value:.4f}")

pairwise_df = pd.DataFrame(pairwise_results)
pairwise_df.to_csv(os.path.join(TABLES, "pairwise_logrank.csv"), index=False)

# ══════════════════════════════════════════════════════════════════════════════
# 4. SCATTER PLOT: ROS vs ALTITUDE RISK
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. DUAL RISK SCORE SCATTER PLOT")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

# Panel A: Colored by dual group
ax = axes[0]
for label in labels:
    grp = df[df["dual_group"] == label]
    ax.scatter(grp["alt_risk_score"], grp["ros_risk_score"],
               c=colors[label], alpha=0.6, s=40, label=label, edgecolors='white', linewidth=0.5)
ax.axhline(ros_med, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.axvline(alt_med, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.set_xlabel("Altitude Risk Score", fontsize=12)
ax.set_ylabel("ROS/Ferroptosis Risk Score", fontsize=12)
ax.set_title("A. Dual-Signature Patient Classification", fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.2)

# Add correlation annotation
rho = stats.spearmanr(df["alt_risk_score"], df["ros_risk_score"])[0]
ax.text(0.95, 0.05, f"Spearman rho = {rho:.3f}", transform=ax.transAxes,
        ha='right', fontsize=10, style='italic',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Panel B: Colored by survival status
ax = axes[1]
alive = df[df["OS_event"] == 0]
dead = df[df["OS_event"] == 1]
ax.scatter(alive["alt_risk_score"], alive["ros_risk_score"],
           c='#2ca02c', alpha=0.4, s=30, label=f"Alive (n={len(alive)})", edgecolors='white', linewidth=0.5)
ax.scatter(dead["alt_risk_score"], dead["ros_risk_score"],
           c='#d62728', alpha=0.6, s=40, label=f"Dead (n={len(dead)})", marker='x', linewidth=1.5)
ax.axhline(ros_med, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.axvline(alt_med, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.set_xlabel("Altitude Risk Score", fontsize=12)
ax.set_ylabel("ROS/Ferroptosis Risk Score", fontsize=12)
ax.set_title("B. Survival Status in Dual-Signature Space", fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig2b_dual_scatter.png"), dpi=300, bbox_inches='tight')
plt.close()
print("Saved scatter plot.")

# ══════════════════════════════════════════════════════════════════════════════
# 5. COX REGRESSION — UNIVARIATE & MULTIVARIATE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. COX REGRESSION ANALYSIS")
print("=" * 70)

# Prepare data for Cox
cox_df = df[["OS_months", "OS_event", "ros_risk_score", "alt_risk_score",
             "dual_group", "gender", "tumor_stage", "tumor_grade", "age_at_diagnosis"]].copy()

# Encode dual group as dummies (reference: D = Concordant Low)
cox_df["group_A"] = (cox_df["dual_group"] == "A: Concordant High").astype(int)
cox_df["group_B"] = (cox_df["dual_group"] == "B: Ferroptosis-dominant").astype(int)
cox_df["group_C"] = (cox_df["dual_group"] == "C: Hypoxia-dominant").astype(int)

# Encode clinical covariates
cox_df["is_male"] = (cox_df["gender"] == "male").astype(int)
stage_map = {"Stage I": 1, "Stage II": 2, "Stage III": 3, "Stage IIIA": 3,
             "Stage IIIB": 3, "Stage IIIC": 3, "Stage IV": 4, "Stage IVA": 4, "Stage IVB": 4}
cox_df["stage_num"] = cox_df["tumor_stage"].map(stage_map)
cox_df["age_years"] = cox_df["age_at_diagnosis"] / 365.25

# --- 5a. Univariate: dual groups only ---
print("\n--- Univariate Cox (Dual Groups vs Concordant Low) ---")
cph_uni = CoxPHFitter()
uni_data = cox_df[["OS_months", "OS_event", "group_A", "group_B", "group_C"]].dropna()
cph_uni.fit(uni_data, duration_col="OS_months", event_col="OS_event")
cph_uni.print_summary()

# --- 5b. Multivariate: dual groups + clinical ---
print("\n--- Multivariate Cox (Dual Groups + Age + Sex + Stage) ---")
# Check which clinical variables have enough data
for col in ["age_years", "is_male", "stage_num"]:
    n_valid = cox_df[col].notna().sum()
    print(f"  {col}: {n_valid}/{len(cox_df)} non-missing")

# Use available covariates — drop stage_num if too sparse
mv_cols = ["OS_months", "OS_event", "group_A", "group_B", "group_C"]
clinical_cols = []
for col in ["age_years", "is_male", "stage_num"]:
    if cox_df[col].notna().sum() > len(cox_df) * 0.5:  # require >50% coverage
        mv_cols.append(col)
        clinical_cols.append(col)
    else:
        print(f"  Excluding {col} (<50% coverage)")

mv_data = cox_df[mv_cols].dropna()
print(f"  Patients in multivariate model: {len(mv_data)} (covariates: {clinical_cols})")

if len(mv_data) >= 30:
    cph_mv = CoxPHFitter()
    cph_mv.fit(mv_data, duration_col="OS_months", event_col="OS_event")
    cph_mv.print_summary()
else:
    print("  WARNING: Too few patients for multivariate model, skipping")
    cph_mv = None

# Save Cox results
cox_results = []
models_to_save = [("Univariate", cph_uni, len(uni_data))]
if cph_mv is not None:
    models_to_save.append(("Multivariate", cph_mv, len(mv_data)))
for model_name, cph, n in models_to_save:
    for var in cph.summary.index:
        row = cph.summary.loc[var]
        cox_results.append({
            "model": model_name,
            "variable": var,
            "n": n,
            "coef": row["coef"],
            "HR": row["exp(coef)"],
            "HR_lower": row["exp(coef) lower 95%"],
            "HR_upper": row["exp(coef) upper 95%"],
            "p_value": row["p"],
        })

cox_df_out = pd.DataFrame(cox_results)
cox_df_out.to_csv(os.path.join(TABLES, "dual_group_cox.csv"), index=False)
print(f"\nSaved: results/tables/dual_group_cox.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 6. MEDIAN SURVIVAL BY GROUP
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("6. MEDIAN SURVIVAL & EVENT RATES")
print("=" * 70)

summary_rows = []
kmf = KaplanMeierFitter()
for label in labels:
    grp = df[df["dual_group"] == label]
    kmf.fit(grp["OS_months"], event_observed=grp["OS_event"])
    med_surv = kmf.median_survival_time_
    surv_1y = float(kmf.predict(12))
    surv_3y = float(kmf.predict(36))
    surv_5y = float(kmf.predict(60))
    summary_rows.append({
        "group": label,
        "n": len(grp),
        "events": int(grp["OS_event"].sum()),
        "event_rate_pct": 100 * grp["OS_event"].mean(),
        "median_survival_months": med_surv,
        "surv_1yr": surv_1y,
        "surv_3yr": surv_3y,
        "surv_5yr": surv_5y,
    })
    print(f"  {label}: n={len(grp)}, events={int(grp['OS_event'].sum())}, "
          f"median={med_surv:.1f}mo, 3yr={surv_3y*100:.1f}%")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(TABLES, "dual_group_summary.csv"), index=False)

# ══════════════════════════════════════════════════════════════════════════════
# 7. CONCORDANCE INDEX COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("7. C-INDEX COMPARISON: DUAL vs INDIVIDUAL SIGNATURES")
print("=" * 70)

from lifelines.utils import concordance_index

ci_ros = concordance_index(df["OS_months"], -df["ros_risk_score"], df["OS_event"])
ci_alt = concordance_index(df["OS_months"], -df["alt_risk_score"], df["OS_event"])

# Combined score (simple average of z-scored scores)
df["combined_score"] = (df["ros_risk_score"] / df["ros_risk_score"].std() +
                        df["alt_risk_score"] / df["alt_risk_score"].std()) / 2
ci_combined = concordance_index(df["OS_months"], -df["combined_score"], df["OS_event"])

print(f"  ROS signature alone:      C-index = {ci_ros:.4f}")
print(f"  Altitude signature alone:  C-index = {ci_alt:.4f}")
print(f"  Combined score:           C-index = {ci_combined:.4f}")
print(f"  Improvement over ROS:     +{ci_combined - ci_ros:.4f}")
print(f"  Improvement over altitude: +{ci_combined - ci_alt:.4f}")

# Save C-index comparison
cindex_df = pd.DataFrame([
    {"signature": "ROS/Ferroptosis (11-gene)", "c_index": ci_ros},
    {"signature": "Altitude (9-gene)", "c_index": ci_alt},
    {"signature": "Combined dual-axis", "c_index": ci_combined},
])
cindex_df.to_csv(os.path.join(TABLES, "cindex_comparison.csv"), index=False)

# ══════════════════════════════════════════════════════════════════════════════
# 8. SAVE UPDATED MASTER WITH GROUP LABELS
# ══════════════════════════════════════════════════════════════════════════════
df.to_csv(os.path.join(DATA, "tcga_convergence_master.csv"), index=False)
print(f"\nUpdated master dataframe with dual_group and combined_score columns.")

print("\n" + "=" * 70)
print("DONE — Script 02 complete")
print("=" * 70)
