"""
05_hmox_immunomodulation.py — HMOX1/HMOX2 heme oxygenase bridge analysis

Characterises the HMOX1/HMOX2 heme oxygenase system as the functional
bridge between the ROS/ferroptosis and altitude/hypoxia signatures.

  HMOX1 = inducible heme oxygenase (NRF2-driven, stress-responsive)
          In ROS signature: risk gene, coeff = +0.042
          Products: CO (anti-inflammatory), biliverdin, free iron (pro-ferroptotic)

  HMOX2 = constitutive heme oxygenase (baseline expression)
          In altitude signature: protective gene, coeff = -0.256

  HMOX1/HMOX2 ratio = shift from constitutive to stress-induced heme metabolism

Outputs:
  - Updated master CSV with hmox_ratio and log2_hmox_ratio columns
  - results/tables/hmox_correlations.csv
  - results/tables/hmox_survival.csv
  - Figures to results/figures/main/
"""
import pandas as pd
import numpy as np
from scipy import stats
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
FIGS = os.path.join(BASE, "results", "figures", "main")
TABLES = os.path.join(BASE, "results", "tables")

os.makedirs(FIGS, exist_ok=True)
os.makedirs(TABLES, exist_ok=True)

# Load paths for expression data
with open(os.path.join(DATA, "paths.json")) as f:
    PATHS = json.load(f)

# ======================================================================
# 1. LOAD DATA & COMPUTE HMOX RATIO
# ======================================================================
print("=" * 70)
print("1. LOADING DATA & COMPUTING HMOX1/HMOX2 RATIO")
print("=" * 70)

df = pd.read_csv(os.path.join(DATA, "tcga_convergence_master.csv"))
df = df.dropna(subset=["OS_months", "OS_event"]).copy()
print(f"Patients with survival data: {len(df)}")

# Verify HMOX columns exist
for col in ["HMOX1", "HMOX2"]:
    assert col in df.columns, f"Missing column: {col}"
    print(f"  {col}: mean={df[col].mean():.3f}, median={df[col].median():.3f}, "
          f"range=[{df[col].min():.3f}, {df[col].max():.3f}]")

# Compute HMOX1/HMOX2 ratio with pseudocount to avoid division by zero
pseudocount = 0.01
df["hmox_ratio"] = (df["HMOX1"] + pseudocount) / (df["HMOX2"] + pseudocount)
df["log2_hmox_ratio"] = np.log2(df["hmox_ratio"])

print(f"\nHMOX1/HMOX2 ratio:")
print(f"  Mean:   {df['hmox_ratio'].mean():.3f}")
print(f"  Median: {df['hmox_ratio'].median():.3f}")
print(f"  Range:  [{df['hmox_ratio'].min():.3f}, {df['hmox_ratio'].max():.3f}]")
print(f"\nlog2(HMOX1/HMOX2):")
print(f"  Mean:   {df['log2_hmox_ratio'].mean():.3f}")
print(f"  Median: {df['log2_hmox_ratio'].median():.3f}")
print(f"  Range:  [{df['log2_hmox_ratio'].min():.3f}, {df['log2_hmox_ratio'].max():.3f}]")

# Compute tertiles
df["hmox_tertile"] = pd.qcut(df["hmox_ratio"], q=3, labels=["Low", "Mid", "High"])
for t in ["Low", "Mid", "High"]:
    sub = df[df["hmox_tertile"] == t]
    print(f"  Tertile {t}: n={len(sub)}, ratio range=[{sub['hmox_ratio'].min():.3f}, "
          f"{sub['hmox_ratio'].max():.3f}]")

# ======================================================================
# 2. CORRELATE HMOX RATIO WITH KEY VARIABLES
# ======================================================================
print("\n" + "=" * 70)
print("2. HMOX RATIO CORRELATIONS WITH SIGNATURE VARIABLES")
print("=" * 70)

correlation_results = []

# Define targets to correlate with
corr_targets = {
    "ros_risk_score": "ROS/Ferroptosis risk score",
    "alt_risk_score": "Altitude risk score",
    "NFE2L2": "NRF2 (NFE2L2) expression",
    "KEAP1": "KEAP1 expression",
}

# Check for optional columns from scripts 03/04
optional_targets = {
    "nrf2_activity": "NRF2 activity score",
    "ferroptosis_vulnerability": "Ferroptosis vulnerability",
}
for col, label in optional_targets.items():
    if col in df.columns:
        corr_targets[col] = label
        print(f"  Found optional column: {col}")
    else:
        print(f"  Optional column not found (script 03/04 not yet run): {col}")

# Also correlate HMOX1 and HMOX2 individually
hmox_vars = {
    "log2_hmox_ratio": "log2(HMOX1/HMOX2) ratio",
    "HMOX1": "HMOX1 expression",
    "HMOX2": "HMOX2 expression",
}

for hmox_name, hmox_label in hmox_vars.items():
    print(f"\n--- Correlations for {hmox_label} ---")
    for target_col, target_label in corr_targets.items():
        if target_col not in df.columns:
            continue
        valid = df[[hmox_name, target_col]].dropna()
        if len(valid) < 10:
            continue
        rho, pval = stats.spearmanr(valid[hmox_name], valid[target_col])
        correlation_results.append({
            "hmox_variable": hmox_label,
            "target_variable": target_label,
            "target_column": target_col,
            "spearman_rho": rho,
            "p_value": pval,
            "n": len(valid),
            "significant": "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns",
        })
        print(f"  vs {target_label:40s}: rho={rho:+.3f}, p={pval:.2e} "
              f"{'***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'}")

# Check for immune checkpoint columns already in master
checkpoint_cols = ["CD274", "PDCD1", "CTLA4", "LAG3", "HAVCR2", "TIGIT",
                   "SIGLEC15", "IDO1", "CD276"]
available_checkpoints = [c for c in checkpoint_cols if c in df.columns]
if available_checkpoints:
    print(f"\n--- HMOX1 vs Immune Checkpoints (n={len(available_checkpoints)}) ---")
    for ckpt in available_checkpoints:
        valid = df[["HMOX1", ckpt]].dropna()
        if len(valid) < 10:
            continue
        rho, pval = stats.spearmanr(valid["HMOX1"], valid[ckpt])
        correlation_results.append({
            "hmox_variable": "HMOX1 expression",
            "target_variable": f"Checkpoint: {ckpt}",
            "target_column": ckpt,
            "spearman_rho": rho,
            "p_value": pval,
            "n": len(valid),
            "significant": "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns",
        })
        print(f"  HMOX1 vs {ckpt:12s}: rho={rho:+.3f}, p={pval:.2e} "
              f"{'***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'}")

# ======================================================================
# 3. SURVIVAL ANALYSIS BY HMOX RATIO TERTILES
# ======================================================================
print("\n" + "=" * 70)
print("3. SURVIVAL ANALYSIS BY HMOX1/HMOX2 RATIO TERTILES")
print("=" * 70)

survival_results = []

# --- 3a. Kaplan-Meier by tertiles ---
tertile_labels = ["Low", "Mid", "High"]
tertile_colors = {"Low": "#2ca02c", "Mid": "#ff7f0e", "High": "#d62728"}

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
kmf = KaplanMeierFitter()

for tert in tertile_labels:
    sub = df[df["hmox_tertile"] == tert]
    kmf.fit(sub["OS_months"], event_observed=sub["OS_event"],
            label=f"{tert} (n={len(sub)})")
    kmf.plot_survival_function(ax=ax, ci_show=True, color=tertile_colors[tert],
                               linewidth=2)
    med_surv = kmf.median_survival_time_
    surv_3y = float(kmf.predict(36))
    survival_results.append({
        "analysis": "KM by HMOX ratio tertile",
        "group": tert,
        "n": len(sub),
        "events": int(sub["OS_event"].sum()),
        "median_survival_months": med_surv,
        "surv_3yr": surv_3y,
    })
    print(f"  Tertile {tert}: n={len(sub)}, events={int(sub['OS_event'].sum())}, "
          f"median={med_surv:.1f}mo, 3yr-surv={surv_3y*100:.1f}%")

# Log-rank test
lr_result = multivariate_logrank_test(df["OS_months"], df["hmox_tertile"],
                                       df["OS_event"])
lr_p = lr_result.p_value
print(f"\n  3-group log-rank p = {lr_p:.2e}")

ax.set_title(f"Overall Survival by HMOX1/HMOX2 Ratio Tertile\n"
             f"(log-rank p = {lr_p:.2e})", fontsize=13, fontweight='bold')
ax.set_xlabel("Time (months)", fontsize=12)
ax.set_ylabel("Overall Survival Probability", fontsize=12)
ax.legend(fontsize=10, loc="lower left")
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig5a_km_hmox_ratio_tertiles.png"),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig5a_km_hmox_ratio_tertiles.png")

# --- 3b. Univariate Cox: log2(HMOX1/HMOX2) continuous ---
print("\n--- Univariate Cox: log2(HMOX1/HMOX2) ratio ---")
cox_uni_data = df[["OS_months", "OS_event", "log2_hmox_ratio"]].dropna()
cph_uni = CoxPHFitter()
cph_uni.fit(cox_uni_data, duration_col="OS_months", event_col="OS_event")
cph_uni.print_summary()

uni_summary = cph_uni.summary.loc["log2_hmox_ratio"]
survival_results.append({
    "analysis": "Univariate Cox (log2 HMOX ratio)",
    "group": "continuous",
    "n": len(cox_uni_data),
    "events": int(cox_uni_data["OS_event"].sum()),
    "HR": uni_summary["exp(coef)"],
    "HR_lower": uni_summary["exp(coef) lower 95%"],
    "HR_upper": uni_summary["exp(coef) upper 95%"],
    "coef": uni_summary["coef"],
    "p_value": uni_summary["p"],
    "concordance_index": cph_uni.concordance_index_,
})
print(f"  HR = {uni_summary['exp(coef)']:.3f} "
      f"(95% CI: {uni_summary['exp(coef) lower 95%']:.3f}-"
      f"{uni_summary['exp(coef) upper 95%']:.3f}), "
      f"p = {uni_summary['p']:.2e}")

# --- 3c. Multivariate Cox: adjusting for age and sex ---
print("\n--- Multivariate Cox: log2(HMOX ratio) + age + sex ---")
cox_mv_data = df[["OS_months", "OS_event", "log2_hmox_ratio",
                   "age_at_diagnosis", "gender"]].copy()
cox_mv_data["age_years"] = cox_mv_data["age_at_diagnosis"] / 365.25
cox_mv_data["is_male"] = (cox_mv_data["gender"] == "male").astype(int)
cox_mv_data = cox_mv_data[["OS_months", "OS_event", "log2_hmox_ratio",
                            "age_years", "is_male"]].dropna()

if len(cox_mv_data) >= 30:
    cph_mv = CoxPHFitter()
    cph_mv.fit(cox_mv_data, duration_col="OS_months", event_col="OS_event")
    cph_mv.print_summary()

    mv_summary = cph_mv.summary.loc["log2_hmox_ratio"]
    survival_results.append({
        "analysis": "Multivariate Cox (log2 HMOX ratio + age + sex)",
        "group": "continuous",
        "n": len(cox_mv_data),
        "events": int(cox_mv_data["OS_event"].sum()),
        "HR": mv_summary["exp(coef)"],
        "HR_lower": mv_summary["exp(coef) lower 95%"],
        "HR_upper": mv_summary["exp(coef) upper 95%"],
        "coef": mv_summary["coef"],
        "p_value": mv_summary["p"],
        "concordance_index": cph_mv.concordance_index_,
    })
    # Also save age and sex results
    for var in ["age_years", "is_male"]:
        row = cph_mv.summary.loc[var]
        survival_results.append({
            "analysis": "Multivariate Cox (covariate)",
            "group": var,
            "n": len(cox_mv_data),
            "events": int(cox_mv_data["OS_event"].sum()),
            "HR": row["exp(coef)"],
            "HR_lower": row["exp(coef) lower 95%"],
            "HR_upper": row["exp(coef) upper 95%"],
            "coef": row["coef"],
            "p_value": row["p"],
        })
    print(f"  Adjusted HR = {mv_summary['exp(coef)']:.3f} "
          f"(95% CI: {mv_summary['exp(coef) lower 95%']:.3f}-"
          f"{mv_summary['exp(coef) upper 95%']:.3f}), "
          f"p = {mv_summary['p']:.2e}")
else:
    print(f"  WARNING: Too few patients ({len(cox_mv_data)}) for multivariate model")

# ======================================================================
# 4. HMOX1 AND IMMUNE MODULATION
# ======================================================================
print("\n" + "=" * 70)
print("4. HMOX1 AND IMMUNE MODULATION ANALYSIS")
print("=" * 70)

# Check if immune cell scores already exist in master (from script 04)
immune_score_cols = [c for c in df.columns if c.startswith("immune_") or
                     c.endswith("_score") and "immune" in c.lower()]

# Define immune marker genes to load from full expression matrix
nk_markers = ["NCAM1", "NKG7", "KLRD1"]
dc_markers = ["ITGAX", "CD1C", "HLA-DRA"]
tcell_markers = ["CD8A", "CD4", "IFNG"]
all_immune_markers = nk_markers + dc_markers + tcell_markers

marker_categories = {}
for g in nk_markers:
    marker_categories[g] = "NK cell"
for g in dc_markers:
    marker_categories[g] = "Dendritic cell"
for g in tcell_markers:
    marker_categories[g] = "T cell"

# Load immune markers from full expression matrix
print("Loading immune marker genes from full expression matrix...")
expr_path = PATHS["expression_full"]
expr_full = pd.read_csv(expr_path, index_col=0)
print(f"  Full expression matrix: {expr_full.shape[0]} genes x {expr_full.shape[1]} samples")

# Extract immune markers
found_markers = [g for g in all_immune_markers if g in expr_full.index]
missing_markers = [g for g in all_immune_markers if g not in expr_full.index]
if missing_markers:
    print(f"  WARNING: Missing markers in expression matrix: {missing_markers}")
print(f"  Found {len(found_markers)}/{len(all_immune_markers)} immune markers")

# Transpose: patients as rows
immune_expr = expr_full.loc[found_markers].T
immune_expr.index.name = "patientId"
immune_expr = immune_expr.reset_index()

# Merge with master dataframe
df_immune = df.merge(immune_expr, on="patientId", how="inner", suffixes=("", "_immune"))
print(f"  Patients with immune marker data: {len(df_immune)}")

# Correlate HMOX1 with each immune marker
immune_corr_results = []
print("\n--- HMOX1 vs Immune Cell Markers ---")
for marker in found_markers:
    col = marker if marker in df_immune.columns else f"{marker}_immune"
    if col not in df_immune.columns:
        continue
    valid = df_immune[["HMOX1", col]].dropna()
    if len(valid) < 10:
        continue
    rho, pval = stats.spearmanr(valid["HMOX1"], valid[col])
    cat = marker_categories.get(marker, "Other")
    immune_corr_results.append({
        "marker": marker,
        "category": cat,
        "spearman_rho": rho,
        "p_value": pval,
        "n": len(valid),
        "significant": "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns",
    })
    print(f"  HMOX1 vs {marker:10s} ({cat:15s}): rho={rho:+.3f}, p={pval:.2e} "
          f"{'***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'}")

# Also correlate log2(HMOX ratio) with immune markers
print("\n--- log2(HMOX1/HMOX2 ratio) vs Immune Cell Markers ---")
for marker in found_markers:
    col = marker if marker in df_immune.columns else f"{marker}_immune"
    if col not in df_immune.columns:
        continue
    valid = df_immune[["log2_hmox_ratio", col]].dropna()
    if len(valid) < 10:
        continue
    rho, pval = stats.spearmanr(valid["log2_hmox_ratio"], valid[col])
    cat = marker_categories.get(marker, "Other")
    immune_corr_results.append({
        "marker": f"ratio_vs_{marker}",
        "category": cat,
        "spearman_rho": rho,
        "p_value": pval,
        "n": len(valid),
        "significant": "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns",
    })
    print(f"  Ratio vs {marker:10s} ({cat:15s}): rho={rho:+.3f}, p={pval:.2e} "
          f"{'***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'}")

# If immune cell score columns exist from script 04, use them too
if immune_score_cols:
    print(f"\n--- HMOX1 vs Immune Cell Scores (from script 04) ---")
    for col in immune_score_cols:
        valid = df[["HMOX1", col]].dropna()
        if len(valid) < 10:
            continue
        rho, pval = stats.spearmanr(valid["HMOX1"], valid[col])
        immune_corr_results.append({
            "marker": col,
            "category": "Immune score",
            "spearman_rho": rho,
            "p_value": pval,
            "n": len(valid),
            "significant": "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns",
        })
        print(f"  HMOX1 vs {col}: rho={rho:+.3f}, p={pval:.2e}")

# ======================================================================
# 5. COMPARE HMOX1, HMOX2, RATIO ACROSS 4 DUAL GROUPS
# ======================================================================
print("\n" + "=" * 70)
print("5. HMOX EXPRESSION ACROSS DUAL-SIGNATURE GROUPS")
print("=" * 70)

group_labels = [
    "A: Concordant High",
    "B: Ferroptosis-dominant",
    "C: Hypoxia-dominant",
    "D: Concordant Low",
]

# Filter to patients in known groups
df_grp = df[df["dual_group"].isin(group_labels)].copy()

group_comparison_results = []
for var, var_label in [("HMOX1", "HMOX1 expression"),
                        ("HMOX2", "HMOX2 expression"),
                        ("hmox_ratio", "HMOX1/HMOX2 ratio"),
                        ("log2_hmox_ratio", "log2(HMOX1/HMOX2)")]:
    groups = [df_grp[df_grp["dual_group"] == g][var].dropna().values
              for g in group_labels]
    stat_kw, p_kw = stats.kruskal(*groups)

    group_means = {}
    for g, vals in zip(group_labels, groups):
        group_means[g] = np.median(vals)

    group_comparison_results.append({
        "variable": var_label,
        "kruskal_wallis_H": stat_kw,
        "p_value": p_kw,
        "median_A": group_means.get("A: Concordant High", np.nan),
        "median_B": group_means.get("B: Ferroptosis-dominant", np.nan),
        "median_C": group_means.get("C: Hypoxia-dominant", np.nan),
        "median_D": group_means.get("D: Concordant Low", np.nan),
    })
    print(f"\n  {var_label}:")
    print(f"    Kruskal-Wallis H={stat_kw:.2f}, p={p_kw:.2e}")
    for g in group_labels:
        sub = df_grp[df_grp["dual_group"] == g][var]
        print(f"    {g}: median={sub.median():.3f}, IQR=[{sub.quantile(0.25):.3f}, "
              f"{sub.quantile(0.75):.3f}]")

# Pairwise Mann-Whitney U tests for HMOX ratio across groups
print("\n--- Pairwise Mann-Whitney U tests for log2(HMOX1/HMOX2) ---")
from itertools import combinations
for g1, g2 in combinations(group_labels, 2):
    vals1 = df_grp[df_grp["dual_group"] == g1]["log2_hmox_ratio"].dropna()
    vals2 = df_grp[df_grp["dual_group"] == g2]["log2_hmox_ratio"].dropna()
    u_stat, u_p = stats.mannwhitneyu(vals1, vals2, alternative='two-sided')
    short1 = g1.split(":")[0].strip()
    short2 = g2.split(":")[0].strip()
    print(f"  {short1} vs {short2}: U={u_stat:.1f}, p={u_p:.4f} "
          f"{'*' if u_p < 0.05 else 'ns'}")

# ======================================================================
# 6. GENERATE FIGURES
# ======================================================================
print("\n" + "=" * 70)
print("6. GENERATING FIGURES")
print("=" * 70)

group_colors = {
    "A: Concordant High": "#d62728",
    "B: Ferroptosis-dominant": "#ff7f0e",
    "C: Hypoxia-dominant": "#9467bd",
    "D: Concordant Low": "#2ca02c",
}

# --- 6a. HMOX1 vs HMOX2 scatter colored by dual group ---
fig, ax = plt.subplots(1, 1, figsize=(8, 7))
for grp_label in group_labels:
    sub = df_grp[df_grp["dual_group"] == grp_label]
    ax.scatter(sub["HMOX2"], sub["HMOX1"],
               c=group_colors[grp_label], alpha=0.6, s=45, label=grp_label,
               edgecolors='white', linewidth=0.5)

# Add identity line
lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1])]
ax.plot(lims, lims, '--', color='gray', alpha=0.4, linewidth=1, label='y = x')

# Annotate correlation
rho_h, p_h = stats.spearmanr(df_grp["HMOX2"], df_grp["HMOX1"])
ax.text(0.05, 0.95, f"Spearman rho = {rho_h:.3f}\np = {p_h:.2e}",
        transform=ax.transAxes, ha='left', va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_xlabel("HMOX2 Expression (constitutive)", fontsize=12)
ax.set_ylabel("HMOX1 Expression (inducible, NRF2-driven)", fontsize=12)
ax.set_title("HMOX1 vs HMOX2 Expression by Dual-Signature Group",
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig5b_hmox1_vs_hmox2_scatter.png"),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig5b_hmox1_vs_hmox2_scatter.png")

# --- 6b. Boxplot: log2(HMOX1/HMOX2) by dual group ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel 1: HMOX1 by group
ax = axes[0]
bp_data_1 = [df_grp[df_grp["dual_group"] == g]["HMOX1"].dropna().values
             for g in group_labels]
bp1 = ax.boxplot(bp_data_1, labels=["A", "B", "C", "D"], patch_artist=True,
                  widths=0.6, showfliers=True,
                  flierprops=dict(marker='o', markersize=3, alpha=0.4))
for patch, grp in zip(bp1['boxes'], group_labels):
    patch.set_facecolor(group_colors[grp])
    patch.set_alpha(0.7)
stat_h1, p_h1 = stats.kruskal(*bp_data_1)
ax.set_title(f"HMOX1 (Inducible)\nKruskal-Wallis p = {p_h1:.2e}",
             fontsize=12, fontweight='bold')
ax.set_ylabel("Expression", fontsize=11)
ax.set_xlabel("Dual-Signature Group", fontsize=11)
ax.grid(True, alpha=0.2, axis='y')

# Panel 2: HMOX2 by group
ax = axes[1]
bp_data_2 = [df_grp[df_grp["dual_group"] == g]["HMOX2"].dropna().values
             for g in group_labels]
bp2 = ax.boxplot(bp_data_2, labels=["A", "B", "C", "D"], patch_artist=True,
                  widths=0.6, showfliers=True,
                  flierprops=dict(marker='o', markersize=3, alpha=0.4))
for patch, grp in zip(bp2['boxes'], group_labels):
    patch.set_facecolor(group_colors[grp])
    patch.set_alpha(0.7)
stat_h2, p_h2 = stats.kruskal(*bp_data_2)
ax.set_title(f"HMOX2 (Constitutive)\nKruskal-Wallis p = {p_h2:.2e}",
             fontsize=12, fontweight='bold')
ax.set_ylabel("Expression", fontsize=11)
ax.set_xlabel("Dual-Signature Group", fontsize=11)
ax.grid(True, alpha=0.2, axis='y')

# Panel 3: log2 ratio by group
ax = axes[2]
bp_data_3 = [df_grp[df_grp["dual_group"] == g]["log2_hmox_ratio"].dropna().values
             for g in group_labels]
bp3 = ax.boxplot(bp_data_3, labels=["A", "B", "C", "D"], patch_artist=True,
                  widths=0.6, showfliers=True,
                  flierprops=dict(marker='o', markersize=3, alpha=0.4))
for patch, grp in zip(bp3['boxes'], group_labels):
    patch.set_facecolor(group_colors[grp])
    patch.set_alpha(0.7)
stat_ratio, p_ratio = stats.kruskal(*bp_data_3)
ax.set_title(f"log2(HMOX1/HMOX2) Ratio\nKruskal-Wallis p = {p_ratio:.2e}",
             fontsize=12, fontweight='bold')
ax.set_ylabel("log2(HMOX1/HMOX2)", fontsize=11)
ax.set_xlabel("Dual-Signature Group", fontsize=11)
ax.axhline(0, color='gray', linestyle='--', alpha=0.4, linewidth=1)
ax.grid(True, alpha=0.2, axis='y')

# Legend
legend_patches = [plt.Rectangle((0, 0), 1, 1, facecolor=group_colors[g], alpha=0.7)
                  for g in group_labels]
fig.legend(legend_patches, group_labels, loc='lower center', ncol=4,
           fontsize=9, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
plt.savefig(os.path.join(FIGS, "fig5c_hmox_boxplots_by_group.png"),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig5c_hmox_boxplots_by_group.png")

# --- 6d. Scatter: HMOX1 expression vs NK cell markers ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
nk_found = [m for m in nk_markers if m in df_immune.columns or
            f"{m}_immune" in df_immune.columns]

for idx, marker in enumerate(nk_found[:3]):
    ax = axes[idx]
    col = marker if marker in df_immune.columns else f"{marker}_immune"
    valid = df_immune[["HMOX1", col, "dual_group"]].dropna()

    for grp_label in group_labels:
        sub = valid[valid["dual_group"] == grp_label]
        if len(sub) == 0:
            continue
        ax.scatter(sub["HMOX1"], sub[col],
                   c=group_colors[grp_label], alpha=0.5, s=35,
                   label=grp_label if idx == 0 else None,
                   edgecolors='white', linewidth=0.3)

    # Fit regression line
    slope, intercept, r_val, p_reg, se = stats.linregress(valid["HMOX1"], valid[col])
    x_line = np.linspace(valid["HMOX1"].min(), valid["HMOX1"].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, '-', color='black',
            linewidth=1.5, alpha=0.7)

    rho_mk, p_mk = stats.spearmanr(valid["HMOX1"], valid[col])
    ax.text(0.05, 0.95, f"rho = {rho_mk:+.3f}\np = {p_mk:.2e}",
            transform=ax.transAxes, ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    marker_name = {"NCAM1": "NCAM1 (CD56)", "NKG7": "NKG7", "KLRD1": "KLRD1 (CD94)"}
    ax.set_xlabel("HMOX1 Expression", fontsize=11)
    ax.set_ylabel(f"{marker_name.get(marker, marker)} Expression", fontsize=11)
    ax.set_title(f"HMOX1 vs {marker_name.get(marker, marker)}\n(NK cell marker)",
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2)

# Fill empty panels if fewer than 3 NK markers
for idx in range(len(nk_found), 3):
    axes[idx].set_visible(False)

if nk_found:
    axes[0].legend(fontsize=8, loc="lower left")

plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig5d_hmox1_vs_nk_markers.png"),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig5d_hmox1_vs_nk_markers.png")

# --- 6e. Correlation bar plot: HMOX1 vs all immune markers ---
hmox1_immune = [r for r in immune_corr_results
                if not r["marker"].startswith("ratio_vs_")]
if hmox1_immune:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    bar_df = pd.DataFrame(hmox1_immune).sort_values("spearman_rho")
    colors_bar = []
    cat_colors = {"NK cell": "#e41a1c", "Dendritic cell": "#377eb8",
                  "T cell": "#4daf4a", "Immune score": "#984ea3"}
    for _, row in bar_df.iterrows():
        colors_bar.append(cat_colors.get(row["category"], "#999999"))

    bars = ax.barh(range(len(bar_df)), bar_df["spearman_rho"].values,
                   color=colors_bar, edgecolor='white', linewidth=0.5, height=0.7)

    # Add significance stars
    for i, (_, row) in enumerate(bar_df.iterrows()):
        x_pos = row["spearman_rho"]
        offset = 0.01 if x_pos >= 0 else -0.01
        ha = 'left' if x_pos >= 0 else 'right'
        ax.text(x_pos + offset, i, row["significant"],
                ha=ha, va='center', fontsize=9, fontweight='bold')

    ax.set_yticks(range(len(bar_df)))
    ax.set_yticklabels(bar_df["marker"].values, fontsize=10)
    ax.set_xlabel("Spearman Correlation with HMOX1", fontsize=12)
    ax.set_title("HMOX1 Correlation with Immune Cell Markers",
                 fontsize=13, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(True, alpha=0.2, axis='x')

    # Legend for categories
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=cat)
                       for cat, c in cat_colors.items()
                       if cat in bar_df["category"].values]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "fig5e_hmox1_immune_correlations.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: fig5e_hmox1_immune_correlations.png")

# --- 6f. Comprehensive summary figure ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel A: HMOX1 vs NRF2 (NFE2L2)
ax = axes[0, 0]
valid = df[["HMOX1", "NFE2L2", "dual_group"]].dropna()
for grp_label in group_labels:
    sub = valid[valid["dual_group"] == grp_label]
    ax.scatter(sub["NFE2L2"], sub["HMOX1"],
               c=group_colors[grp_label], alpha=0.5, s=35, label=grp_label,
               edgecolors='white', linewidth=0.3)
rho_nrf2, p_nrf2 = stats.spearmanr(valid["NFE2L2"], valid["HMOX1"])
slope, intercept, _, _, _ = stats.linregress(valid["NFE2L2"], valid["HMOX1"])
x_line = np.linspace(valid["NFE2L2"].min(), valid["NFE2L2"].max(), 100)
ax.plot(x_line, slope * x_line + intercept, '-', color='black', linewidth=1.5, alpha=0.7)
ax.text(0.05, 0.95, f"rho = {rho_nrf2:+.3f}\np = {p_nrf2:.2e}",
        transform=ax.transAxes, ha='left', va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.set_xlabel("NFE2L2 (NRF2) Expression", fontsize=11)
ax.set_ylabel("HMOX1 Expression", fontsize=11)
ax.set_title("A. NRF2 Drives HMOX1 Induction", fontsize=12, fontweight='bold')
ax.legend(fontsize=7, loc="lower right")
ax.grid(True, alpha=0.2)

# Panel B: HMOX ratio vs ROS risk score
ax = axes[0, 1]
valid = df[["log2_hmox_ratio", "ros_risk_score", "dual_group"]].dropna()
for grp_label in group_labels:
    sub = valid[valid["dual_group"] == grp_label]
    ax.scatter(sub["log2_hmox_ratio"], sub["ros_risk_score"],
               c=group_colors[grp_label], alpha=0.5, s=35,
               edgecolors='white', linewidth=0.3)
rho_ros, p_ros = stats.spearmanr(valid["log2_hmox_ratio"], valid["ros_risk_score"])
ax.text(0.05, 0.95, f"rho = {rho_ros:+.3f}\np = {p_ros:.2e}",
        transform=ax.transAxes, ha='left', va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.set_xlabel("log2(HMOX1/HMOX2) Ratio", fontsize=11)
ax.set_ylabel("ROS/Ferroptosis Risk Score", fontsize=11)
ax.set_title("B. HMOX Ratio Correlates with ROS Risk", fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.2)

# Panel C: HMOX ratio vs Altitude risk score
ax = axes[1, 0]
valid = df[["log2_hmox_ratio", "alt_risk_score", "dual_group"]].dropna()
for grp_label in group_labels:
    sub = valid[valid["dual_group"] == grp_label]
    ax.scatter(sub["log2_hmox_ratio"], sub["alt_risk_score"],
               c=group_colors[grp_label], alpha=0.5, s=35,
               edgecolors='white', linewidth=0.3)
rho_alt, p_alt = stats.spearmanr(valid["log2_hmox_ratio"], valid["alt_risk_score"])
ax.text(0.05, 0.95, f"rho = {rho_alt:+.3f}\np = {p_alt:.2e}",
        transform=ax.transAxes, ha='left', va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.set_xlabel("log2(HMOX1/HMOX2) Ratio", fontsize=11)
ax.set_ylabel("Altitude Risk Score", fontsize=11)
ax.set_title("C. HMOX Ratio Correlates with Altitude Risk", fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.2)

# Panel D: HMOX1 vs immune checkpoint (CD274 = PD-L1)
ax = axes[1, 1]
if "CD274" in df.columns:
    valid = df[["HMOX1", "CD274", "dual_group"]].dropna()
    for grp_label in group_labels:
        sub = valid[valid["dual_group"] == grp_label]
        ax.scatter(sub["HMOX1"], sub["CD274"],
                   c=group_colors[grp_label], alpha=0.5, s=35,
                   edgecolors='white', linewidth=0.3)
    rho_cd274, p_cd274 = stats.spearmanr(valid["HMOX1"], valid["CD274"])
    slope, intercept, _, _, _ = stats.linregress(valid["HMOX1"], valid["CD274"])
    x_line = np.linspace(valid["HMOX1"].min(), valid["HMOX1"].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, '-', color='black', linewidth=1.5, alpha=0.7)
    ax.text(0.05, 0.95, f"rho = {rho_cd274:+.3f}\np = {p_cd274:.2e}",
            transform=ax.transAxes, ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_xlabel("HMOX1 Expression", fontsize=11)
    ax.set_ylabel("CD274 (PD-L1) Expression", fontsize=11)
    ax.set_title("D. HMOX1 vs PD-L1 (Immune Checkpoint)", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2)
else:
    ax.text(0.5, 0.5, "CD274 data not available", ha='center', va='center',
            transform=ax.transAxes, fontsize=12)
    ax.set_title("D. HMOX1 vs Immune Checkpoint", fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig5f_hmox_summary_panel.png"),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig5f_hmox_summary_panel.png")

# ======================================================================
# 7. SAVE ALL OUTPUT TABLES
# ======================================================================
print("\n" + "=" * 70)
print("7. SAVING OUTPUT TABLES")
print("=" * 70)

# --- 7a. Correlation table ---
all_corr = correlation_results.copy()
for r in immune_corr_results:
    all_corr.append({
        "hmox_variable": "HMOX1 expression" if not r["marker"].startswith("ratio_vs_") else "log2(HMOX1/HMOX2) ratio",
        "target_variable": f"Immune: {r['marker'].replace('ratio_vs_', '')}",
        "target_column": r["marker"],
        "spearman_rho": r["spearman_rho"],
        "p_value": r["p_value"],
        "n": r["n"],
        "significant": r["significant"],
    })

corr_df = pd.DataFrame(all_corr)
corr_df = corr_df.sort_values("p_value")
corr_df.to_csv(os.path.join(TABLES, "hmox_correlations.csv"), index=False)
print(f"  Saved: results/tables/hmox_correlations.csv ({len(corr_df)} rows)")

# --- 7b. Survival table ---
surv_df = pd.DataFrame(survival_results)
surv_df.to_csv(os.path.join(TABLES, "hmox_survival.csv"), index=False)
print(f"  Saved: results/tables/hmox_survival.csv ({len(surv_df)} rows)")

# --- 7c. Group comparison table ---
grp_comp_df = pd.DataFrame(group_comparison_results)
grp_comp_df.to_csv(os.path.join(TABLES, "hmox_group_comparisons.csv"), index=False)
print(f"  Saved: results/tables/hmox_group_comparisons.csv ({len(grp_comp_df)} rows)")

# --- 7d. Update master CSV with HMOX ratio columns ---
master = pd.read_csv(os.path.join(DATA, "tcga_convergence_master.csv"))

# Compute ratio for full master (including patients without survival data)
master["hmox_ratio"] = (master["HMOX1"] + pseudocount) / (master["HMOX2"] + pseudocount)
master["log2_hmox_ratio"] = np.log2(master["hmox_ratio"])

master.to_csv(os.path.join(DATA, "tcga_convergence_master.csv"), index=False)
print(f"  Updated master CSV with hmox_ratio and log2_hmox_ratio columns "
      f"({len(master)} patients)")

# ======================================================================
# SUMMARY
# ======================================================================
print("\n" + "=" * 70)
print("SUMMARY: HMOX1/HMOX2 HEME OXYGENASE BRIDGE ANALYSIS")
print("=" * 70)

print(f"\nKey findings:")
print(f"  1. HMOX1/HMOX2 ratio reflects NRF2-driven shift to inducible heme metabolism")

# Report key correlations
for r in correlation_results:
    if r["target_column"] in ["ros_risk_score", "alt_risk_score", "NFE2L2"] and \
       r["hmox_variable"] == "log2(HMOX1/HMOX2) ratio":
        print(f"  2. Ratio vs {r['target_variable']}: rho={r['spearman_rho']:+.3f} (p={r['p_value']:.2e})")

# Report survival
for r in survival_results:
    if r["analysis"] == "Univariate Cox (log2 HMOX ratio)":
        print(f"  3. Univariate Cox: HR={r.get('HR', 'N/A'):.3f}, p={r.get('p_value', 'N/A'):.2e}")
    if r["analysis"] == "Multivariate Cox (log2 HMOX ratio + age + sex)":
        print(f"  4. Multivariate Cox (adj age+sex): HR={r.get('HR', 'N/A'):.3f}, p={r.get('p_value', 'N/A'):.2e}")

# Report top immune correlations
nk_corrs = [r for r in immune_corr_results
            if r["category"] == "NK cell" and not r["marker"].startswith("ratio_vs_")]
if nk_corrs:
    avg_nk_rho = np.mean([r["spearman_rho"] for r in nk_corrs])
    print(f"  5. Mean HMOX1-NK marker correlation: rho={avg_nk_rho:+.3f} "
          f"(hypothesis: negative = NK depletion)")

dc_corrs = [r for r in immune_corr_results
            if r["category"] == "Dendritic cell" and not r["marker"].startswith("ratio_vs_")]
if dc_corrs:
    avg_dc_rho = np.mean([r["spearman_rho"] for r in dc_corrs])
    print(f"  6. Mean HMOX1-DC marker correlation: rho={avg_dc_rho:+.3f} "
          f"(hypothesis: negative = DC depletion)")

print(f"\nFigures saved to: results/figures/main/")
print(f"Tables saved to:  results/tables/")

print("\n" + "=" * 70)
print("DONE -- Script 05 complete")
print("=" * 70)
