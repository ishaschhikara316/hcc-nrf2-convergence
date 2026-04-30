"""
03_nrf2_activity_scoring.py — NRF2 transcriptional activity and ferroptosis vulnerability

Computes per-patient NRF2 activity score (mean z-score of 16 NRF2/ARE target genes)
and ferroptosis vulnerability index, then correlates with risk scores and dual groups.

Inputs:
  - data/tcga_convergence_master.csv (306 patients, dual groups, gene expression)
  - Full expression matrix via data/paths.json (for z-scoring across all patients)

Outputs:
  - Updated data/tcga_convergence_master.csv (+ nrf2_activity, ferroptosis_vulnerability)
  - results/tables/nrf2_group_comparison.csv
  - results/tables/nrf2_correlations.csv
  - results/figures/main/fig3a_nrf2_by_group.png
  - results/figures/main/fig3b_ferroptosis_by_group.png
  - results/figures/main/fig3c_nrf2_vs_ferroptosis.png
  - results/figures/main/fig3d_nrf2_vs_ros_risk.png
"""
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import json
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
TABLES = os.path.join(BASE, "results", "tables")

# Ensure output directories exist
os.makedirs(FIGS, exist_ok=True)
os.makedirs(TABLES, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("1. LOADING DATA")
print("=" * 70)

df = pd.read_csv(os.path.join(DATA, "tcga_convergence_master.csv"))
df = df.dropna(subset=["OS_months", "OS_event"]).copy()
print(f"Master dataframe: {len(df)} patients")

# Load full expression matrix for z-scoring across all samples
with open(os.path.join(DATA, "paths.json")) as f:
    paths = json.load(f)

expr_full = pd.read_csv(paths["expression_full"], index_col=0)  # genes x samples
print(f"Full expression matrix: {expr_full.shape[0]} genes x {expr_full.shape[1]} samples")

# Transpose to samples x genes
expr = expr_full.T
expr.index.name = "patientId"
print(f"Transposed: {expr.shape[0]} samples x {expr.shape[1]} genes")

# Keep only patients in master dataframe
common_patients = df["patientId"].values
expr_sub = expr.loc[expr.index.isin(common_patients)].copy()
print(f"Patients in both master and expression: {len(expr_sub)}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. COMPUTE NRF2 ACTIVITY SCORE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. COMPUTING NRF2 ACTIVITY SCORE (16 NRF2/ARE TARGET GENES)")
print("=" * 70)

NRF2_TARGETS = [
    "NQO1", "HMOX1", "SLC7A11", "TXNRD1", "G6PD", "GSR",
    "GCLC", "GCLM", "FTH1", "FTL", "SQSTM1", "SRXN1",
    "AKR1C1", "AKR1B10", "ME1", "ABCC2",
]

# Check which NRF2 targets are available
available_nrf2 = [g for g in NRF2_TARGETS if g in expr_sub.columns]
missing_nrf2 = [g for g in NRF2_TARGETS if g not in expr_sub.columns]
print(f"NRF2 target genes found: {len(available_nrf2)}/{len(NRF2_TARGETS)}")
if missing_nrf2:
    print(f"  Missing: {missing_nrf2}")

# Z-score each gene across all patients in the expression subset
nrf2_zscores = pd.DataFrame(index=expr_sub.index)
for gene in available_nrf2:
    vals = expr_sub[gene].values
    mu = np.mean(vals)
    sigma = np.std(vals, ddof=1)
    if sigma > 0:
        nrf2_zscores[gene] = (vals - mu) / sigma
    else:
        print(f"  WARNING: std=0 for {gene}, excluding from NRF2 score")
        nrf2_zscores[gene] = 0.0

# NRF2 activity = mean z-score across all 16 target genes
nrf2_activity = nrf2_zscores[available_nrf2].mean(axis=1)

# Map back to master dataframe by patientId
nrf2_map = pd.Series(nrf2_activity.values, index=nrf2_activity.index, name="nrf2_activity")
df["nrf2_activity"] = df["patientId"].map(nrf2_map)

print(f"\nNRF2 activity score:")
print(f"  Mean:  {df['nrf2_activity'].mean():.4f}")
print(f"  Std:   {df['nrf2_activity'].std():.4f}")
print(f"  Range: [{df['nrf2_activity'].min():.4f}, {df['nrf2_activity'].max():.4f}]")

# ══════════════════════════════════════════════════════════════════════════════
# 3. COMPUTE FERROPTOSIS VULNERABILITY INDEX
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. COMPUTING FERROPTOSIS VULNERABILITY INDEX")
print("=" * 70)

PRO_FERROPTOSIS = ["ACSL4", "LPCAT3", "TFRC"]
ANTI_FERROPTOSIS = ["GPX4", "SLC7A11", "FTH1"]

# Check availability
avail_pro = [g for g in PRO_FERROPTOSIS if g in expr_sub.columns]
avail_anti = [g for g in ANTI_FERROPTOSIS if g in expr_sub.columns]
print(f"Pro-ferroptosis genes found:  {avail_pro}")
print(f"Anti-ferroptosis genes found: {avail_anti}")

# Z-score pro-ferroptosis genes
pro_zscores = pd.DataFrame(index=expr_sub.index)
for gene in avail_pro:
    vals = expr_sub[gene].values
    mu = np.mean(vals)
    sigma = np.std(vals, ddof=1)
    if sigma > 0:
        pro_zscores[gene] = (vals - mu) / sigma
    else:
        pro_zscores[gene] = 0.0

# Z-score anti-ferroptosis genes
anti_zscores = pd.DataFrame(index=expr_sub.index)
for gene in avail_anti:
    vals = expr_sub[gene].values
    mu = np.mean(vals)
    sigma = np.std(vals, ddof=1)
    if sigma > 0:
        anti_zscores[gene] = (vals - mu) / sigma
    else:
        anti_zscores[gene] = 0.0

# Ferroptosis vulnerability = mean(pro z-scores) - mean(anti z-scores)
# Higher = more vulnerable to ferroptosis
ferro_vuln = pro_zscores[avail_pro].mean(axis=1) - anti_zscores[avail_anti].mean(axis=1)

# Map to master dataframe
ferro_map = pd.Series(ferro_vuln.values, index=ferro_vuln.index, name="ferroptosis_vulnerability")
df["ferroptosis_vulnerability"] = df["patientId"].map(ferro_map)

print(f"\nFerroptosis vulnerability index:")
print(f"  Mean:  {df['ferroptosis_vulnerability'].mean():.4f}")
print(f"  Std:   {df['ferroptosis_vulnerability'].std():.4f}")
print(f"  Range: [{df['ferroptosis_vulnerability'].min():.4f}, {df['ferroptosis_vulnerability'].max():.4f}]")
print(f"  Interpretation: higher = more ferroptosis-vulnerable")

# ══════════════════════════════════════════════════════════════════════════════
# 4. VALIDATE NRF2 SCORE vs NFE2L2 AND KEAP1
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. VALIDATING NRF2 ACTIVITY SCORE")
print("=" * 70)

validation_results = []

# NRF2 activity vs NFE2L2 (NRF2 gene itself) — expect positive correlation
if "NFE2L2" in df.columns:
    rho_nfe2l2, p_nfe2l2 = stats.spearmanr(df["nrf2_activity"].dropna(),
                                             df.loc[df["nrf2_activity"].notna(), "NFE2L2"])
    print(f"NRF2 activity vs NFE2L2 expression:")
    print(f"  Spearman rho = {rho_nfe2l2:.4f}, p = {p_nfe2l2:.2e}")
    if rho_nfe2l2 > 0 and p_nfe2l2 < 0.05:
        print(f"  VALIDATED: Positive correlation confirms NRF2 score reflects transcriptional activity")
    validation_results.append({
        "comparison": "nrf2_activity vs NFE2L2",
        "spearman_rho": rho_nfe2l2,
        "p_value": p_nfe2l2,
        "expected_direction": "positive",
        "validated": rho_nfe2l2 > 0 and p_nfe2l2 < 0.05,
    })
else:
    print("  WARNING: NFE2L2 not found in master dataframe")

# NRF2 activity vs KEAP1 — expect negative correlation (KEAP1 degrades NRF2)
if "KEAP1" in df.columns:
    rho_keap1, p_keap1 = stats.spearmanr(df["nrf2_activity"].dropna(),
                                           df.loc[df["nrf2_activity"].notna(), "KEAP1"])
    print(f"\nNRF2 activity vs KEAP1 expression:")
    print(f"  Spearman rho = {rho_keap1:.4f}, p = {p_keap1:.2e}")
    if rho_keap1 < 0:
        print(f"  EXPECTED: Negative correlation (KEAP1 suppresses NRF2)")
    else:
        print(f"  NOTE: Positive correlation — may reflect compensatory KEAP1 upregulation in NRF2-active tumors")
    validation_results.append({
        "comparison": "nrf2_activity vs KEAP1",
        "spearman_rho": rho_keap1,
        "p_value": p_keap1,
        "expected_direction": "negative",
        "validated": rho_keap1 < 0 and p_keap1 < 0.05,
    })
else:
    print("  WARNING: KEAP1 not found in master dataframe")

# ══════════════════════════════════════════════════════════════════════════════
# 5. CORRELATE NRF2 ACTIVITY WITH RISK SCORES AND FERROPTOSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. CORRELATING NRF2 ACTIVITY WITH RISK SCORES & FERROPTOSIS VULNERABILITY")
print("=" * 70)

correlation_results = []
valid_mask = df["nrf2_activity"].notna() & df["ferroptosis_vulnerability"].notna()
df_valid = df[valid_mask].copy()

# --- NRF2 activity vs ROS risk score ---
rho, pval = stats.spearmanr(df_valid["nrf2_activity"], df_valid["ros_risk_score"])
print(f"NRF2 activity vs ROS risk score:")
print(f"  Spearman rho = {rho:.4f}, p = {pval:.2e}")
correlation_results.append({
    "variable_1": "nrf2_activity",
    "variable_2": "ros_risk_score",
    "spearman_rho": rho,
    "p_value": pval,
    "n": len(df_valid),
})

# --- NRF2 activity vs Altitude risk score ---
rho, pval = stats.spearmanr(df_valid["nrf2_activity"], df_valid["alt_risk_score"])
print(f"\nNRF2 activity vs Altitude risk score:")
print(f"  Spearman rho = {rho:.4f}, p = {pval:.2e}")
correlation_results.append({
    "variable_1": "nrf2_activity",
    "variable_2": "alt_risk_score",
    "spearman_rho": rho,
    "p_value": pval,
    "n": len(df_valid),
})

# --- NRF2 activity vs Ferroptosis vulnerability ---
rho, pval = stats.spearmanr(df_valid["nrf2_activity"], df_valid["ferroptosis_vulnerability"])
print(f"\nNRF2 activity vs Ferroptosis vulnerability:")
print(f"  Spearman rho = {rho:.4f}, p = {pval:.2e}")
if rho < 0:
    print(f"  EXPECTED: Negative correlation — NRF2-high tumors are ferroptosis-resistant")
correlation_results.append({
    "variable_1": "nrf2_activity",
    "variable_2": "ferroptosis_vulnerability",
    "spearman_rho": rho,
    "p_value": pval,
    "n": len(df_valid),
})

# --- NRF2 activity vs HMOX1 ---
if "HMOX1" in df_valid.columns:
    rho, pval = stats.spearmanr(df_valid["nrf2_activity"], df_valid["HMOX1"])
    print(f"\nNRF2 activity vs HMOX1 expression:")
    print(f"  Spearman rho = {rho:.4f}, p = {pval:.2e}")
    correlation_results.append({
        "variable_1": "nrf2_activity",
        "variable_2": "HMOX1",
        "spearman_rho": rho,
        "p_value": pval,
        "n": len(df_valid),
    })

# --- NRF2 activity vs HMOX2 ---
if "HMOX2" in df_valid.columns:
    rho, pval = stats.spearmanr(df_valid["nrf2_activity"], df_valid["HMOX2"])
    print(f"\nNRF2 activity vs HMOX2 expression:")
    print(f"  Spearman rho = {rho:.4f}, p = {pval:.2e}")
    correlation_results.append({
        "variable_1": "nrf2_activity",
        "variable_2": "HMOX2",
        "spearman_rho": rho,
        "p_value": pval,
        "n": len(df_valid),
    })

# Apply BH FDR correction across all correlations
corr_df = pd.DataFrame(correlation_results)
_, fdr_pvals, _, _ = multipletests(corr_df["p_value"].values, method="fdr_bh")
corr_df["p_fdr"] = fdr_pvals

# Also add validation results
val_df = pd.DataFrame(validation_results)
if len(val_df) > 0:
    val_corr = val_df.rename(columns={"comparison": "variable_1"}).copy()
    val_corr["variable_2"] = ""
    val_corr["n"] = len(df_valid)
    val_corr = val_corr[["variable_1", "variable_2", "spearman_rho", "p_value", "n"]]
    _, val_fdr, _, _ = multipletests(val_corr["p_value"].values, method="fdr_bh")
    val_corr["p_fdr"] = val_fdr

# Combine all correlation results and save
all_corr = pd.concat([corr_df, val_corr], ignore_index=True) if len(val_df) > 0 else corr_df
all_corr.to_csv(os.path.join(TABLES, "nrf2_correlations.csv"), index=False)
print(f"\nSaved: results/tables/nrf2_correlations.csv ({len(all_corr)} correlations)")

# ══════════════════════════════════════════════════════════════════════════════
# 6. COMPARE NRF2 ACTIVITY ACROSS 4 DUAL GROUPS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("6. NRF2 ACTIVITY BY DUAL GROUP (KRUSKAL-WALLIS + POST-HOC)")
print("=" * 70)

group_labels = [
    "A: Concordant High",
    "B: Ferroptosis-dominant",
    "C: Hypoxia-dominant",
    "D: Concordant Low",
]

# Kruskal-Wallis test for NRF2 activity
nrf2_groups = [df_valid.loc[df_valid["dual_group"] == g, "nrf2_activity"].values for g in group_labels]
kw_stat, kw_p = stats.kruskal(*nrf2_groups)
print(f"Kruskal-Wallis test for NRF2 activity across groups:")
print(f"  H = {kw_stat:.4f}, p = {kw_p:.2e}")

# Group statistics
group_comparison = []
for g_label, g_vals in zip(group_labels, nrf2_groups):
    group_comparison.append({
        "group": g_label,
        "metric": "nrf2_activity",
        "n": len(g_vals),
        "mean": np.mean(g_vals),
        "median": np.median(g_vals),
        "std": np.std(g_vals, ddof=1),
    })
    print(f"  {g_label}: n={len(g_vals)}, mean={np.mean(g_vals):.4f}, median={np.median(g_vals):.4f}")

# Post-hoc: pairwise Mann-Whitney U with Bonferroni correction
print(f"\nPost-hoc pairwise Mann-Whitney U tests (Bonferroni-corrected):")
posthoc_results = []
pairs = list(combinations(range(len(group_labels)), 2))
n_comparisons = len(pairs)

for i, j in pairs:
    u_stat, p_raw = stats.mannwhitneyu(nrf2_groups[i], nrf2_groups[j], alternative='two-sided')
    p_bonf = min(p_raw * n_comparisons, 1.0)
    posthoc_results.append({
        "group_1": group_labels[i],
        "group_2": group_labels[j],
        "metric": "nrf2_activity",
        "U_statistic": u_stat,
        "p_raw": p_raw,
        "p_bonferroni": p_bonf,
        "significant": p_bonf < 0.05,
    })
    sig_str = "*" if p_bonf < 0.05 else "ns"
    print(f"  {group_labels[i]} vs {group_labels[j]}: U={u_stat:.1f}, "
          f"p_raw={p_raw:.4f}, p_bonf={p_bonf:.4f} {sig_str}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. COMPARE FERROPTOSIS VULNERABILITY ACROSS 4 DUAL GROUPS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("7. FERROPTOSIS VULNERABILITY BY DUAL GROUP (KRUSKAL-WALLIS + POST-HOC)")
print("=" * 70)

ferro_groups = [df_valid.loc[df_valid["dual_group"] == g, "ferroptosis_vulnerability"].values
                for g in group_labels]
kw_stat_f, kw_p_f = stats.kruskal(*ferro_groups)
print(f"Kruskal-Wallis test for ferroptosis vulnerability across groups:")
print(f"  H = {kw_stat_f:.4f}, p = {kw_p_f:.2e}")

for g_label, g_vals in zip(group_labels, ferro_groups):
    group_comparison.append({
        "group": g_label,
        "metric": "ferroptosis_vulnerability",
        "n": len(g_vals),
        "mean": np.mean(g_vals),
        "median": np.median(g_vals),
        "std": np.std(g_vals, ddof=1),
    })
    print(f"  {g_label}: n={len(g_vals)}, mean={np.mean(g_vals):.4f}, median={np.median(g_vals):.4f}")

# Post-hoc for ferroptosis
print(f"\nPost-hoc pairwise Mann-Whitney U tests (Bonferroni-corrected):")
for i, j in pairs:
    u_stat, p_raw = stats.mannwhitneyu(ferro_groups[i], ferro_groups[j], alternative='two-sided')
    p_bonf = min(p_raw * n_comparisons, 1.0)
    posthoc_results.append({
        "group_1": group_labels[i],
        "group_2": group_labels[j],
        "metric": "ferroptosis_vulnerability",
        "U_statistic": u_stat,
        "p_raw": p_raw,
        "p_bonferroni": p_bonf,
        "significant": p_bonf < 0.05,
    })
    sig_str = "*" if p_bonf < 0.05 else "ns"
    print(f"  {group_labels[i]} vs {group_labels[j]}: U={u_stat:.1f}, "
          f"p_raw={p_raw:.4f}, p_bonf={p_bonf:.4f} {sig_str}")

# Save group comparison table
group_comp_df = pd.DataFrame(group_comparison)
posthoc_df = pd.DataFrame(posthoc_results)

# Combine summary + posthoc into one output file
with open(os.path.join(TABLES, "nrf2_group_comparison.csv"), "w") as fout:
    fout.write("# Group summary statistics\n")
group_comp_df.to_csv(os.path.join(TABLES, "nrf2_group_comparison.csv"), index=False)

# Also save posthoc results
posthoc_df.to_csv(os.path.join(TABLES, "nrf2_posthoc_pairwise.csv"), index=False)
print(f"\nSaved: results/tables/nrf2_group_comparison.csv")
print(f"Saved: results/tables/nrf2_posthoc_pairwise.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 8. FIGURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("8. GENERATING FIGURES")
print("=" * 70)

colors = {
    "A: Concordant High": "#d62728",
    "B: Ferroptosis-dominant": "#ff7f0e",
    "C: Hypoxia-dominant": "#9467bd",
    "D: Concordant Low": "#2ca02c",
}
palette = [colors[g] for g in group_labels]

# --- Fig 3a: NRF2 activity by dual group ---
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(data=df_valid, x="dual_group", y="nrf2_activity",
            order=group_labels, palette=palette, width=0.6,
            fliersize=3, linewidth=1.2, ax=ax)
sns.stripplot(data=df_valid, x="dual_group", y="nrf2_activity",
              order=group_labels, color="black", alpha=0.3, size=3, jitter=True, ax=ax)

ax.set_title(f"NRF2 Transcriptional Activity by Dual Group\n(Kruskal-Wallis p = {kw_p:.2e})",
             fontsize=13, fontweight='bold')
ax.set_xlabel("Dual-Signature Group", fontsize=12)
ax.set_ylabel("NRF2 Activity Score (mean z-score)", fontsize=12)
ax.set_xticklabels([g.split(": ")[1] for g in group_labels], fontsize=10)
ax.grid(True, alpha=0.2, axis='y')

# Add significance brackets for notable comparisons
# Find the A vs D comparison
for res in posthoc_results:
    if (res["group_1"] == "A: Concordant High" and
            res["group_2"] == "D: Concordant Low" and
            res["metric"] == "nrf2_activity"):
        p_ad = res["p_bonferroni"]
        if p_ad < 0.001:
            sig_label = "***"
        elif p_ad < 0.01:
            sig_label = "**"
        elif p_ad < 0.05:
            sig_label = "*"
        else:
            sig_label = "ns"

        y_max = df_valid["nrf2_activity"].max()
        y_bar = y_max + 0.1 * (df_valid["nrf2_activity"].max() - df_valid["nrf2_activity"].min())
        ax.plot([0, 0, 3, 3], [y_bar - 0.02, y_bar, y_bar, y_bar - 0.02],
                color='black', linewidth=1)
        ax.text(1.5, y_bar + 0.01, sig_label, ha='center', fontsize=12, fontweight='bold')
        break

plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig3a_nrf2_by_group.png"), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: fig3a_nrf2_by_group.png")

# --- Fig 3b: Ferroptosis vulnerability by dual group ---
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(data=df_valid, x="dual_group", y="ferroptosis_vulnerability",
            order=group_labels, palette=palette, width=0.6,
            fliersize=3, linewidth=1.2, ax=ax)
sns.stripplot(data=df_valid, x="dual_group", y="ferroptosis_vulnerability",
              order=group_labels, color="black", alpha=0.3, size=3, jitter=True, ax=ax)

ax.set_title(f"Ferroptosis Vulnerability by Dual Group\n(Kruskal-Wallis p = {kw_p_f:.2e})",
             fontsize=13, fontweight='bold')
ax.set_xlabel("Dual-Signature Group", fontsize=12)
ax.set_ylabel("Ferroptosis Vulnerability Index", fontsize=12)
ax.set_xticklabels([g.split(": ")[1] for g in group_labels], fontsize=10)
ax.grid(True, alpha=0.2, axis='y')

# Add significance bracket for A vs D
for res in posthoc_results:
    if (res["group_1"] == "A: Concordant High" and
            res["group_2"] == "D: Concordant Low" and
            res["metric"] == "ferroptosis_vulnerability"):
        p_ad = res["p_bonferroni"]
        if p_ad < 0.001:
            sig_label = "***"
        elif p_ad < 0.01:
            sig_label = "**"
        elif p_ad < 0.05:
            sig_label = "*"
        else:
            sig_label = "ns"

        y_max = df_valid["ferroptosis_vulnerability"].max()
        y_bar = y_max + 0.1 * (df_valid["ferroptosis_vulnerability"].max() -
                                df_valid["ferroptosis_vulnerability"].min())
        ax.plot([0, 0, 3, 3], [y_bar - 0.02, y_bar, y_bar, y_bar - 0.02],
                color='black', linewidth=1)
        ax.text(1.5, y_bar + 0.01, sig_label, ha='center', fontsize=12, fontweight='bold')
        break

plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig3b_ferroptosis_by_group.png"), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: fig3b_ferroptosis_by_group.png")

# --- Fig 3c: NRF2 activity vs Ferroptosis vulnerability scatter ---
fig, ax = plt.subplots(figsize=(8, 6))
for g_label in group_labels:
    grp = df_valid[df_valid["dual_group"] == g_label]
    ax.scatter(grp["nrf2_activity"], grp["ferroptosis_vulnerability"],
               c=colors[g_label], alpha=0.6, s=40, label=g_label,
               edgecolors='white', linewidth=0.5)

# Add regression line
x_vals = df_valid["nrf2_activity"].values
y_vals = df_valid["ferroptosis_vulnerability"].values
slope, intercept, r_val, p_val, std_err = stats.linregress(x_vals, y_vals)
x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
ax.plot(x_line, slope * x_line + intercept, 'k--', alpha=0.5, linewidth=1.5)

# Spearman correlation annotation
rho_nf, p_nf = stats.spearmanr(x_vals, y_vals)
ax.text(0.05, 0.95, f"Spearman rho = {rho_nf:.3f}\np = {p_nf:.2e}",
        transform=ax.transAxes, ha='left', va='top', fontsize=10, style='italic',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_title("NRF2 Activity vs Ferroptosis Vulnerability", fontsize=13, fontweight='bold')
ax.set_xlabel("NRF2 Activity Score", fontsize=12)
ax.set_ylabel("Ferroptosis Vulnerability Index", fontsize=12)
ax.legend(fontsize=9, loc="lower left")
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig3c_nrf2_vs_ferroptosis.png"), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: fig3c_nrf2_vs_ferroptosis.png")

# --- Fig 3d: NRF2 activity vs ROS risk score scatter ---
fig, ax = plt.subplots(figsize=(8, 6))
for g_label in group_labels:
    grp = df_valid[df_valid["dual_group"] == g_label]
    ax.scatter(grp["nrf2_activity"], grp["ros_risk_score"],
               c=colors[g_label], alpha=0.6, s=40, label=g_label,
               edgecolors='white', linewidth=0.5)

# Regression line
x_vals = df_valid["nrf2_activity"].values
y_vals = df_valid["ros_risk_score"].values
slope, intercept, r_val, p_val, std_err = stats.linregress(x_vals, y_vals)
x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
ax.plot(x_line, slope * x_line + intercept, 'k--', alpha=0.5, linewidth=1.5)

# Spearman annotation
rho_nr, p_nr = stats.spearmanr(x_vals, y_vals)
ax.text(0.05, 0.95, f"Spearman rho = {rho_nr:.3f}\np = {p_nr:.2e}",
        transform=ax.transAxes, ha='left', va='top', fontsize=10, style='italic',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_title("NRF2 Activity vs ROS/Ferroptosis Risk Score", fontsize=13, fontweight='bold')
ax.set_xlabel("NRF2 Activity Score", fontsize=12)
ax.set_ylabel("ROS/Ferroptosis Risk Score", fontsize=12)
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig3d_nrf2_vs_ros_risk.png"), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: fig3d_nrf2_vs_ros_risk.png")

# ══════════════════════════════════════════════════════════════════════════════
# 9. SAVE UPDATED MASTER DATAFRAME
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("9. SAVING UPDATED MASTER DATAFRAME")
print("=" * 70)

df.to_csv(os.path.join(DATA, "tcga_convergence_master.csv"), index=False)
print(f"Updated: data/tcga_convergence_master.csv")
print(f"  New columns: nrf2_activity, ferroptosis_vulnerability")
print(f"  Shape: {df.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# 10. SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("10. SUMMARY OF KEY FINDINGS")
print("=" * 70)

# NRF2 by group
print("\nNRF2 activity by group (mean):")
for g_label in group_labels:
    grp = df_valid[df_valid["dual_group"] == g_label]
    print(f"  {g_label}: {grp['nrf2_activity'].mean():.4f}")

# Key correlations
print(f"\nKey correlations:")
for _, row in corr_df.iterrows():
    sig_str = "***" if row["p_fdr"] < 0.001 else "**" if row["p_fdr"] < 0.01 else "*" if row["p_fdr"] < 0.05 else "ns"
    print(f"  {row['variable_1']} vs {row['variable_2']}: "
          f"rho={row['spearman_rho']:.3f}, p_fdr={row['p_fdr']:.2e} {sig_str}")

# Hypothesis check
nrf2_a = df_valid.loc[df_valid["dual_group"] == "A: Concordant High", "nrf2_activity"].mean()
nrf2_d = df_valid.loc[df_valid["dual_group"] == "D: Concordant Low", "nrf2_activity"].mean()
print(f"\nHypothesis test: Group A (Concordant High) has highest NRF2?")
print(f"  Group A mean = {nrf2_a:.4f}")
print(f"  Group D mean = {nrf2_d:.4f}")
print(f"  Difference: {nrf2_a - nrf2_d:+.4f}")

print("\n" + "=" * 70)
print("DONE — Script 03 complete")
print("=" * 70)
