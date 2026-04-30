"""
06_patient_archetypes.py — Molecularly distinct HCC subtypes via consensus clustering

Defines patient archetypes by consensus clustering on 10 convergence features:
  - nrf2_activity, ferroptosis_vulnerability, sting_score
  - immune_NK_cells, immune_Dendritic_cells, immune_CD8_T_cells
  - HMOX1, HMOX2
  - ros_risk_score, alt_risk_score

Uses 500-iteration consensus clustering (Ward linkage, Euclidean distance)
for k=2..5, selects optimal k via CDF area analysis, then characterises
each archetype by molecular profile, survival, and mutation landscape.

Outputs:
  - Updated master CSV with archetype column
  - results/tables/archetype_features.csv
  - results/tables/archetype_survival.csv
  - results/model/consensus_matrices.npz
  - Figures to results/figures/main/
"""
import pandas as pd
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index
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
MODEL = os.path.join(BASE, "results", "model")

os.makedirs(FIGS, exist_ok=True)
os.makedirs(TABLES, exist_ok=True)
os.makedirs(MODEL, exist_ok=True)

# ======================================================================
# 1. LOAD DATA & SELECT CLUSTERING FEATURES
# ======================================================================
print("=" * 70)
print("1. LOADING DATA & SELECTING CLUSTERING FEATURES")
print("=" * 70)

df = pd.read_csv(os.path.join(DATA, "tcga_convergence_master.csv"))
df = df.dropna(subset=["OS_months", "OS_event"]).copy()
print(f"Patients with survival data: {len(df)}")

# Define the 10 clustering features
CLUSTER_FEATURES = [
    "nrf2_activity",
    "ferroptosis_vulnerability",
    "sting_score",
    "immune_NK_cells",
    "immune_Dendritic_cells",
    "immune_CD8_T_cells",
    "HMOX1",
    "HMOX2",
    "ros_risk_score",
    "alt_risk_score",
]

# Verify all features exist
missing_features = [f for f in CLUSTER_FEATURES if f not in df.columns]
if missing_features:
    raise ValueError(f"Missing clustering features: {missing_features}")
print(f"Clustering features ({len(CLUSTER_FEATURES)}): {CLUSTER_FEATURES}")

# Drop patients with any NaN in clustering features
df_clust = df.dropna(subset=CLUSTER_FEATURES).copy()
print(f"Patients after dropping NaN in clustering features: {len(df_clust)}")

# Z-score normalise
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(df_clust[CLUSTER_FEATURES].values)
feature_means = df_clust[CLUSTER_FEATURES].mean()
feature_stds = df_clust[CLUSTER_FEATURES].std()

print("\nFeature summary (pre-normalisation):")
for feat in CLUSTER_FEATURES:
    vals = df_clust[feat]
    print(f"  {feat:30s}: mean={vals.mean():.3f}, std={vals.std():.3f}, "
          f"range=[{vals.min():.3f}, {vals.max():.3f}]")

patient_ids = df_clust["patientId"].values
n_patients = X.shape[0]
print(f"\nZ-scored feature matrix: {X.shape[0]} patients x {X.shape[1]} features")

# ======================================================================
# 2. CONSENSUS CLUSTERING (k=2 to k=5)
# ======================================================================
print("\n" + "=" * 70)
print("2. CONSENSUS CLUSTERING (k=2 to k=5, 500 iterations)")
print("=" * 70)

N_ITER = 500
SUBSAMPLE_FRAC = 0.80
K_RANGE = range(2, 6)
np.random.seed(42)

# Store consensus matrices for each k
consensus_matrices = {}

for k in K_RANGE:
    print(f"\n  k = {k}: running {N_ITER} iterations ...", end="", flush=True)

    # Co-clustering count matrix and indicator matrix
    cooccur = np.zeros((n_patients, n_patients), dtype=np.float64)
    indicator = np.zeros((n_patients, n_patients), dtype=np.float64)

    for iteration in range(N_ITER):
        # Randomly sample 80% of patients
        n_sub = int(n_patients * SUBSAMPLE_FRAC)
        idx = np.random.choice(n_patients, size=n_sub, replace=False)
        idx_sorted = np.sort(idx)

        X_sub = X[idx_sorted]

        # Hierarchical clustering with Ward's method
        dist_vec = pdist(X_sub, metric='euclidean')
        Z = linkage(dist_vec, method='ward')
        labels_sub = fcluster(Z, t=k, criterion='maxclust')

        # Update co-clustering and indicator matrices
        for i_local in range(n_sub):
            for j_local in range(i_local + 1, n_sub):
                gi = idx_sorted[i_local]
                gj = idx_sorted[j_local]
                indicator[gi, gj] += 1.0
                indicator[gj, gi] += 1.0
                if labels_sub[i_local] == labels_sub[j_local]:
                    cooccur[gi, gj] += 1.0
                    cooccur[gj, gi] += 1.0

        if (iteration + 1) % 100 == 0:
            print(f" {iteration + 1}", end="", flush=True)

    # Compute consensus matrix: fraction of times pairs co-clustered
    # when both were sampled
    mask = indicator > 0
    consensus = np.zeros_like(cooccur)
    consensus[mask] = cooccur[mask] / indicator[mask]
    np.fill_diagonal(consensus, 1.0)

    consensus_matrices[k] = consensus
    print(f" done.")
    print(f"    Consensus matrix range: [{consensus[mask].min():.3f}, {consensus[mask].max():.3f}]")

# Save consensus matrices
np.savez_compressed(
    os.path.join(MODEL, "consensus_matrices.npz"),
    **{f"k{k}": consensus_matrices[k] for k in K_RANGE}
)
print(f"\n  Saved: results/model/consensus_matrices.npz")

# ======================================================================
# 3. SELECT OPTIMAL k (CDF & DELTA AREA)
# ======================================================================
print("\n" + "=" * 70)
print("3. SELECTING OPTIMAL k VIA CDF AREA ANALYSIS")
print("=" * 70)

# For each k, compute CDF of consensus values and area under the CDF
cdf_data = {}
areas = {}

for k in K_RANGE:
    C = consensus_matrices[k]
    # Extract upper triangle values (excluding diagonal)
    upper_idx = np.triu_indices_from(C, k=1)
    vals = C[upper_idx]

    # Sort values and compute empirical CDF
    sorted_vals = np.sort(vals)
    cdf_y = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

    cdf_data[k] = (sorted_vals, cdf_y)

    # Area under CDF curve (trapezoidal integration)
    area = np.trapz(cdf_y, sorted_vals)
    areas[k] = area
    print(f"  k={k}: CDF area = {area:.4f}")

# Delta area: relative change in area
delta_areas = {}
k_list = sorted(K_RANGE)
for i in range(1, len(k_list)):
    k_curr = k_list[i]
    k_prev = k_list[i - 1]
    if areas[k_prev] != 0:
        delta = (areas[k_curr] - areas[k_prev]) / areas[k_prev]
    else:
        delta = 0.0
    delta_areas[k_curr] = delta
    print(f"  delta(k={k_curr}): {delta:.4f}")

# Select optimal k: the k where delta area shows diminishing returns
# (smallest relative increase above a threshold, or the elbow)
# First check if k=3 or k=4 gives substantial improvement over k=2
# Use heuristic: pick k where the proportional gain drops below 5%
# or where the consensus matrix is most clearly bimodal
optimal_k = 3  # default
for k_curr in k_list[1:]:
    if k_curr in delta_areas:
        if abs(delta_areas[k_curr]) < 0.02:
            optimal_k = k_list[k_list.index(k_curr) - 1]
            break
else:
    # If no clear plateau, pick k with maximum area
    optimal_k = max(areas, key=areas.get)

# Ensure optimal_k is at least 3 for biological interpretability
if optimal_k < 3:
    optimal_k = 3

print(f"\n  Selected optimal k = {optimal_k}")

# ======================================================================
# 4. ASSIGN ARCHETYPES FOR OPTIMAL k
# ======================================================================
print("\n" + "=" * 70)
print(f"4. ASSIGNING ARCHETYPES (k={optimal_k})")
print("=" * 70)

# Cluster the consensus matrix itself to get final assignments
C_opt = consensus_matrices[optimal_k]
dist_consensus = 1.0 - C_opt
np.fill_diagonal(dist_consensus, 0.0)
# Ensure symmetry and no negative values
dist_consensus = np.maximum(dist_consensus, 0.0)
dist_consensus = (dist_consensus + dist_consensus.T) / 2.0

dist_vec_consensus = squareform(dist_consensus, checks=False)
Z_final = linkage(dist_vec_consensus, method='ward')
cluster_labels = fcluster(Z_final, t=optimal_k, criterion='maxclust')

# Add cluster labels to dataframe
df_clust = df_clust.copy()
df_clust["cluster"] = cluster_labels

print(f"  Cluster assignments:")
for cl in sorted(df_clust["cluster"].unique()):
    n_cl = (df_clust["cluster"] == cl).sum()
    print(f"    Cluster {cl}: {n_cl} patients ({100 * n_cl / len(df_clust):.1f}%)")

# Compute mean z-scores per cluster for naming
X_df = pd.DataFrame(X, columns=CLUSTER_FEATURES, index=df_clust.index)
X_df["cluster"] = cluster_labels
cluster_means = X_df.groupby("cluster")[CLUSTER_FEATURES].mean()

print(f"\n  Mean z-scores per cluster:")
print(cluster_means.round(3).to_string())

# Name archetypes based on dominant molecular features
# Determine names by looking at which features are most elevated in each cluster
archetype_names = {}
for cl in sorted(cluster_means.index):
    profile = cluster_means.loc[cl]
    top_features = profile.nlargest(3)
    bottom_features = profile.nsmallest(2)

    # Naming heuristics based on biological interpretation
    high_nrf2 = profile["nrf2_activity"] > 0.5
    high_ros = profile["ros_risk_score"] > 0.5
    high_immune = (profile["immune_CD8_T_cells"] > 0.3 or
                   profile["immune_NK_cells"] > 0.3)
    low_nrf2 = profile["nrf2_activity"] < -0.3
    low_immune = (profile["immune_CD8_T_cells"] < -0.3 and
                  profile["immune_NK_cells"] < -0.3)
    high_sting = profile["sting_score"] > 0.3
    high_ferroptosis = profile["ferroptosis_vulnerability"] > 0.3
    high_hmox1 = profile["HMOX1"] > 0.3
    balanced = all(abs(v) < 0.5 for v in profile.values)

    if high_nrf2 and high_ros and high_hmox1:
        name = "NRF2-Dominant"
    elif high_nrf2 and high_ros:
        name = "NRF2-Dominant"
    elif high_immune and high_sting:
        name = "Immune-Active"
    elif high_immune:
        name = "Immune-Active"
    elif low_nrf2 and low_immune:
        name = "Cold-Quiescent"
    elif high_ferroptosis and not high_nrf2:
        name = "Ferroptosis-Vulnerable"
    elif balanced:
        name = "Redox-Balanced"
    else:
        # Fallback: name by top feature
        top_feat = top_features.index[0]
        feat_name_map = {
            "nrf2_activity": "NRF2-Dominant",
            "ferroptosis_vulnerability": "Ferroptosis-Vulnerable",
            "sting_score": "STING-Active",
            "immune_NK_cells": "Immune-Active",
            "immune_CD8_T_cells": "Immune-Active",
            "immune_Dendritic_cells": "Immune-Active",
            "HMOX1": "HMOX1-High",
            "HMOX2": "HMOX2-Driven",
            "ros_risk_score": "ROS-High",
            "alt_risk_score": "Hypoxia-Adaptive",
        }
        name = feat_name_map.get(top_feat, f"Archetype-{cl}")

    # Ensure uniqueness
    if name in archetype_names.values():
        name = f"{name}-{cl}"
    archetype_names[cl] = name

print(f"\n  Archetype names:")
for cl, name in sorted(archetype_names.items()):
    print(f"    Cluster {cl} -> {name}")

# Map cluster numbers to archetype names
df_clust["archetype"] = df_clust["cluster"].map(archetype_names)

# ======================================================================
# 5. ARCHETYPE CHARACTERISATION
# ======================================================================
print("\n" + "=" * 70)
print("5. ARCHETYPE CHARACTERISATION (Kruskal-Wallis per feature)")
print("=" * 70)

archetype_labels = sorted(df_clust["archetype"].unique())
n_archetypes = len(archetype_labels)

# Compute mean and median per archetype per feature (on original scale)
feature_stats = []
kw_results = []

for feat in CLUSTER_FEATURES:
    groups = [df_clust[df_clust["archetype"] == a][feat].dropna().values
              for a in archetype_labels]
    groups_nonempty = [g for g in groups if len(g) > 0]

    if len(groups_nonempty) >= 2:
        stat_kw, p_kw = stats.kruskal(*groups_nonempty)
    else:
        stat_kw, p_kw = np.nan, np.nan

    kw_results.append({
        "feature": feat,
        "kruskal_wallis_H": stat_kw,
        "p_value": p_kw,
        "significant": "***" if p_kw < 0.001 else "**" if p_kw < 0.01 else "*" if p_kw < 0.05 else "ns",
    })
    print(f"  {feat:30s}: H={stat_kw:8.2f}, p={p_kw:.2e} "
          f"{'***' if p_kw < 0.001 else '**' if p_kw < 0.01 else '*' if p_kw < 0.05 else 'ns'}")

    for a in archetype_labels:
        vals = df_clust[df_clust["archetype"] == a][feat].dropna()
        feature_stats.append({
            "archetype": a,
            "feature": feat,
            "mean": vals.mean(),
            "median": vals.median(),
            "std": vals.std(),
            "q25": vals.quantile(0.25),
            "q75": vals.quantile(0.75),
            "n": len(vals),
        })

feature_stats_df = pd.DataFrame(feature_stats)
feature_stats_df.to_csv(os.path.join(TABLES, "archetype_features.csv"), index=False)
print(f"\n  Saved: results/tables/archetype_features.csv ({len(feature_stats_df)} rows)")

# ======================================================================
# 6. SURVIVAL ANALYSIS
# ======================================================================
print("\n" + "=" * 70)
print("6. SURVIVAL ANALYSIS BY ARCHETYPE")
print("=" * 70)

survival_results = []

# --- 6a. Kaplan-Meier by archetype ---
surv_data = df_clust.dropna(subset=["OS_months", "OS_event"]).copy()
print(f"  Patients with survival data: {len(surv_data)}")

# Log-rank test across all archetypes
try:
    lr_result = multivariate_logrank_test(
        surv_data["OS_months"],
        surv_data["archetype"],
        surv_data["OS_event"]
    )
    lr_p = lr_result.p_value
    lr_stat = lr_result.test_statistic
    print(f"  Log-rank test (all archetypes): chi2={lr_stat:.2f}, p={lr_p:.2e}")
except Exception as e:
    lr_p = np.nan
    lr_stat = np.nan
    print(f"  Log-rank test failed: {e}")

survival_results.append({
    "analysis": "Log-rank (all archetypes)",
    "variable": "archetype",
    "n": len(surv_data),
    "events": int(surv_data["OS_event"].sum()),
    "test_statistic": lr_stat,
    "p_value": lr_p,
})

# Pairwise log-rank tests
print("\n  Pairwise log-rank tests:")
for a1, a2 in combinations(archetype_labels, 2):
    sub = surv_data[surv_data["archetype"].isin([a1, a2])]
    if len(sub) < 5:
        continue
    try:
        pw_result = multivariate_logrank_test(
            sub["OS_months"], sub["archetype"], sub["OS_event"]
        )
        pw_p = pw_result.p_value
        pw_stat = pw_result.test_statistic
    except Exception:
        pw_p = np.nan
        pw_stat = np.nan
    sig = "***" if pw_p < 0.001 else "**" if pw_p < 0.01 else "*" if pw_p < 0.05 else "ns"
    print(f"    {a1} vs {a2}: chi2={pw_stat:.2f}, p={pw_p:.4f} {sig}")
    survival_results.append({
        "analysis": f"Pairwise log-rank: {a1} vs {a2}",
        "variable": "archetype",
        "n": len(sub),
        "events": int(sub["OS_event"].sum()),
        "test_statistic": pw_stat,
        "p_value": pw_p,
    })

# Median survival per archetype
print("\n  Median survival by archetype:")
for a in archetype_labels:
    sub = surv_data[surv_data["archetype"] == a]
    kmf = KaplanMeierFitter()
    kmf.fit(sub["OS_months"], sub["OS_event"])
    med = kmf.median_survival_time_
    print(f"    {a}: median OS = {med:.1f} months (n={len(sub)}, events={int(sub['OS_event'].sum())})")
    survival_results.append({
        "analysis": "Median survival",
        "variable": a,
        "n": len(sub),
        "events": int(sub["OS_event"].sum()),
        "test_statistic": np.nan,
        "p_value": np.nan,
        "median_OS_months": med,
    })

# --- 6b. Univariate Cox with archetype (dummy-coded) ---
print("\n  Univariate Cox (archetype as categorical):")
cox_uni = CoxPHFitter()
cox_uni_data = surv_data[["OS_months", "OS_event", "archetype"]].copy()
cox_dummies = pd.get_dummies(cox_uni_data["archetype"], prefix="arch", drop_first=True)
cox_uni_data = pd.concat([cox_uni_data[["OS_months", "OS_event"]], cox_dummies], axis=1)
cox_uni_data = cox_uni_data.dropna()

try:
    cox_uni.fit(cox_uni_data, duration_col="OS_months", event_col="OS_event")
    print(cox_uni.summary[["coef", "exp(coef)", "exp(coef) lower 95%",
                           "exp(coef) upper 95%", "p"]].to_string())
    uni_cindex = cox_uni.concordance_index_
    print(f"  C-index (archetype only): {uni_cindex:.4f}")

    for var_name, row in cox_uni.summary.iterrows():
        survival_results.append({
            "analysis": "Univariate Cox",
            "variable": var_name,
            "n": len(cox_uni_data),
            "events": int(cox_uni_data["OS_event"].sum()),
            "HR": row["exp(coef)"],
            "HR_lower": row["exp(coef) lower 95%"],
            "HR_upper": row["exp(coef) upper 95%"],
            "coef": row["coef"],
            "p_value": row["p"],
            "c_index": uni_cindex,
        })
except Exception as e:
    print(f"  Univariate Cox failed: {e}")
    uni_cindex = np.nan

# --- 6c. Multivariate Cox adjusting for age and sex ---
print("\n  Multivariate Cox (archetype + age + sex):")
cox_mv_data = surv_data.copy()
cox_mv_data["is_male"] = (cox_mv_data["gender"] == "male").astype(int)
cox_mv_data["age_years"] = cox_mv_data["age_at_diagnosis"] / 365.25

cox_mv_dummies = pd.get_dummies(cox_mv_data["archetype"], prefix="arch", drop_first=True)
arch_cols = list(cox_mv_dummies.columns)
cox_mv_data = pd.concat([cox_mv_data[["OS_months", "OS_event", "age_years", "is_male"]],
                          cox_mv_dummies], axis=1)
cox_mv_data = cox_mv_data.dropna()

try:
    cox_mv = CoxPHFitter()
    cox_mv.fit(cox_mv_data, duration_col="OS_months", event_col="OS_event")
    print(cox_mv.summary[["coef", "exp(coef)", "exp(coef) lower 95%",
                          "exp(coef) upper 95%", "p"]].to_string())
    mv_cindex = cox_mv.concordance_index_
    print(f"  C-index (multivariate): {mv_cindex:.4f}")

    for var_name, row in cox_mv.summary.iterrows():
        survival_results.append({
            "analysis": "Multivariate Cox (adj age+sex)",
            "variable": var_name,
            "n": len(cox_mv_data),
            "events": int(cox_mv_data["OS_event"].sum()),
            "HR": row["exp(coef)"],
            "HR_lower": row["exp(coef) lower 95%"],
            "HR_upper": row["exp(coef) upper 95%"],
            "coef": row["coef"],
            "p_value": row["p"],
            "c_index": mv_cindex,
        })
except Exception as e:
    print(f"  Multivariate Cox failed: {e}")
    mv_cindex = np.nan

# --- 6d. C-index comparison: archetype vs individual signatures ---
print("\n  C-index comparison (archetype vs individual signatures):")
cindex_comparison = []

# Archetype c-index already computed above
cindex_comparison.append({
    "model": "Archetype (univariate)",
    "c_index": uni_cindex,
})
cindex_comparison.append({
    "model": "Archetype + age + sex",
    "c_index": mv_cindex,
})

# Individual signature c-indices
for sig_name in ["ros_risk_score", "alt_risk_score", "nrf2_activity",
                 "ferroptosis_vulnerability", "sting_score"]:
    sig_data = surv_data[["OS_months", "OS_event", sig_name]].dropna()
    if len(sig_data) > 10:
        try:
            cph_sig = CoxPHFitter()
            cph_sig.fit(sig_data, duration_col="OS_months", event_col="OS_event")
            ci = cph_sig.concordance_index_
        except Exception:
            ci = np.nan
        cindex_comparison.append({
            "model": sig_name,
            "c_index": ci,
        })
        print(f"    {sig_name:35s}: C-index = {ci:.4f}")

cindex_df = pd.DataFrame(cindex_comparison)
print(f"\n    Archetype (univariate) C-index:     {uni_cindex:.4f}")
print(f"    Archetype (multivariate) C-index:   {mv_cindex:.4f}")

# ======================================================================
# 7. MUTATION ANALYSIS
# ======================================================================
print("\n" + "=" * 70)
print("7. MUTATION ANALYSIS ACROSS ARCHETYPES")
print("=" * 70)

mutation_genes = ["TP53", "KEAP1"]
mutation_cols_found = []

for gene in mutation_genes:
    # Check for mutation column variants
    possible_cols = [
        f"{gene}_mutation", f"{gene}_mut", f"{gene}_mutated",
        f"mut_{gene}", f"mutation_{gene}",
    ]
    found = None
    for col_name in possible_cols:
        if col_name in df_clust.columns:
            found = col_name
            break
    if found:
        mutation_cols_found.append((gene, found))
        print(f"  Found mutation data: {found}")
    else:
        print(f"  NOTE: No {gene} mutation column found in master CSV - skipping")

if mutation_cols_found:
    print("\n  Mutation frequencies by archetype:")
    for gene, col_name in mutation_cols_found:
        print(f"\n  --- {gene} ---")
        contingency_data = []
        for a in archetype_labels:
            sub = df_clust[df_clust["archetype"] == a]
            n_mut = sub[col_name].sum()
            n_total = len(sub)
            freq = n_mut / n_total if n_total > 0 else 0
            contingency_data.append((a, int(n_mut), int(n_total - n_mut)))
            print(f"    {a}: {int(n_mut)}/{n_total} ({100*freq:.1f}%)")

        # Fisher exact test (for 2x2) or chi-square (for larger tables)
        if n_archetypes == 2:
            table_2x2 = np.array([[contingency_data[0][1], contingency_data[0][2]],
                                   [contingency_data[1][1], contingency_data[1][2]]])
            odds_ratio, fisher_p = stats.fisher_exact(table_2x2)
            print(f"    Fisher exact: OR={odds_ratio:.3f}, p={fisher_p:.4f}")
            survival_results.append({
                "analysis": f"Fisher exact ({gene} mutation)",
                "variable": col_name,
                "test_statistic": odds_ratio,
                "p_value": fisher_p,
            })
        else:
            # Use chi-square for larger contingency tables
            contingency_table = np.array([[d[1], d[2]] for d in contingency_data])
            chi2_stat, chi2_p, dof, expected = stats.chi2_contingency(contingency_table)
            print(f"    Chi-square: chi2={chi2_stat:.2f}, p={chi2_p:.4f}, dof={dof}")
            survival_results.append({
                "analysis": f"Chi-square ({gene} mutation)",
                "variable": col_name,
                "test_statistic": chi2_stat,
                "p_value": chi2_p,
            })

            # Also perform pairwise Fisher exact tests
            print(f"    Pairwise Fisher exact tests:")
            for (a1, m1, w1), (a2, m2, w2) in combinations(contingency_data, 2):
                table_pw = np.array([[m1, w1], [m2, w2]])
                or_pw, p_pw = stats.fisher_exact(table_pw)
                sig = "***" if p_pw < 0.001 else "**" if p_pw < 0.01 else "*" if p_pw < 0.05 else "ns"
                print(f"      {a1} vs {a2}: OR={or_pw:.3f}, p={p_pw:.4f} {sig}")
else:
    print("  No mutation data available for analysis.")

# ======================================================================
# 8. GENERATE FIGURES
# ======================================================================
print("\n" + "=" * 70)
print("8. GENERATING FIGURES")
print("=" * 70)

# Colour palette for archetypes
archetype_palette = {}
base_colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
for idx, a in enumerate(archetype_labels):
    archetype_palette[a] = base_colors[idx % len(base_colors)]

# --- 8a. Consensus matrix heatmap for optimal k ---
print("  Generating consensus matrix heatmap...")
fig, ax = plt.subplots(1, 1, figsize=(10, 9))

# Reorder patients by cluster assignment for visual clarity
order = np.argsort(cluster_labels)
C_ordered = C_opt[np.ix_(order, order)]

im = ax.imshow(C_ordered, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto',
               interpolation='nearest')
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Consensus Index", fontsize=11)

# Mark cluster boundaries
sorted_labels = cluster_labels[order]
boundaries = []
for cl in range(1, optimal_k + 1):
    cl_mask = sorted_labels == cl
    cl_indices = np.where(cl_mask)[0]
    if len(cl_indices) > 0:
        boundaries.append(cl_indices[-1] + 0.5)

for b in boundaries[:-1]:
    ax.axhline(b, color='black', linewidth=1.5, alpha=0.8)
    ax.axvline(b, color='black', linewidth=1.5, alpha=0.8)

ax.set_title(f"Consensus Matrix (k={optimal_k}, {N_ITER} iterations)",
             fontsize=14, fontweight='bold')
ax.set_xlabel("Patients (ordered by cluster)", fontsize=12)
ax.set_ylabel("Patients (ordered by cluster)", fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig6a_consensus_matrix.png"),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig6a_consensus_matrix.png")

# --- 8b. CDF plot for k=2 to k=5 ---
print("  Generating CDF plot...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# CDF curves
ax = axes[0]
cdf_colors = {2: "#d62728", 3: "#1f77b4", 4: "#2ca02c", 5: "#ff7f0e"}
for k in K_RANGE:
    sorted_vals, cdf_y = cdf_data[k]
    ax.plot(sorted_vals, cdf_y, '-', color=cdf_colors[k], linewidth=2,
            label=f"k={k} (area={areas[k]:.3f})")
ax.set_xlabel("Consensus Index", fontsize=12)
ax.set_ylabel("CDF", fontsize=12)
ax.set_title("A. Consensus CDF", fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Delta area plot
ax = axes[1]
delta_k = sorted(delta_areas.keys())
delta_vals = [delta_areas[k] for k in delta_k]
ax.bar(delta_k, delta_vals, color=[cdf_colors[k] for k in delta_k],
       edgecolor='white', linewidth=1.5, width=0.6)
ax.set_xlabel("k (number of clusters)", fontsize=12)
ax.set_ylabel("Relative Change in CDF Area", fontsize=12)
ax.set_title("B. Delta Area", fontsize=13, fontweight='bold')
ax.set_xticks(delta_k)
ax.axhline(0, color='black', linewidth=0.8)
ax.grid(True, alpha=0.2, axis='y')

# Mark optimal k
for i, k in enumerate(delta_k):
    if k == optimal_k:
        ax.bar(k, delta_areas[k], color=cdf_colors[k], edgecolor='black',
               linewidth=2.5, width=0.6, zorder=5)
        ax.annotate(f"optimal k={k}", xy=(k, delta_areas[k]),
                    xytext=(k + 0.3, delta_areas[k] + 0.01),
                    fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='black'))

plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig6b_consensus_cdf.png"),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig6b_consensus_cdf.png")

# --- 8c. KM curves by archetype ---
print("  Generating KM curves...")
fig, ax = plt.subplots(1, 1, figsize=(9, 7))
kmf = KaplanMeierFitter()

for a in archetype_labels:
    sub = surv_data[surv_data["archetype"] == a]
    kmf.fit(sub["OS_months"], sub["OS_event"], label=f"{a} (n={len(sub)})")
    kmf.plot_survival_function(ax=ax, color=archetype_palette[a], linewidth=2,
                                ci_alpha=0.15)

ax.set_xlabel("Time (months)", fontsize=12)
ax.set_ylabel("Overall Survival Probability", fontsize=12)
p_str = f"{lr_p:.2e}" if not np.isnan(lr_p) else "N/A"
ax.set_title(f"Overall Survival by Archetype\nLog-rank p = {p_str}",
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='lower left')
ax.grid(True, alpha=0.2)
ax.set_ylim(0, 1.05)
ax.set_xlim(0, None)

plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig6c_km_archetypes.png"),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig6c_km_archetypes.png")

# --- 8d. Heatmap: features x archetypes (mean z-scores, clustered) ---
print("  Generating feature heatmap...")
# Compute mean z-scores per archetype
z_scores_per_archetype = {}
for a in archetype_labels:
    mask = df_clust["archetype"] == a
    z_scores_per_archetype[a] = X[mask.values].mean(axis=0)

heatmap_data = pd.DataFrame(z_scores_per_archetype, index=CLUSTER_FEATURES)

# Cluster features for better visualisation
from scipy.cluster.hierarchy import leaves_list
feat_dist = pdist(heatmap_data.values, metric='euclidean')
if len(feat_dist) > 0:
    feat_linkage = linkage(feat_dist, method='average')
    feat_order = leaves_list(feat_linkage)
    heatmap_data = heatmap_data.iloc[feat_order]

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
            linewidths=0.5, linecolor='white', ax=ax,
            cbar_kws={'label': 'Mean Z-score', 'shrink': 0.8},
            xticklabels=True, yticklabels=True)
ax.set_title(f"Archetype Molecular Profiles (Mean Z-scores, k={optimal_k})",
             fontsize=13, fontweight='bold')
ax.set_xlabel("Archetype", fontsize=12)
ax.set_ylabel("Feature", fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=10)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig6d_archetype_heatmap.png"),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig6d_archetype_heatmap.png")

# --- 8e. Boxplots: key features by archetype ---
print("  Generating boxplots...")
key_features = ["nrf2_activity", "ferroptosis_vulnerability", "sting_score",
                "immune_CD8_T_cells", "HMOX1", "ros_risk_score"]

n_feats = len(key_features)
n_cols = 3
n_rows = (n_feats + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
axes = axes.flatten()

for idx, feat in enumerate(key_features):
    ax = axes[idx]
    plot_data = []
    plot_labels = []
    plot_colors = []
    for a in archetype_labels:
        vals = df_clust[df_clust["archetype"] == a][feat].dropna()
        plot_data.append(vals.values)
        plot_labels.append(a)
        plot_colors.append(archetype_palette[a])

    bp = ax.boxplot(plot_data, patch_artist=True, widths=0.6,
                    flierprops=dict(marker='o', markersize=3, alpha=0.4))
    for patch, color in zip(bp['boxes'], plot_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # KW test
    nonempty = [d for d in plot_data if len(d) > 0]
    if len(nonempty) >= 2:
        stat_kw, p_kw = stats.kruskal(*nonempty)
        ax.set_title(f"{feat}\nKruskal-Wallis p = {p_kw:.2e}",
                     fontsize=11, fontweight='bold')
    else:
        ax.set_title(f"{feat}", fontsize=11, fontweight='bold')

    ax.set_xticklabels([a.replace("-", "-\n") for a in plot_labels],
                       fontsize=8, rotation=0, ha='center')
    ax.set_ylabel(feat, fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')

# Hide unused axes
for idx in range(n_feats, len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig6e_archetype_boxplots.png"),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig6e_archetype_boxplots.png")

# ======================================================================
# 9. SAVE OUTPUT TABLES & UPDATE MASTER CSV
# ======================================================================
print("\n" + "=" * 70)
print("9. SAVING OUTPUT TABLES")
print("=" * 70)

# --- 9a. Survival table ---
surv_out = pd.DataFrame(survival_results)
surv_out.to_csv(os.path.join(TABLES, "archetype_survival.csv"), index=False)
print(f"  Saved: results/tables/archetype_survival.csv ({len(surv_out)} rows)")

# --- 9b. C-index comparison ---
cindex_df.to_csv(os.path.join(TABLES, "cindex_comparison.csv"), index=False)
print(f"  Saved: results/tables/cindex_comparison.csv ({len(cindex_df)} rows)")

# --- 9c. Update master CSV with archetype column ---
master = pd.read_csv(os.path.join(DATA, "tcga_convergence_master.csv"))

# Build a mapping: patientId -> archetype
archetype_map = df_clust.set_index("patientId")["archetype"].to_dict()

# Drop existing archetype column if present
if "archetype" in master.columns:
    master = master.drop(columns=["archetype"])

master["archetype"] = master["patientId"].map(archetype_map)
master.to_csv(os.path.join(DATA, "tcga_convergence_master.csv"), index=False)
n_assigned = master["archetype"].notna().sum()
print(f"  Updated master CSV with archetype column "
      f"({n_assigned}/{len(master)} patients assigned)")

# ======================================================================
# SUMMARY
# ======================================================================
print("\n" + "=" * 70)
print("SUMMARY: PATIENT ARCHETYPE ANALYSIS")
print("=" * 70)

print(f"\n  Consensus clustering: {N_ITER} iterations, 80% subsampling")
print(f"  Features used: {len(CLUSTER_FEATURES)} convergence features")
print(f"  Patients clustered: {n_patients}")
print(f"  Optimal k: {optimal_k}")

print(f"\n  Archetype composition:")
for a in archetype_labels:
    n_a = (df_clust["archetype"] == a).sum()
    pct = 100 * n_a / len(df_clust)
    print(f"    {a}: {n_a} patients ({pct:.1f}%)")

print(f"\n  Survival:")
print(f"    Log-rank p = {lr_p:.2e}")
if not np.isnan(uni_cindex):
    print(f"    C-index (archetype univariate): {uni_cindex:.4f}")
if not np.isnan(mv_cindex):
    print(f"    C-index (archetype + age + sex): {mv_cindex:.4f}")

print(f"\n  Features with significant archetype differences (p < 0.05):")
for r in kw_results:
    if r["p_value"] < 0.05:
        print(f"    {r['feature']:30s}: p={r['p_value']:.2e} {r['significant']}")

print(f"\n  Outputs saved:")
print(f"    - results/tables/archetype_features.csv")
print(f"    - results/tables/archetype_survival.csv")
print(f"    - results/tables/cindex_comparison.csv")
print(f"    - results/model/consensus_matrices.npz")
print(f"    - results/figures/main/fig6a_consensus_matrix.png")
print(f"    - results/figures/main/fig6b_consensus_cdf.png")
print(f"    - results/figures/main/fig6c_km_archetypes.png")
print(f"    - results/figures/main/fig6d_archetype_heatmap.png")
print(f"    - results/figures/main/fig6e_archetype_boxplots.png")
print(f"    - data/tcga_convergence_master.csv (updated with archetype)")

print("\n" + "=" * 70)
print("DONE: 06_patient_archetypes.py")
print("=" * 70)
