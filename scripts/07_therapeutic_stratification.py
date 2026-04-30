"""
07_therapeutic_stratification.py — Drug sensitivity mapping & clinical prediction

Maps drug sensitivity across patient archetypes and builds clinical
prediction tools for therapeutic stratification.

1. Drug sensitivity analysis: correlate risk scores with drug target genes
2. Drug-relevant gene panel across archetypes (or dual_group fallback)
3. Immunotherapy response prediction (immunophenoscore, cytolytic activity)
4. Nomogram construction (Cox model, C-index comparison)
5. Decision curve analysis (3-year and 5-year OS)

Outputs:
  - results/tables/drug_targets_by_archetype.csv
  - results/tables/immunotherapy_prediction.csv
  - results/tables/cindex_comparison_full.csv
  - results/tables/dca_results.csv
  - Figures to results/figures/main/
"""
import pandas as pd
import numpy as np
from scipy import stats
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
# LOAD DATA
# ======================================================================
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

master = pd.read_csv(os.path.join(DATA, "tcga_convergence_master.csv"))
print(f"Master dataframe: {master.shape[0]} patients, {master.shape[1]} columns")

# Load full expression matrix for drug target genes
expr_full = pd.read_csv(PATHS["expression_full"], index_col=0)
expr_full = expr_full[expr_full.index.notna()]
print(f"Expression matrix: {expr_full.shape[0]} genes x {expr_full.shape[1]} samples")

# Determine grouping variable: archetype if available, otherwise dual_group
if "archetype" in master.columns:
    GROUP_COL = "archetype"
    print(f"Using archetype column for grouping")
else:
    GROUP_COL = "dual_group"
    print(f"Archetype column not found; using dual_group as fallback")

df = master.dropna(subset=["OS_months", "OS_event"]).copy()
df = df[df["OS_months"] > 0].copy()
print(f"Working with {len(df)} patients ({int(df['OS_event'].sum())} events)")
print(f"Groups ({GROUP_COL}): {df[GROUP_COL].value_counts().to_dict()}")

# Common patients between master and expression matrix
common_patients = sorted(set(df["patientId"]) & set(expr_full.columns))
print(f"Patients with expression data: {len(common_patients)}")

# ======================================================================
# DRUG-RELEVANT GENE PANEL
# ======================================================================
DRUG_TARGETS = {
    "SLC7A11":  {"drug": "Erastin", "mechanism": "Ferroptosis (system Xc- inhibition)"},
    "GPX4":     {"drug": "RSL3", "mechanism": "Ferroptosis (GPX4 inhibition)"},
    "VEGFA":    {"drug": "Sorafenib / Lenvatinib", "mechanism": "Anti-angiogenic TKI"},
    "KDR":      {"drug": "Sorafenib / Lenvatinib", "mechanism": "Anti-angiogenic TKI (VEGFR2)"},
    "MTOR":     {"drug": "Everolimus", "mechanism": "mTOR inhibition"},
    "RPS6KB1":  {"drug": "Everolimus", "mechanism": "mTOR pathway (S6K1)"},
    "CD274":    {"drug": "Atezolizumab / Nivolumab", "mechanism": "PD-L1 ICB"},
    "PDCD1":    {"drug": "Pembrolizumab", "mechanism": "PD-1 ICB"},
    "TOP2A":    {"drug": "Doxorubicin", "mechanism": "Topoisomerase II inhibition"},
}

# Immunotherapy-related gene sets
EFFECTOR_GENES = ["CD8A", "GZMA", "GZMB", "PRF1", "IFNG"]
CHECKPOINT_GENES = ["CTLA4", "PDCD1", "LAG3", "HAVCR2", "TIGIT"]
CYT_GENES = ["GZMA", "PRF1"]

# ======================================================================
# 1. DRUG SENSITIVITY ANALYSIS
# ======================================================================
print("\n" + "=" * 70)
print("1. DRUG SENSITIVITY ANALYSIS")
print("=" * 70)

# Extract drug target expression for common patients
drug_gene_list = list(DRUG_TARGETS.keys())
available_drug_genes = [g for g in drug_gene_list if g in expr_full.index]
missing_drug_genes = [g for g in drug_gene_list if g not in expr_full.index]
if missing_drug_genes:
    print(f"  Warning: genes not found in expression matrix: {missing_drug_genes}")
print(f"  Available drug target genes: {len(available_drug_genes)}/{len(drug_gene_list)}")

# Build drug target expression dataframe for common patients
drug_expr = expr_full.loc[available_drug_genes, common_patients].T
drug_expr.index.name = "patientId"
drug_expr = drug_expr.reset_index()

# Merge with master data
drug_merged = pd.merge(
    df[["patientId", "ros_risk_score", "alt_risk_score", "combined_score", GROUP_COL]],
    drug_expr, on="patientId", how="inner"
)
print(f"  Drug analysis dataframe: {len(drug_merged)} patients")

# Correlate risk scores with drug target expression
print("\n  Spearman correlations: risk scores vs drug target expression")
print(f"  {'Gene':<10} {'Drug':<30} {'vs combined_score':>18} {'vs ros_risk':>12} {'vs alt_risk':>12}")
print("  " + "-" * 82)

corr_results = []
for gene in available_drug_genes:
    info = DRUG_TARGETS[gene]
    row = {"gene": gene, "drug": info["drug"], "mechanism": info["mechanism"]}
    for score_col, label in [("combined_score", "combined"), ("ros_risk_score", "ros"), ("alt_risk_score", "alt")]:
        valid = drug_merged[[score_col, gene]].dropna()
        if len(valid) > 30:
            r, p = stats.spearmanr(valid[score_col], valid[gene])
            row[f"spearman_r_{label}"] = r
            row[f"p_value_{label}"] = p
        else:
            row[f"spearman_r_{label}"] = np.nan
            row[f"p_value_{label}"] = np.nan
    corr_results.append(row)

    sig_c = "*" if row.get("p_value_combined", 1) < 0.05 else " "
    sig_r = "*" if row.get("p_value_ros", 1) < 0.05 else " "
    sig_a = "*" if row.get("p_value_alt", 1) < 0.05 else " "
    print(f"  {gene:<10} {info['drug']:<30} r={row.get('spearman_r_combined', 0):>6.3f}{sig_c}"
          f"  r={row.get('spearman_r_ros', 0):>6.3f}{sig_r}"
          f"  r={row.get('spearman_r_alt', 0):>6.3f}{sig_a}")

corr_df = pd.DataFrame(corr_results)

# ======================================================================
# 2. DRUG TARGET EXPRESSION BY ARCHETYPE / GROUP
# ======================================================================
print("\n" + "=" * 70)
print("2. DRUG TARGET EXPRESSION BY GROUP")
print("=" * 70)

groups = sorted(drug_merged[GROUP_COL].dropna().unique())
print(f"  Groups: {groups}")

# Compute mean expression per group, plus Kruskal-Wallis test
group_stats = []
for gene in available_drug_genes:
    row = {"gene": gene, "drug": DRUG_TARGETS[gene]["drug"],
           "mechanism": DRUG_TARGETS[gene]["mechanism"]}
    group_vals = []
    for grp in groups:
        vals = drug_merged.loc[drug_merged[GROUP_COL] == grp, gene].dropna()
        row[f"mean_{grp}"] = vals.mean()
        row[f"std_{grp}"] = vals.std()
        row[f"n_{grp}"] = len(vals)
        group_vals.append(vals.values)

    # Kruskal-Wallis across groups
    valid_groups = [v for v in group_vals if len(v) >= 3]
    if len(valid_groups) >= 2:
        h_stat, kw_p = stats.kruskal(*valid_groups)
        row["kruskal_H"] = h_stat
        row["kruskal_p"] = kw_p
    else:
        row["kruskal_H"] = np.nan
        row["kruskal_p"] = np.nan
    group_stats.append(row)

    sig = "*" if row.get("kruskal_p", 1) < 0.05 else " "
    means_str = "  ".join([f"{grp[:12]:>12}: {row.get(f'mean_{grp}', 0):.2f}" for grp in groups])
    print(f"  {gene:<10} KW p={row.get('kruskal_p', 1):.4f}{sig}   {means_str}")

drug_targets_df = pd.DataFrame(group_stats)
drug_targets_df.to_csv(os.path.join(TABLES, "drug_targets_by_archetype.csv"), index=False)
print(f"\n  Saved: drug_targets_by_archetype.csv")

# ======================================================================
# 3. IMMUNOTHERAPY RESPONSE PREDICTION
# ======================================================================
print("\n" + "=" * 70)
print("3. IMMUNOTHERAPY RESPONSE PREDICTION")
print("=" * 70)

# Get effector and checkpoint gene expression from full expression matrix
all_immuno_genes = EFFECTOR_GENES + CHECKPOINT_GENES
avail_effector = [g for g in EFFECTOR_GENES if g in expr_full.index]
avail_checkpoint = [g for g in CHECKPOINT_GENES if g in expr_full.index]
avail_cyt = [g for g in CYT_GENES if g in expr_full.index]

print(f"  Effector genes available: {avail_effector}")
print(f"  Checkpoint genes available: {avail_checkpoint}")
print(f"  Cytolytic genes available: {avail_cyt}")

# Extract expression for common patients
immuno_expr = expr_full.loc[
    [g for g in all_immuno_genes if g in expr_full.index], common_patients
].T.copy()
immuno_expr.index.name = "patientId"
immuno_expr = immuno_expr.reset_index()

# Merge with master
immuno_merged = pd.merge(
    df[["patientId", GROUP_COL, "combined_score", "ros_risk_score",
        "alt_risk_score", "OS_months", "OS_event"]],
    immuno_expr, on="patientId", how="inner"
)

# Z-score each gene
for gene in avail_effector + avail_checkpoint:
    col = immuno_merged[gene].astype(float)
    immuno_merged[f"{gene}_z"] = (col - col.mean()) / (col.std() + 1e-10)

# Immunophenoscore-like metric:
# mean z-score(effector genes) - mean z-score(checkpoint genes)
effector_z_cols = [f"{g}_z" for g in avail_effector]
checkpoint_z_cols = [f"{g}_z" for g in avail_checkpoint]

immuno_merged["effector_score"] = immuno_merged[effector_z_cols].mean(axis=1)
immuno_merged["checkpoint_score"] = immuno_merged[checkpoint_z_cols].mean(axis=1)
immuno_merged["immunophenoscore"] = immuno_merged["effector_score"] - immuno_merged["checkpoint_score"]

# Cytolytic activity score: geometric mean of GZMA and PRF1
if len(avail_cyt) == 2:
    gzma_vals = immuno_merged["GZMA"].astype(float).clip(lower=1e-6)
    prf1_vals = immuno_merged["PRF1"].astype(float).clip(lower=1e-6)
    immuno_merged["cytolytic_activity"] = np.sqrt(gzma_vals * prf1_vals)
elif len(avail_cyt) == 1:
    immuno_merged["cytolytic_activity"] = immuno_merged[avail_cyt[0]].astype(float)
else:
    immuno_merged["cytolytic_activity"] = np.nan

# Compare across groups
print(f"\n  Immunophenoscore by group:")
immuno_group_results = []
for grp in groups:
    mask = immuno_merged[GROUP_COL] == grp
    ips = immuno_merged.loc[mask, "immunophenoscore"]
    cyt = immuno_merged.loc[mask, "cytolytic_activity"]
    eff = immuno_merged.loc[mask, "effector_score"]
    chk = immuno_merged.loc[mask, "checkpoint_score"]
    row = {
        "group": grp, "n": int(mask.sum()),
        "immunophenoscore_mean": ips.mean(), "immunophenoscore_std": ips.std(),
        "effector_mean": eff.mean(), "checkpoint_mean": chk.mean(),
        "cytolytic_mean": cyt.mean(), "cytolytic_std": cyt.std(),
    }
    immuno_group_results.append(row)
    print(f"    {grp}: IPS={ips.mean():.3f} +/- {ips.std():.3f}, "
          f"CYT={cyt.mean():.3f}, n={int(mask.sum())}")

# Kruskal-Wallis for immunophenoscore
ips_groups = [immuno_merged.loc[immuno_merged[GROUP_COL] == g, "immunophenoscore"].dropna().values
              for g in groups]
valid_ips = [v for v in ips_groups if len(v) >= 3]
if len(valid_ips) >= 2:
    h_stat, kw_p = stats.kruskal(*valid_ips)
    print(f"\n  Kruskal-Wallis (immunophenoscore): H={h_stat:.2f}, p={kw_p:.2e}")

# Kruskal-Wallis for cytolytic activity
cyt_groups = [immuno_merged.loc[immuno_merged[GROUP_COL] == g, "cytolytic_activity"].dropna().values
              for g in groups]
valid_cyt = [v for v in cyt_groups if len(v) >= 3]
if len(valid_cyt) >= 2:
    h_stat_c, kw_p_c = stats.kruskal(*valid_cyt)
    print(f"  Kruskal-Wallis (cytolytic activity): H={h_stat_c:.2f}, p={kw_p_c:.2e}")

immuno_results_df = pd.DataFrame(immuno_group_results)
immuno_results_df.to_csv(os.path.join(TABLES, "immunotherapy_prediction.csv"), index=False)
print(f"\n  Saved: immunotherapy_prediction.csv")

# ======================================================================
# 4. NOMOGRAM CONSTRUCTION
# ======================================================================
print("\n" + "=" * 70)
print("4. NOMOGRAM CONSTRUCTION")
print("=" * 70)

# Prepare clinical variables
df["male"] = (df["gender"].str.lower() == "male").astype(int) if "gender" in df.columns else np.nan

if "age_at_diagnosis" in df.columns:
    age = df["age_at_diagnosis"]
    df["age_years"] = np.where(age > 200, age / 365.25, age)
elif "age" in df.columns:
    df["age_years"] = df["age"]
else:
    df["age_years"] = np.nan

# Create dummy variables for groups
group_dummies = pd.get_dummies(df[GROUP_COL], prefix="grp", drop_first=True)
group_dummy_cols = group_dummies.columns.tolist()
df = pd.concat([df, group_dummies], axis=1)

# Build nomogram Cox model: group dummies + age + sex
nomo_vars = group_dummy_cols.copy()
if "age_years" in df.columns and df["age_years"].notna().sum() > 100:
    nomo_vars.append("age_years")
if "male" in df.columns and df["male"].notna().sum() > 100:
    nomo_vars.append("male")

nomo_cols = ["OS_months", "OS_event"] + nomo_vars
nomo_df = df[nomo_cols].dropna()
nomo_df = nomo_df[nomo_df["OS_months"] > 0].copy()

print(f"  Nomogram variables: {nomo_vars}")
print(f"  Patients for nomogram: {len(nomo_df)}")

cph_nomo = CoxPHFitter()
cph_nomo.fit(nomo_df, duration_col="OS_months", event_col="OS_event")
print("\n  Cox model summary (archetype nomogram):")
summary = cph_nomo.summary[["coef", "exp(coef)", "p"]].copy()
print(summary.to_string())

# Compute C-index for the archetype model
pred_nomo = cph_nomo.predict_partial_hazard(nomo_df)
ci_nomo = concordance_index(nomo_df["OS_months"], -pred_nomo.values.ravel(), nomo_df["OS_event"])
print(f"\n  C-index (archetype model): {ci_nomo:.4f}")

# ======================================================================
# 4b. C-INDEX COMPARISON
# ======================================================================
print("\n" + "=" * 70)
print("4b. C-INDEX COMPARISON (bootstrap)")
print("=" * 70)

# Prepare comparison models
# Model 1: Archetype model (already fitted above)
# Model 2: Combined score only
# Model 3: ROS risk score only
# Model 4: Alt risk score only

# Ensure all patients have all score columns
score_df = df[["OS_months", "OS_event", "combined_score", "ros_risk_score", "alt_risk_score"]
              + nomo_vars].dropna()
score_df = score_df[score_df["OS_months"] > 0].copy()

models_config = {
    "Archetype + Age + Sex": nomo_vars,
    "Combined Score": ["combined_score"],
    "ROS Risk Score": ["ros_risk_score"],
    "Alt Risk Score": ["alt_risk_score"],
    "Archetype Only": group_dummy_cols,
}

# Point estimate C-indices
print(f"\n  Point-estimate C-indices:")
point_cindices = {}
for model_name, pred_vars in models_config.items():
    try:
        model_df = score_df[["OS_months", "OS_event"] + pred_vars].dropna()
        model_df = model_df[model_df["OS_months"] > 0]
        if len(model_df) < 50:
            continue
        cph_tmp = CoxPHFitter()
        cph_tmp.fit(model_df, duration_col="OS_months", event_col="OS_event")
        pred_tmp = cph_tmp.predict_partial_hazard(model_df)
        ci_tmp = concordance_index(model_df["OS_months"], -pred_tmp.values.ravel(), model_df["OS_event"])
        point_cindices[model_name] = ci_tmp
        print(f"    {model_name:<30} C-index: {ci_tmp:.4f}")
    except Exception as e:
        print(f"    {model_name:<30} FAILED: {e}")

# Bootstrap C-index comparison
n_boot = 500
boot_cindices = {name: [] for name in models_config}

print(f"\n  Running {n_boot} bootstrap iterations...")
for b in range(n_boot):
    idx = np.random.choice(len(score_df), len(score_df), replace=True)
    boot_data = score_df.iloc[idx].copy()

    for model_name, pred_vars in models_config.items():
        try:
            model_df = boot_data[["OS_months", "OS_event"] + pred_vars].dropna()
            model_df = model_df[model_df["OS_months"] > 0]
            if len(model_df) < 50:
                continue
            cph_b = CoxPHFitter()
            cph_b.fit(model_df, duration_col="OS_months", event_col="OS_event")
            pred_b = cph_b.predict_partial_hazard(model_df)
            ci_b = concordance_index(model_df["OS_months"], -pred_b.values.ravel(), model_df["OS_event"])
            boot_cindices[model_name].append(ci_b)
        except Exception:
            pass

    if (b + 1) % 100 == 0:
        print(f"    Bootstrap {b + 1}/{n_boot}")

print("\n  Bootstrap C-index comparison:")
cindex_results = []
for model_name, cis in boot_cindices.items():
    if len(cis) > 10:
        mean_ci = np.mean(cis)
        lo, hi = np.percentile(cis, [2.5, 97.5])
        cindex_results.append({
            "model": model_name,
            "c_index_mean": mean_ci,
            "c_index_95CI_lo": lo,
            "c_index_95CI_hi": hi,
            "n_successful_boots": len(cis),
        })
        print(f"    {model_name:<30} C-index: {mean_ci:.4f} ({lo:.4f}-{hi:.4f})")

cindex_df = pd.DataFrame(cindex_results)
cindex_df.to_csv(os.path.join(TABLES, "cindex_comparison_full.csv"), index=False)
print(f"\n  Saved: cindex_comparison_full.csv")

# ======================================================================
# 5. DECISION CURVE ANALYSIS
# ======================================================================
print("\n" + "=" * 70)
print("5. DECISION CURVE ANALYSIS")
print("=" * 70)


def decision_curve_analysis(time, event, predicted_prob, timepoint, thresholds=None):
    """
    Compute net benefit for a model at a given timepoint.
    Net benefit = TP/N - FP/N * (pt / (1-pt))
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 0.61, 0.01)

    # Binary outcome at timepoint: event happened before timepoint
    outcome = ((time <= timepoint) & (event == 1)).astype(int)
    # Predicted probability of event = 1 - survival probability
    pred_event = 1 - predicted_prob

    n = len(outcome)
    prevalence = outcome.mean()

    net_benefits_model = []
    net_benefits_all = []
    net_benefits_none = []

    for pt in thresholds:
        # Treat all
        nb_all = prevalence - (1 - prevalence) * pt / (1 - pt + 1e-10)
        net_benefits_all.append(nb_all)

        # Treat none
        net_benefits_none.append(0)

        # Model
        predicted_positive = pred_event >= pt
        tp = (predicted_positive & (outcome == 1)).sum()
        fp = (predicted_positive & (outcome == 0)).sum()
        nb_model = tp / n - fp / n * pt / (1 - pt + 1e-10)
        net_benefits_model.append(nb_model)

    return thresholds, net_benefits_model, net_benefits_all, net_benefits_none


# Fit models for DCA
# Archetype model
dca_nomo_df = score_df[["OS_months", "OS_event"] + nomo_vars].dropna()
dca_nomo_df = dca_nomo_df[dca_nomo_df["OS_months"] > 0].copy()
cph_dca_arch = CoxPHFitter()
cph_dca_arch.fit(dca_nomo_df, duration_col="OS_months", event_col="OS_event")
surv_arch = cph_dca_arch.predict_survival_function(dca_nomo_df)

# Combined score model
dca_score_df = score_df[["OS_months", "OS_event", "combined_score"]].dropna()
dca_score_df = dca_score_df[dca_score_df["OS_months"] > 0].copy()
cph_dca_score = CoxPHFitter()
cph_dca_score.fit(dca_score_df, duration_col="OS_months", event_col="OS_event")
surv_score = cph_dca_score.predict_survival_function(dca_score_df)

# We need common patient indices for fair comparison
# Use the archetype model patients (subset of score_df)
arch_idx = dca_nomo_df.index
# The score model needs to be evaluated on same patients
dca_score_common = dca_score_df.loc[dca_score_df.index.isin(arch_idx)].copy()
# Re-fit combined score on same patients
cph_dca_score2 = CoxPHFitter()
cph_dca_score2.fit(dca_score_common, duration_col="OS_months", event_col="OS_event")
surv_score_common = cph_dca_score2.predict_survival_function(dca_score_common)

timepoints = {"3-year": 36, "5-year": 60}
dca_all_results = []

fig_dca, axes_dca = plt.subplots(1, 2, figsize=(14, 6))

for ax, (tp_name, tp_months) in zip(axes_dca, timepoints.items()):
    # Get predicted survival at timepoint for archetype model
    if tp_months <= surv_arch.index.max():
        closest_idx_a = surv_arch.index[np.abs(surv_arch.index - tp_months).argmin()]
        pred_surv_arch = surv_arch.loc[closest_idx_a].values

        thresholds, nb_arch, nb_all, nb_none = decision_curve_analysis(
            dca_nomo_df["OS_months"].values, dca_nomo_df["OS_event"].values,
            pred_surv_arch, tp_months
        )

        ax.plot(thresholds, nb_arch, color='steelblue', linewidth=2.5,
                label='Archetype Model')
        ax.plot(thresholds, nb_all, color='gray', linewidth=1.5,
                linestyle='--', label='Treat All')
        ax.plot(thresholds, nb_none, color='black', linewidth=1.5,
                linestyle=':', label='Treat None')

        # Combined score model (on common patients)
        if tp_months <= surv_score_common.index.max():
            closest_idx_s = surv_score_common.index[
                np.abs(surv_score_common.index - tp_months).argmin()
            ]
            pred_surv_score = surv_score_common.loc[closest_idx_s].values

            # Match patient indices
            _, nb_score, _, _ = decision_curve_analysis(
                dca_score_common["OS_months"].values,
                dca_score_common["OS_event"].values,
                pred_surv_score, tp_months
            )
            ax.plot(thresholds, nb_score, color='darkorange', linewidth=2,
                    linestyle='-', alpha=0.9, label='Combined Score')

        ax.set_xlabel("Threshold Probability", fontsize=12)
        ax.set_ylabel("Net Benefit", fontsize=12)
        ax.set_title(f"{tp_name} Overall Survival", fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.set_xlim(0, 0.6)
        y_max = max(max(nb_arch), max(nb_all)) * 1.3 if max(nb_arch) > 0 else 0.3
        ax.set_ylim(-0.05, max(0.3, y_max))
        ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Record DCA metrics
        pos_arch = [t for t, nb in zip(thresholds, nb_arch) if nb > 0]
        pos_score_range = ""
        if tp_months <= surv_score_common.index.max():
            pos_score_vals = [t for t, nb in zip(thresholds, nb_score) if nb > 0]
            if pos_score_vals:
                pos_score_range = f"{min(pos_score_vals):.2f}-{max(pos_score_vals):.2f}"

        dca_all_results.append({
            "timepoint": tp_name,
            "model": "Archetype Model",
            "positive_nb_range": f"{min(pos_arch):.2f}-{max(pos_arch):.2f}" if pos_arch else "none",
            "max_nb": max(nb_arch),
        })
        dca_all_results.append({
            "timepoint": tp_name,
            "model": "Combined Score",
            "positive_nb_range": pos_score_range if pos_score_range else "none",
            "max_nb": max(nb_score) if tp_months <= surv_score_common.index.max() else np.nan,
        })

        if pos_arch:
            print(f"  {tp_name} Archetype model: positive NB at thresholds "
                  f"{min(pos_arch):.2f}-{max(pos_arch):.2f}, max NB={max(nb_arch):.4f}")
    else:
        print(f"  {tp_name}: timepoint exceeds max follow-up, skipping")
        ax.text(0.5, 0.5, f"{tp_name}\nInsufficient follow-up", ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color='gray')

plt.suptitle("Decision Curve Analysis", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig7e_dca_curves.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: fig7e_dca_curves.png")

dca_results_df = pd.DataFrame(dca_all_results)
dca_results_df.to_csv(os.path.join(TABLES, "dca_results.csv"), index=False)
print(f"  Saved: dca_results.csv")

# ======================================================================
# 6. FIGURES
# ======================================================================
print("\n" + "=" * 70)
print("6. GENERATING FIGURES")
print("=" * 70)

# ── 6a. Heatmap: drug target expression by archetype ─────────────────

# Compute mean z-scored expression per group for heatmap
heatmap_data = {}
for gene in available_drug_genes:
    gene_vals = drug_merged[gene].astype(float)
    gene_z = (gene_vals - gene_vals.mean()) / (gene_vals.std() + 1e-10)
    drug_merged[f"{gene}_z"] = gene_z
    for grp in groups:
        mask = drug_merged[GROUP_COL] == grp
        heatmap_data.setdefault(grp, {})[gene] = gene_z[mask].mean()

heatmap_df = pd.DataFrame(heatmap_data).T
# Reorder columns by drug type
col_order = [g for g in ["SLC7A11", "GPX4", "VEGFA", "KDR", "MTOR", "RPS6KB1",
                         "CD274", "PDCD1", "TOP2A"] if g in heatmap_df.columns]
heatmap_df = heatmap_df[col_order]

# Drug labels for annotation
drug_labels = [DRUG_TARGETS.get(g, {}).get("drug", g) for g in col_order]

fig, ax = plt.subplots(figsize=(12, max(4, len(groups) * 0.8 + 2)))
sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            linewidths=0.8, ax=ax, cbar_kws={"label": "Mean z-score"},
            xticklabels=col_order, yticklabels=heatmap_df.index)

# Add drug names as secondary x-axis labels
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks([i + 0.5 for i in range(len(col_order))])
ax2.set_xticklabels(drug_labels, fontsize=8, rotation=30, ha='left')
ax2.tick_params(length=0)

ax.set_title("Drug Target Expression by Patient Group\n", fontsize=14, fontweight='bold', pad=30)
ax.set_xlabel("Drug Target Gene", fontsize=11)
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig7a_drug_target_heatmap.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig7a_drug_target_heatmap.png")

# ── 6b. Boxplots: key drug targets by group ──────────────────────────

key_targets = [g for g in ["SLC7A11", "GPX4", "VEGFA", "CD274"] if g in available_drug_genes]
n_key = len(key_targets)

fig, axes = plt.subplots(1, n_key, figsize=(4 * n_key, 5))
if n_key == 1:
    axes = [axes]

# Define a color palette for groups
group_palette = sns.color_palette("Set2", len(groups))
group_colors = dict(zip(groups, group_palette))

for i, gene in enumerate(key_targets):
    ax = axes[i]
    plot_data = drug_merged[[GROUP_COL, gene]].dropna()
    plot_data[gene] = plot_data[gene].astype(float)

    bp = ax.boxplot(
        [plot_data.loc[plot_data[GROUP_COL] == g, gene].values for g in groups],
        labels=[g.split(": ")[-1] if ": " in g else g for g in groups],
        patch_artist=True,
        widths=0.6,
        boxprops=dict(linewidth=1.2),
        medianprops=dict(color='black', linewidth=1.5),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
    )
    for patch, grp in zip(bp['boxes'], groups):
        patch.set_facecolor(group_colors[grp])
        patch.set_alpha(0.75)

    # Kruskal-Wallis p-value
    grp_vals = [plot_data.loc[plot_data[GROUP_COL] == g, gene].values for g in groups]
    valid_g = [v for v in grp_vals if len(v) >= 3]
    if len(valid_g) >= 2:
        _, kw_p = stats.kruskal(*valid_g)
        p_str = f"p={kw_p:.2e}" if kw_p < 0.001 else f"p={kw_p:.3f}"
        ax.set_title(f"{gene}\n({DRUG_TARGETS[gene]['drug']})\n{p_str}",
                     fontsize=10, fontweight='bold')
    else:
        ax.set_title(f"{gene}\n({DRUG_TARGETS[gene]['drug']})", fontsize=10, fontweight='bold')

    ax.set_ylabel("Expression (log2 TPM)", fontsize=10)
    ax.tick_params(axis='x', rotation=25, labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle("Key Drug Target Expression by Patient Group",
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig7b_drug_target_boxplots.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig7b_drug_target_boxplots.png")

# ── 6c. Immunophenoscore + cytolytic activity by group ────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Immunophenoscore
ax = axes[0]
ips_plot_data = immuno_merged[[GROUP_COL, "immunophenoscore"]].dropna()
bp_ips = ax.boxplot(
    [ips_plot_data.loc[ips_plot_data[GROUP_COL] == g, "immunophenoscore"].values for g in groups],
    labels=[g.split(": ")[-1] if ": " in g else g for g in groups],
    patch_artist=True, widths=0.6,
    medianprops=dict(color='black', linewidth=1.5),
)
for patch, grp in zip(bp_ips['boxes'], groups):
    patch.set_facecolor(group_colors[grp])
    patch.set_alpha(0.75)

if len(valid_ips) >= 2:
    _, kw_p_ips = stats.kruskal(*valid_ips)
    p_str_ips = f"p={kw_p_ips:.2e}" if kw_p_ips < 0.001 else f"p={kw_p_ips:.3f}"
    ax.set_title(f"Immunophenoscore\n{p_str_ips}", fontsize=12, fontweight='bold')
else:
    ax.set_title("Immunophenoscore", fontsize=12, fontweight='bold')
ax.set_ylabel("Score (effector - checkpoint z-scores)", fontsize=10)
ax.tick_params(axis='x', rotation=25, labelsize=9)
ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Cytolytic activity
ax = axes[1]
cyt_plot_data = immuno_merged[[GROUP_COL, "cytolytic_activity"]].dropna()
bp_cyt = ax.boxplot(
    [cyt_plot_data.loc[cyt_plot_data[GROUP_COL] == g, "cytolytic_activity"].values for g in groups],
    labels=[g.split(": ")[-1] if ": " in g else g for g in groups],
    patch_artist=True, widths=0.6,
    medianprops=dict(color='black', linewidth=1.5),
)
for patch, grp in zip(bp_cyt['boxes'], groups):
    patch.set_facecolor(group_colors[grp])
    patch.set_alpha(0.75)

if len(valid_cyt) >= 2:
    _, kw_p_cyt = stats.kruskal(*valid_cyt)
    p_str_cyt = f"p={kw_p_cyt:.2e}" if kw_p_cyt < 0.001 else f"p={kw_p_cyt:.3f}"
    ax.set_title(f"Cytolytic Activity\n{p_str_cyt}", fontsize=12, fontweight='bold')
else:
    ax.set_title("Cytolytic Activity", fontsize=12, fontweight='bold')
ax.set_ylabel("Geometric mean (GZMA, PRF1)", fontsize=10)
ax.tick_params(axis='x', rotation=25, labelsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.suptitle("Immunotherapy Response Prediction by Patient Group",
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig7c_immunotherapy_prediction.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig7c_immunotherapy_prediction.png")

# ── 6d. C-index comparison bar chart ─────────────────────────────────

if cindex_results:
    fig, ax = plt.subplots(figsize=(8, 5))
    ci_df = pd.DataFrame(cindex_results).sort_values("c_index_mean", ascending=True)

    y_pos = range(len(ci_df))
    bars = ax.barh(y_pos, ci_df["c_index_mean"],
                   xerr=[ci_df["c_index_mean"] - ci_df["c_index_95CI_lo"],
                         ci_df["c_index_95CI_hi"] - ci_df["c_index_mean"]],
                   color=sns.color_palette("viridis", len(ci_df)),
                   edgecolor='black', linewidth=0.5, capsize=4, height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ci_df["model"], fontsize=10)
    ax.set_xlabel("C-index (95% CI)", fontsize=12)
    ax.set_title("Model Comparison: Concordance Index", fontsize=13, fontweight='bold')
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='No discrimination')
    ax.set_xlim(0.45, max(ci_df["c_index_95CI_hi"].max() + 0.03, 0.75))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotate bars with values
    for i, (_, row) in enumerate(ci_df.iterrows()):
        ax.text(row["c_index_mean"] + 0.005, i,
                f"{row['c_index_mean']:.3f}",
                va='center', fontsize=9, fontweight='bold')

    ax.legend(fontsize=9, loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "fig7d_cindex_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: fig7d_cindex_comparison.png")

# ── 6f. Nomogram-style point assignment visualization ─────────────────

print("\n  Building nomogram visualization...")

# Recompute on the nomogram df
nomo_cph = cph_nomo
nomo_data = nomo_df.copy()

var_ranges = {}
for var in nomo_vars:
    coef = nomo_cph.params_[var]
    col = nomo_data[var].astype(float)
    v_min = float(col.quantile(0.01))
    v_max = float(col.quantile(0.99))
    var_ranges[var] = (v_min, v_max, coef)

max_total_beta = sum(abs(c * (vmax - vmin)) for (vmin, vmax, c) in var_ranges.values())
points_per_unit_beta = 100 / (max_total_beta + 1e-10)

n_panel_rows = len(nomo_vars) + 3  # points scale + vars + total + survival
fig = plt.figure(figsize=(14, max(8, 2 + len(nomo_vars) * 1.5 + 3)))
gs = gridspec.GridSpec(n_panel_rows, 1,
                       height_ratios=[1] * (len(nomo_vars) + 1) + [0.5, 1.5])

# Points scale (top row)
ax_points = fig.add_subplot(gs[0])
ax_points.set_xlim(0, 100)
ax_points.set_xticks(np.arange(0, 101, 10))
ax_points.set_title("Points", fontsize=11, fontweight='bold')
ax_points.yaxis.set_visible(False)
ax_points.spines['left'].set_visible(False)
ax_points.spines['right'].set_visible(False)

# Variable scales
total_points_data = np.zeros(len(nomo_data))

for i, var in enumerate(nomo_vars):
    ax = fig.add_subplot(gs[i + 1])
    coef = nomo_cph.params_[var]
    v_min, v_max, _ = var_ranges[var]

    beta_contrib = coef * (nomo_data[var].values - v_min)
    points = beta_contrib * points_per_unit_beta
    if coef < 0:
        points = -coef * (v_max - nomo_data[var].values) * points_per_unit_beta
    total_points_data += np.clip(points, 0, 100)

    ax.set_xlim(0, 100)
    n_ticks = min(8, max(3, int(abs(coef * (v_max - v_min)) * points_per_unit_beta / 10) + 2))
    point_ticks = np.linspace(0, abs(coef * (v_max - v_min)) * points_per_unit_beta, n_ticks)

    if coef > 0:
        val_ticks = v_min + point_ticks / (abs(coef) * points_per_unit_beta + 1e-10)
    else:
        val_ticks = v_max - point_ticks / (abs(coef) * points_per_unit_beta + 1e-10)

    ax.set_xticks(point_ticks)

    # Label formatting
    if var == "age_years":
        ax.set_xticklabels([f"{v:.0f}" for v in val_ticks], fontsize=7)
        label = "Age (years)"
    elif var == "male":
        ax.set_xticks([0, abs(coef) * points_per_unit_beta])
        ax.set_xticklabels(["Female", "Male"], fontsize=8)
        label = "Sex"
    elif var.startswith("grp_"):
        # Group dummy variable
        grp_name = var.replace("grp_", "")
        ax.set_xticks([0, abs(coef) * points_per_unit_beta])
        ax.set_xticklabels(["No", "Yes"], fontsize=8)
        label = grp_name[:30]
    else:
        ax.set_xticklabels([f"{v:.1f}" for v in val_ticks], fontsize=7)
        label = var

    ax.set_title(label, fontsize=9, fontweight='bold', loc='left')
    ax.yaxis.set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Total points row
ax_total = fig.add_subplot(gs[len(nomo_vars) + 1])
total_range = (total_points_data.min(), total_points_data.max())
ax_total.set_xlim(0, 100)
ax_total.set_title("Total Points", fontsize=11, fontweight='bold', loc='left')
total_ticks = np.linspace(total_range[0], total_range[1], 10)
tick_pos = (total_ticks - total_range[0]) / (total_range[1] - total_range[0] + 1e-10) * 100
ax_total.set_xticks(tick_pos)
ax_total.set_xticklabels([f"{v:.0f}" for v in total_ticks], fontsize=7)
ax_total.yaxis.set_visible(False)
ax_total.spines['left'].set_visible(False)
ax_total.spines['right'].set_visible(False)

# Survival probability row
ax_surv = fig.add_subplot(gs[len(nomo_vars) + 2])
surv_func = nomo_cph.predict_survival_function(nomo_data)

for tp_months, label, color in [(36, "3-year", "blue"), (60, "5-year", "red")]:
    if tp_months <= surv_func.index.max():
        closest = surv_func.index[np.abs(surv_func.index - tp_months).argmin()]
        surv_at_tp = surv_func.loc[closest]
        normalized_x = (total_points_data - total_range[0]) / (
            total_range[1] - total_range[0] + 1e-10) * 100
        ax_surv.scatter(normalized_x, surv_at_tp, s=3, alpha=0.3, color=color,
                        label=f"{label} Survival")

ax_surv.set_xlim(0, 100)
ax_surv.set_ylim(0, 1.05)
ax_surv.set_xlabel("Total Points (normalized)", fontsize=10)
ax_surv.set_ylabel("Survival Prob.", fontsize=10)
ax_surv.legend(fontsize=8, loc='upper right')
ax_surv.spines['top'].set_visible(False)
ax_surv.spines['right'].set_visible(False)

plt.suptitle("Prognostic Nomogram (Archetype + Clinical)",
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig7f_nomogram.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig7f_nomogram.png")

# ======================================================================
# SUMMARY
# ======================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\n  Grouping variable: {GROUP_COL}")
print(f"  Groups: {groups}")
print(f"  Drug target genes analysed: {len(available_drug_genes)}")

sig_drugs = corr_df[corr_df["p_value_combined"] < 0.05]
print(f"  Genes correlated with combined_score (p<0.05): {len(sig_drugs)}")
for _, r in sig_drugs.iterrows():
    direction = "higher in high-risk" if r["spearman_r_combined"] > 0 else "lower in high-risk"
    print(f"    {r['gene']} ({r['drug']}): r={r['spearman_r_combined']:.3f}, {direction}")

if cindex_results:
    best = max(cindex_results, key=lambda x: x["c_index_mean"])
    print(f"\n  Best model: {best['model']} (C-index={best['c_index_mean']:.4f})")

print(f"\n  Tables saved:")
print(f"    - {os.path.join(TABLES, 'drug_targets_by_archetype.csv')}")
print(f"    - {os.path.join(TABLES, 'immunotherapy_prediction.csv')}")
print(f"    - {os.path.join(TABLES, 'cindex_comparison_full.csv')}")
print(f"    - {os.path.join(TABLES, 'dca_results.csv')}")

print(f"\n  Figures saved:")
print(f"    - fig7a_drug_target_heatmap.png")
print(f"    - fig7b_drug_target_boxplots.png")
print(f"    - fig7c_immunotherapy_prediction.png")
print(f"    - fig7d_cindex_comparison.png")
print(f"    - fig7e_dca_curves.png")
print(f"    - fig7f_nomogram.png")

print(f"\nScript 07 complete.")
