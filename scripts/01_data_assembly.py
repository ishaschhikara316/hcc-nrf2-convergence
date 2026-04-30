"""
01_data_assembly.py — Load both LASSO models, compute dual risk scores, merge with clinical data

Inputs:
  - ROS model: ../hcc-ros-signature/results/model/lasso_model.json (11 genes)
  - Altitude model: ../hcc-altitude-signature/results/corrected/model/lasso_model_corrected.json (9 genes)
  - Expression: ../hcc-ros-signature/data/tcga/tcga_lihc_expression_full.csv (~20K genes x 366 samples)
  - Clinical: ../hcc-ros-signature/data/tcga/tcga_lihc_clinical.csv

Outputs:
  - data/tcga_convergence_master.csv (patient-level: expression + clinical + both risk scores)
  - results/tables/dual_score_correlation.csv
"""
import pandas as pd
import numpy as np
from scipy import stats
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROS_BASE = os.path.join(os.path.dirname(BASE), "hcc-ros-signature")
ALT_BASE = os.path.join(os.path.dirname(BASE), "hcc-altitude-signature")
DATA_OUT = os.path.join(BASE, "data")
TABLES = os.path.join(BASE, "results", "tables")

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD MODELS
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("1. LOADING LASSO MODELS")
print("=" * 70)

with open(os.path.join(ROS_BASE, "results", "model", "lasso_model.json")) as f:
    ros_model = json.load(f)

with open(os.path.join(ALT_BASE, "results", "corrected", "model", "lasso_model_corrected.json")) as f:
    alt_model = json.load(f)

ros_genes = ros_model["genes"]
alt_genes = alt_model["genes"]

print(f"\nROS/Ferroptosis Signature ({len(ros_genes)} genes):")
for g, c in ros_genes.items():
    direction = "Risk" if c > 0 else "Protective"
    print(f"  {g:12s}  coeff={c:+.4f}  ({direction})")

print(f"\nAltitude Signature ({len(alt_genes)} genes):")
for g, c in alt_genes.items():
    direction = "Risk" if c > 0 else "Protective"
    print(f"  {g:12s}  coeff={c:+.4f}  ({direction})")

# Identify shared genes
ros_set = set(ros_genes.keys())
alt_set = set(alt_genes.keys())
shared = ros_set & alt_set
all_sig_genes = sorted(ros_set | alt_set)
print(f"\nShared genes: {shared if shared else 'None (HMOX1 in ROS, HMOX2 in altitude)'}")
print(f"Total unique signature genes: {len(all_sig_genes)}")
print(f"Heme oxygenase system: HMOX1 (ROS, risk) + HMOX2 (altitude, protective)")

# ══════════════════════════════════════════════════════════════════════════════
# 2. LOAD EXPRESSION DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. LOADING EXPRESSION DATA")
print("=" * 70)

expr_path = os.path.join(ROS_BASE, "data", "tcga", "tcga_lihc_expression_full.csv")
expr_raw = pd.read_csv(expr_path, index_col=0)  # genes x samples
print(f"Expression matrix: {expr_raw.shape[0]} genes x {expr_raw.shape[1]} samples")

# Transpose to samples x genes
expr = expr_raw.T
expr.index.name = "patientId"
print(f"Transposed: {expr.shape[0]} samples x {expr.shape[1]} genes")

# Verify all signature genes present
missing_ros = [g for g in ros_genes if g not in expr.columns]
missing_alt = [g for g in alt_genes if g not in expr.columns]
if missing_ros:
    print(f"WARNING: Missing ROS genes: {missing_ros}")
if missing_alt:
    print(f"WARNING: Missing altitude genes: {missing_alt}")
if not missing_ros and not missing_alt:
    print("All signature genes found in expression matrix.")

# ══════════════════════════════════════════════════════════════════════════════
# 3. COMPUTE RISK SCORES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. COMPUTING DUAL RISK SCORES")
print("=" * 70)


def compute_risk_score(expression_df, model_dict):
    """Compute z-scored risk score using model coefficients, means, and stds."""
    genes = model_dict["genes"]
    # Get normalization params — handle both model formats
    if "gene_means" in model_dict:
        means = model_dict["gene_means"]
        stds = model_dict["gene_stds"]
    elif "normalization" in model_dict:
        means = model_dict["normalization"]["mean"]
        stds = model_dict["normalization"]["std"]
    else:
        raise ValueError("Model missing normalization parameters")

    risk = np.zeros(len(expression_df))
    for gene, coef in genes.items():
        if gene in expression_df.columns and gene in means and gene in stds:
            m, s = means[gene], stds[gene]
            if s > 0:
                risk += coef * ((expression_df[gene].values - m) / s)
            else:
                print(f"  WARNING: std=0 for {gene}, skipping")
        else:
            print(f"  WARNING: {gene} not found in expression or normalization, skipping")
    return risk


# ROS risk score
ros_risk = compute_risk_score(expr, ros_model)
print(f"ROS risk score: mean={ros_risk.mean():.4f}, std={ros_risk.std():.4f}, "
      f"range=[{ros_risk.min():.4f}, {ros_risk.max():.4f}]")

# Altitude risk score
alt_risk = compute_risk_score(expr, alt_model)
print(f"Altitude risk score: mean={alt_risk.mean():.4f}, std={alt_risk.std():.4f}, "
      f"range=[{alt_risk.min():.4f}, {alt_risk.max():.4f}]")

# ══════════════════════════════════════════════════════════════════════════════
# 4. LOAD CLINICAL DATA & MERGE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. MERGING WITH CLINICAL DATA")
print("=" * 70)

clinical = pd.read_csv(os.path.join(ROS_BASE, "data", "tcga", "tcga_lihc_clinical.csv"))
print(f"Clinical data: {len(clinical)} patients")
print(f"Columns: {list(clinical.columns)}")

# Build master dataframe
master = pd.DataFrame({
    "patientId": expr.index,
    "ros_risk_score": ros_risk,
    "alt_risk_score": alt_risk,
})

# Merge clinical
master = master.merge(clinical, on="patientId", how="inner")
print(f"Merged: {len(master)} patients with clinical + dual risk scores")

# Add key gene expression values for downstream analysis
key_genes = [
    # Both signatures
    "HMOX1", "HMOX2",
    # ROS signature
    "TXNRD1", "MAFG", "G6PD", "SQSTM1", "SLC7A11", "GSR", "NCF2", "MSRA", "GLRX2", "BACH1",
    # Altitude signature
    "ARNT2", "GRB2", "ITGA6", "TBX5", "HK2", "LDHA", "EPO", "GC",
    # NRF2 targets (for scoring in script 03)
    "NQO1", "GCLC", "GCLM", "FTH1", "FTL", "SRXN1", "AKR1C1", "AKR1B10", "ME1", "ABCC2",
    # Ferroptosis markers
    "GPX4", "ACSL4", "LPCAT3", "TFRC",
    # STING pathway
    "TMEM173", "C6orf150", "TBK1", "IRF3", "IFNB1", "CXCL10", "CCL5",
    # Immune checkpoints
    "CD274", "PDCD1", "CTLA4", "LAG3", "HAVCR2", "TIGIT", "SIGLEC15", "IDO1", "CD276", "VSIR",
    # NFE2L2 (NRF2 itself)
    "NFE2L2", "KEAP1",
]

# Add available gene expression values
added_genes = []
for gene in key_genes:
    if gene in expr.columns:
        gene_vals = expr.loc[master["patientId"].values, gene].values
        master[gene] = gene_vals
        added_genes.append(gene)

print(f"Added expression for {len(added_genes)}/{len(key_genes)} key genes")
missing_key = set(key_genes) - set(added_genes)
if missing_key:
    print(f"Missing key genes: {missing_key}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. CORRELATION BETWEEN RISK SCORES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. RISK SCORE CORRELATION")
print("=" * 70)

df = master.dropna(subset=["OS_months", "OS_event"]).copy()
print(f"Patients with survival data: {len(df)}")
print(f"  Events (deaths): {int(df['OS_event'].sum())}")
print(f"  Median follow-up: {df['OS_months'].median():.1f} months")

rho, pval = stats.spearmanr(df["ros_risk_score"], df["alt_risk_score"])
print(f"\nSpearman correlation between ROS and altitude risk scores:")
print(f"  rho = {rho:.4f}, p = {pval:.2e}")

pearson_r, pearson_p = stats.pearsonr(df["ros_risk_score"], df["alt_risk_score"])
print(f"Pearson correlation:")
print(f"  r = {pearson_r:.4f}, p = {pearson_p:.2e}")

# Interpretation
if abs(rho) < 0.3:
    interp = "weak — signatures capture largely independent biology"
elif abs(rho) < 0.6:
    interp = "moderate — partially overlapping but distinct axes"
else:
    interp = "strong — substantial shared biology"
print(f"Interpretation: {interp}")

# Save correlation stats
corr_df = pd.DataFrame([{
    "metric": "Spearman_rho",
    "value": rho,
    "p_value": pval,
}, {
    "metric": "Pearson_r",
    "value": pearson_r,
    "p_value": pearson_p,
}, {
    "metric": "n_patients",
    "value": len(df),
    "p_value": np.nan,
}, {
    "metric": "n_events",
    "value": int(df["OS_event"].sum()),
    "p_value": np.nan,
}])
corr_df.to_csv(os.path.join(TABLES, "dual_score_correlation.csv"), index=False)
print(f"\nSaved: results/tables/dual_score_correlation.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 6. SAVE MASTER DATAFRAME
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("6. SAVING MASTER DATAFRAME")
print("=" * 70)

master.to_csv(os.path.join(DATA_OUT, "tcga_convergence_master.csv"), index=False)
print(f"Saved: data/tcga_convergence_master.csv")
print(f"  Shape: {master.shape}")
print(f"  Columns: {list(master.columns[:10])} ... + {master.shape[1] - 10} more")

# Also save the full expression matrix path for scripts that need genome-wide data
with open(os.path.join(DATA_OUT, "paths.json"), "w") as f:
    json.dump({
        "expression_full": expr_path,
        "clinical": os.path.join(ROS_BASE, "data", "tcga", "tcga_lihc_clinical.csv"),
        "ros_model": os.path.join(ROS_BASE, "results", "model", "lasso_model.json"),
        "alt_model": os.path.join(ALT_BASE, "results", "corrected", "model", "lasso_model_corrected.json"),
        "geo_cohorts": os.path.join(ROS_BASE, "data", "geo_cohorts"),
    }, f, indent=2)
print(f"Saved: data/paths.json (data path references)")

print("\n" + "=" * 70)
print("DONE — Script 01 complete")
print("=" * 70)
