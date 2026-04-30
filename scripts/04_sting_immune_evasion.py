"""
04_sting_immune_evasion.py — NRF2 -> STING suppression -> immune evasion axis

Tests the hypothesis that NRF2-high tumors suppress STING signalling and
evade immune surveillance.

Analyses:
  1. STING pathway scoring & NRF2-STING correlations
  2. Immune cell deconvolution via ssGSEA (or mean z-score fallback)
  3. Immune checkpoint landscape across dual groups
  4. NRF2-immune correlation matrix
  5. Publication-quality figures

Inputs:
  - data/tcga_convergence_master.csv (after scripts 01-03)
  - Full expression matrix via data/paths.json -> expression_full

Outputs:
  - Updated master CSV with sting_score + immune cell scores
  - results/tables/nrf2_sting_correlations.csv
  - results/tables/immune_group_comparison.csv
  - results/tables/checkpoint_comparison.csv
  - results/figures/main/fig4*.png
"""
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
FIGS = os.path.join(BASE, "results", "figures", "main")
SFIGS = os.path.join(BASE, "results", "figures", "supplementary")
TABLES = os.path.join(BASE, "results", "tables")

for d in [FIGS, SFIGS, TABLES]:
    os.makedirs(d, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# IMMUNE CELL GENE SIGNATURES (Bindea / Charoentong approach)
# ══════════════════════════════════════════════════════════════════════════════
IMMUNE_SIGNATURES = {
    "CD8_T_cells": ["CD8A", "CD8B", "GZMA", "GZMB", "PRF1", "IFNG", "TBX21", "EOMES"],
    "CD4_T_cells": ["CD4", "IL7R", "LEF1", "TCF7", "CCR7"],
    "Th1": ["TBX21", "IFNG", "IL12RB2", "STAT4", "CXCR3"],
    "Th2": ["GATA3", "IL4", "IL5", "IL13", "CCR4"],
    "Th17": ["RORC", "IL17A", "IL17F", "IL22", "CCR6"],
    "Treg": ["FOXP3", "IL2RA", "CTLA4", "IKZF2", "TNFRSF18"],
    "NK_cells": ["NCAM1", "NKG7", "KLRD1", "KLRB1", "NCR1", "NCR3", "GNLY"],
    "B_cells": ["CD19", "MS4A1", "CD79A", "CD79B", "PAX5"],
    "Plasma_cells": ["SDC1", "XBP1", "PRDM1", "IRF4", "MZB1"],
    "Macrophages_M1": ["CD68", "NOS2", "IL1B", "IL6", "TNF", "CXCL10", "CXCL11"],
    "Macrophages_M2": ["CD68", "CD163", "MRC1", "MSR1", "CD200R1", "IL10"],
    "Dendritic_cells": ["ITGAX", "CD1C", "CLEC10A", "FCER1A", "HLA-DRA"],
    "pDCs": ["CLEC4C", "IL3RA", "TCF4", "IRF7"],
    "Neutrophils": ["CEACAM8", "FCGR3B", "CSF3R", "CXCR1", "CXCR2"],
    "Mast_cells": ["KIT", "TPSAB1", "TPSB2", "CPA3", "HDC"],
}

STING_GENES = ["TMEM173", "C6orf150", "TBK1", "IRF3", "CXCL10", "CCL5"]
# IFNB1 excluded: mostly zeros in TCGA-LIHC (>85% zero expression)

CHECKPOINT_GENES = ["CD274", "PDCD1", "CTLA4", "LAG3", "HAVCR2",
                    "TIGIT", "SIGLEC15", "IDO1", "CD276"]


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def compute_mean_zscore(df, genes):
    """Compute composite score as mean of z-scored gene expressions."""
    available = [g for g in genes if g in df.columns]
    if len(available) == 0:
        return pd.Series(np.nan, index=df.index)
    zscores = df[available].apply(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)
    return zscores.mean(axis=1)


def run_ssgsea(expr_matrix, gene_sets):
    """
    Run ssGSEA using gseapy. Falls back to mean z-score if gseapy fails.

    Parameters
    ----------
    expr_matrix : pd.DataFrame
        Genes (rows) x Samples (columns), with Hugo_Symbol as index.
    gene_sets : dict
        {cell_type: [gene1, gene2, ...]}

    Returns
    -------
    pd.DataFrame : samples x cell_types with enrichment scores
    """
    try:
        import gseapy as gp

        # gseapy ssgsea expects genes x samples dataframe
        # Filter to genes that exist in the expression matrix
        filtered_sets = {}
        for name, genes in gene_sets.items():
            present = [g for g in genes if g in expr_matrix.index]
            if len(present) >= 3:
                filtered_sets[name] = present
            else:
                print(f"    WARNING: {name} has only {len(present)} genes in expression matrix, skipping")

        if len(filtered_sets) == 0:
            raise ValueError("No gene sets with sufficient genes")

        print(f"    Running gseapy ssGSEA with {len(filtered_sets)} gene sets...")
        ss = gp.ssgsea(
            data=expr_matrix,
            gene_sets=filtered_sets,
            outdir=None,
            no_plot=True,
            min_size=3,
            max_size=500,
            threads=4,
        )

        # gseapy >= 1.0 returns object with .res2d attribute
        if hasattr(ss, 'res2d'):
            result_df = ss.res2d
            # Pivot to samples x cell_types
            if 'Name' in result_df.columns and 'Term' in result_df.columns:
                scores = result_df.pivot(index='Name', columns='Term', values='NES')
                if scores is not None and not scores.empty:
                    print(f"    ssGSEA complete: {scores.shape[0]} samples x {scores.shape[1]} cell types")
                    return scores
            # Alternative format
            elif 'ES' in result_df.columns or 'NES' in result_df.columns:
                val_col = 'NES' if 'NES' in result_df.columns else 'ES'
                scores = result_df.pivot(index='Name', columns='Term', values=val_col)
                if scores is not None and not scores.empty:
                    print(f"    ssGSEA complete: {scores.shape[0]} samples x {scores.shape[1]} cell types")
                    return scores

        # Try older gseapy API format
        if hasattr(ss, 'resultsOnSamples'):
            scores = ss.resultsOnSamples
            print(f"    ssGSEA complete: {scores.shape}")
            return scores.T  # transpose to samples x cell_types

        raise ValueError("Could not parse gseapy ssGSEA output")

    except Exception as e:
        print(f"    ssGSEA via gseapy failed ({type(e).__name__}: {e})")
        print("    Falling back to mean z-score approach...")
        return _fallback_mean_zscore(expr_matrix, gene_sets)


def _fallback_mean_zscore(expr_matrix, gene_sets):
    """Fallback: compute immune scores as mean z-score of marker genes."""
    expr_t = expr_matrix.T  # samples x genes
    scores = {}
    for name, genes in gene_sets.items():
        present = [g for g in genes if g in expr_t.columns]
        if len(present) >= 2:
            z = expr_t[present].apply(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)
            scores[name] = z.mean(axis=1)
        else:
            print(f"    WARNING: {name} only {len(present)} genes found, filling NaN")
            scores[name] = pd.Series(np.nan, index=expr_t.index)
    result = pd.DataFrame(scores)
    print(f"    Fallback complete: {result.shape[0]} samples x {result.shape[1]} cell types")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 0. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("04_sting_immune_evasion.py — NRF2 -> STING -> Immune Evasion Axis")
print("=" * 70)

df = pd.read_csv(os.path.join(DATA, "tcga_convergence_master.csv"))
print(f"Master dataframe: {df.shape[0]} patients, {df.shape[1]} columns")

# Load paths for full expression matrix
with open(os.path.join(DATA, "paths.json")) as f:
    paths = json.load(f)

# Check for nrf2_activity column (from script 03)
if "nrf2_activity" not in df.columns:
    print("\nWARNING: nrf2_activity column not found — computing from NRF2 target genes")
    nrf2_targets = ["NQO1", "GCLC", "GCLM", "FTH1", "FTL", "SRXN1",
                    "AKR1C1", "AKR1B10", "ME1", "ABCC2", "TXNRD1",
                    "G6PD", "SLC7A11", "GSR", "SQSTM1"]
    available_targets = [g for g in nrf2_targets if g in df.columns]
    print(f"  Using {len(available_targets)} NRF2 target genes: {available_targets}")
    df["nrf2_activity"] = compute_mean_zscore(df, available_targets)
    print(f"  NRF2 activity: mean={df['nrf2_activity'].mean():.4f}, "
          f"std={df['nrf2_activity'].std():.4f}")
else:
    print(f"NRF2 activity column found: mean={df['nrf2_activity'].mean():.4f}")

# Create NRF2 tertiles
df["nrf2_tertile"] = pd.qcut(df["nrf2_activity"], q=3,
                              labels=["NRF2-Low", "NRF2-Mid", "NRF2-High"])
print(f"NRF2 tertiles: {df['nrf2_tertile'].value_counts().to_dict()}")

# Verify dual_group column
if "dual_group" not in df.columns:
    raise ValueError("dual_group column missing — run script 02 first")

group_labels = sorted(df["dual_group"].unique())
print(f"Dual groups: {group_labels}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. STING PATHWAY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("1. STING PATHWAY ANALYSIS")
print("=" * 70)

# --- 1a. Compute STING composite score ---
print("\n--- 1a. Computing STING pathway score ---")
# Check IFNB1 separately
if "IFNB1" in df.columns:
    n_zero = (df["IFNB1"] == 0).sum()
    pct_zero = 100 * n_zero / len(df)
    print(f"  IFNB1: {n_zero}/{len(df)} zeros ({pct_zero:.1f}%) — {'EXCLUDED' if pct_zero > 50 else 'included'}")

sting_available = [g for g in STING_GENES if g in df.columns]
print(f"  STING genes used ({len(sting_available)}): {sting_available}")

df["sting_score"] = compute_mean_zscore(df, sting_available)
print(f"  STING score: mean={df['sting_score'].mean():.4f}, "
      f"std={df['sting_score'].std():.4f}, "
      f"range=[{df['sting_score'].min():.4f}, {df['sting_score'].max():.4f}]")

# --- 1b. Individual gene correlations with NRF2 ---
print("\n--- 1b. NRF2 vs individual STING gene correlations (Spearman) ---")
sting_corr_results = []

# Individual genes
all_sting = STING_GENES + (["IFNB1"] if "IFNB1" in df.columns else [])
for gene in all_sting:
    if gene not in df.columns:
        continue
    mask = df[gene].notna() & df["nrf2_activity"].notna()
    if mask.sum() < 20:
        continue
    rho, pval = stats.spearmanr(df.loc[mask, "nrf2_activity"], df.loc[mask, gene])
    sting_corr_results.append({
        "gene": gene,
        "spearman_rho": rho,
        "p_value": pval,
        "n": int(mask.sum()),
        "type": "individual_gene",
    })
    print(f"  NRF2 vs {gene:12s}: rho={rho:+.4f}, p={pval:.2e}")

# Composite STING score
mask = df["sting_score"].notna() & df["nrf2_activity"].notna()
rho_sting, p_sting = stats.spearmanr(df.loc[mask, "nrf2_activity"],
                                      df.loc[mask, "sting_score"])
sting_corr_results.append({
    "gene": "STING_composite",
    "spearman_rho": rho_sting,
    "p_value": p_sting,
    "n": int(mask.sum()),
    "type": "composite_score",
})
print(f"\n  NRF2 vs STING composite:  rho={rho_sting:+.4f}, p={p_sting:.2e}")

# BH-FDR correction
corr_df = pd.DataFrame(sting_corr_results)
if len(corr_df) > 1:
    reject, fdr_pvals, _, _ = multipletests(corr_df["p_value"], method="fdr_bh")
    corr_df["fdr_q"] = fdr_pvals
    corr_df["significant"] = reject
else:
    corr_df["fdr_q"] = corr_df["p_value"]
    corr_df["significant"] = corr_df["p_value"] < 0.05

corr_df.to_csv(os.path.join(TABLES, "nrf2_sting_correlations.csv"), index=False)
print(f"\nSaved: results/tables/nrf2_sting_correlations.csv")

n_sig = corr_df["significant"].sum()
print(f"Significant after FDR: {n_sig}/{len(corr_df)}")

# Interpret direction
if rho_sting < 0:
    print("HYPOTHESIS SUPPORTED: NRF2 activity negatively correlates with STING pathway")
else:
    print("NOTE: NRF2-STING correlation is positive (contrary to suppression hypothesis)")


# ══════════════════════════════════════════════════════════════════════════════
# 2. IMMUNE CELL DECONVOLUTION (ssGSEA)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. IMMUNE CELL DECONVOLUTION (ssGSEA)")
print("=" * 70)

# Load full expression matrix
print(f"\nLoading full expression matrix from: {paths['expression_full']}")
expr_full = pd.read_csv(paths["expression_full"], index_col=0)
print(f"  Raw shape: {expr_full.shape[0]} genes x {expr_full.shape[1]} samples")

# Clean: drop rows with NaN index (unnamed genes)
expr_full = expr_full[expr_full.index.notna()].copy()

# Handle duplicate gene symbols by keeping the one with highest mean expression
if expr_full.index.duplicated().any():
    n_dup = expr_full.index.duplicated().sum()
    print(f"  Removing {n_dup} duplicate gene symbols (keeping highest mean expression)")
    expr_full["_mean"] = expr_full.mean(axis=1)
    expr_full = expr_full.sort_values("_mean", ascending=False)
    expr_full = expr_full[~expr_full.index.duplicated(keep='first')]
    expr_full = expr_full.drop(columns=["_mean"])

print(f"  Clean shape: {expr_full.shape[0]} genes x {expr_full.shape[1]} samples")

# Check availability of immune signature genes
print("\n  Immune signature gene availability:")
for celltype, genes in IMMUNE_SIGNATURES.items():
    present = [g for g in genes if g in expr_full.index]
    missing = [g for g in genes if g not in expr_full.index]
    status = "OK" if len(present) >= 3 else "LOW"
    print(f"    {celltype:20s}: {len(present)}/{len(genes)} genes "
          f"{'(' + ', '.join(missing) + ' missing)' if missing else ''} [{status}]")

# Restrict expression matrix to samples in our master dataframe
common_samples = [s for s in df["patientId"] if s in expr_full.columns]
print(f"\n  Samples in common: {len(common_samples)}/{len(df)}")
expr_subset = expr_full[common_samples]

# Run ssGSEA
print("\n  Computing immune cell enrichment scores...")
immune_scores = run_ssgsea(expr_subset, IMMUNE_SIGNATURES)

# Align immune scores with master dataframe
immune_scores.index = immune_scores.index.astype(str)
immune_cols = list(immune_scores.columns)
print(f"  Immune cell types scored: {len(immune_cols)}")

# Merge into master dataframe
for col in immune_cols:
    col_name = f"immune_{col}"
    mapping = immune_scores[col].to_dict()
    df[col_name] = df["patientId"].map(mapping)

print(f"  Added {len(immune_cols)} immune score columns to master dataframe")

# --- 2b. Compare immune scores across dual groups (Kruskal-Wallis) ---
print("\n--- 2b. Immune scores across dual groups (Kruskal-Wallis) ---")
immune_group_results = []
for celltype in immune_cols:
    col_name = f"immune_{celltype}"
    if col_name not in df.columns:
        continue
    groups_data = []
    group_names = sorted(df["dual_group"].unique())
    for grp in group_names:
        vals = df.loc[df["dual_group"] == grp, col_name].dropna()
        groups_data.append(vals)

    if all(len(g) >= 3 for g in groups_data):
        h_stat, p_val = stats.kruskal(*groups_data)
    else:
        h_stat, p_val = np.nan, np.nan

    row = {
        "cell_type": celltype,
        "kruskal_H": h_stat,
        "p_value": p_val,
    }
    # Add group means
    for i, grp in enumerate(group_names):
        row[f"mean_{grp}"] = groups_data[i].mean() if len(groups_data[i]) > 0 else np.nan
    immune_group_results.append(row)

immune_grp_df = pd.DataFrame(immune_group_results)

# BH-FDR correction
valid_p = immune_grp_df["p_value"].notna()
if valid_p.sum() > 0:
    reject, fdr_pvals, _, _ = multipletests(
        immune_grp_df.loc[valid_p, "p_value"], method="fdr_bh"
    )
    immune_grp_df.loc[valid_p, "fdr_q"] = fdr_pvals
    immune_grp_df.loc[valid_p, "significant"] = reject
else:
    immune_grp_df["fdr_q"] = np.nan
    immune_grp_df["significant"] = False

immune_grp_df = immune_grp_df.sort_values("p_value")
immune_grp_df.to_csv(os.path.join(TABLES, "immune_group_comparison.csv"), index=False)
print(f"\nSaved: results/tables/immune_group_comparison.csv")

print("\nImmune cell type group comparison results:")
for _, row in immune_grp_df.iterrows():
    sig = "*" if row.get("significant", False) else " "
    print(f"  {sig} {row['cell_type']:20s}: H={row['kruskal_H']:.2f}, "
          f"p={row['p_value']:.2e}, FDR={row.get('fdr_q', np.nan):.2e}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. IMMUNE CHECKPOINT LANDSCAPE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. IMMUNE CHECKPOINT LANDSCAPE")
print("=" * 70)

available_checkpoints = [g for g in CHECKPOINT_GENES if g in df.columns]
print(f"Checkpoint genes available: {len(available_checkpoints)}/{len(CHECKPOINT_GENES)}")
print(f"  {available_checkpoints}")

# --- 3a. Compare across dual groups ---
print("\n--- 3a. Checkpoints across dual groups (Kruskal-Wallis) ---")
checkpoint_results = []
for gene in available_checkpoints:
    groups_data = []
    group_names = sorted(df["dual_group"].unique())
    for grp in group_names:
        vals = df.loc[df["dual_group"] == grp, gene].dropna()
        groups_data.append(vals)

    if all(len(g) >= 3 for g in groups_data):
        h_stat, p_val = stats.kruskal(*groups_data)
    else:
        h_stat, p_val = np.nan, np.nan

    row = {
        "gene": gene,
        "comparison": "dual_group",
        "kruskal_H": h_stat,
        "p_value": p_val,
    }
    for i, grp in enumerate(group_names):
        row[f"mean_{grp}"] = groups_data[i].mean() if len(groups_data[i]) > 0 else np.nan
        row[f"median_{grp}"] = groups_data[i].median() if len(groups_data[i]) > 0 else np.nan
    checkpoint_results.append(row)

# --- 3b. Compare across NRF2 tertiles ---
print("--- 3b. Checkpoints across NRF2 tertiles (Kruskal-Wallis) ---")
tertile_labels = ["NRF2-Low", "NRF2-Mid", "NRF2-High"]
for gene in available_checkpoints:
    groups_data = []
    for tert in tertile_labels:
        vals = df.loc[df["nrf2_tertile"] == tert, gene].dropna()
        groups_data.append(vals)

    if all(len(g) >= 3 for g in groups_data):
        h_stat, p_val = stats.kruskal(*groups_data)
    else:
        h_stat, p_val = np.nan, np.nan

    row = {
        "gene": gene,
        "comparison": "nrf2_tertile",
        "kruskal_H": h_stat,
        "p_value": p_val,
    }
    for i, tert in enumerate(tertile_labels):
        row[f"mean_{tert}"] = groups_data[i].mean() if len(groups_data[i]) > 0 else np.nan
        row[f"median_{tert}"] = groups_data[i].median() if len(groups_data[i]) > 0 else np.nan
    checkpoint_results.append(row)

ckpt_df = pd.DataFrame(checkpoint_results)

# BH-FDR correction (separately for each comparison type)
for comp in ["dual_group", "nrf2_tertile"]:
    mask = (ckpt_df["comparison"] == comp) & ckpt_df["p_value"].notna()
    if mask.sum() > 0:
        reject, fdr_pvals, _, _ = multipletests(
            ckpt_df.loc[mask, "p_value"], method="fdr_bh"
        )
        ckpt_df.loc[mask, "fdr_q"] = fdr_pvals
        ckpt_df.loc[mask, "significant"] = reject

ckpt_df.to_csv(os.path.join(TABLES, "checkpoint_comparison.csv"), index=False)
print(f"\nSaved: results/tables/checkpoint_comparison.csv")

print("\nCheckpoint results (dual group):")
for _, row in ckpt_df[ckpt_df["comparison"] == "dual_group"].iterrows():
    sig = "*" if row.get("significant", False) else " "
    print(f"  {sig} {row['gene']:12s}: H={row['kruskal_H']:.2f}, "
          f"p={row['p_value']:.2e}, FDR={row.get('fdr_q', np.nan):.2e}")

print("\nCheckpoint results (NRF2 tertile):")
for _, row in ckpt_df[ckpt_df["comparison"] == "nrf2_tertile"].iterrows():
    sig = "*" if row.get("significant", False) else " "
    print(f"  {sig} {row['gene']:12s}: H={row['kruskal_H']:.2f}, "
          f"p={row['p_value']:.2e}, FDR={row.get('fdr_q', np.nan):.2e}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. NRF2-IMMUNE CORRELATION MATRIX
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. NRF2 vs IMMUNE CELL TYPE CORRELATIONS (Spearman)")
print("=" * 70)

nrf2_immune_corrs = []
for celltype in immune_cols:
    col_name = f"immune_{celltype}"
    if col_name not in df.columns:
        continue
    mask = df[col_name].notna() & df["nrf2_activity"].notna()
    if mask.sum() < 20:
        continue
    rho, pval = stats.spearmanr(df.loc[mask, "nrf2_activity"],
                                 df.loc[mask, col_name])
    nrf2_immune_corrs.append({
        "cell_type": celltype,
        "spearman_rho": rho,
        "p_value": pval,
        "n": int(mask.sum()),
    })

nrf2_immune_df = pd.DataFrame(nrf2_immune_corrs)
if len(nrf2_immune_df) > 1:
    reject, fdr_pvals, _, _ = multipletests(nrf2_immune_df["p_value"], method="fdr_bh")
    nrf2_immune_df["fdr_q"] = fdr_pvals
    nrf2_immune_df["significant"] = reject
else:
    nrf2_immune_df["fdr_q"] = nrf2_immune_df["p_value"]
    nrf2_immune_df["significant"] = nrf2_immune_df["p_value"] < 0.05

nrf2_immune_df = nrf2_immune_df.sort_values("spearman_rho")
print("\nNRF2 vs immune cell correlations (sorted by rho):")
for _, row in nrf2_immune_df.iterrows():
    sig = "*" if row["significant"] else " "
    direction = "DEPLETED" if row["spearman_rho"] < -0.1 else "ENRICHED" if row["spearman_rho"] > 0.1 else "neutral"
    print(f"  {sig} {row['cell_type']:20s}: rho={row['spearman_rho']:+.4f}, "
          f"FDR={row['fdr_q']:.2e} [{direction} in NRF2-high]")

# Focus cells
focus_cells = ["NK_cells", "Dendritic_cells", "CD8_T_cells", "pDCs"]
print("\nFocus cells (expected depleted in NRF2-high):")
for cell in focus_cells:
    match = nrf2_immune_df[nrf2_immune_df["cell_type"] == cell]
    if len(match) > 0:
        r = match.iloc[0]
        print(f"  {cell}: rho={r['spearman_rho']:+.4f}, p={r['p_value']:.2e}, FDR={r['fdr_q']:.2e}")
    else:
        print(f"  {cell}: not scored")


# ══════════════════════════════════════════════════════════════════════════════
# 5. GENERATE FIGURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. GENERATING FIGURES")
print("=" * 70)

# Set global style
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# --- Figure 4a: Heatmap — NRF2 tertile x immune cell types ---
print("\n--- Fig 4a: Immune infiltration heatmap by NRF2 tertile ---")
heatmap_data = {}
for celltype in immune_cols:
    col_name = f"immune_{celltype}"
    if col_name not in df.columns:
        continue
    row_data = {}
    for tert in tertile_labels:
        vals = df.loc[df["nrf2_tertile"] == tert, col_name].dropna()
        row_data[tert] = vals.mean()
    heatmap_data[celltype] = row_data

heatmap_df = pd.DataFrame(heatmap_data).T
# Z-score across tertiles for visualization
heatmap_z = heatmap_df.apply(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0, axis=1)

fig, ax = plt.subplots(figsize=(8, 10))
sns.heatmap(
    heatmap_z,
    cmap="RdBu_r",
    center=0,
    annot=heatmap_df.round(3),
    fmt=".3f",
    linewidths=0.5,
    linecolor="white",
    ax=ax,
    cbar_kws={"label": "Z-score (across tertiles)", "shrink": 0.8},
    yticklabels=[ct.replace("_", " ") for ct in heatmap_z.index],
)
ax.set_title("Immune Cell Infiltration by NRF2 Activity Tertile\n(mean ssGSEA scores)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("NRF2 Activity Tertile", fontsize=12)
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig4a_immune_heatmap_nrf2_tertile.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: fig4a_immune_heatmap_nrf2_tertile.png")


# --- Figure 4b: STING score by dual group ---
print("--- Fig 4b: STING score by dual group ---")
fig, ax = plt.subplots(figsize=(8, 6))

group_order = sorted(df["dual_group"].unique())
colors_groups = {
    "A: Concordant High": "#d62728",
    "B: Ferroptosis-dominant": "#ff7f0e",
    "C: Hypoxia-dominant": "#9467bd",
    "D: Concordant Low": "#2ca02c",
}
palette = [colors_groups.get(g, "#888888") for g in group_order]

bp = sns.boxplot(
    data=df,
    x="dual_group",
    y="sting_score",
    order=group_order,
    palette=palette,
    width=0.6,
    fliersize=3,
    ax=ax,
)
sns.stripplot(
    data=df,
    x="dual_group",
    y="sting_score",
    order=group_order,
    color="black",
    alpha=0.3,
    size=3,
    jitter=True,
    ax=ax,
)

# Kruskal-Wallis test
groups_data = [df.loc[df["dual_group"] == g, "sting_score"].dropna() for g in group_order]
h_stat, p_kw = stats.kruskal(*groups_data)

ax.set_title(f"STING Pathway Score by Dual-Signature Group\n(Kruskal-Wallis H={h_stat:.2f}, p={p_kw:.2e})",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Dual-Signature Group", fontsize=12)
ax.set_ylabel("STING Pathway Score (mean z-score)", fontsize=12)
ax.set_xticklabels([g.replace(": ", ":\n") for g in group_order], fontsize=10)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig4b_sting_by_group.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: fig4b_sting_by_group.png (KW p={p_kw:.2e})")


# --- Figure 4c: NRF2 activity vs STING score scatter ---
print("--- Fig 4c: NRF2 vs STING scatter ---")
fig, ax = plt.subplots(figsize=(7, 6))

mask = df["nrf2_activity"].notna() & df["sting_score"].notna()
plot_df = df.loc[mask].copy()

# Color by dual group
for grp in group_order:
    subset = plot_df[plot_df["dual_group"] == grp]
    ax.scatter(
        subset["nrf2_activity"],
        subset["sting_score"],
        c=colors_groups.get(grp, "#888888"),
        alpha=0.6,
        s=40,
        label=grp,
        edgecolors="white",
        linewidth=0.5,
    )

# Regression line
from numpy.polynomial.polynomial import polyfit
x = plot_df["nrf2_activity"].values
y = plot_df["sting_score"].values
slope, intercept = np.polyfit(x, y, 1)
x_line = np.linspace(x.min(), x.max(), 100)
ax.plot(x_line, slope * x_line + intercept, "k--", alpha=0.7, linewidth=2,
        label=f"Linear fit (slope={slope:.3f})")

rho_s, p_s = stats.spearmanr(x, y)
ax.text(0.05, 0.95,
        f"Spearman rho = {rho_s:+.3f}\np = {p_s:.2e}\nn = {len(plot_df)}",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

ax.set_title("NRF2 Activity vs STING Pathway Score",
             fontsize=13, fontweight="bold")
ax.set_xlabel("NRF2 Activity (composite z-score)", fontsize=12)
ax.set_ylabel("STING Pathway Score (composite z-score)", fontsize=12)
ax.legend(fontsize=9, loc="lower left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig4c_nrf2_vs_sting_scatter.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: fig4c_nrf2_vs_sting_scatter.png (rho={rho_s:+.3f})")


# --- Figure 4d: Checkpoint expression heatmap by dual group ---
print("--- Fig 4d: Checkpoint heatmap by dual group ---")
ckpt_heatmap_data = {}
for gene in available_checkpoints:
    row_data = {}
    for grp in group_order:
        vals = df.loc[df["dual_group"] == grp, gene].dropna()
        row_data[grp] = vals.mean()
    ckpt_heatmap_data[gene] = row_data

ckpt_heatmap = pd.DataFrame(ckpt_heatmap_data).T
# Z-score across groups for visualization
ckpt_z = ckpt_heatmap.apply(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0, axis=1)

# Add significance markers
ckpt_annot = ckpt_heatmap.round(1).astype(str)
for gene in available_checkpoints:
    match = ckpt_df[(ckpt_df["gene"] == gene) & (ckpt_df["comparison"] == "dual_group")]
    if len(match) > 0 and match.iloc[0].get("significant", False):
        for col in ckpt_annot.columns:
            ckpt_annot.loc[gene, col] = ckpt_annot.loc[gene, col] + "*"

fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(
    ckpt_z,
    cmap="YlOrRd",
    annot=ckpt_annot,
    fmt="",
    linewidths=0.5,
    linecolor="white",
    ax=ax,
    cbar_kws={"label": "Z-score (across groups)", "shrink": 0.8},
    xticklabels=[g.split(":")[0] + ":\n" + g.split(": ")[1] if ": " in g else g
                 for g in ckpt_z.columns],
)
ax.set_title("Immune Checkpoint Expression by Dual-Signature Group\n"
             "(* = significant after FDR correction, Kruskal-Wallis)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Dual-Signature Group", fontsize=12)
ax.set_ylabel("Immune Checkpoint Gene", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig4d_checkpoint_heatmap.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: fig4d_checkpoint_heatmap.png")


# --- Figure 4e: NRF2 vs immune cells correlation bar plot ---
print("--- Fig 4e: NRF2-immune correlation bar plot ---")
fig, ax = plt.subplots(figsize=(10, 7))

plot_data = nrf2_immune_df.sort_values("spearman_rho").copy()
colors_bar = []
for _, row in plot_data.iterrows():
    if row["significant"] and row["spearman_rho"] < 0:
        colors_bar.append("#2166ac")   # Significant negative (depleted)
    elif row["significant"] and row["spearman_rho"] > 0:
        colors_bar.append("#b2182b")   # Significant positive (enriched)
    else:
        colors_bar.append("#cccccc")   # Not significant

bars = ax.barh(
    range(len(plot_data)),
    plot_data["spearman_rho"],
    color=colors_bar,
    edgecolor="white",
    linewidth=0.5,
    height=0.7,
)

ax.set_yticks(range(len(plot_data)))
ax.set_yticklabels([ct.replace("_", " ") for ct in plot_data["cell_type"]], fontsize=10)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Spearman rho (NRF2 activity vs immune score)", fontsize=12)
ax.set_title("NRF2 Activity Correlation with Immune Cell Infiltration\n"
             "(Blue = depleted in NRF2-high, Red = enriched, Grey = n.s.)",
             fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3, axis="x")

# Add significance stars
for i, (_, row) in enumerate(plot_data.iterrows()):
    if row["significant"]:
        x_pos = row["spearman_rho"]
        offset = 0.01 if x_pos >= 0 else -0.01
        ha = "left" if x_pos >= 0 else "right"
        ax.text(x_pos + offset, i, "*", fontsize=14, fontweight="bold",
                va="center", ha=ha, color="black")

plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig4e_nrf2_immune_correlations.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: fig4e_nrf2_immune_correlations.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6. SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("6. SAVING UPDATED MASTER DATAFRAME")
print("=" * 70)

# Columns added: sting_score, nrf2_tertile, nrf2_activity (if computed), immune_*
new_cols = ["sting_score", "nrf2_tertile"]
if "nrf2_activity" in df.columns:
    new_cols.append("nrf2_activity")
immune_added = [c for c in df.columns if c.startswith("immune_")]
new_cols.extend(immune_added)

print(f"New columns added to master dataframe:")
for col in new_cols:
    n_valid = df[col].notna().sum()
    print(f"  {col}: {n_valid}/{len(df)} non-null")

df.to_csv(os.path.join(DATA, "tcga_convergence_master.csv"), index=False)
print(f"\nSaved: data/tcga_convergence_master.csv ({df.shape[0]} rows, {df.shape[1]} columns)")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY — Script 04 Results")
print("=" * 70)

print(f"\n1. STING Pathway:")
print(f"   NRF2 vs STING composite: rho={rho_sting:+.4f}, p={p_sting:.2e}")
n_neg_sting = sum(1 for r in sting_corr_results
                  if r["type"] == "individual_gene" and r["spearman_rho"] < 0)
n_total_sting = sum(1 for r in sting_corr_results if r["type"] == "individual_gene")
print(f"   Individual STING genes with negative NRF2 correlation: "
      f"{n_neg_sting}/{n_total_sting}")

print(f"\n2. Immune Deconvolution:")
n_sig_immune = immune_grp_df["significant"].sum() if "significant" in immune_grp_df.columns else 0
print(f"   Cell types significantly different across groups (FDR<0.05): "
      f"{n_sig_immune}/{len(immune_grp_df)}")

print(f"\n3. Immune Checkpoints:")
n_sig_ckpt_grp = ckpt_df.loc[ckpt_df["comparison"] == "dual_group", "significant"].sum() \
    if "significant" in ckpt_df.columns else 0
n_sig_ckpt_nrf2 = ckpt_df.loc[ckpt_df["comparison"] == "nrf2_tertile", "significant"].sum() \
    if "significant" in ckpt_df.columns else 0
print(f"   Significant across dual groups: {n_sig_ckpt_grp}/{len(available_checkpoints)}")
print(f"   Significant across NRF2 tertiles: {n_sig_ckpt_nrf2}/{len(available_checkpoints)}")

print(f"\n4. NRF2-Immune Correlations:")
n_depleted = len(nrf2_immune_df[(nrf2_immune_df["spearman_rho"] < 0) &
                                 (nrf2_immune_df["significant"])])
n_enriched = len(nrf2_immune_df[(nrf2_immune_df["spearman_rho"] > 0) &
                                 (nrf2_immune_df["significant"])])
print(f"   Immune types depleted in NRF2-high (sig.): {n_depleted}")
print(f"   Immune types enriched in NRF2-high (sig.): {n_enriched}")

print(f"\n5. Files saved:")
print(f"   - results/tables/nrf2_sting_correlations.csv")
print(f"   - results/tables/immune_group_comparison.csv")
print(f"   - results/tables/checkpoint_comparison.csv")
print(f"   - results/figures/main/fig4a_immune_heatmap_nrf2_tertile.png")
print(f"   - results/figures/main/fig4b_sting_by_group.png")
print(f"   - results/figures/main/fig4c_nrf2_vs_sting_scatter.png")
print(f"   - results/figures/main/fig4d_checkpoint_heatmap.png")
print(f"   - results/figures/main/fig4e_nrf2_immune_correlations.png")
print(f"   - data/tcga_convergence_master.csv (updated)")

print("\n" + "=" * 70)
print("DONE — Script 04 complete")
print("=" * 70)
