"""
11_convergence_lasso.py — Build a joint LASSO-Cox model from all 20 signature genes

Instead of combining two risk scores, combine the RAW GENES from both signatures
into a single LASSO-Cox regression. This finds the optimal convergence signature.

Gene pool: 11 ROS + 9 Altitude = 20 unique genes (no overlap)
"""
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.utils import concordance_index
from scipy import stats
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import gzip
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
TABLES = os.path.join(BASE, "results", "tables")
FIGS = os.path.join(BASE, "results", "figures", "main")
MODEL_DIR = os.path.join(BASE, "results", "model")

with open(os.path.join(DATA, "paths.json")) as f:
    paths = json.load(f)
with open(paths["ros_model"]) as f:
    ros_model = json.load(f)
with open(paths["alt_model"]) as f:
    alt_model = json.load(f)

GEO = paths["geo_cohorts"]

# ══════════════════════════════════════════════════════════════════════════════
# 1. DEFINE GENE POOL & LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("1. GENE POOL & DATA")
print("=" * 70)

ROS_GENES = list(ros_model["genes"].keys())
ALT_GENES = list(alt_model["genes"].keys())
ALL_GENES = sorted(set(ROS_GENES + ALT_GENES))

print(f"ROS genes ({len(ROS_GENES)}): {ROS_GENES}")
print(f"Alt genes ({len(ALT_GENES)}): {ALT_GENES}")
print(f"Combined pool ({len(ALL_GENES)} unique): {ALL_GENES}")

# Load full expression matrix (need raw gene values, not pre-computed scores)
expr_full = pd.read_csv(paths["expression_full"], index_col=0).T  # samples x genes
clinical = pd.read_csv(os.path.join(paths["clinical"]))

# Merge — find common patient IDs between expression and clinical
df = clinical[["patientId", "OS_months", "OS_event"]].dropna().copy()
df = df[df["OS_months"] > 0]
common_ids = sorted(set(df["patientId"]) & set(expr_full.index))
df = df[df["patientId"].isin(common_ids)].set_index("patientId")
print(f"Patients with expression + clinical: {len(df)}")

# Add gene expression
available_genes = [g for g in ALL_GENES if g in expr_full.columns]
print(f"Available in expression: {len(available_genes)}/{len(ALL_GENES)}")
missing = set(ALL_GENES) - set(available_genes)
if missing:
    print(f"Missing: {missing}")

for g in available_genes:
    df[g] = expr_full.loc[df.index, g].values

df = df.dropna()
print(f"Patients: {len(df)}, Events: {int(df['OS_event'].sum())}")

# Z-score normalize genes
gene_means = {}
gene_stds = {}
for g in available_genes:
    m, s = df[g].mean(), df[g].std()
    gene_means[g] = float(m)
    gene_stds[g] = float(s)
    if s > 0:
        df[f"z_{g}"] = (df[g] - m) / s
    else:
        df[f"z_{g}"] = 0

z_genes = [f"z_{g}" for g in available_genes]

# ══════════════════════════════════════════════════════════════════════════════
# 2. LASSO-COX WITH CROSS-VALIDATION FOR LAMBDA SELECTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. LASSO-COX (L1-penalized Cox regression)")
print("=" * 70)

# Test a range of penalizers (lambda values)
# Focus on lambda range that selects 5-18 genes (avoid 0-gene and all-gene extremes)
penalizers = [0.001, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.07, 0.1]
cv_results = []

for pen in penalizers:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_cis = []
    fold_ngenes = []

    for train_idx, test_idx in kf.split(df):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]

        try:
            cph = CoxPHFitter(penalizer=pen, l1_ratio=1.0)
            cph.fit(train[["OS_months", "OS_event"] + z_genes],
                    duration_col="OS_months", event_col="OS_event")

            # Count non-zero genes
            n_genes = (cph.params_.abs() > 1e-6).sum()

            # Predict on test
            pred = cph.predict_partial_hazard(test[z_genes]).values.flatten()
            ci = concordance_index(test["OS_months"], -pred, test["OS_event"])
            fold_cis.append(ci)
            fold_ngenes.append(n_genes)
        except Exception:
            pass

    if fold_cis:
        mean_ci = np.mean(fold_cis)
        std_ci = np.std(fold_cis)
        mean_ng = np.mean(fold_ngenes)
        cv_results.append({
            "penalizer": pen, "mean_cindex": mean_ci, "std_cindex": std_ci,
            "mean_ngenes": mean_ng, "n_folds": len(fold_cis),
        })
        print(f"  lambda={pen:.3f}: CV C-index={mean_ci:.4f} +/- {std_ci:.4f}, genes={mean_ng:.1f}")

cv_df = pd.DataFrame(cv_results)

# Only consider lambdas where at least 3 genes are selected
cv_valid = cv_df[cv_df["mean_ngenes"] >= 3].copy()
if len(cv_valid) == 0:
    cv_valid = cv_df[cv_df["mean_ngenes"] >= 1].copy()
if len(cv_valid) == 0:
    cv_valid = cv_df.copy()

best_idx = cv_valid["mean_cindex"].idxmax()
best_lambda = cv_valid.loc[best_idx, "penalizer"]
best_cv_ci = cv_valid.loc[best_idx, "mean_cindex"]
best_cv_std = cv_valid.loc[best_idx, "std_cindex"]
print(f"\n  Best lambda (genes>=3): {best_lambda} (CV C-index = {best_cv_ci:.4f} +/- {best_cv_std:.4f})")

# 1-SE rule: most parsimonious within 1 SE of best, but keep >=8 genes for biological richness
se_threshold = best_cv_ci - best_cv_std
candidates_1se = cv_valid[(cv_valid["mean_cindex"] >= se_threshold) & (cv_valid["mean_ngenes"] >= 8)]
if len(candidates_1se) > 0:
    lambda_1se = candidates_1se["penalizer"].max()
else:
    # Fallback: use lambda_min (best CV C-index)
    lambda_1se = best_lambda
print(f"  1-SE lambda (>=8 genes): {lambda_1se} ({cv_valid.loc[cv_valid['penalizer']==lambda_1se, 'mean_ngenes'].values[0]:.0f} genes)")

# Also report lambda_min for comparison
print(f"  Lambda_min: {best_lambda} ({cv_valid.loc[cv_valid['penalizer']==best_lambda, 'mean_ngenes'].values[0]:.0f} genes)")

selected_lambda = lambda_1se
print(f"  SELECTED: lambda = {selected_lambda}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. FIT FINAL MODEL ON FULL TRAINING DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. FINAL CONVERGENCE MODEL")
print("=" * 70)

cph_final = CoxPHFitter(penalizer=selected_lambda, l1_ratio=1.0)
cph_final.fit(df[["OS_months", "OS_event"] + z_genes],
              duration_col="OS_months", event_col="OS_event")

# Extract selected genes
selected_genes = {}
for zg in z_genes:
    coef = float(cph_final.params_[zg])
    if abs(coef) > 1e-6:
        gene = zg[2:]  # strip "z_"
        selected_genes[gene] = coef

print(f"\nConvergence signature: {len(selected_genes)} genes")
print(f"{'Gene':<12s} {'Coeff':>8s} {'HR':>8s} {'Source':>15s} {'Direction':>12s}")
print("-" * 60)
for gene, coef in sorted(selected_genes.items(), key=lambda x: -abs(x[1])):
    hr = np.exp(coef)
    source = "ROS" if gene in ROS_GENES else "Altitude"
    if gene in ROS_GENES and gene in ALT_GENES:
        source = "Both"
    direction = "Risk" if coef > 0 else "Protective"
    print(f"{gene:<12s} {coef:>+8.4f} {hr:>8.3f} {source:>15s} {direction:>12s}")

# Training C-index
pred_train = cph_final.predict_partial_hazard(df[z_genes]).values.flatten()
ci_train = concordance_index(df["OS_months"], -pred_train, df["OS_event"])
print(f"\nTraining C-index: {ci_train:.4f}")
print(f"AIC: {cph_final.AIC_partial_:.1f}")

# Composition
n_ros = sum(1 for g in selected_genes if g in ROS_GENES)
n_alt = sum(1 for g in selected_genes if g in ALT_GENES)
print(f"\nComposition: {n_ros} ROS genes + {n_alt} altitude genes = {len(selected_genes)} total")

# ══════════════════════════════════════════════════════════════════════════════
# 4. COMPARE WITH PREVIOUS SCORES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. COMPARISON WITH PREVIOUS SCORES")
print("=" * 70)

# Previous scores
master = pd.read_csv(os.path.join(DATA, "tcga_convergence_master.csv"))
master = master.dropna(subset=["OS_months", "OS_event"])
master = master[master["OS_months"] > 0]

ci_ros_old = concordance_index(master["OS_months"], -master["ros_risk_score"], master["OS_event"])
ci_alt_old = concordance_index(master["OS_months"], -master["alt_risk_score"], master["OS_event"])
ci_naive = concordance_index(master["OS_months"], -master["combined_score"], master["OS_event"])

print(f"  ROS signature (11-gene):    C = {ci_ros_old:.4f}")
print(f"  Altitude signature (9-gene): C = {ci_alt_old:.4f}")
print(f"  Naive combined score:       C = {ci_naive:.4f}")
print(f"  CONVERGENCE model ({len(selected_genes)}-gene): C = {ci_train:.4f}")
print(f"  Improvement over naive:     {ci_train - ci_naive:+.4f}")
print(f"  Improvement over ROS:       {ci_train - ci_ros_old:+.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. 10-FOLD CROSS-VALIDATION OF FINAL MODEL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. 10-FOLD CROSS-VALIDATION")
print("=" * 70)

kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_cis = []
cv_naive_cis = []

for train_idx, test_idx in kf.split(df):
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    try:
        cph_cv = CoxPHFitter(penalizer=selected_lambda, l1_ratio=1.0)
        cph_cv.fit(train[["OS_months", "OS_event"] + z_genes],
                    duration_col="OS_months", event_col="OS_event")
        pred = cph_cv.predict_partial_hazard(test[z_genes]).values.flatten()
        ci = concordance_index(test["OS_months"], -pred, test["OS_event"])
        cv_cis.append(ci)
    except:
        pass

    # Naive for comparison
    z_ros_test = (test["ros_risk_score"] if "ros_risk_score" in test.columns
                  else np.zeros(len(test)))
    z_alt_test = (test["alt_risk_score"] if "alt_risk_score" in test.columns
                  else np.zeros(len(test)))

# Re-do with proper naive scores using the master dataframe
master_indexed = master.set_index("patientId")
cv_naive_cis = []
kf2 = KFold(n_splits=10, shuffle=True, random_state=42)
for train_idx, test_idx in kf2.split(df):
    test = df.iloc[test_idx]
    matched = [pid for pid in test.index if pid in master_indexed.index]
    if len(matched) >= 5:
        m_sub = master_indexed.loc[matched]
        ros_std = m_sub["ros_risk_score"].std()
        alt_std = m_sub["alt_risk_score"].std()
        if ros_std > 0 and alt_std > 0:
            naive = (m_sub["ros_risk_score"] / ros_std + m_sub["alt_risk_score"] / alt_std) / 2
            ci_n = concordance_index(m_sub["OS_months"], -naive, m_sub["OS_event"])
            cv_naive_cis.append(ci_n)

cv_mean = np.mean(cv_cis)
cv_std = np.std(cv_cis)
cv_naive_mean = np.mean(cv_naive_cis) if cv_naive_cis else 0

print(f"  Convergence model: CV C-index = {cv_mean:.4f} +/- {cv_std:.4f}")
print(f"  Naive combined:    CV C-index = {cv_naive_mean:.4f}")
print(f"  CV improvement:    {cv_mean - cv_naive_mean:+.4f}")
print(f"  Overfitting gap:   train={ci_train:.4f}, CV={cv_mean:.4f}, gap={ci_train-cv_mean:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. PERMUTATION TEST
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("6. PERMUTATION TEST (500 iterations)")
print("=" * 70)

n_perm = 500
perm_cis = []
for i in range(n_perm):
    perm_events = df["OS_event"].values.copy()
    np.random.shuffle(perm_events)
    ci_p = concordance_index(df["OS_months"], -pred_train, perm_events)
    perm_cis.append(ci_p)

perm_p = np.mean([p >= ci_train for p in perm_cis])
print(f"  Observed C-index: {ci_train:.4f}")
print(f"  Permuted mean:    {np.mean(perm_cis):.4f}")
print(f"  Permutation p:    {perm_p:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. SURVIVAL ANALYSIS WITH CONVERGENCE SCORE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("7. SURVIVAL ANALYSIS (TCGA)")
print("=" * 70)

df["convergence_score"] = pred_train

# Median split
med = df["convergence_score"].median()
df["conv_group"] = np.where(df["convergence_score"] >= med, "High", "Low")

high = df[df["conv_group"] == "High"]
low = df[df["conv_group"] == "Low"]

lr = logrank_test(high["OS_months"], low["OS_months"],
                  event_observed_A=high["OS_event"], event_observed_B=low["OS_event"])

cdf = df[["OS_months", "OS_event"]].copy()
cdf["is_high"] = (df["conv_group"] == "High").astype(int)
cph_bin = CoxPHFitter()
cph_bin.fit(cdf, duration_col="OS_months", event_col="OS_event")
hr = float(cph_bin.summary.loc["is_high", "exp(coef)"])
hr_lo = float(cph_bin.summary.loc["is_high", "exp(coef) lower 95%"])
hr_hi = float(cph_bin.summary.loc["is_high", "exp(coef) upper 95%"])
p_cox = float(cph_bin.summary.loc["is_high", "p"])

print(f"  Median split: HR={hr:.2f} ({hr_lo:.2f}-{hr_hi:.2f}), p={p_cox:.2e}")
print(f"  Log-rank p: {lr.p_value:.2e}")

# Median survival
kmf = KaplanMeierFitter()
kmf.fit(high["OS_months"], event_observed=high["OS_event"])
med_high = kmf.median_survival_time_
kmf.fit(low["OS_months"], event_observed=low["OS_event"])
med_low = kmf.median_survival_time_
print(f"  High risk: median={med_high:.1f} months, n={len(high)}, events={int(high['OS_event'].sum())}")
print(f"  Low risk:  median={med_low:.1f} months, n={len(low)}, events={int(low['OS_event'].sum())}")

# ══════════════════════════════════════════════════════════════════════════════
# 8. EXTERNAL VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("8. EXTERNAL VALIDATION")
print("=" * 70)


def compute_convergence_score(expr_df, selected_genes, gene_means, gene_stds):
    """Compute convergence risk score using cohort-level z-normalization."""
    available = [g for g in selected_genes if g in expr_df.columns]
    if len(available) < len(selected_genes) * 0.5:
        return None, available
    score = np.zeros(len(expr_df))
    for gene in available:
        vals = expr_df[gene].values.astype(float)
        m, s = np.nanmean(vals), np.nanstd(vals)
        if s > 0:
            score += selected_genes[gene] * ((vals - m) / s)
    return score, available


validation_results = []

# ── GSE14520 ──
print("\n--- GSE14520 ---")
try:
    expr14 = pd.read_csv(os.path.join(GEO, "gse14520_gene_expr_cache.csv"), index_col=0)
    if expr14.shape[0] > expr14.shape[1]:
        expr14 = expr14.T

    clin14 = pd.read_csv(os.path.join(GEO, "GSE14520_Extra_Supplement.txt.gz"),
                          sep='\t', compression='gzip').set_index("Affy_GSM")
    common = sorted(set(expr14.index) & set(clin14.index))
    expr14 = expr14.loc[common]
    clin14 = clin14.loc[common]

    conv_score, avail = compute_convergence_score(expr14, selected_genes, gene_means, gene_stds)

    if conv_score is not None:
        vdf = pd.DataFrame({
            "OS_months": pd.to_numeric(clin14["Survival months"], errors='coerce'),
            "OS_event": pd.to_numeric(clin14["Survival status"], errors='coerce'),
            "score": conv_score,
        }, index=common)
        vdf = vdf.dropna()
        vdf = vdf[vdf["OS_months"] > 0]

        ci_v = concordance_index(vdf["OS_months"], -vdf["score"], vdf["OS_event"])

        med_v = vdf["score"].median()
        vdf["grp"] = np.where(vdf["score"] >= med_v, "High", "Low")
        h, l = vdf[vdf["grp"] == "High"], vdf[vdf["grp"] == "Low"]
        lr_v = logrank_test(h["OS_months"], l["OS_months"],
                            event_observed_A=h["OS_event"], event_observed_B=l["OS_event"])
        cd = vdf[["OS_months", "OS_event"]].copy()
        cd["is_high"] = (vdf["grp"] == "High").astype(int)
        try:
            cp = CoxPHFitter()
            cp.fit(cd, duration_col="OS_months", event_col="OS_event")
            hr_v = float(cp.summary.loc["is_high", "exp(coef)"])
            hr_lo_v = float(cp.summary.loc["is_high", "exp(coef) lower 95%"])
            hr_hi_v = float(cp.summary.loc["is_high", "exp(coef) upper 95%"])
            p_v = float(cp.summary.loc["is_high", "p"])
        except Exception:
            hr_v, hr_lo_v, hr_hi_v, p_v = np.nan, np.nan, np.nan, lr_v.p_value

        print(f"  n={len(vdf)}, events={int(vdf['OS_event'].sum())}")
        print(f"  Genes available: {len(avail)}/{len(selected_genes)}: {avail}")
        print(f"  C-index: {ci_v:.4f}")
        print(f"  HR: {hr_v:.2f} ({hr_lo_v:.2f}-{hr_hi_v:.2f}), p={p_v:.4f}")
        print(f"  Log-rank p: {lr_v.p_value:.4e}")

        validation_results.append({
            "cohort": "GSE14520", "n": len(vdf), "events": int(vdf["OS_event"].sum()),
            "genes_available": len(avail), "genes_total": len(selected_genes),
            "c_index": ci_v, "HR": hr_v, "HR_lower": hr_lo_v, "HR_upper": hr_hi_v,
            "cox_p": p_v, "logrank_p": lr_v.p_value,
        })
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback; traceback.print_exc()

# ── ICGC LIRI-JP ──
print("\n--- ICGC LIRI-JP ---")
try:
    expr_liri = pd.read_csv(os.path.join(GEO, "liri_jp_expression_merged.csv"), index_col=0).T
    clin_liri = pd.read_csv(os.path.join(DATA, "liri_clinical_xena.csv")).set_index("donor_id")
    common = sorted(set(expr_liri.index) & set(clin_liri.index))
    expr_liri = expr_liri.loc[common]
    clin_liri = clin_liri.loc[common]

    conv_score, avail = compute_convergence_score(expr_liri, selected_genes, gene_means, gene_stds)

    if conv_score is not None:
        vdf = pd.DataFrame({
            "OS_months": clin_liri["OS_time"].astype(float) / 30.44,
            "OS_event": clin_liri["OS"].astype(float),
            "score": conv_score,
        }, index=common)
        vdf = vdf.dropna()
        vdf = vdf[vdf["OS_months"] > 0]

        ci_v = concordance_index(vdf["OS_months"], -vdf["score"], vdf["OS_event"])

        med_v = vdf["score"].median()
        vdf["grp"] = np.where(vdf["score"] >= med_v, "High", "Low")
        h, l = vdf[vdf["grp"] == "High"], vdf[vdf["grp"] == "Low"]
        lr_v = logrank_test(h["OS_months"], l["OS_months"],
                            event_observed_A=h["OS_event"], event_observed_B=l["OS_event"])
        cd = vdf[["OS_months", "OS_event"]].copy()
        cd["is_high"] = (vdf["grp"] == "High").astype(int)
        try:
            cp = CoxPHFitter()
            cp.fit(cd, duration_col="OS_months", event_col="OS_event")
            hr_v = float(cp.summary.loc["is_high", "exp(coef)"])
            hr_lo_v = float(cp.summary.loc["is_high", "exp(coef) lower 95%"])
            hr_hi_v = float(cp.summary.loc["is_high", "exp(coef) upper 95%"])
            p_v = float(cp.summary.loc["is_high", "p"])
        except Exception:
            hr_v, hr_lo_v, hr_hi_v, p_v = np.nan, np.nan, np.nan, lr_v.p_value

        print(f"  n={len(vdf)}, events={int(vdf['OS_event'].sum())}")
        print(f"  Genes available: {len(avail)}/{len(selected_genes)}: {avail}")
        print(f"  C-index: {ci_v:.4f}")
        print(f"  HR: {hr_v:.2f} ({hr_lo_v:.2f}-{hr_hi_v:.2f}), p={p_v:.4f}")
        print(f"  Log-rank p: {lr_v.p_value:.4e}")

        validation_results.append({
            "cohort": "ICGC_LIRI-JP", "n": len(vdf), "events": int(vdf["OS_event"].sum()),
            "genes_available": len(avail), "genes_total": len(selected_genes),
            "c_index": ci_v, "HR": hr_v, "HR_lower": hr_lo_v, "HR_upper": hr_hi_v,
            "cox_p": p_v, "logrank_p": lr_v.p_value,
        })
    else:
        print(f"  Not enough genes: {avail}")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback; traceback.print_exc()

# ── GSE76427 (from series matrix) ──
print("\n--- GSE76427 ---")
try:
    matrix_path = os.path.join(GEO, "GSE76427_series_matrix.txt.gz")
    annot_path = os.path.join(GEO, "GPL10558_annot.csv")

    with gzip.open(matrix_path, 'rt', errors='replace') as f:
        lines = f.readlines()

    sample_ids = []
    chars = {}
    data_lines = []
    in_data = False
    for line in lines:
        line = line.strip()
        if line.startswith('!Sample_geo_accession'):
            sample_ids = [s.strip('"') for s in line.split('\t')[1:]]
        elif line.startswith('!Sample_characteristics_ch1'):
            parts = [p.strip('"') for p in line.split('\t')[1:]]
            if parts and ':' in parts[0]:
                key = parts[0].split(':')[0].strip()
                vals = [p.split(':', 1)[-1].strip() if ':' in p else p for p in parts]
                orig = key; idx = 1
                while key in chars: key = f"{orig}_{idx}"; idx += 1
                chars[key] = vals
        elif '!series_matrix_table_begin' in line: in_data = True; continue
        elif '!series_matrix_table_end' in line: in_data = False
        elif in_data and line: data_lines.append(line)

    clin76 = pd.DataFrame(chars, index=sample_ids)
    os_ev = [c for c in clin76.columns if 'event_os' in c.lower()]
    os_tm = [c for c in clin76.columns if 'duryears_os' in c.lower()]

    if os_ev and os_tm:
        clin76["OS_event"] = pd.to_numeric(clin76[os_ev[0]], errors='coerce')
        clin76["OS_months"] = pd.to_numeric(clin76[os_tm[0]], errors='coerce') * 12

        # Parse expression and map to gene symbols
        header = data_lines[0].split('\t')
        probes, vals = [], []
        for line in data_lines[1:]:
            parts = line.split('\t')
            probes.append(parts[0].strip('"'))
            vals.append([float(x) if x.strip('"') not in ('', 'null', 'NA') else np.nan for x in parts[1:]])
        expr76 = pd.DataFrame(vals, index=probes, columns=[s.strip('"') for s in header[1:]])

        annot = pd.read_csv(annot_path)
        sym_col = "Gene symbol"
        id_col = annot.columns[0]
        mapping = annot.set_index(id_col)[sym_col].dropna().to_dict()
        expr76["gene"] = [mapping.get(p, None) for p in expr76.index]
        expr76 = expr76.dropna(subset=["gene"])
        expr76 = expr76[expr76["gene"] != ""]
        expr76 = expr76.groupby("gene").mean().T  # samples x genes

        common76 = sorted(set(expr76.index) & set(clin76.index))
        expr76 = expr76.loc[common76]
        clin76 = clin76.loc[common76]

        conv_score, avail = compute_convergence_score(expr76, selected_genes, gene_means, gene_stds)

        if conv_score is not None:
            vdf = pd.DataFrame({
                "OS_months": clin76["OS_months"].values,
                "OS_event": clin76["OS_event"].values,
                "score": conv_score,
            }, index=common76)
            vdf = vdf.dropna()
            vdf = vdf[vdf["OS_months"] > 0]

            ci_v = concordance_index(vdf["OS_months"], -vdf["score"], vdf["OS_event"])

            med_v = vdf["score"].median()
            vdf["grp"] = np.where(vdf["score"] >= med_v, "High", "Low")
            h, l = vdf[vdf["grp"] == "High"], vdf[vdf["grp"] == "Low"]
            lr_v = logrank_test(h["OS_months"], l["OS_months"],
                                event_observed_A=h["OS_event"], event_observed_B=l["OS_event"])
            cd = vdf[["OS_months", "OS_event"]].copy()
            cd["is_high"] = (vdf["grp"] == "High").astype(int)
            cp = CoxPHFitter()
            cp.fit(cd, duration_col="OS_months", event_col="OS_event")
            hr_v = float(cp.summary.loc["is_high", "exp(coef)"])
            hr_lo_v = float(cp.summary.loc["is_high", "exp(coef) lower 95%"])
            hr_hi_v = float(cp.summary.loc["is_high", "exp(coef) upper 95%"])
            p_v = float(cp.summary.loc["is_high", "p"])

            print(f"  n={len(vdf)}, events={int(vdf['OS_event'].sum())}")
            print(f"  Genes available: {len(avail)}/{len(selected_genes)}: {avail}")
            print(f"  C-index: {ci_v:.4f}")
            print(f"  HR: {hr_v:.2f} ({hr_lo_v:.2f}-{hr_hi_v:.2f}), p={p_v:.4f}")
            print(f"  Log-rank p: {lr_v.p_value:.4e}")

            validation_results.append({
                "cohort": "GSE76427", "n": len(vdf), "events": int(vdf["OS_event"].sum()),
                "genes_available": len(avail), "genes_total": len(selected_genes),
                "c_index": ci_v, "HR": hr_v, "HR_lower": hr_lo_v, "HR_upper": hr_hi_v,
                "cox_p": p_v, "logrank_p": lr_v.p_value,
            })

except Exception as e:
    print(f"  ERROR: {e}")
    import traceback; traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════════════
# 9. FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("9. FINAL SUMMARY — CONVERGENCE SIGNATURE")
print("=" * 70)

print(f"\n  Model: {len(selected_genes)}-gene convergence signature (LASSO-Cox, lambda={selected_lambda})")
print(f"  Genes: {list(selected_genes.keys())}")
print(f"\n  {'Cohort':<20s} {'n':>5s} {'Events':>7s} {'C-index':>8s} {'HR':>8s} {'p-value':>10s}")
print(f"  {'-'*60}")
print(f"  {'TCGA-LIHC (train)':<20s} {len(df):>5d} {int(df['OS_event'].sum()):>7d} {ci_train:>8.4f} {hr:>8.2f} {p_cox:>10.2e}")
for r in validation_results:
    print(f"  {r['cohort']:<20s} {r['n']:>5d} {r['events']:>7d} {r['c_index']:>8.4f} {r['HR']:>8.2f} {r['cox_p']:>10.4f}")

# Fisher combined p
valid_p = [r["logrank_p"] for r in validation_results if r["logrank_p"] < 1]
if len(valid_p) >= 2:
    chi2 = -2 * sum(np.log(p) for p in valid_p)
    fisher_p = 1 - stats.chi2.cdf(chi2, 2 * len(valid_p))
    print(f"\n  Fisher combined p ({len(valid_p)} validation cohorts): {fisher_p:.2e}")

# ══════════════════════════════════════════════════════════════════════════════
# 10. SAVE MODEL & RESULTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("10. SAVING MODEL & RESULTS")
print("=" * 70)

# Save model
model_out = {
    "model_type": "LASSO-Cox convergence signature",
    "penalizer": selected_lambda,
    "l1_ratio": 1.0,
    "n_candidate_genes": len(ALL_GENES),
    "genes": selected_genes,
    "gene_means": gene_means,
    "gene_stds": gene_stds,
    "c_index_train": ci_train,
    "cv_cindex_mean": cv_mean,
    "cv_cindex_std": cv_std,
    "perm_p": perm_p,
    "n_patients": len(df),
    "n_events": int(df["OS_event"].sum()),
    "aic": float(cph_final.AIC_partial_),
    "cv_results": cv_df.to_dict(orient="records"),
}
with open(os.path.join(MODEL_DIR, "convergence_lasso_model.json"), "w") as f:
    json.dump(model_out, f, indent=2)
print(f"  Saved: results/model/convergence_lasso_model.json")

# Save validation results
val_df = pd.DataFrame([{
    "cohort": "TCGA-LIHC (train)", "n": len(df), "events": int(df["OS_event"].sum()),
    "genes_available": len(selected_genes), "genes_total": len(selected_genes),
    "c_index": ci_train, "HR": hr, "HR_lower": hr_lo, "HR_upper": hr_hi,
    "cox_p": p_cox, "logrank_p": lr.p_value,
}] + validation_results)
val_df.to_csv(os.path.join(TABLES, "convergence_validation.csv"), index=False)
print(f"  Saved: results/tables/convergence_validation.csv")

# Save CV results
cv_df.to_csv(os.path.join(TABLES, "convergence_cv_lambda.csv"), index=False)

# ══════════════════════════════════════════════════════════════════════════════
# 11. FIGURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("11. FIGURES")
print("=" * 70)

# Lambda CV curve
fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(cv_df["penalizer"], cv_df["mean_cindex"], yerr=cv_df["std_cindex"],
            fmt='o-', color='#4c72b0', capsize=4, linewidth=2, markersize=6)
ax.axvline(selected_lambda, color='red', linestyle='--', alpha=0.7,
           label=f'Selected (1-SE): lambda={selected_lambda}')
ax2 = ax.twinx()
ax2.plot(cv_df["penalizer"], cv_df["mean_ngenes"], 's--', color='#dd8452', alpha=0.7, markersize=5)
ax2.set_ylabel("Number of genes", fontsize=11, color='#dd8452')
ax.set_xlabel("Penalizer (lambda)", fontsize=12)
ax.set_ylabel("CV C-index", fontsize=12)
ax.set_title("LASSO-Cox Cross-Validation: Lambda Selection", fontweight='bold')
ax.set_xscale('log')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig11a_lasso_cv_lambda.png"), dpi=300)
plt.close()
print("  Saved: fig11a_lasso_cv_lambda.png")

# Gene coefficients
fig, ax = plt.subplots(figsize=(10, max(4, len(selected_genes) * 0.5)))
genes_sorted = sorted(selected_genes.items(), key=lambda x: x[1])
gene_names = [g for g, c in genes_sorted]
coefs = [c for g, c in genes_sorted]
colors = ['#d62728' if c > 0 else '#2ca02c' for c in coefs]
y_pos = range(len(gene_names))
ax.barh(y_pos, coefs, color=colors, edgecolor='white', height=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{g} ({'ROS' if g in ROS_GENES else 'Alt'})" for g in gene_names], fontsize=10)
ax.axvline(0, color='gray', linewidth=1)
ax.set_xlabel("LASSO-Cox Coefficient", fontsize=12)
ax.set_title(f"Convergence Signature: {len(selected_genes)} Genes", fontweight='bold')
ax.grid(True, alpha=0.2, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig11b_convergence_coefficients.png"), dpi=300)
plt.close()
print("  Saved: fig11b_convergence_coefficients.png")

# KM survival
fig, axes = plt.subplots(1, 1 + len(validation_results), figsize=(6 * (1 + len(validation_results)), 5))
if not isinstance(axes, np.ndarray):
    axes = [axes]

# TCGA
ax = axes[0]
kmf = KaplanMeierFitter()
for lbl, col in [("High", "#d62728"), ("Low", "#2ca02c")]:
    g = df[df["conv_group"] == lbl]
    kmf.fit(g["OS_months"], event_observed=g["OS_event"], label=f"{lbl} (n={len(g)})")
    kmf.plot_survival_function(ax=ax, ci_show=True, color=col, linewidth=2)
ax.set_title(f"TCGA-LIHC (n={len(df)})\nHR={hr:.2f}, p={lr.p_value:.2e}", fontweight='bold')
ax.set_xlabel("Time (months)"); ax.set_ylabel("Survival Probability")
ax.legend(fontsize=9, loc="lower left"); ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)

# External cohorts
for idx, r in enumerate(validation_results):
    ax = axes[idx + 1]
    ax.text(0.5, 0.5, f'{r["cohort"]}\nn={r["n"]}, events={r["events"]}\n'
            f'C-index: {r["c_index"]:.4f}\n'
            f'HR: {r["HR"]:.2f} ({r["HR_lower"]:.2f}-{r["HR_upper"]:.2f})\n'
            f'p = {r["cox_p"]:.4f}',
            transform=ax.transAxes, ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title(f'{r["cohort"]} ({r["genes_available"]}/{r["genes_total"]} genes)', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig11c_convergence_km.png"), dpi=300)
plt.close()
print("  Saved: fig11c_convergence_km.png")

print("\n" + "=" * 70)
print("DONE — Script 11 complete")
print("=" * 70)
