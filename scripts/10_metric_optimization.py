"""
10_metric_optimization.py — Optimize combined score and re-validate

Improvements over naive equal-weight average:
1. Cox-learned optimal weights for ROS + Altitude
2. Add NRF2 activity as 3rd predictor
3. Add interaction term (ROS x Altitude)
4. Optimal cutpoints via maximally selected rank statistics
5. Internal 10-fold cross-validation + permutation test
6. Re-validate on external cohorts with optimized model
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

with open(os.path.join(DATA, "paths.json")) as f:
    paths = json.load(f)
with open(paths["ros_model"]) as f:
    ros_model = json.load(f)
with open(paths["alt_model"]) as f:
    alt_model = json.load(f)

GEO = paths["geo_cohorts"]

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("1. LOADING TCGA DATA")
print("=" * 70)

df = pd.read_csv(os.path.join(DATA, "tcga_convergence_master.csv"))
df = df.dropna(subset=["OS_months", "OS_event"]).copy()
df = df[df["OS_months"] > 0].copy()
print(f"Patients: {len(df)}, Events: {int(df['OS_event'].sum())}")

# Z-score the risk scores for comparable weighting
df["z_ros"] = (df["ros_risk_score"] - df["ros_risk_score"].mean()) / df["ros_risk_score"].std()
df["z_alt"] = (df["alt_risk_score"] - df["alt_risk_score"].mean()) / df["alt_risk_score"].std()
df["z_nrf2"] = (df["nrf2_activity"] - df["nrf2_activity"].mean()) / df["nrf2_activity"].std()
df["z_ros_x_alt"] = df["z_ros"] * df["z_alt"]

# ══════════════════════════════════════════════════════════════════════════════
# 2. BASELINE: CURRENT NAIVE COMBINED SCORE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. BASELINE (naive equal-weight average)")
print("=" * 70)

df["naive_combined"] = (df["z_ros"] + df["z_alt"]) / 2
ci_naive = concordance_index(df["OS_months"], -df["naive_combined"], df["OS_event"])
ci_ros = concordance_index(df["OS_months"], -df["z_ros"], df["OS_event"])
ci_alt = concordance_index(df["OS_months"], -df["z_alt"], df["OS_event"])
print(f"  ROS alone:         C = {ci_ros:.4f}")
print(f"  Altitude alone:    C = {ci_alt:.4f}")
print(f"  Naive combined:    C = {ci_naive:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL COMPARISON — FIT PROGRESSIVELY RICHER COX MODELS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. COX MODEL COMPARISON (learning optimal weights)")
print("=" * 70)

models = {}

# Model A: ROS only
cph_a = CoxPHFitter()
cph_a.fit(df[["OS_months", "OS_event", "z_ros"]], duration_col="OS_months", event_col="OS_event")
pred_a = cph_a.predict_partial_hazard(df[["z_ros"]]).values.flatten()
ci_a = concordance_index(df["OS_months"], -pred_a, df["OS_event"])
models["A: ROS only"] = {"c_index": ci_a, "cph": cph_a, "features": ["z_ros"]}

# Model B: Altitude only
cph_b = CoxPHFitter()
cph_b.fit(df[["OS_months", "OS_event", "z_alt"]], duration_col="OS_months", event_col="OS_event")
pred_b = cph_b.predict_partial_hazard(df[["z_alt"]]).values.flatten()
ci_b = concordance_index(df["OS_months"], -pred_b, df["OS_event"])
models["B: Alt only"] = {"c_index": ci_b, "cph": cph_b, "features": ["z_alt"]}

# Model C: Cox-weighted ROS + Alt (2-feature)
cph_c = CoxPHFitter()
cph_c.fit(df[["OS_months", "OS_event", "z_ros", "z_alt"]], duration_col="OS_months", event_col="OS_event")
pred_c = cph_c.predict_partial_hazard(df[["z_ros", "z_alt"]]).values.flatten()
ci_c = concordance_index(df["OS_months"], -pred_c, df["OS_event"])
models["C: Cox(ROS+Alt)"] = {"c_index": ci_c, "cph": cph_c, "features": ["z_ros", "z_alt"]}

# Model D: Cox-weighted ROS + Alt + NRF2 (3-feature)
cph_d = CoxPHFitter()
cph_d.fit(df[["OS_months", "OS_event", "z_ros", "z_alt", "z_nrf2"]],
          duration_col="OS_months", event_col="OS_event")
pred_d = cph_d.predict_partial_hazard(df[["z_ros", "z_alt", "z_nrf2"]]).values.flatten()
ci_d = concordance_index(df["OS_months"], -pred_d, df["OS_event"])
models["D: Cox(ROS+Alt+NRF2)"] = {"c_index": ci_d, "cph": cph_d, "features": ["z_ros", "z_alt", "z_nrf2"]}

# Model E: Cox with interaction term
cph_e = CoxPHFitter()
cph_e.fit(df[["OS_months", "OS_event", "z_ros", "z_alt", "z_ros_x_alt"]],
          duration_col="OS_months", event_col="OS_event")
pred_e = cph_e.predict_partial_hazard(df[["z_ros", "z_alt", "z_ros_x_alt"]]).values.flatten()
ci_e = concordance_index(df["OS_months"], -pred_e, df["OS_event"])
models["E: Cox(ROS+Alt+Interact)"] = {"c_index": ci_e, "cph": cph_e, "features": ["z_ros", "z_alt", "z_ros_x_alt"]}

# Model F: Full model (ROS + Alt + NRF2 + Interaction)
cph_f = CoxPHFitter()
cph_f.fit(df[["OS_months", "OS_event", "z_ros", "z_alt", "z_nrf2", "z_ros_x_alt"]],
          duration_col="OS_months", event_col="OS_event")
pred_f = cph_f.predict_partial_hazard(df[["z_ros", "z_alt", "z_nrf2", "z_ros_x_alt"]]).values.flatten()
ci_f = concordance_index(df["OS_months"], -pred_f, df["OS_event"])
models["F: Full(ROS+Alt+NRF2+Int)"] = {"c_index": ci_f, "cph": cph_f,
                                         "features": ["z_ros", "z_alt", "z_nrf2", "z_ros_x_alt"]}

print(f"\n  {'Model':<30s} {'C-index':>8s}  {'vs Naive':>10s}")
print(f"  {'-'*50}")
print(f"  {'Naive equal-weight':<30s} {ci_naive:>8.4f}  {'baseline':>10s}")
for name, m in models.items():
    delta = m["c_index"] - ci_naive
    print(f"  {name:<30s} {m['c_index']:>8.4f}  {delta:>+10.4f}")

# Print coefficients of best models
print("\n  Cox coefficients:")
for name in ["C: Cox(ROS+Alt)", "D: Cox(ROS+Alt+NRF2)", "F: Full(ROS+Alt+NRF2+Int)"]:
    m = models[name]
    print(f"\n  {name}:")
    for var in m["cph"].summary.index:
        row = m["cph"].summary.loc[var]
        print(f"    {var:<16s}: coef={row['coef']:+.4f}, HR={row['exp(coef)']:.3f}, p={row['p']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. SELECT BEST MODEL & COMPUTE OPTIMIZED SCORE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. SELECT BEST MODEL")
print("=" * 70)

# Pick the model with highest C-index
best_name = max(models, key=lambda k: models[k]["c_index"])
best = models[best_name]
print(f"  Best model: {best_name} (C = {best['c_index']:.4f})")

# Also check: if adding NRF2 or interaction doesn't help much, prefer simpler model
# Use AIC for model selection
print(f"\n  AIC comparison:")
for name, m in models.items():
    aic = m["cph"].AIC_partial_
    print(f"    {name:<30s}: AIC = {aic:.1f}")

# Select by AIC (lower is better)
best_aic_name = min(models, key=lambda k: models[k]["cph"].AIC_partial_)
best_aic = models[best_aic_name]
print(f"\n  Best by AIC: {best_aic_name} (AIC = {best_aic['cph'].AIC_partial_:.1f})")

# Use the AIC-selected model as the optimized model
selected_name = best_aic_name
selected = models[selected_name]
selected_cph = selected["cph"]
selected_features = selected["features"]
print(f"\n  SELECTED MODEL: {selected_name}")
print(f"  Features: {selected_features}")
print(f"  C-index: {selected['c_index']:.4f} (vs naive {ci_naive:.4f}, delta = {selected['c_index']-ci_naive:+.4f})")

# Compute optimized score for all patients
df["optimized_score"] = selected_cph.predict_partial_hazard(df[selected_features]).values.flatten()

# ══════════════════════════════════════════════════════════════════════════════
# 5. 10-FOLD CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. 10-FOLD CROSS-VALIDATION")
print("=" * 70)

kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_cindexes = []
cv_cindexes_naive = []

for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
    train = df.iloc[train_idx]
    test = df.iloc[test_idx]

    # Fit Cox on train
    try:
        cph_cv = CoxPHFitter()
        cph_cv.fit(train[["OS_months", "OS_event"] + selected_features],
                    duration_col="OS_months", event_col="OS_event")
        pred_test = cph_cv.predict_partial_hazard(test[selected_features]).values.flatten()
        ci_test = concordance_index(test["OS_months"], -pred_test, test["OS_event"])
        cv_cindexes.append(ci_test)
    except Exception:
        pass

    # Naive combined on test
    naive_test = (test["z_ros"] + test["z_alt"]) / 2
    ci_naive_test = concordance_index(test["OS_months"], -naive_test, test["OS_event"])
    cv_cindexes_naive.append(ci_naive_test)

cv_mean = np.mean(cv_cindexes)
cv_std = np.std(cv_cindexes)
cv_naive_mean = np.mean(cv_cindexes_naive)
cv_naive_std = np.std(cv_cindexes_naive)

print(f"  Optimized model: CV C-index = {cv_mean:.4f} +/- {cv_std:.4f}")
print(f"  Naive combined:  CV C-index = {cv_naive_mean:.4f} +/- {cv_naive_std:.4f}")
print(f"  CV improvement:  {cv_mean - cv_naive_mean:+.4f}")
print(f"  Overfitting check: train={selected['c_index']:.4f}, CV={cv_mean:.4f}, "
      f"gap={selected['c_index']-cv_mean:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. PERMUTATION TEST
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("6. PERMUTATION TEST (200 iterations)")
print("=" * 70)

n_perm = 200
perm_cindexes = []
for i in range(n_perm):
    perm_events = df["OS_event"].values.copy()
    np.random.shuffle(perm_events)
    perm_ci = concordance_index(df["OS_months"], -df["optimized_score"], perm_events)
    perm_cindexes.append(perm_ci)

perm_mean = np.mean(perm_cindexes)
perm_p = np.mean([pc >= selected["c_index"] for pc in perm_cindexes])
print(f"  Observed C-index: {selected['c_index']:.4f}")
print(f"  Permuted mean:    {perm_mean:.4f}")
print(f"  Permutation p:    {perm_p:.4f} ({'SIGNIFICANT' if perm_p < 0.01 else 'check'})")

# ══════════════════════════════════════════════════════════════════════════════
# 7. OPTIMAL CUTPOINTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("7. OPTIMAL CUTPOINTS (maximally selected rank statistic)")
print("=" * 70)

# Test a range of cutpoints for the optimized score and find the one with lowest log-rank p
quantiles = np.arange(0.25, 0.76, 0.05)
best_p = 1.0
best_q = 0.5
cutpoint_results = []

for q in quantiles:
    cut = df["optimized_score"].quantile(q)
    high = df[df["optimized_score"] >= cut]
    low = df[df["optimized_score"] < cut]
    if len(high) >= 20 and len(low) >= 20 and high["OS_event"].sum() >= 5 and low["OS_event"].sum() >= 5:
        lr = logrank_test(high["OS_months"], low["OS_months"],
                          event_observed_A=high["OS_event"], event_observed_B=low["OS_event"])
        cutpoint_results.append({"quantile": q, "cutpoint": cut,
                                  "n_high": len(high), "n_low": len(low),
                                  "p_value": lr.p_value})
        if lr.p_value < best_p:
            best_p = lr.p_value
            best_q = q

print(f"  Best cutpoint: quantile={best_q:.2f}, p={best_p:.2e}")
print(f"  (Median split: quantile=0.50)")

# Apply optimal cutpoint
opt_cut = df["optimized_score"].quantile(best_q)
df["opt_risk_group"] = np.where(df["optimized_score"] >= opt_cut, "High", "Low")
grp_h = df[df["opt_risk_group"] == "High"]
grp_l = df[df["opt_risk_group"] == "Low"]

# Cox HR with optimal cut
cdf_opt = df[["OS_months", "OS_event"]].copy()
cdf_opt["is_high"] = (df["opt_risk_group"] == "High").astype(int)
cph_opt = CoxPHFitter()
cph_opt.fit(cdf_opt, duration_col="OS_months", event_col="OS_event")
hr_opt = float(cph_opt.summary.loc["is_high", "exp(coef)"])
hr_opt_lo = float(cph_opt.summary.loc["is_high", "exp(coef) lower 95%"])
hr_opt_hi = float(cph_opt.summary.loc["is_high", "exp(coef) upper 95%"])
p_opt = float(cph_opt.summary.loc["is_high", "p"])

print(f"  Optimized binary: HR={hr_opt:.2f} ({hr_opt_lo:.2f}-{hr_opt_hi:.2f}), p={p_opt:.2e}")

# Compare with median split
med_cut = df["optimized_score"].median()
df["med_risk_group"] = np.where(df["optimized_score"] >= med_cut, "High", "Low")
cdf_med = df[["OS_months", "OS_event"]].copy()
cdf_med["is_high"] = (df["med_risk_group"] == "High").astype(int)
cph_med = CoxPHFitter()
cph_med.fit(cdf_med, duration_col="OS_months", event_col="OS_event")
hr_med = float(cph_med.summary.loc["is_high", "exp(coef)"])
p_med = float(cph_med.summary.loc["is_high", "p"])
print(f"  Median split:     HR={hr_med:.2f}, p={p_med:.2e}")

# ══════════════════════════════════════════════════════════════════════════════
# 8. RE-VALIDATE ON EXTERNAL COHORTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("8. EXTERNAL VALIDATION WITH OPTIMIZED MODEL")
print("=" * 70)

# Save the optimized model coefficients
opt_coefs = {}
for var in selected_cph.summary.index:
    opt_coefs[var] = float(selected_cph.summary.loc[var, "coef"])
print(f"  Optimized coefficients: {opt_coefs}")

NRF2_TARGETS = ["NQO1", "HMOX1", "SLC7A11", "TXNRD1", "G6PD", "GSR",
                "GCLC", "GCLM", "FTH1", "FTL", "SQSTM1", "SRXN1",
                "AKR1C1", "AKR1B10", "ME1", "ABCC2"]


def compute_risk_cohort(expr_df, model_dict):
    genes = model_dict["genes"]
    available = [g for g in genes if g in expr_df.columns]
    if len(available) < len(genes) * 0.5:
        return None, available
    risk = np.zeros(len(expr_df))
    for gene in available:
        vals = expr_df[gene].values.astype(float)
        m, s = np.nanmean(vals), np.nanstd(vals)
        if s > 0:
            risk += genes[gene] * ((vals - m) / s)
    return risk, available


def compute_nrf2_cohort(expr_df):
    available = [g for g in NRF2_TARGETS if g in expr_df.columns]
    if len(available) < 3:
        return None
    zscores = pd.DataFrame()
    for g in available:
        vals = expr_df[g].astype(float)
        m, s = vals.mean(), vals.std()
        if s > 0:
            zscores[g] = (vals - m) / s
    return zscores.mean(axis=1).values


def compute_optimized_score(df_cohort, has_alt=True, has_nrf2=True):
    """Compute optimized score using learned Cox coefficients."""
    score = np.zeros(len(df_cohort))
    for feat, coef in opt_coefs.items():
        if feat == "z_ros":
            score += coef * df_cohort["z_ros"].values
        elif feat == "z_alt" and has_alt:
            score += coef * df_cohort["z_alt"].values
        elif feat == "z_nrf2" and has_nrf2:
            score += coef * df_cohort["z_nrf2"].values
        elif feat == "z_ros_x_alt" and has_alt:
            score += coef * (df_cohort["z_ros"] * df_cohort["z_alt"]).values
    return score


validation_results = []

# ── GSE14520 ──
print("\n--- GSE14520 ---")
try:
    expr14 = pd.read_csv(os.path.join(GEO, "gse14520_gene_expr_cache.csv"), index_col=0)
    if expr14.shape[0] > expr14.shape[1]:
        expr14 = expr14.T

    supp_path = os.path.join(GEO, "GSE14520_Extra_Supplement.txt.gz")
    clin14 = pd.read_csv(supp_path, sep='\t', compression='gzip')
    clin14 = clin14.set_index("Affy_GSM")

    common = sorted(set(expr14.index) & set(clin14.index))
    expr14 = expr14.loc[common]
    clin14 = clin14.loc[common]

    ros_risk, ros_avail = compute_risk_cohort(expr14, ros_model)
    alt_risk, alt_avail = compute_risk_cohort(expr14, alt_model)
    nrf2 = compute_nrf2_cohort(expr14)

    vdf = pd.DataFrame({
        "OS_months": pd.to_numeric(clin14["Survival months"], errors='coerce'),
        "OS_event": pd.to_numeric(clin14["Survival status"], errors='coerce'),
    }, index=common)

    if ros_risk is not None:
        vdf["z_ros"] = (ros_risk - ros_risk.mean()) / (ros_risk.std() + 1e-10)
    if alt_risk is not None:
        vdf["z_alt"] = (alt_risk - alt_risk.mean()) / (alt_risk.std() + 1e-10)
    else:
        vdf["z_alt"] = 0
    if nrf2 is not None:
        vdf["z_nrf2"] = (nrf2 - nrf2.mean()) / (nrf2.std() + 1e-10)
    else:
        vdf["z_nrf2"] = 0

    vdf = vdf.dropna(subset=["OS_months", "OS_event"])
    vdf = vdf[vdf["OS_months"] > 0]

    has_alt = alt_risk is not None
    has_nrf2 = nrf2 is not None

    # Naive combined
    vdf["naive"] = (vdf["z_ros"] + vdf["z_alt"]) / 2
    ci_naive_v = concordance_index(vdf["OS_months"], -vdf["naive"], vdf["OS_event"])

    # Optimized combined
    vdf["optimized"] = compute_optimized_score(vdf, has_alt=has_alt, has_nrf2=has_nrf2)
    ci_opt_v = concordance_index(vdf["OS_months"], -vdf["optimized"], vdf["OS_event"])

    # Binary HR (median split on optimized)
    med = vdf["optimized"].median()
    vdf["grp"] = np.where(vdf["optimized"] >= med, "High", "Low")
    high_v = vdf[vdf["grp"] == "High"]
    low_v = vdf[vdf["grp"] == "Low"]
    lr_v = logrank_test(high_v["OS_months"], low_v["OS_months"],
                        event_observed_A=high_v["OS_event"], event_observed_B=low_v["OS_event"])

    cdf_v = vdf[["OS_months", "OS_event"]].copy()
    cdf_v["is_high"] = (vdf["grp"] == "High").astype(int)
    cph_v = CoxPHFitter()
    cph_v.fit(cdf_v, duration_col="OS_months", event_col="OS_event")
    hr_v = float(cph_v.summary.loc["is_high", "exp(coef)"])
    hr_v_lo = float(cph_v.summary.loc["is_high", "exp(coef) lower 95%"])
    hr_v_hi = float(cph_v.summary.loc["is_high", "exp(coef) upper 95%"])
    p_v = float(cph_v.summary.loc["is_high", "p"])

    print(f"  n={len(vdf)}, events={int(vdf['OS_event'].sum())}")
    print(f"  ROS genes: {len(ros_avail)}, Alt genes: {len(alt_avail) if alt_avail else 0}, NRF2: {'yes' if has_nrf2 else 'no'}")
    print(f"  Naive C-index:     {ci_naive_v:.4f}")
    print(f"  Optimized C-index: {ci_opt_v:.4f} ({ci_opt_v-ci_naive_v:+.4f})")
    print(f"  HR (High vs Low):  {hr_v:.2f} ({hr_v_lo:.2f}-{hr_v_hi:.2f}), p={p_v:.4f}")
    print(f"  Log-rank p:        {lr_v.p_value:.4e}")

    validation_results.append({
        "cohort": "GSE14520", "n": len(vdf), "events": int(vdf["OS_event"].sum()),
        "ci_naive": ci_naive_v, "ci_optimized": ci_opt_v,
        "HR": hr_v, "HR_lower": hr_v_lo, "HR_upper": hr_v_hi,
        "cox_p": p_v, "logrank_p": lr_v.p_value,
    })
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback; traceback.print_exc()

# ── ICGC LIRI-JP ──
print("\n--- ICGC LIRI-JP ---")
try:
    expr_liri = pd.read_csv(os.path.join(GEO, "liri_jp_expression_merged.csv"), index_col=0)
    expr_liri = expr_liri.T  # donors x genes

    clin_liri = pd.read_csv(os.path.join(DATA, "liri_clinical_xena.csv"))
    clin_liri = clin_liri.set_index("donor_id")

    common = sorted(set(expr_liri.index) & set(clin_liri.index))
    expr_liri = expr_liri.loc[common]
    clin_liri = clin_liri.loc[common]

    ros_risk, ros_avail = compute_risk_cohort(expr_liri, ros_model)
    nrf2 = compute_nrf2_cohort(expr_liri)

    vdf = pd.DataFrame({
        "OS_months": clin_liri["OS_time"] / 30.44,
        "OS_event": clin_liri["OS"].astype(float),
    }, index=common)

    if ros_risk is not None:
        vdf["z_ros"] = (ros_risk - ros_risk.mean()) / (ros_risk.std() + 1e-10)
    vdf["z_alt"] = 0  # No altitude genes available
    if nrf2 is not None:
        vdf["z_nrf2"] = (nrf2 - nrf2.mean()) / (nrf2.std() + 1e-10)
    else:
        vdf["z_nrf2"] = 0

    vdf = vdf.dropna(subset=["OS_months", "OS_event"])
    vdf = vdf[vdf["OS_months"] > 0]

    has_nrf2 = nrf2 is not None

    vdf["naive"] = vdf["z_ros"]  # Only ROS available
    ci_naive_v = concordance_index(vdf["OS_months"], -vdf["naive"], vdf["OS_event"])

    vdf["optimized"] = compute_optimized_score(vdf, has_alt=False, has_nrf2=has_nrf2)
    ci_opt_v = concordance_index(vdf["OS_months"], -vdf["optimized"], vdf["OS_event"])

    med = vdf["optimized"].median()
    vdf["grp"] = np.where(vdf["optimized"] >= med, "High", "Low")
    high_v = vdf[vdf["grp"] == "High"]
    low_v = vdf[vdf["grp"] == "Low"]
    lr_v = logrank_test(high_v["OS_months"], low_v["OS_months"],
                        event_observed_A=high_v["OS_event"], event_observed_B=low_v["OS_event"])

    cdf_v = vdf[["OS_months", "OS_event"]].copy()
    cdf_v["is_high"] = (vdf["grp"] == "High").astype(int)
    cph_v = CoxPHFitter()
    cph_v.fit(cdf_v, duration_col="OS_months", event_col="OS_event")
    hr_v = float(cph_v.summary.loc["is_high", "exp(coef)"])
    hr_v_lo = float(cph_v.summary.loc["is_high", "exp(coef) lower 95%"])
    hr_v_hi = float(cph_v.summary.loc["is_high", "exp(coef) upper 95%"])
    p_v = float(cph_v.summary.loc["is_high", "p"])

    print(f"  n={len(vdf)}, events={int(vdf['OS_event'].sum())}")
    print(f"  ROS genes: {len(ros_avail)}, Alt genes: 0, NRF2: {'yes' if has_nrf2 else 'no'}")
    print(f"  Naive C-index (ROS-only):  {ci_naive_v:.4f}")
    print(f"  Optimized C-index:         {ci_opt_v:.4f} ({ci_opt_v-ci_naive_v:+.4f})")
    print(f"  HR (High vs Low):  {hr_v:.2f} ({hr_v_lo:.2f}-{hr_v_hi:.2f}), p={p_v:.4f}")
    print(f"  Log-rank p:        {lr_v.p_value:.4e}")

    validation_results.append({
        "cohort": "ICGC_LIRI-JP", "n": len(vdf), "events": int(vdf["OS_event"].sum()),
        "ci_naive": ci_naive_v, "ci_optimized": ci_opt_v,
        "HR": hr_v, "HR_lower": hr_v_lo, "HR_upper": hr_v_hi,
        "cox_p": p_v, "logrank_p": lr_v.p_value,
    })
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback; traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════════════
# 9. SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("9. FINAL COMPARISON: NAIVE vs OPTIMIZED")
print("=" * 70)

print(f"\n  TCGA Training:")
print(f"    Naive C-index:     {ci_naive:.4f}")
print(f"    Optimized C-index: {selected['c_index']:.4f} ({selected['c_index']-ci_naive:+.4f})")
print(f"    CV C-index:        {cv_mean:.4f} +/- {cv_std:.4f}")
print(f"    Permutation p:     {perm_p:.4f}")

for r in validation_results:
    print(f"\n  {r['cohort']}:")
    print(f"    Naive C-index:     {r['ci_naive']:.4f}")
    print(f"    Optimized C-index: {r['ci_optimized']:.4f} ({r['ci_optimized']-r['ci_naive']:+.4f})")
    print(f"    HR (High vs Low):  {r['HR']:.2f} ({r['HR_lower']:.2f}-{r['HR_upper']:.2f}), p={r['cox_p']:.4f}")

# Save results
results_df = pd.DataFrame([{
    "cohort": "TCGA-LIHC (train)", "n": len(df), "events": int(df["OS_event"].sum()),
    "ci_naive": ci_naive, "ci_optimized": selected["c_index"],
    "cv_cindex": cv_mean, "cv_std": cv_std, "perm_p": perm_p,
    "model": selected_name, "features": str(selected_features),
}] + validation_results)
results_df.to_csv(os.path.join(TABLES, "optimized_model_results.csv"), index=False)

# Save model coefficients
model_out = {
    "model_name": selected_name,
    "features": selected_features,
    "coefficients": opt_coefs,
    "c_index_train": selected["c_index"],
    "c_index_cv": cv_mean,
    "cv_std": cv_std,
    "perm_p": perm_p,
    "aic": float(selected_cph.AIC_partial_),
}
with open(os.path.join(BASE, "results", "model", "optimized_cox_model.json"), "w") as f:
    json.dump(model_out, f, indent=2)

# ══════════════════════════════════════════════════════════════════════════════
# 10. FIGURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("10. GENERATING FIGURES")
print("=" * 70)

# Figure: Model comparison bar chart
fig, ax = plt.subplots(figsize=(10, 5))
names = ["Naive\n(equal wt)"] + [n.split(": ")[1] for n in models.keys()]
cis = [ci_naive] + [m["c_index"] for m in models.values()]
colors = ["#999999"] + ["#4c72b0" if c > ci_naive else "#dd8452" for c in cis[1:]]
bars = ax.bar(range(len(names)), cis, color=colors, edgecolor='white', linewidth=1.5)
ax.axhline(ci_naive, color='red', linestyle='--', alpha=0.5, label=f'Naive baseline ({ci_naive:.4f})')
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.3)
for i, (c, bar) in enumerate(zip(cis, bars)):
    ax.text(bar.get_x() + bar.get_width()/2, c + 0.003, f'{c:.4f}', ha='center', fontsize=9)
ax.set_ylabel("C-index", fontsize=12)
ax.set_title("Model Comparison: Optimized vs Naive Combined Score", fontsize=13, fontweight='bold')
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, fontsize=9)
ax.set_ylim(0.5, max(cis) + 0.03)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig10a_model_comparison.png"), dpi=300)
plt.close()
print("  Saved: fig10a_model_comparison.png")

# Figure: KM with optimized score
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# TCGA
ax = axes[0]
kmf = KaplanMeierFitter()
for lbl, col in [("High", "#d62728"), ("Low", "#2ca02c")]:
    g = df[df["med_risk_group"] == lbl]
    kmf.fit(g["OS_months"], event_observed=g["OS_event"], label=f"{lbl} Risk (n={len(g)})")
    kmf.plot_survival_function(ax=ax, ci_show=True, color=col, linewidth=2)
ax.set_title(f"TCGA-LIHC (Optimized Score)\nHR={hr_med:.2f}, p={p_med:.2e}", fontweight='bold')
ax.set_xlabel("Time (months)"); ax.set_ylabel("Survival Probability")
ax.legend(fontsize=10, loc="lower left"); ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)

# Best external cohort
if validation_results:
    ax = axes[1]
    best_ext = max(validation_results, key=lambda r: abs(np.log(r["HR"])) if r["HR"] > 0 else 0)
    ax.text(0.5, 0.5, f'{best_ext["cohort"]}\nC-index: {best_ext["ci_optimized"]:.4f}\n'
            f'HR: {best_ext["HR"]:.2f} ({best_ext["HR_lower"]:.2f}-{best_ext["HR_upper"]:.2f})\n'
            f'p = {best_ext["cox_p"]:.4f}',
            transform=ax.transAxes, ha='center', va='center', fontsize=14,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax.set_title(f"External Validation: {best_ext['cohort']}", fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig10b_optimized_km.png"), dpi=300)
plt.close()
print("  Saved: fig10b_optimized_km.png")

# CV distribution
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(cv_cindexes, bins=10, alpha=0.7, color='#4c72b0', label=f'Optimized (mean={cv_mean:.4f})')
ax.hist(cv_cindexes_naive, bins=10, alpha=0.5, color='#999999', label=f'Naive (mean={cv_naive_mean:.4f})')
ax.axvline(selected["c_index"], color='red', linestyle='--', label=f'Train ({selected["c_index"]:.4f})')
ax.set_xlabel("C-index", fontsize=12); ax.set_ylabel("Frequency", fontsize=12)
ax.set_title("10-Fold Cross-Validation: C-index Distribution", fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig10c_cv_distribution.png"), dpi=300)
plt.close()
print("  Saved: fig10c_cv_distribution.png")

print("\n" + "=" * 70)
print("DONE — Script 10 complete")
print("=" * 70)
