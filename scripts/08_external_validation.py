"""
08_external_validation.py — Validate dual-axis classification in external cohorts

Cohorts (all already downloaded):
  1. GSE14520 (n=221, Chinese HBV-HCC)
  2. ICGC LIRI-JP (n=231, Japanese HCC)
  3. GSE54236 (n=80, European HCC)
"""
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.utils import concordance_index
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gzip
import json
import os
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
TABLES = os.path.join(BASE, "results", "tables")
FIGS = os.path.join(BASE, "results", "figures", "main")

with open(os.path.join(DATA, "paths.json")) as f:
    paths = json.load(f)

GEO = paths["geo_cohorts"]

# Load models
with open(paths["ros_model"]) as f:
    ros_model = json.load(f)
with open(paths["alt_model"]) as f:
    alt_model = json.load(f)

ros_genes = ros_model["genes"]
alt_genes = alt_model["genes"]

# NRF2 target genes for activity scoring
NRF2_TARGETS = ["NQO1", "HMOX1", "SLC7A11", "TXNRD1", "G6PD", "GSR",
                "GCLC", "GCLM", "FTH1", "FTL", "SQSTM1", "SRXN1",
                "AKR1C1", "AKR1B10", "ME1", "ABCC2"]


def compute_risk_score_cohort(expr_df, model_dict):
    """Compute risk score with cohort-level z-normalization."""
    genes = model_dict["genes"]
    available = [g for g in genes if g in expr_df.columns]
    if len(available) < len(genes) * 0.5:
        return None, available
    risk = np.zeros(len(expr_df))
    for gene in available:
        coef = genes[gene]
        vals = expr_df[gene].values.astype(float)
        m, s = np.nanmean(vals), np.nanstd(vals)
        if s > 0:
            risk += coef * ((vals - m) / s)
    return risk, available


def compute_nrf2_score(expr_df):
    """Compute NRF2 activity as mean z-score of available targets."""
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


def validate_cohort(name, expr_df, clinical_df, time_col, event_col):
    """Run full validation on one cohort."""
    print(f"\n{'─' * 60}")
    print(f"  COHORT: {name}")
    print(f"{'─' * 60}")

    # Compute risk scores
    ros_risk, ros_avail = compute_risk_score_cohort(expr_df, ros_model)
    alt_risk, alt_avail = compute_risk_score_cohort(expr_df, alt_model)

    print(f"  ROS genes available: {len(ros_avail)}/{len(ros_genes)} {ros_avail}")
    print(f"  Alt genes available: {len(alt_avail)}/{len(alt_genes)} {alt_avail}")

    if ros_risk is None:
        print(f"  SKIPPING: Too few ROS genes (<50%)")
        return None
    if alt_risk is None:
        print(f"  NOTE: Too few altitude genes — using ROS-only analysis")
        alt_risk = np.zeros(len(expr_df))  # neutral altitude score

    # Merge expression risk scores with clinical
    df = clinical_df.copy()
    df["ros_risk_score"] = ros_risk
    df["alt_risk_score"] = alt_risk
    df = df.dropna(subset=[time_col, event_col]).copy()
    df = df[df[time_col] > 0].copy()
    print(f"  Patients with survival: {len(df)}, Events: {int(df[event_col].sum())}")

    if len(df) < 30 or df[event_col].sum() < 10:
        print(f"  SKIPPING: Too few patients or events")
        return None

    # 2x2 classification
    ros_med = df["ros_risk_score"].median()
    alt_med = df["alt_risk_score"].median()
    conditions = [
        (df["ros_risk_score"] >= ros_med) & (df["alt_risk_score"] >= alt_med),
        (df["ros_risk_score"] >= ros_med) & (df["alt_risk_score"] < alt_med),
        (df["ros_risk_score"] < ros_med) & (df["alt_risk_score"] >= alt_med),
        (df["ros_risk_score"] < ros_med) & (df["alt_risk_score"] < alt_med),
    ]
    labels = ["A: Concordant High", "B: Ferroptosis-dom", "C: Hypoxia-dom", "D: Concordant Low"]
    df["dual_group"] = np.select(conditions, labels, default="D: Concordant Low")

    for lbl in labels:
        g = df[df["dual_group"] == lbl]
        print(f"    {lbl}: n={len(g)}, events={int(g[event_col].sum())}")

    # Combined score
    rs, als = df["ros_risk_score"].std(), df["alt_risk_score"].std()
    if rs > 0 and als > 0:
        df["combined_score"] = (df["ros_risk_score"] / rs + df["alt_risk_score"] / als) / 2
    else:
        df["combined_score"] = df["ros_risk_score"] + df["alt_risk_score"]

    # NRF2 activity
    nrf2 = compute_nrf2_score(expr_df)
    if nrf2 is not None:
        df["nrf2_activity"] = nrf2[:len(df)] if len(nrf2) >= len(df) else None

    # C-indices
    ci_ros = concordance_index(df[time_col], -df["ros_risk_score"], df[event_col])
    ci_alt = concordance_index(df[time_col], -df["alt_risk_score"], df[event_col])
    ci_comb = concordance_index(df[time_col], -df["combined_score"], df[event_col])

    print(f"  C-index ROS: {ci_ros:.4f}")
    print(f"  C-index Alt: {ci_alt:.4f}")
    print(f"  C-index Combined: {ci_comb:.4f}")

    # Log-rank test (A vs D)
    grp_a = df[df["dual_group"] == "A: Concordant High"]
    grp_d = df[df["dual_group"] == "D: Concordant Low"]
    if len(grp_a) >= 5 and len(grp_d) >= 5:
        lr = logrank_test(grp_a[time_col], grp_d[time_col],
                          event_observed_A=grp_a[event_col],
                          event_observed_B=grp_d[event_col])
        p_ad = lr.p_value
        print(f"  Log-rank A vs D: p={p_ad:.4f}")
    else:
        p_ad = np.nan

    # Overall log-rank
    try:
        mlr = multivariate_logrank_test(df[time_col], df["dual_group"], df[event_col])
        p_overall = mlr.p_value
    except Exception:
        p_overall = np.nan
    print(f"  Overall 4-group log-rank: p={p_overall:.4f}" if not np.isnan(p_overall) else "  Overall log-rank: NA")

    # HR for A vs D (univariate Cox)
    hr_ad, hr_ad_lo, hr_ad_hi, p_cox = np.nan, np.nan, np.nan, np.nan
    if len(grp_a) >= 5 and len(grp_d) >= 5:
        try:
            cox_df = pd.concat([grp_a, grp_d]).copy()
            cox_df["is_A"] = (cox_df["dual_group"] == "A: Concordant High").astype(int)
            cph = CoxPHFitter()
            cph.fit(cox_df[[time_col, event_col, "is_A"]], duration_col=time_col, event_col=event_col)
            hr_ad = float(cph.summary.loc["is_A", "exp(coef)"])
            hr_ad_lo = float(cph.summary.loc["is_A", "exp(coef) lower 95%"])
            hr_ad_hi = float(cph.summary.loc["is_A", "exp(coef) upper 95%"])
            p_cox = float(cph.summary.loc["is_A", "p"])
            print(f"  HR (A vs D): {hr_ad:.2f} ({hr_ad_lo:.2f}-{hr_ad_hi:.2f}), p={p_cox:.4f}")
        except Exception as e:
            print(f"  Cox failed: {e}")

    return {
        "cohort": name,
        "n": len(df),
        "events": int(df[event_col].sum()),
        "ros_genes_avail": len(ros_avail),
        "alt_genes_avail": len(alt_avail),
        "c_index_ros": ci_ros,
        "c_index_alt": ci_alt,
        "c_index_combined": ci_comb,
        "logrank_p_overall": p_overall,
        "logrank_p_A_vs_D": p_ad,
        "HR_A_vs_D": hr_ad,
        "HR_lower": hr_ad_lo,
        "HR_upper": hr_ad_hi,
        "cox_p_A_vs_D": p_cox,
        "df": df,
        "time_col": time_col,
        "event_col": event_col,
    }


# ══════════════════════════════════════════════════════════════════════════════
# COHORT 1: GSE14520
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("EXTERNAL VALIDATION")
print("=" * 70)

results = []

try:
    print("\n--- Loading GSE14520 ---")
    expr_path = os.path.join(GEO, "gse14520_gene_expr_cache.csv")
    expr14 = pd.read_csv(expr_path, index_col=0)
    # Determine orientation
    if expr14.shape[0] > expr14.shape[1]:
        # genes as rows
        expr14 = expr14.T
    print(f"  Expression: {expr14.shape[0]} samples x {expr14.shape[1]} genes")

    # Load clinical from supplement
    supp_path = os.path.join(GEO, "GSE14520_Extra_Supplement.txt.gz")
    if os.path.exists(supp_path):
        clin14 = pd.read_csv(supp_path, sep='\t', compression='gzip')
        print(f"  Clinical columns: {list(clin14.columns[:10])}")

        # Find time and event columns
        time_candidates = [c for c in clin14.columns if 'surviv' in c.lower() or 'time' in c.lower() or 'month' in c.lower()]
        event_candidates = [c for c in clin14.columns if 'status' in c.lower() or 'event' in c.lower() or 'dead' in c.lower()]
        print(f"  Time candidates: {time_candidates}")
        print(f"  Event candidates: {event_candidates}")

        # Try standard column names
        time_col = event_col = None
        for tc in ["Survival months", "OS_months", "Survival.months", "survival_months"]:
            if tc in clin14.columns:
                time_col = tc
                break
        for ec in ["Survival status", "OS_event", "Survival.status", "survival_status"]:
            if ec in clin14.columns:
                event_col = ec
                break

        if time_col and event_col:
            # Align expression and clinical
            # Find common sample IDs
            expr_samples = set(expr14.index)
            if "Affy_GSM" in clin14.columns:
                clin14 = clin14.set_index("Affy_GSM")
            elif clin14.index.dtype == object:
                pass  # already indexed

            common = sorted(expr_samples & set(clin14.index))
            if len(common) < 20:
                # Try matching differently
                print(f"  Direct match: {len(common)} samples. Trying positional alignment...")
                clin14_aligned = clin14.iloc[:len(expr14)].copy()
                clin14_aligned.index = expr14.index[:len(clin14_aligned)]
                clin14 = clin14_aligned
                common = sorted(set(expr14.index) & set(clin14.index))

            print(f"  Matched samples: {len(common)}")
            if len(common) >= 30:
                expr14_matched = expr14.loc[common]
                clin14_matched = clin14.loc[common].copy()
                clin14_matched[event_col] = pd.to_numeric(clin14_matched[event_col], errors='coerce')
                clin14_matched[time_col] = pd.to_numeric(clin14_matched[time_col], errors='coerce')
                # Convert status if needed (Dead=1, Alive=0)
                if clin14_matched[event_col].max() > 1:
                    clin14_matched[event_col] = (clin14_matched[event_col] > 0).astype(int)

                r = validate_cohort("GSE14520", expr14_matched, clin14_matched, time_col, event_col)
                if r:
                    results.append(r)
            else:
                print("  SKIPPING GSE14520: Not enough matched samples")
        else:
            print(f"  SKIPPING GSE14520: Could not identify time/event columns")
    else:
        print(f"  SKIPPING GSE14520: Supplement file not found")
except Exception as e:
    print(f"  ERROR in GSE14520: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# COHORT 2: ICGC LIRI-JP
# ══════════════════════════════════════════════════════════════════════════════
try:
    print("\n--- Loading ICGC LIRI-JP ---")
    expr_liri = pd.read_csv(os.path.join(GEO, "liri_jp_expression_merged.csv"), index_col=0)
    clin_liri = pd.read_csv(os.path.join(GEO, "liri_jp_donors.csv"))

    # This file has genes as rows (gene_id index), samples as columns
    # 12 rows x 233 cols = 11 genes x 232 samples
    if expr_liri.shape[0] < expr_liri.shape[1]:
        expr_liri = expr_liri.T  # now samples x genes
    print(f"  Expression: {expr_liri.shape[0]} samples x {expr_liri.shape[1]} genes")
    print(f"  Clinical: {len(clin_liri)} patients, columns: {list(clin_liri.columns[:8])}")

    # Find time/event columns
    time_col = event_col = donor_col = None
    for c in clin_liri.columns:
        cl = c.lower()
        if 'surviv' in cl and 'time' in cl:
            time_col = c
        elif 'surviv' in cl and 'status' in cl:
            event_col = c
        elif cl in ['os_months', 'os_time']:
            time_col = c
        elif cl in ['os_event', 'os_status']:
            event_col = c
        elif 'donor' in cl and 'id' in cl:
            donor_col = c

    # Fallback: check for donor_survival_time and donor_vital_status
    if time_col is None:
        for c in clin_liri.columns:
            if 'vital_status' in c.lower():
                event_col = c
            if 'survival_time' in c.lower():
                time_col = c

    # Use icgc_donor_id for matching (matches expression column headers)
    if "icgc_donor_id" in clin_liri.columns:
        donor_col = "icgc_donor_id"
    print(f"  Time col: {time_col}, Event col: {event_col}, Donor col: {donor_col}")

    if time_col and event_col:
        # Align using icgc_donor_id which matches expression headers
        if donor_col and donor_col in clin_liri.columns:
            clin_liri = clin_liri.drop_duplicates(subset=[donor_col])
            clin_liri = clin_liri.set_index(donor_col)

        # Try matching
        expr_samples = set(expr_liri.index)
        clin_samples = set(clin_liri.index)
        common = sorted(expr_samples & clin_samples)

        if len(common) < 20:
            # Try matching by stripping prefixes
            expr_ids = {s.split('-')[-1] if '-' in str(s) else s: s for s in expr_liri.index}
            clin_ids = {s.split('-')[-1] if '-' in str(s) else s: s for s in clin_liri.index}
            matched = set(expr_ids.keys()) & set(clin_ids.keys())
            if len(matched) > len(common):
                common_expr = [expr_ids[m] for m in sorted(matched)]
                common_clin = [clin_ids[m] for m in sorted(matched)]
                expr_liri = expr_liri.loc[common_expr]
                clin_liri_sub = clin_liri.loc[common_clin].copy()
                clin_liri_sub.index = expr_liri.index
                clin_liri = clin_liri_sub
                common = list(expr_liri.index)

        print(f"  Matched: {len(common)}")
        if len(common) >= 30:
            expr_matched = expr_liri.loc[common] if len(common) < len(expr_liri) else expr_liri
            clin_matched = clin_liri.loc[common].copy() if len(common) < len(clin_liri) else clin_liri.copy()

            # Convert event col
            clin_matched[time_col] = pd.to_numeric(clin_matched[time_col], errors='coerce')
            if clin_matched[event_col].dtype == object:
                clin_matched[event_col] = clin_matched[event_col].map(
                    {"alive": 0, "dead": 1, "deceased": 1, "Alive": 0, "Dead": 1}
                ).fillna(0).astype(int)
            else:
                clin_matched[event_col] = pd.to_numeric(clin_matched[event_col], errors='coerce')

            # Convert days to months if values seem like days
            if clin_matched[time_col].median() > 365:
                clin_matched[time_col] = clin_matched[time_col] / 30.44
                print(f"  Converted days to months (median was > 365)")

            r = validate_cohort("ICGC_LIRI-JP", expr_matched, clin_matched, time_col, event_col)
            if r:
                results.append(r)
        else:
            print("  SKIPPING ICGC: Not enough matched samples")
    else:
        print(f"  SKIPPING ICGC: Could not find time/event columns")
except Exception as e:
    print(f"  ERROR in ICGC: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# COHORT 3: GSE54236
# ══════════════════════════════════════════════════════════════════════════════
try:
    print("\n--- Loading GSE54236 ---")
    expr54 = pd.read_csv(os.path.join(GEO, "gse54236_expr.csv"), index_col=0)
    clin54 = pd.read_csv(os.path.join(GEO, "gse54236_clinical.csv"), index_col=0)

    if expr54.shape[0] > expr54.shape[1]:
        expr54 = expr54.T
    print(f"  Expression: {expr54.shape[0]} samples x {expr54.shape[1]} genes")
    print(f"  Clinical: {len(clin54)} patients, columns: {list(clin54.columns[:8])}")

    # Find time/event
    time_col = event_col = None
    for c in clin54.columns:
        cl = c.lower()
        if any(x in cl for x in ['surviv', 'time', 'month', 'os_month']):
            if 'status' not in cl and 'event' not in cl:
                time_col = c
        if any(x in cl for x in ['status', 'event', 'dead', 'vital']):
            event_col = c

    # Try common names
    if time_col is None:
        for tc in ["OS_months", "survival_months", "time", "os_time"]:
            if tc in clin54.columns:
                time_col = tc
                break
    if event_col is None:
        for ec in ["OS_event", "status", "vital_status", "event"]:
            if ec in clin54.columns:
                event_col = ec
                break

    # GSE54236 has no event column — all patients treated as events (per altitude paper)
    if event_col is None and time_col is not None:
        print(f"  No event column found — treating all patients as events (no censoring)")
        event_col = "OS_event_imputed"
        clin54[event_col] = 1

    print(f"  Time col: {time_col}, Event col: {event_col}")

    if time_col and event_col:
        common = sorted(set(expr54.index) & set(clin54.index))
        if len(common) < 20:
            # Positional alignment
            min_len = min(len(expr54), len(clin54))
            expr54 = expr54.iloc[:min_len]
            clin54_aligned = clin54.iloc[:min_len].copy()
            clin54_aligned.index = expr54.index
            clin54 = clin54_aligned
            common = list(expr54.index)

        print(f"  Matched: {len(common)}")
        if len(common) >= 20:
            expr_m = expr54.loc[common]
            clin_m = clin54.loc[common].copy()
            clin_m[time_col] = pd.to_numeric(clin_m[time_col], errors='coerce')
            clin_m[event_col] = pd.to_numeric(clin_m[event_col], errors='coerce')
            if clin_m[event_col].max() > 1:
                clin_m[event_col] = (clin_m[event_col] > 0).astype(int)
            if clin_m[time_col].median() > 365:
                clin_m[time_col] = clin_m[time_col] / 30.44

            r = validate_cohort("GSE54236", expr_m, clin_m, time_col, event_col)
            if r:
                results.append(r)
    else:
        print("  SKIPPING GSE54236: Could not find time/event columns")
except Exception as e:
    print(f"  ERROR in GSE54236: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)

if results:
    summary_rows = []
    for r in results:
        row = {k: v for k, v in r.items() if k != "df" and k != "time_col" and k != "event_col"}
        summary_rows.append(row)
        print(f"\n  {r['cohort']}:")
        print(f"    n={r['n']}, events={r['events']}")
        print(f"    C-index: ROS={r['c_index_ros']:.3f}, Alt={r['c_index_alt']:.3f}, Combined={r['c_index_combined']:.3f}")
        print(f"    HR (A vs D): {r['HR_A_vs_D']:.2f} ({r['HR_lower']:.2f}-{r['HR_upper']:.2f}), p={r['cox_p_A_vs_D']:.4f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(TABLES, "external_validation_summary.csv"), index=False)
    print(f"\n  Saved: external_validation_summary.csv")

    # Fisher combined p-value
    valid_p = [r["logrank_p_A_vs_D"] for r in results if not np.isnan(r["logrank_p_A_vs_D"])]
    if len(valid_p) >= 2:
        chi2_combined = -2 * sum(np.log(p) for p in valid_p)
        df_combined = 2 * len(valid_p)
        fisher_p = 1 - stats.chi2.cdf(chi2_combined, df_combined)
        print(f"\n  Fisher combined p-value (A vs D): {fisher_p:.2e} (k={len(valid_p)} cohorts)")

# ══════════════════════════════════════════════════════════════════════════════
# FOREST PLOT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GENERATING FIGURES")
print("=" * 70)

# Add TCGA result
tcga_master = pd.read_csv(os.path.join(DATA, "tcga_convergence_master.csv"))
tcga_master = tcga_master.dropna(subset=["OS_months", "OS_event"])
grp_a_tcga = tcga_master[tcga_master["dual_group"] == "A: Concordant High"]
grp_d_tcga = tcga_master[tcga_master["dual_group"] == "D: Concordant Low"]
try:
    cox_tcga = pd.concat([grp_a_tcga, grp_d_tcga]).copy()
    cox_tcga["is_A"] = (cox_tcga["dual_group"] == "A: Concordant High").astype(int)
    cph_tcga = CoxPHFitter()
    cph_tcga.fit(cox_tcga[["OS_months", "OS_event", "is_A"]], duration_col="OS_months", event_col="OS_event")
    tcga_hr = float(cph_tcga.summary.loc["is_A", "exp(coef)"])
    tcga_lo = float(cph_tcga.summary.loc["is_A", "exp(coef) lower 95%"])
    tcga_hi = float(cph_tcga.summary.loc["is_A", "exp(coef) upper 95%"])
    tcga_p = float(cph_tcga.summary.loc["is_A", "p"])
except Exception:
    tcga_hr, tcga_lo, tcga_hi, tcga_p = np.nan, np.nan, np.nan, np.nan

# Build forest plot data
forest_data = [{"cohort": "TCGA-LIHC (training)", "n": len(tcga_master),
                "HR": tcga_hr, "HR_lower": tcga_lo, "HR_upper": tcga_hi, "p": tcga_p}]
for r in results:
    forest_data.append({"cohort": r["cohort"], "n": r["n"],
                        "HR": r["HR_A_vs_D"], "HR_lower": r["HR_lower"],
                        "HR_upper": r["HR_upper"], "p": r["cox_p_A_vs_D"]})

if len(forest_data) > 0:
    fig, ax = plt.subplots(figsize=(10, max(4, len(forest_data) * 1.2 + 1)))
    y_positions = list(range(len(forest_data)))[::-1]

    for i, d in enumerate(forest_data):
        y = y_positions[i]
        color = '#1f77b4' if i == 0 else '#d62728'
        if not np.isnan(d["HR"]):
            ax.plot(d["HR"], y, 'o', color=color, markersize=8, zorder=5)
            ax.plot([d["HR_lower"], d["HR_upper"]], [y, y], '-', color=color, linewidth=2, zorder=4)
            label = f'{d["cohort"]} (n={d["n"]}): HR={d["HR"]:.2f} ({d["HR_lower"]:.2f}-{d["HR_upper"]:.2f}), p={d["p"]:.4f}'
        else:
            label = f'{d["cohort"]} (n={d["n"]}): HR=NA'
        ax.text(0.02, y, label, transform=ax.get_yaxis_transform(), va='center', fontsize=10)

    ax.axvline(1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel("Hazard Ratio (Concordant High vs Concordant Low)", fontsize=12)
    ax.set_yticks([])
    ax.set_title("Forest Plot: HR for Group A vs D Across Cohorts", fontsize=13, fontweight='bold')
    ax.set_xlim(0, max(d["HR_upper"] for d in forest_data if not np.isnan(d.get("HR_upper", np.nan))) + 1)
    ax.grid(True, alpha=0.2, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "fig8a_forest_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: fig8a_forest_plot.png")

# KM curves for each validation cohort
n_cohorts = len(results)
if n_cohorts > 0:
    fig, axes = plt.subplots(1, n_cohorts, figsize=(6 * n_cohorts, 5))
    if n_cohorts == 1:
        axes = [axes]

    colors = {
        "A: Concordant High": "#d62728",
        "B: Ferroptosis-dom": "#ff7f0e",
        "C: Hypoxia-dom": "#9467bd",
        "D: Concordant Low": "#2ca02c",
    }

    for idx, r in enumerate(results):
        ax = axes[idx]
        df = r["df"]
        tc, ec = r["time_col"], r["event_col"]
        kmf = KaplanMeierFitter()

        for lbl in ["A: Concordant High", "D: Concordant Low"]:
            grp = df[df["dual_group"] == lbl]
            if len(grp) >= 3:
                kmf.fit(grp[tc], event_observed=grp[ec], label=lbl)
                kmf.plot_survival_function(ax=ax, ci_show=True, color=colors.get(lbl, 'gray'), linewidth=2)

        p_val = r["logrank_p_A_vs_D"]
        ax.set_title(f'{r["cohort"]} (n={r["n"]})\nA vs D: p={p_val:.4f}', fontsize=11, fontweight='bold')
        ax.set_xlabel("Time (months)", fontsize=10)
        ax.set_ylabel("Survival Probability", fontsize=10)
        ax.legend(fontsize=9, loc="lower left")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "fig8b_validation_km.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: fig8b_validation_km.png")

# C-index comparison across cohorts
if results:
    fig, ax = plt.subplots(figsize=(10, 5))
    cohort_names = ["TCGA-LIHC"] + [r["cohort"] for r in results]
    ci_ros_vals = [0.6999] + [r["c_index_ros"] for r in results]
    ci_alt_vals = [0.6724] + [r["c_index_alt"] for r in results]
    ci_comb_vals = [0.7170] + [r["c_index_combined"] for r in results]

    x = np.arange(len(cohort_names))
    w = 0.25
    ax.bar(x - w, ci_ros_vals, w, label="ROS Signature", color="#d62728", alpha=0.8)
    ax.bar(x, ci_alt_vals, w, label="Altitude Signature", color="#1f77b4", alpha=0.8)
    ax.bar(x + w, ci_comb_vals, w, label="Combined Dual-Axis", color="#2ca02c", alpha=0.8)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cohort_names, rotation=15, ha='right')
    ax.set_ylabel("C-index", fontsize=12)
    ax.set_title("Concordance Index Across Cohorts", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0.4, 0.8)
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "fig8c_cindex_cohorts.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: fig8c_cindex_cohorts.png")

print("\n" + "=" * 70)
print("DONE — Script 08 complete")
print("=" * 70)
