"""
09_additional_validation.py — Add more validation cohorts for stronger metrics

New cohorts:
  1. GSE76427 (n=115, Singapore, Illumina HT-12, OS + RFS confirmed)
  2. ICGC LIRI-JP full expression via UCSC Xena (n~230, RNA-seq)
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
import requests
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
TABLES = os.path.join(BASE, "results", "tables")
FIGS = os.path.join(BASE, "results", "figures", "main")

with open(os.path.join(DATA, "paths.json")) as f:
    paths = json.load(f)
GEO = paths["geo_cohorts"]

with open(paths["ros_model"]) as f:
    ros_model = json.load(f)
with open(paths["alt_model"]) as f:
    alt_model = json.load(f)

NRF2_TARGETS = ["NQO1", "HMOX1", "SLC7A11", "TXNRD1", "G6PD", "GSR",
                "GCLC", "GCLM", "FTH1", "FTL", "SQSTM1", "SRXN1",
                "AKR1C1", "AKR1B10", "ME1", "ABCC2"]


def compute_risk(expr_df, model_dict):
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


def validate_and_report(name, df, time_col, event_col):
    """Run dual-axis validation on a prepared dataframe with risk scores."""
    print(f"\n  {name}: n={len(df)}, events={int(df[event_col].sum())}")

    ros_med = df["ros_risk"].median()
    alt_med = df["alt_risk"].median()
    conditions = [
        (df["ros_risk"] >= ros_med) & (df["alt_risk"] >= alt_med),
        (df["ros_risk"] >= ros_med) & (df["alt_risk"] < alt_med),
        (df["ros_risk"] < ros_med) & (df["alt_risk"] >= alt_med),
        (df["ros_risk"] < ros_med) & (df["alt_risk"] < alt_med),
    ]
    labels = ["A: Concordant High", "B: Ferroptosis-dom", "C: Hypoxia-dom", "D: Concordant Low"]
    df["dual_group"] = np.select(conditions, labels, default="D: Concordant Low")

    for lbl in labels:
        g = df[df["dual_group"] == lbl]
        print(f"    {lbl}: n={len(g)}, events={int(g[event_col].sum())}")

    # Combined score
    rs, als = df["ros_risk"].std(), df["alt_risk"].std()
    if rs > 0 and als > 0:
        df["combined"] = (df["ros_risk"] / rs + df["alt_risk"] / als) / 2
    else:
        df["combined"] = df["ros_risk"]

    # C-indices
    ci_ros = concordance_index(df[time_col], -df["ros_risk"], df[event_col])
    ci_alt = concordance_index(df[time_col], -df["alt_risk"], df[event_col])
    ci_comb = concordance_index(df[time_col], -df["combined"], df[event_col])
    print(f"  C-index: ROS={ci_ros:.4f}, Alt={ci_alt:.4f}, Combined={ci_comb:.4f}")

    # Log-rank A vs D
    grp_a = df[df["dual_group"] == "A: Concordant High"]
    grp_d = df[df["dual_group"] == "D: Concordant Low"]
    hr, hr_lo, hr_hi, p_cox, p_lr = np.nan, np.nan, np.nan, np.nan, np.nan

    if len(grp_a) >= 5 and len(grp_d) >= 5 and grp_a[event_col].sum() >= 2 and grp_d[event_col].sum() >= 1:
        lr = logrank_test(grp_a[time_col], grp_d[time_col],
                          event_observed_A=grp_a[event_col], event_observed_B=grp_d[event_col])
        p_lr = lr.p_value
        try:
            cdf = pd.concat([grp_a, grp_d]).copy()
            cdf["is_A"] = (cdf["dual_group"] == "A: Concordant High").astype(int)
            cph = CoxPHFitter()
            cph.fit(cdf[[time_col, event_col, "is_A"]], duration_col=time_col, event_col=event_col)
            hr = float(cph.summary.loc["is_A", "exp(coef)"])
            hr_lo = float(cph.summary.loc["is_A", "exp(coef) lower 95%"])
            hr_hi = float(cph.summary.loc["is_A", "exp(coef) upper 95%"])
            p_cox = float(cph.summary.loc["is_A", "p"])
        except Exception as e:
            print(f"    Cox failed: {e}")
        print(f"  HR (A vs D): {hr:.2f} ({hr_lo:.2f}-{hr_hi:.2f}), p={p_cox:.4f}")
        print(f"  Log-rank A vs D: p={p_lr:.4f}")

    # Overall log-rank
    try:
        mlr = multivariate_logrank_test(df[time_col], df["dual_group"], df[event_col])
        p_overall = mlr.p_value
        print(f"  Overall 4-group log-rank: p={p_overall:.4e}")
    except:
        p_overall = np.nan

    return {
        "cohort": name, "n": len(df), "events": int(df[event_col].sum()),
        "c_index_ros": ci_ros, "c_index_alt": ci_alt, "c_index_combined": ci_comb,
        "logrank_p_overall": p_overall, "logrank_p_A_vs_D": p_lr,
        "HR_A_vs_D": hr, "HR_lower": hr_lo, "HR_upper": hr_hi, "cox_p": p_cox,
        "df": df, "time_col": time_col, "event_col": event_col,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 1. GSE76427 — Parse from series matrix
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("1. GSE76427 (Singapore, n=115, Illumina HT-12)")
print("=" * 70)

results = []

try:
    matrix_path = os.path.join(GEO, "GSE76427_series_matrix.txt.gz")
    annot_path = os.path.join(GEO, "GPL10558_annot.csv")

    # Parse series matrix
    with gzip.open(matrix_path, 'rt', errors='replace') as f:
        lines = f.readlines()

    sample_ids = []
    characteristics = {}
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
                # Handle duplicate keys
                orig = key
                idx = 1
                while key in characteristics:
                    key = f"{orig}_{idx}"
                    idx += 1
                characteristics[key] = vals
        elif line.startswith('!series_matrix_table_begin'):
            in_data = True
            continue
        elif line.startswith('!series_matrix_table_end'):
            in_data = False
        elif in_data and line:
            data_lines.append(line)

    print(f"  Samples: {len(sample_ids)}")
    print(f"  Clinical fields: {list(characteristics.keys())}")

    # Build clinical dataframe
    clin76 = pd.DataFrame(characteristics, index=sample_ids)
    # Extract OS data
    os_event_col = [c for c in clin76.columns if 'event_os' in c.lower()]
    os_time_col = [c for c in clin76.columns if 'duryears_os' in c.lower()]
    print(f"  OS event col: {os_event_col}, OS time col: {os_time_col}")

    if os_event_col and os_time_col:
        clin76["OS_event"] = pd.to_numeric(clin76[os_event_col[0]], errors='coerce')
        clin76["OS_months"] = pd.to_numeric(clin76[os_time_col[0]], errors='coerce') * 12  # years to months
        print(f"  OS data: {clin76['OS_event'].notna().sum()} patients, {int(clin76['OS_event'].sum())} events")

    # Parse expression data
    header = data_lines[0].split('\t')
    probe_ids = []
    expr_data = []
    for line in data_lines[1:]:
        parts = line.split('\t')
        probe_ids.append(parts[0].strip('"'))
        expr_data.append([float(x) if x.strip('"') not in ('', 'null', 'NA') else np.nan
                          for x in parts[1:]])

    expr76_probes = pd.DataFrame(expr_data, index=probe_ids,
                                  columns=[s.strip('"') for s in header[1:]])
    print(f"  Expression (probes): {expr76_probes.shape}")

    # Map probes to gene symbols using annotation
    if os.path.exists(annot_path):
        annot = pd.read_csv(annot_path)
        # Find the gene symbol column
        symbol_col = None
        for c in annot.columns:
            if c == "Gene symbol":
                symbol_col = c
                break
        if symbol_col is None:
            for c in annot.columns:
                if 'symbol' in c.lower():
                    symbol_col = c
                    break
        id_col = None
        for c in annot.columns:
            if 'id' in c.lower() or 'probe' in c.lower():
                id_col = c
                break
        print(f"  Annotation: {len(annot)} probes, symbol_col={symbol_col}, id_col={id_col}")

        if symbol_col and id_col:
            annot_map = annot.set_index(id_col)[symbol_col].dropna().to_dict()
            # Map probe IDs to gene symbols
            expr76_probes.index = expr76_probes.index.astype(str)
            mapped_genes = [annot_map.get(pid, None) for pid in expr76_probes.index]
            expr76_probes["gene"] = mapped_genes
            expr76_probes = expr76_probes.dropna(subset=["gene"])
            expr76_probes = expr76_probes[expr76_probes["gene"] != ""]
            # Average duplicates
            expr76 = expr76_probes.groupby("gene").mean()
            expr76 = expr76.T  # samples x genes
            print(f"  Expression (genes): {expr76.shape[0]} samples x {expr76.shape[1]} genes")

            # Check signature gene coverage
            ros_avail = [g for g in ros_model["genes"] if g in expr76.columns]
            alt_avail = [g for g in alt_model["genes"] if g in expr76.columns]
            print(f"  ROS genes: {len(ros_avail)}/{len(ros_model['genes'])}: {ros_avail}")
            print(f"  Alt genes: {len(alt_avail)}/{len(alt_model['genes'])}: {alt_avail}")

            # Compute risk scores
            ros_risk, _ = compute_risk(expr76, ros_model)
            alt_risk, _ = compute_risk(expr76, alt_model)

            if ros_risk is not None and alt_risk is not None:
                # Align clinical and expression
                common = sorted(set(expr76.index) & set(clin76.index))
                print(f"  Matched samples: {len(common)}")

                df76 = clin76.loc[common, ["OS_months", "OS_event"]].copy()
                df76["ros_risk"] = ros_risk[:len(common)] if len(expr76) == len(common) else \
                    pd.Series(ros_risk, index=expr76.index).loc[common].values
                df76["alt_risk"] = alt_risk[:len(common)] if len(expr76) == len(common) else \
                    pd.Series(alt_risk, index=expr76.index).loc[common].values
                df76 = df76.dropna(subset=["OS_months", "OS_event"])
                df76 = df76[df76["OS_months"] > 0]

                r = validate_and_report("GSE76427", df76, "OS_months", "OS_event")
                if r:
                    results.append(r)
            else:
                print("  SKIPPING: Not enough genes for risk scores")
    else:
        print(f"  Annotation file not found: {annot_path}")

except Exception as e:
    print(f"  ERROR: {e}")
    import traceback; traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════════════
# 2. ICGC LIRI-JP via UCSC Xena
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. ICGC LIRI-JP (via UCSC Xena download)")
print("=" * 70)

try:
    xena_dir = os.path.join(DATA, "xena_icgc")
    os.makedirs(xena_dir, exist_ok=True)

    # Try multiple Xena URLs (the hub sometimes changes)
    expr_urls = [
        "https://icgc.xenahubs.net/download/LIRI-JP/exp_seq.LIRI-JP.tsv.gz",
        "https://icgcxena.xenahubs.net/download/LIRI-JP/exp_seq.LIRI-JP.tsv.gz",
        "https://gdc.xenahubs.net/download/LIRI-JP/exp_seq.LIRI-JP.tsv.gz",
    ]
    clin_urls = [
        "https://icgc.xenahubs.net/download/LIRI-JP/donor.LIRI-JP.tsv.gz",
        "https://icgcxena.xenahubs.net/download/LIRI-JP/donor.LIRI-JP.tsv.gz",
    ]
    expr_url = expr_urls[0]
    clin_url = clin_urls[0]
    expr_file = os.path.join(xena_dir, "exp_seq.LIRI-JP.tsv.gz")
    clin_file = os.path.join(xena_dir, "donor.LIRI-JP.tsv.gz")

    def try_download(urls, fpath, desc):
        if os.path.exists(fpath):
            print(f"  Cached: {desc}")
            return True
        for url in urls:
            try:
                print(f"  Trying {url}...")
                resp = requests.get(url, timeout=300)
                resp.raise_for_status()
                with open(fpath, 'wb') as f:
                    f.write(resp.content)
                print(f"  Downloaded: {os.path.getsize(fpath)} bytes")
                return True
            except Exception as e:
                print(f"    Failed: {e}")
        return False

    if not try_download(expr_urls, expr_file, "LIRI expression"):
        print("  WARNING: Could not download LIRI-JP expression — skipping this cohort")
        print("  The ICGC Xena hub may be down. Try manually from https://xenabrowser.net")
        raise RuntimeError("LIRI-JP download failed")
    if not try_download(clin_urls, clin_file, "LIRI clinical"):
        raise RuntimeError("LIRI-JP clinical download failed")

    # Load expression
    expr_liri = pd.read_csv(expr_file, sep='\t', index_col=0, compression='gzip')
    print(f"  Expression raw: {expr_liri.shape}")
    if expr_liri.shape[0] > expr_liri.shape[1]:
        expr_liri = expr_liri.T  # make samples x genes
    print(f"  Expression: {expr_liri.shape[0]} samples x {expr_liri.shape[1]} genes")

    # Load clinical
    clin_liri = pd.read_csv(clin_file, sep='\t', compression='gzip')
    print(f"  Clinical: {len(clin_liri)} rows, columns: {list(clin_liri.columns[:10])}")

    # Find ID, time, event columns
    id_col = clin_liri.columns[0]  # usually icgc_donor_id or sampleID
    time_col = event_col = None
    for c in clin_liri.columns:
        cl = c.lower()
        if 'survival_time' in cl or 'os_time' in cl:
            time_col = c
        elif 'vital_status' in cl or 'os_event' in cl or 'os_status' in cl:
            event_col = c

    print(f"  ID: {id_col}, Time: {time_col}, Event: {event_col}")

    if time_col and event_col:
        clin_liri = clin_liri.set_index(id_col)
        clin_liri[time_col] = pd.to_numeric(clin_liri[time_col], errors='coerce')

        # Convert vital_status to numeric
        if clin_liri[event_col].dtype == object:
            clin_liri["OS_event"] = clin_liri[event_col].map(
                {"alive": 0, "dead": 1, "deceased": 1}).fillna(0).astype(int)
        else:
            clin_liri["OS_event"] = pd.to_numeric(clin_liri[event_col], errors='coerce')

        # Convert to months if needed
        if clin_liri[time_col].median() > 365:
            clin_liri["OS_months"] = clin_liri[time_col] / 30.44
        else:
            clin_liri["OS_months"] = clin_liri[time_col]

        # Match expression and clinical
        common = sorted(set(expr_liri.index) & set(clin_liri.index))
        print(f"  Matched: {len(common)}")

        if len(common) >= 30:
            expr_m = expr_liri.loc[common]
            clin_m = clin_liri.loc[common]

            ros_avail = [g for g in ros_model["genes"] if g in expr_m.columns]
            alt_avail = [g for g in alt_model["genes"] if g in expr_m.columns]
            print(f"  ROS genes: {len(ros_avail)}/{len(ros_model['genes'])}")
            print(f"  Alt genes: {len(alt_avail)}/{len(alt_model['genes'])}")

            ros_risk, _ = compute_risk(expr_m, ros_model)
            alt_risk, _ = compute_risk(expr_m, alt_model)

            if ros_risk is not None:
                df_liri = clin_m[["OS_months", "OS_event"]].copy()
                df_liri["ros_risk"] = ros_risk
                df_liri["alt_risk"] = alt_risk if alt_risk is not None else np.zeros(len(df_liri))
                df_liri = df_liri.dropna(subset=["OS_months", "OS_event"])
                df_liri = df_liri[df_liri["OS_months"] > 0]

                r = validate_and_report("ICGC_LIRI-JP_Xena", df_liri, "OS_months", "OS_event")
                if r:
                    results.append(r)
        else:
            print("  Not enough matched samples")
    else:
        print("  Could not find time/event columns")

except Exception as e:
    print(f"  ERROR: {e}")
    import traceback; traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════════════
# 3. COMBINED RESULTS WITH PREVIOUS VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. COMBINED VALIDATION SUMMARY")
print("=" * 70)

# Load previous results
prev_path = os.path.join(TABLES, "external_validation_summary.csv")
if os.path.exists(prev_path):
    prev_df = pd.read_csv(prev_path)
    print(f"  Previous validations: {list(prev_df['cohort'])}")
else:
    prev_df = pd.DataFrame()

# Combine
new_rows = []
for r in results:
    new_rows.append({k: v for k, v in r.items() if k not in ("df", "time_col", "event_col")})
new_df = pd.DataFrame(new_rows) if new_rows else pd.DataFrame()

if len(prev_df) > 0 and len(new_df) > 0:
    existing = set(prev_df["cohort"])
    new_only = new_df[~new_df["cohort"].isin(existing)]
    combined = pd.concat([prev_df, new_only], ignore_index=True)
elif len(prev_df) > 0:
    combined = prev_df
elif len(new_df) > 0:
    combined = new_df
else:
    combined = pd.DataFrame()

combined.to_csv(os.path.join(TABLES, "external_validation_summary.csv"), index=False)
print(f"\n  Total validated cohorts: {len(combined)}")
for _, row in combined.iterrows():
    hr_str = f"HR={row['HR_A_vs_D']:.2f}" if not np.isnan(row.get('HR_A_vs_D', np.nan)) else "HR=NA"
    p_str = f"p={row.get('cox_p', row.get('cox_p_A_vs_D', np.nan)):.4f}" if not np.isnan(row.get('cox_p', row.get('cox_p_A_vs_D', np.nan))) else "p=NA"
    print(f"    {row['cohort']}: n={row['n']}, C-comb={row['c_index_combined']:.3f}, {hr_str}, {p_str}")

# Fisher combined p
all_p = []
for _, row in combined.iterrows():
    p = row.get("logrank_p_A_vs_D", row.get("logrank_p_A_vs_D", np.nan))
    if not np.isnan(p) and p > 0:
        all_p.append(p)
if len(all_p) >= 2:
    chi2 = -2 * sum(np.log(p) for p in all_p)
    fisher_p = 1 - stats.chi2.cdf(chi2, 2 * len(all_p))
    print(f"\n  Fisher combined p (A vs D, {len(all_p)} cohorts): {fisher_p:.2e}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. UPDATED FIGURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. GENERATING UPDATED FIGURES")
print("=" * 70)

# Forest plot with all cohorts
all_forest = []

# Add TCGA
tcga = pd.read_csv(os.path.join(DATA, "tcga_convergence_master.csv")).dropna(subset=["OS_months", "OS_event"])
ga = tcga[tcga["dual_group"] == "A: Concordant High"]
gd = tcga[tcga["dual_group"] == "D: Concordant Low"]
try:
    cd = pd.concat([ga, gd]).copy()
    cd["is_A"] = (cd["dual_group"] == "A: Concordant High").astype(int)
    cph = CoxPHFitter()
    cph.fit(cd[["OS_months", "OS_event", "is_A"]], duration_col="OS_months", event_col="OS_event")
    all_forest.append({"cohort": "TCGA-LIHC (training)", "n": len(tcga),
                        "HR": float(cph.summary.loc["is_A", "exp(coef)"]),
                        "lo": float(cph.summary.loc["is_A", "exp(coef) lower 95%"]),
                        "hi": float(cph.summary.loc["is_A", "exp(coef) upper 95%"]),
                        "p": float(cph.summary.loc["is_A", "p"])})
except:
    pass

# Add all validation cohorts
for _, row in combined.iterrows():
    all_forest.append({"cohort": row["cohort"], "n": int(row["n"]),
                        "HR": row["HR_A_vs_D"], "lo": row["HR_lower"], "hi": row["HR_upper"],
                        "p": row.get("cox_p", row.get("cox_p_A_vs_D", np.nan))})

fig, ax = plt.subplots(figsize=(10, max(4, len(all_forest) * 1.5)))
y_pos = list(range(len(all_forest)))[::-1]
for i, d in enumerate(all_forest):
    y = y_pos[i]
    color = '#1f77b4' if i == 0 else '#d62728'
    marker = 's' if i == 0 else 'o'
    if not np.isnan(d["HR"]):
        ax.plot(d["HR"], y, marker, color=color, markersize=10, zorder=5)
        ax.plot([d["lo"], d["hi"]], [y, y], '-', color=color, linewidth=2.5, zorder=4)
        sig = "*" if d["p"] < 0.05 else ""
        ax.text(max(d["hi"] + 0.3, 1.5), y,
                f'n={d["n"]}, HR={d["HR"]:.2f} ({d["lo"]:.2f}-{d["hi"]:.2f}), p={d["p"]:.4f}{sig}',
                va='center', fontsize=10)
    ax.text(-0.5, y, d["cohort"], va='center', ha='right', fontsize=11, fontweight='bold' if i == 0 else 'normal')

ax.axvline(1, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
ax.set_xlabel("Hazard Ratio (Group A: Concordant High vs Group D: Concordant Low)", fontsize=12)
ax.set_yticks([])
ax.set_title("Forest Plot: Dual-Axis Classification Across Cohorts", fontsize=14, fontweight='bold')
ax.set_xlim(-0.5, max(d.get("hi", 2) for d in all_forest if not np.isnan(d.get("hi", np.nan))) + 3)
ax.grid(True, alpha=0.2, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig8a_forest_plot_updated.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig8a_forest_plot_updated.png")

# Multi-panel KM
valid_results = [r for r in results if r is not None]
if valid_results:
    n_panels = len(valid_results)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    colors = {"A: Concordant High": "#d62728", "D: Concordant Low": "#2ca02c"}

    for idx, r in enumerate(valid_results):
        ax = axes[idx]
        df = r["df"]
        tc, ec = r["time_col"], r["event_col"]
        kmf = KaplanMeierFitter()
        for lbl, col in colors.items():
            grp = df[df["dual_group"] == lbl]
            if len(grp) >= 3:
                kmf.fit(grp[tc], event_observed=grp[ec], label=lbl)
                kmf.plot_survival_function(ax=ax, ci_show=True, color=col, linewidth=2)
        p = r["logrank_p_A_vs_D"]
        ax.set_title(f'{r["cohort"]} (n={r["n"]})\nA vs D: p={p:.4f}' if not np.isnan(p) else r["cohort"],
                     fontsize=11, fontweight='bold')
        ax.set_xlabel("Time (months)", fontsize=10)
        ax.set_ylabel("Survival Probability", fontsize=10)
        ax.legend(fontsize=9, loc="lower left")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "fig8b_validation_km_updated.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: fig8b_validation_km_updated.png")

print("\n" + "=" * 70)
print("DONE — Script 09 complete")
print("=" * 70)
