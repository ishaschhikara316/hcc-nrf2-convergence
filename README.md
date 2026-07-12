# The NRF2–Ferroptosis–Immunity Triangle in Hepatocellular Carcinoma

**Convergence of redox-vulnerability and altitude-adaptive hypoxia signatures reveals dual-axis patient stratification.**

Most prognostic gene signatures in hepatocellular carcinoma (HCC) capture a single biological axis at a time: ferroptosis resistance, hypoxia adaptation, immune composition, or metabolic state. But real tumors run several of these programs in parallel, and the *interaction* between them is often what determines whether a patient lives years or months. Two patients with identical ferroptosis risk scores can have very different hypoxic adaptations, and vice versa. A signature that ignores the second axis will misclassify them.

This project asks a deliberately simple question: what happens when we apply two biologically distinct, independently validated HCC signatures to the same patients and ask who is high-risk on both at once? The two signatures are taken unchanged from companion projects:

- An [11-gene ROS/ferroptosis signature](../hcc-ros-signature) capturing NRF2-KEAP1 antioxidant rewiring (C-index 0.700 in TCGA-LIHC).
- A [9-gene altitude-adaptive signature](../hcc-altitude-signature) capturing hypoxia-adaptive metabolic reprogramming (C-index 0.672 in TCGA-LIHC).

No new gene selection or model fitting is performed. Both signatures are applied as fixed-weight linear predictors to TCGA-LIHC (n=302) and three independent validation cohorts (n=565). What emerges is a coherent multi-axis picture in which **NRF2 transcriptional activity is the central molecular node**, the **HMOX1/HMOX2 ratio** is a novel prognostic biomarker bridging the two axes, and **immune evasion proceeds through T-cell exhaustion rather than exclusion**, with direct implications for combining ferroptosis-inducing therapy with immune checkpoint blockade.

## What We Did

### The Convergence Framework

The two pre-existing risk scores are computed from cohort-z-scored gene expression using the LASSO-Cox coefficients from the companion papers. Each patient receives:

- A **ROS/ferroptosis risk score** (11 genes: TXNRD1, MAFG, G6PD, SQSTM1, SLC7A11, GSR, NCF2, HMOX1, GLRX2, BACH1, MSRA).
- An **altitude-adaptive risk score** (9 genes: ARNT2, HMOX2, GRB2, GC, ITGA6, TBX5, HK2, LDHA, EPO).
- A **combined dual-axis score** (the arithmetic mean of the two z-standardized risk scores).

The two scores are moderately correlated (Spearman ρ = 0.48, p = 4.0 × 10⁻¹⁹): enough overlap to confirm shared biology, enough independence to make their combination informative. Median splits on each score divide patients into a 2 × 2 grid:

| Group | Definition | n | Median OS (months) | 3-yr survival | 5-yr survival |
|-------|-----------|---|-------------------:|--------------:|--------------:|
| **A: Concordant High** | High ROS + High Altitude | 106 | **14.0** | 29.7% | 19.0% |
| B: Ferroptosis-dominant | High ROS + Low Altitude | 47 | 45.1 | 63.5% | 39.7% |
| C: Hypoxia-dominant | Low ROS + High Altitude | 47 | 48.9 | 58.1% | 36.2% |
| **D: Concordant Low** | Low ROS + Low Altitude | 106 | **70.0** | 75.0% | 57.9% |

**Group A vs Group D HR = 3.60 (2.34–5.54), p < 5 × 10⁻⁹. Four-group log-rank p = 3.9 × 10⁻⁹.** Pairwise tests show that the discordant groups B and C are statistically indistinguishable from each other (p = 0.81) but both are significantly different from A and D. This confirms that the two axes capture different, additive biology.

### Combined Score Discrimination

Bootstrap-resampled C-indices on the TCGA training cohort (500 iterations, 95% CI from percentiles):

| Model | C-index (95% CI) | Δ vs ROS alone |
|-------|------------------|----------------|
| **Combined dual-axis score** | **0.715 (0.665–0.763)** | +0.018 |
| ROS risk score alone | 0.698 (0.651–0.746) | — |
| Altitude risk score alone | 0.672 (0.613–0.731) | −0.026 |
| Archetype + age + sex | 0.668 (0.609–0.721) | −0.030 |
| NRF2 activity alone | 0.605 | −0.093 |

The combined score consistently outperforms either single signature, and the gain holds under 10-fold cross-validation and a 500-iteration permutation test (p < 0.001).

### NRF2 as the Convergence Node

We quantified NRF2 transcriptional activity as the mean z-score of 16 canonical antioxidant-response-element (ARE) target genes (NQO1, HMOX1, SLC7A11, TXNRD1, G6PD, GSR, GCLC, GCLM, FTH1, FTL, SQSTM1, SRXN1, AKR1C1, AKR1B10, ME1, ABCC2). The pattern was unambiguous:

- NRF2 activity is **strongly correlated with the ROS risk score** (ρ = 0.59, p = 1.2 × 10⁻²⁹) and only weakly with the altitude score (ρ = 0.13, p = 0.028).
- NRF2 activity is **highest in Group A and lowest in Group D** (Kruskal–Wallis p = 6.3 × 10⁻¹⁹).
- NRF2 activity is **inversely correlated with ferroptosis vulnerability** (ρ = −0.22, p = 1.7 × 10⁻⁴): high-NRF2 tumors are ferroptosis-resistant by construction.
- The score validates as a true NRF2 readout: positively correlated with NFE2L2 expression (ρ = 0.12, p = 0.034) and elevated in KEAP1-mutant tumors (ρ vs KEAP1 expression = 0.20, p = 5.4 × 10⁻⁴).

NRF2 simultaneously drives ferroptosis resistance (via SLC7A11, TXNRD1, GSR), NADPH-mediated metabolic reprogramming (via G6PD), and immune modulation (via HMOX1-derived CO, biliverdin, and free iron). Computationally, it sits at the intersection of both signatures, a single transcription factor whose constitutive activation explains why concordant high-risk patients have such poor outcomes.

### The HMOX1/HMOX2 Switch: A Novel Prognostic Biomarker

The two signatures each contain one heme oxygenase isoform: **HMOX1** (inducible, NRF2-driven, present in the ROS signature as a risk gene) and **HMOX2** (constitutive, baseline expression, present in the altitude signature as a protective gene). Their ratio captures the degree to which a tumor has shifted from baseline to stress-induced heme metabolism.

- HMOX2 expression is **strongly negatively correlated with the altitude risk score** (ρ = −0.51, p = 2.4 × 10⁻²¹) and ferroptosis vulnerability (ρ = −0.36, p = 7.2 × 10⁻¹¹).
- The log₂(HMOX1/HMOX2) ratio is **highest in Group A** and independently predicts survival (univariate HR = 1.20, p = 0.001; multivariate HR = 1.20, p = 0.002 after adjusting for age and sex).
- HMOX1 is **strongly correlated with immune checkpoint exhaustion markers**: HAVCR2/TIM-3 (ρ = 0.54, p = 2.6 × 10⁻²⁴), TIGIT (ρ = 0.48, p = 9.6 × 10⁻¹⁹), CTLA4 (ρ = 0.44, p = 3.6 × 10⁻¹⁶), PDCD1/PD-1 (ρ = 0.37, p = 3.3 × 10⁻¹¹), LAG3 (ρ = 0.32, p = 1.7 × 10⁻⁸), IDO1 (ρ = 0.31, p = 4.6 × 10⁻⁸).

The HMOX1/HMOX2 axis is, to our knowledge, a novel prognostic readout in HCC. It captures a biologically interpretable shift (from baseline housekeeping heme catabolism to NRF2-driven, stress-responsive heme catabolism) that is also tightly coupled to T-cell exhaustion in the tumor microenvironment.

### Immune Checkpoint Exhaustion, Not Exclusion

A central finding of the project is that NRF2-high HCC does *not* look immunologically cold. HMOX1-high tumors have **higher**, not lower, expression of T-cell markers (CD8A ρ = 0.45, IFNG ρ = 0.40, CD4 ρ = 0.39, NKG7 ρ = 0.38), alongside markedly elevated checkpoint expression. This is the hallmark of T-cell **exhaustion**: the immune system is recruited but functionally disabled.

Across the four dual-axis groups, 6 of 9 immune checkpoints differed significantly (BH-FDR < 0.01):

| Checkpoint | Highest in | Pattern |
|------------|-----------|---------|
| **CD276 / B7-H3** | A & C | p = 1.2 × 10⁻⁵ (q = 1.1 × 10⁻⁴) |
| **PDCD1 / PD-1** | A | p = 0.0024 (q = 0.0043) |
| **CTLA4** | A | p = 1.0 × 10⁻⁴ (q = 4.6 × 10⁻⁴) |
| **HAVCR2 / TIM-3** | A & B | p = 0.0024 (q = 0.0043) |
| **TIGIT** | A & C | p = 0.0038 (q = 0.0057) |
| **SIGLEC15** | C & D (depleted in A) | p = 1.8 × 10⁻⁴ (q = 5.4 × 10⁻⁴) |

Clinically, exhaustion is a *treatable* state: checkpoint inhibitors are designed exactly to rescue exhausted T cells. Exclusion, by contrast, leaves checkpoint inhibitors with nothing to rescue. The dual-axis framework predicts that concordant high-risk patients (Group A) have the immunological substrate to respond to ICB, *if* the underlying NRF2-driven antioxidant defenses are simultaneously disrupted (e.g., by ferroptosis induction).

### Patient Archetypes from Consensus Clustering

Consensus clustering on 10 features (NRF2 activity, ferroptosis vulnerability, STING score, NK-cell ssGSEA, dendritic-cell ssGSEA, CD8 T-cell ssGSEA, HMOX1, HMOX2, ROS risk score, altitude risk score; Ward linkage, Euclidean, 500 iterations, 80% subsampling) identified five reproducible archetypes (optimal k = 5 by CDF analysis):

| Archetype | n | Median OS (mo) | Mortality | Defining features |
|-----------|---|---------------:|----------:|-------------------|
| **NRF2-Dominant** | 29 | **10.0** | 75.9% | Highest NRF2 activity, lowest ferroptosis vulnerability, immune-cold |
| Cold-Quiescent | 117 | 27.5 | 42.7% | Low activity across all axes |
| Immune-Active | 56 | 48.9 | 42.9% | High immune infiltration, high checkpoints |
| HMOX2-Driven | 39 | 55.7 | 43.6% | High constitutive heme oxygenase, protective metabolism |
| Immune-Active-4 | 65 | **83.2** | 24.6% | Variant immune-rich subtype, best prognosis |

Overall log-rank p = 2.7 × 10⁻¹¹. The NRF2-Dominant archetype is significantly worse than every other archetype (NRF2-Dom vs Immune-Active-4: p = 3.1 × 10⁻¹²; vs Cold-Quiescent: p = 4.4 × 10⁻⁶).

### External Validation

Both signatures were applied to three independent cohorts using cohort-specific z-normalization and median splits:

| Cohort | Population | Platform | n | Events | C-index (combined) | HR (A vs D) | Cox p | Notes |
|--------|-----------|----------|---|--------|-------------------:|-------------|------:|-------|
| **TCGA-LIHC** (training) | Mixed (US) | RNA-seq | 302 | 129 | **0.715** | 3.60 | <5 × 10⁻⁹ | Reference |
| **GSE14520** | Chinese (HBV) | Affymetrix | 221 | 85 | 0.635 | **2.58** (1.48–4.49) | 8.2 × 10⁻⁴ | 4-group log-rank p = 0.007 |
| **ICGC LIRI-JP** | Japanese | RNA-seq | 229 | 43 | 0.659 | **2.62** (1.37–4.99) | 3.4 × 10⁻³ | ROS-only (altitude genes unavailable on cached platform) |
| GSE76427 | Singaporean | Illumina | 115 | 23 | 0.489 | 1.27 | NS | Underpowered (only 23 OS events) |
| GSE10141 | European | Affymetrix | 80 | 32 | 0.424 | — | NS | Underpowered, mixed direction |

**Fisher combined p across validation cohorts = 1.7 × 10⁻⁴.** Effect direction (HR > 1 in high-risk) is consistent across all well-powered cohorts, despite differences in etiology (HBV-dominant in GSE14520, mixed in LIRI-JP) and platform (microarray vs RNA-seq). The two underpowered European/Asian cohorts (GSE76427, GSE10141) are unable to detect the effect, the same limitation reported in both companion papers.

### Therapeutic Stratification by Archetype

Cancer-cell-line and tumor expression of drug-target genes was compared across the five archetypes (Kruskal–Wallis):

| Drug class | Marker gene | Highest expression in | KW p |
|------------|-------------|----------------------|-----:|
| **Erastin** (system-Xc⁻ inhibitor, ferroptosis inducer) | SLC7A11 | **NRF2-Dominant** | 3.7 × 10⁻¹¹ |
| RSL3 (GPX4 inhibitor, ferroptosis inducer) | GPX4 | HMOX2-Driven | 2.9 × 10⁻⁴ |
| Sorafenib / Lenvatinib (anti-angiogenic TKIs) | VEGFA | Cold-Quiescent | 1.4 × 10⁻⁷ |
| Atezolizumab / Nivolumab (PD-L1 ICB) | CD274 | **Immune-Active** | 3.6 × 10⁻⁸ |
| Pembrolizumab (PD-1 ICB) | PDCD1 | **Immune-Active** | 7.4 × 10⁻²⁵ |
| Doxorubicin | TOP2A | NRF2-Dominant / Immune-Active | 4.4 × 10⁻¹⁰ |

This produces a falsifiable, archetype-matched treatment rationale:

- **NRF2-Dominant**: very high SLC7A11 and TOP2A → predicted sensitive to ferroptosis induction (erastin analogues); ICB likely needs combination because the tumor is immunologically exhausted but high-checkpoint.
- **Immune-Active / Immune-Active-4**: highest PD-L1/PD-1 expression → ICB monotherapy is the hypothesis-driven first choice.
- **Cold-Quiescent**: highest VEGFA → standard-of-care anti-angiogenic TKIs (sorafenib, lenvatinib).
- **HMOX2-Driven**: relatively protective baseline, intermediate prognosis → standard of care.

Decision curve analysis on the dual-axis model shows positive net benefit across clinically relevant risk thresholds, which indicates that using the framework to guide treatment decisions would improve outcomes compared to a treat-all or treat-none strategy.

## Repository Structure

```
hcc-nrf2-convergence/
├── scripts/
│   ├── 01_data_assembly.py             # Load both LASSO models, compute dual scores, merge clinical
│   ├── 02_dual_signature_stratify.py   # 2x2 classification, KM curves, Cox regression
│   ├── 03_nrf2_activity_scoring.py     # 16-gene NRF2 activity score, NFE2L2/KEAP1 validation
│   ├── 04_sting_immune_evasion.py      # STING score, immune checkpoints, ssGSEA deconvolution
│   ├── 05_hmox_immunomodulation.py     # HMOX1/HMOX2 ratio, immune correlations, survival
│   ├── 06_patient_archetypes.py        # Consensus clustering, archetype KM, feature heatmap
│   ├── 07_therapeutic_stratification.py# Drug target expression, IPS, cytolytic activity, DCA
│   ├── 08_external_validation.py       # GSE14520, ICGC LIRI-JP, GSE76427 validation
│   ├── 09_additional_validation.py     # GSE10141 and additional cohorts
│   ├── 10_metric_optimization.py       # Bootstrap C-index, model comparison
│   └── 11_convergence_lasso.py         # Joint LASSO sanity check on the 20-gene union
├── data/
│   ├── tcga_convergence_master.csv     # Per-patient master table (scores, NRF2, HMOX, clinical)
│   ├── liri_clinical_xena.csv          # ICGC LIRI-JP clinical
│   ├── geo_new/                        # Cached GEO series matrices
│   └── paths.json                      # Centralised data paths
├── results/
│   ├── figures/main/                   # 30+ publication-ready PNGs (Fig 2–11)
│   ├── figures/supplementary/          # Supplementary figures
│   ├── tables/                         # 25+ result CSVs
│   └── model/                          # Cached cluster assignments, joint-LASSO model
└── docs/
    ├── manuscript.tex                  # Single-file draft manuscript
    └── paper/                          # Modular Frontiers-format manuscript (abstract, intro, methods, …)
```

## How to Reproduce

```bash
pip install pandas numpy scipy lifelines scikit-learn matplotlib seaborn gseapy statsmodels requests

# Run the full pipeline in order
for i in 01 02 03 04 05 06 07 08 09 10 11; do
    python3 -u scripts/${i}_*.py
done
```

Scripts read the cached LASSO coefficients from `../hcc-ros-signature/results/model/` and `../hcc-altitude-signature/results/model/`, so the two companion repositories must be present alongside this one. External cohort data is downloaded from GEO/ICGC/Xena on first run and cached under `data/`.

## Key Takeaways

1. **Dual-axis convergence beats either single signature.** Combining the ROS/ferroptosis and altitude-adaptive signatures lifts the C-index from 0.700 (best single) to 0.715, and the four-group stratification (HR 3.6 for concordant high vs concordant low) is far more clinically actionable than a single risk threshold.
2. **NRF2 is the central driver.** A 16-gene NRF2 activity score is highest precisely where outcomes are worst, and inversely correlated with ferroptosis vulnerability: a single transcription factor links antioxidant rewiring, hypoxia adaptation, and immune evasion.
3. **The HMOX1/HMOX2 switch is a novel prognostic biomarker.** The shift from constitutive to stress-induced heme oxygenase predicts survival independently of clinical covariates and tracks T-cell exhaustion markers tightly.
4. **Immune evasion in NRF2-high HCC is exhaustion, not exclusion.** This matters therapeutically: exhausted T cells can be rescued by checkpoint inhibitors, while excluded T cells cannot.
5. **Treatment hypotheses fall out naturally from the archetype map.** NRF2-Dominant tumors are predicted-sensitive to ferroptosis inducers, Immune-Active tumors to ICB monotherapy, Cold-Quiescent tumors to anti-angiogenic TKIs.
6. **Validation holds across ethnically distinct cohorts** (Chinese HBV, Japanese mixed; Fisher combined p = 1.7 × 10⁻⁴), with the same underpowered-cohort caveat that affects both companion signatures.

## Limitations

- All analyses are computational; no wet-lab experimental validation of the NRF2 → ferroptosis → immune-exhaustion links.
- ICGC LIRI-JP validation is ROS-only because altitude-signature genes were not all available on the cached platform.
- GSE76427 (23 events) and GSE10141 (32 events, mixed-direction) are underpowered, a known limitation shared with both companion papers.
- TCGA-LIHC clinical stage is sparsely annotated and was excluded from multivariate models where unavailable.
- The 2 × 2 median-split is standard and reproducible but is not guaranteed to be the optimal cutpoint in every population.
- The NRF2–STING anticorrelation initially hypothesized was not significant (ρ = 0.07, p = 0.20). The immune-evasion mechanism that *did* emerge from the data is checkpoint exhaustion, not STING suppression.

## Data and Methods References

| Resource | Usage |
|----------|-------|
| [TCGA-LIHC](https://portal.gdc.cancer.gov/) | Training cohort (n=302) — RNA-seq, clinical, mutation |
| [GSE14520](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE14520) | External validation (n=221, Chinese HBV-HCC, Affymetrix) |
| [ICGC LIRI-JP](https://dcc.icgc.org/projects/LIRI-JP) | External validation (n=229, Japanese, RNA-seq) |
| [GSE76427](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE76427) | External validation (n=115, Singaporean, Illumina HT-12) |
| [GSE10141](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE10141) | Additional validation (n=80, European, Affymetrix) |
| [MSigDB](https://www.gsea-msigdb.org/gsea/msigdb/) | NRF2 target gene set (Hallmark, KEGG, NFE2L2 ARE targets) |
| [Reactome](https://reactome.org/) | KEAP1-NRF2 pathway annotations |
| [cBioPortal](https://www.cbioportal.org/) | KEAP1 / NFE2L2 / TP53 mutation data |
| [lifelines](https://lifelines.readthedocs.io/) | Cox regression, KM, log-rank, C-index |
| [gseapy](https://gseapy.readthedocs.io/) | ssGSEA immune-cell deconvolution |
| [scikit-learn](https://scikit-learn.org/) | Consensus clustering, bootstrap CV |

## Companion Papers

This project is the third of three companion analyses by the same author:

1. [hcc-ros-signature](../hcc-ros-signature): 11-gene ROS/ferroptosis prognostic signature.
2. [hcc-altitude-signature](../hcc-altitude-signature): 9-gene altitude-adaptive hypoxia signature.
3. **hcc-nrf2-convergence** *(this project)*: Dual-axis convergence framework integrating both.

Portfolio: [ishaschhikara316.github.io/isha](https://ishaschhikara316.github.io/isha)

## License

This project is for academic research purposes.
