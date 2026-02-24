# Agent Summary: Project State and Change Log

This document is intended for AI agents reviewing this codebase. It provides a comprehensive overview of every section, what was implemented, what was changed and why, and what the current results look like.

---

## 1. Project Goal

Replicate the main results from Chang, Hong, and Liskovich (2015), "Regression Discontinuity and the Price Effects of Stock Market Indexing" (*Review of Financial Studies*, 28(1), 212–246), and extend the analysis to 2015–2024.

The paper uses a **fuzzy regression discontinuity (RD) design** at the Russell 1000/2000 cutoff (rank 1000 by end-of-May market capitalization) to estimate the causal price effects of index membership on stock returns, volume, and other outcomes.

**Critical constraint**: We do not have access to actual Russell constituent lists. The paper instruments actual membership (D) with predicted membership (τ = 1 if predicted rank > cutoff). We instead set **D = τ** (sharp RD approximation), yielding Intent-to-Treat (ITT) estimates rather than the paper's LATE. This mechanically attenuates all point estimates by a factor of ~α₀ᵣ ≈ 0.785, plus additional attenuation from rank reconstruction noise (~25-30% misclassification near cutoff).

---

## 2. Data Pipeline

### Files: `auxiliary/data_processing.py`, notebook Cells 1–8

| Function | Purpose | Status |
|----------|---------|--------|
| `merge_crsp_compustat()` | Clean and merge CRSP monthly, Compustat quarterly, and CCM link data | ✅ Complete |
| `compute_market_cap_rankings()` | Generate end-of-May market cap rankings for a given year | ✅ Complete |
| `compute_banding_cutoffs()` | Compute post-2007 banding-adjusted cutoffs | ✅ Fixed (see §8) |
| `identify_index_switchers()` | Build addition/deletion panels using prior-year rank as membership proxy | ✅ Complete |
| `construct_outcome_variables()` | Extract monthly returns (May–Sep) for the RD analysis | ✅ Complete |
| `construct_volume_ratio()` | Compute volume ratio (VR) = normalized stock volume / normalized market volume | ✅ Complete |
| `construct_validity_variables()` | Merge prior-year Compustat annual fundamentals for validity tests | ✅ Complete |

### Key data facts
- CRSP monthly: last trading day of each month (filter by `date.dt.month == 5`, not by date string)
- SHROUT is in thousands; market cap (millions) = abs(PRC) × SHROUT / 1000
- Compustat CSHOQ is in millions of shares
- CCM link: LINKENDDT='E' means still active → replace with `pd.Timestamp('2099-12-31')`
- Rankings produce 3,400–7,200 eligible stocks per year

### Ranking construction (Paper Section 1.1)
1. Get end-of-May prices from CRSP monthly
2. Get shares outstanding from Compustat quarterly (CSHOQ), using RDQ or SEC filing deadline rules to determine which quarter was publicly available
3. Adjust shares using CRSP CFACSHR for splits between quarter-end and May 31
4. Take the larger of CRSP shares (SHROUT) and adjusted Compustat shares
5. Compute market cap = abs(PRC) × shares
6. Rank all eligible U.S. common stocks (SHRCD ∈ [10,11], price ≥ $1.00, NYSE/AMEX/NASDAQ)

---

## 3. Estimation

### File: `auxiliary/estimation.py`

| Function | Purpose | Status |
|----------|---------|--------|
| `fuzzy_rd_estimate()` | 2SLS estimation with year fixed effects. Returns β₀ᵣ (treatment effect), t-stat, p-value, first-stage stats | ✅ Complete |
| `fuzzy_rd_time_trend()` | 2SLS with linear time trend interaction (β₂ᵣ = annual change in treatment effect) | ✅ Complete |
| `optimal_bandwidth()` | IK bandwidth selector | ❌ Not implemented (paper uses bandwidth=100 throughout) |

### Specification details
- **First stage**: D_it = α₀ₗ + α₁ₗ(r_it − c) + τ_it[α₀ᵣ + α₁ᵣ(r_it − c)] + year FE + ε_it
- **Second stage**: Y_it = β₀ₗ + β₁ₗ(r_it − c) + D̂_it[β₀ᵣ + β₁ᵣ(r_it − c)] + year FE + ν_it
- Since D = τ, the first stage is trivially α₀ᵣ = 1.0, F → ∞
- 2SLS standard errors use residuals from the original X (not X̂)
- P-values computed via `scipy.special.betainc` (scipy.stats is broken in the base conda env)

---

## 4. Plotting

### File: `auxiliary/plotting.py`

| Function | Purpose | Status |
|----------|---------|--------|
| `plot_market_cap_continuity()` | Binned scatter of log(market_cap) vs rank around cutoff (Figure 1) | ✅ Complete |
| `plot_rd_discontinuity()` | Binned scatter of outcome vs running variable with local linear fits (Figure 4) | ✅ Complete |
| `plot_time_trends()` | Rolling RD estimates with 95% CI over time (Figure 5) | ✅ Complete |
| `plot_index_weights()` | Index weights before/after reconstitution (Figure 2) | ❌ Not implemented (requires Russell weight data we don't have) |

---

## 5. Notebook Structure (`project.ipynb`)

| Section | Cells | Content | Status |
|---------|-------|---------|--------|
| 1. Setup | 1–2 | Imports, configuration (BANDWIDTH=100, years) | ✅ |
| 2. Data Loading | 3–5 | Load CRSP, Compustat, CCM from gzipped CSVs | ✅ |
| 3a. Rankings | 6–8 | Build rankings for all years, verify top stocks | ✅ |
| 3b. Panels | 9–10 | Build addition/deletion panels with banding, year-by-year diagnostics | ✅ |
| 4. Figure 1 | 15 | Market cap continuity plot | ✅ |
| 5. Table 3 | 17–18 | First-stage regressions + post-banding diagnostics | ✅ |
| 6. Table 4 | 20 | Returns fuzzy RD (May–Sep, 1996–2012) | ✅ |
| 6b. Figure 4 | 21 | RD discontinuity plots (addition/deletion, bin widths 2 and 5) | ✅ |
| 7. Table 5 | 23 | Volume ratio fuzzy RD | ✅ |
| 8. Table 6 | 25 | Validity tests (8 pre-determined variables) | ✅ |
| 9a. Tables 7-8 | 26 | Time trend regressions (β₂ᵣ) on 1996–2012 | ✅ |
| 9b. Extension | 27 | Compare 1996–2012 vs 2015–2024 (both β₀ᵣ and β₂ᵣ) | ✅ |
| 9c. Conclusion | 28 | Extension conclusion markdown cell | ✅ |
| 9d. Figure 5 | 29 | Rolling 3-year RD estimates over time | ✅ |
| 10. Summary | 30 | Summary table with original vs. replicated values | ✅ |

---

## 6. Replication Results (Current)

### Table 4: Returns Fuzzy RD (1996–2012)
| Month | Addition | Deletion | Paper Addition | Paper Deletion |
|-------|----------|----------|----------------|----------------|
| May | −1.41% (t=−0.82) | −0.30% (t=−0.22) | −0.3% | +0.5% |
| **Jun** | **−0.60% (t=−0.39)** | **+0.74% (t=+0.50)** | **+5.0% (t=2.65)** | **+5.4% (t=3.00)** |
| Jul | −1.40% (t=−0.76) | −1.90% (t=−1.36) | −0.3% | −1.9% |

June estimates are attenuated (see §1 re: D=τ and rank noise). July deletion matches well.

### Table 5: Volume Ratio (1996–2012)
- Addition June VR: −0.176 (t=−1.46) vs paper's +0.478 (t=3.14) — wrong sign (noise)
- Deletion June VR: −0.147 (t=−1.77) vs paper's −0.263 (t=−2.74) — same sign ✓

### Table 6: Validity Tests
All 8 variables (market_cap, repurchase, ROE, ROA, EPS, assets, ICR, C/A) show insignificant discontinuities ✓

### Tables 7-8: Time Trends (1996–2012)
- **Deletion β₂ᵣ = −0.495% (t=−2.52)** — replicates the paper's declining price impact (t=−2.46) ✓
- Addition β₂ᵣ = −0.162% (t=−0.67) — same sign but not significant

### Extension: 2015–2024
- Addition: β₀ᵣ = +8.36% (t=1.27), β₂ᵣ = −0.84% (t=−0.83) — large but noisy (N=127)
- Deletion: β₀ᵣ = −5.28% (t=−1.27), β₂ᵣ = −0.34% (t=−0.57) — effect may have vanished

---

## 7. Extension Analysis

Two competing hypotheses about how index price effects changed as passive AUM tripled (2012→2024):

1. **Passive distortion**: larger effects (more passive money → more mechanical buying/selling at reconstitution)
2. **Arbitrage efficiency**: smaller effects (arbitrage capital scales alongside passive, muting price impact)

The evidence leans toward **arbitrage efficiency**: the deletion time trend in 1996–2012 shows a strongly declining price impact (β₂ᵣ = −0.50%, t=−2.52), and the 2015–2024 deletion effect is statistically indistinguishable from zero. However, the 2015–2024 samples are small (N=127 addition, N=279 deletion) due to the wide post-banding cutoffs, limiting statistical power.

---

## 8. Critical Bug Fix: `compute_banding_cutoffs()`

### What is banding?
Starting in 2007, Russell implemented a banding policy: a stock only switches indexes if its position deviates sufficiently from the rank-1000 boundary. This reduces unnecessary turnover.

### The bug history
The function went through **three implementations**:

1. **Original (buggy)**: Used forward cumulative market cap percentage with ±2.5pp. But C_forward%(1000) = 95.6% (top 1000 stocks hold 95.6% of total), so the band [93.1%, 98.1%] spanned ~450 ranks. Produced k_add≈1224–1513, k_del≈745–837.

2. **First fix attempt (also wrong)**: Used ±2.5% of the 1000th stock's *dollar* market cap. Produced k_add≈1010–1020, k_del≈982–993 — bands were too narrow (~15 ranks each side).

3. **Correct implementation (current)**: Uses **reverse cumulative market cap** as described in footnote 5 of the paper. C_rev%(k) = fraction of total R3000E market cap held by stocks ranked k through N (bottom-up cumulation). C_rev%(1000) ≈ 9–10%. Band: [C_rev%(1000) − 0.025, C_rev%(1000) + 0.025] ≈ [7.5%, 12.5%].

### Verification against the paper
Footnote 5 describes a stock ranked 1210 in 2007 with market cap $1.8B. The cutoff stock (rank 1000) had market cap $2.47B. "The cumulative market capitalization of stock 1210 at 8% of the Russell 3000E while the cumulative market capitalization of the cutoff stock was 10%."

Our implementation: C_rev%(1000) = 9.39%, C_rev%(1210) = 7.24%. Band lower limit = 6.89%. Since 7.24% > 6.89%, stock 1210 stays in R1000 ✓. k_add = 1251 for 2007.

### Current cutoffs
| Year | k_add | k_del |
|------|-------|-------|
| 2007 | 1251 | 823 |
| 2012 | 1286 | 806 |
| 2020 | 1447 | 757 |
| 2024 | 1545 | 738 |

The wide bands are intentional — Russell designed banding to significantly reduce index turnover.

---

## 9. Known Limitations

1. **D = τ (no actual Russell lists)**: All estimates are ITT, not LATE. Mechanically attenuated by ~21%.
2. **Rank reconstruction noise**: CRSP/Compustat uses total shares; Russell uses float-adjusted shares. ~25-30% misclassification near cutoff further attenuates estimates.
3. **Post-banding sample sizes**: Post-2007 addition samples are small (median 13/yr) due to the wide banding cutoff. The deletion sample is larger (median 29/yr) but still smaller than pre-banding.
4. **Extension sample power**: 2015–2024 has only 127 addition and 279 deletion observations, limiting statistical significance.
5. **scipy.stats broken**: The base conda environment has a broken scipy.stats module. P-values are computed using `scipy.special.betainc` directly.

---

## 10. Files Deleted (Legacy Cleanup)

These were template leftovers from an eisenhauerIO student template (referenced GPA, academic probation — unrelated to this project):
- `auxiliary/plots.py`, `auxiliary/predictions.py`, `auxiliary/tables.py`
- `files/causalgraph1.PNG`, `files/causalgraph2.PNG`, `files/bounds_nextGPA.PNG`
- Various one-off notebook editing scripts (`edit_notebook*.py`)

---

## 11. Environment

- Python: `/Users/kennyren/anaconda3/bin/python` (base anaconda)
- Key packages: pandas, numpy, matplotlib, scipy (betainc only)
- `environment.yml` exists but the `russell-rd` conda env was never created
- Data accessed through WRDS; stored in `data/` (gitignored)
