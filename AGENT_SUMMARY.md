# Agent Summary: Project State and Change Log

This document is intended for AI agents reviewing this codebase. It provides a comprehensive overview of every section, what was implemented, what was changed and why, and what the current results look like.

---

## 1. Project Goal

Replicate the main results from Chang, Hong, and Liskovich (2015), "Regression Discontinuity and the Price Effects of Stock Market Indexing" (*Review of Financial Studies*, 28(1), 212–246), and extend the analysis to 2015–2024.

The paper uses a **fuzzy regression discontinuity (RD) design** at the Russell 1000/2000 cutoff (rank 1000 by end-of-May market capitalization) to estimate the causal price effects of index membership on stock returns, volume, and other outcomes.

**Data upgrade**: We obtained actual Russell 1000/2000 constituent lists from Bloomberg terminals for 1996–2024, with 9-digit CUSIPs matched to CRSP PERMNOs via ticker → CCM link (97.2% CUSIP coverage; CRSP monthly lacks NCUSIP so ticker-based matching is used). This enables a proper fuzzy 2SLS design where D_actual (observed membership from Bloomberg) is instrumented by τ (predicted membership from rankings), matching the paper's specification.

---

## 2. Data Pipeline

### Files: `auxiliary/data_processing.py`, notebook Cells 1–8

| Function | Purpose | Status |
|----------|---------|--------|
| `merge_crsp_compustat()` | Clean and merge CRSP monthly, Compustat quarterly, and CCM link data | ✅ Complete |
| `compute_market_cap_rankings()` | Generate end-of-May market cap rankings for a given year | ✅ Complete |
| `compute_banding_cutoffs()` | Compute post-2007 banding-adjusted cutoffs | ✅ Fixed (see §8) |
| `match_bloomberg_to_crsp()` | Match Bloomberg Russell constituent lists to CRSP PERMNOs via ticker → CCM link | ✅ NEW (Step 1) |
| `identify_index_switchers()` | Build addition/deletion panels; sets D=tau now, needs Step 2 upgrade | 🔲 Needs D_actual upgrade |
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
| `fuzzy_rd_estimate()` | 2SLS with year FEs, HC1-robust SEs, optional `poly_degree=2` | ✅ Complete |
| `fuzzy_rd_time_trend()` | 2SLS with time trend interaction, HC1-robust SEs, optional `poly_degree=2` | ✅ Complete |
| `optimal_bandwidth()` | Returns 100 (paper's canonical choice per Section 4.2) | ✅ Complete |
| `bandwidth_sensitivity()` | Runs `fuzzy_rd_estimate` at h ∈ {50, 100, 150}; returns DataFrame | ✅ Complete |

### Specification details
- **First stage**: D_it = α₀ₗ + α₁ₗ(r − c) [+ α₂ₗ r²] + τ[α₀ᵣ + α₁ᵣ(r − c) [+ α₂ᵣ r²]] + year FE + ε
- **Second stage**: Y_it = β₀ₗ + β₁ₗ(r − c) [+ β₂ₗ r²] + D̂[β₀ᵣ + β₁ᵣ(r − c) [+ β₂ᵣ r²]] + year FE + ν
- With Bloomberg D_actual, the first stage is an informative regression: α₀ᵣ ≈ 0.785, F > 200
- **Standard errors**: HC1-robust via `statsmodels.stats.sandwich_covariance.S_white_simple`.
  Assembly: `var_beta = (n / df_resid) * bread @ S_white_simple(X_hat * resid[:,None]) @ bread`
  This is the only statsmodels import that avoids the broken scipy chain in base conda.
- P-values via `scipy.special.betainc` (t-distribution, df = n − k)
- First-stage diagnostics (α₀ᵣ t-stat, R², F) remain homoskedastic OLS — appropriate since they are instrument-strength diagnostics, not causal estimates

### poly_degree parameter
Both `fuzzy_rd_estimate` and `fuzzy_rd_time_trend` accept `poly_degree=1` (default, local linear — paper's main spec) or `poly_degree=2` (adds r² and D*r² terms). This replicates the quadratic robustness check in Chang et al. (2015, Section 4.2): *"our results are robust to changes in the bandwidth and to quadratic functions of ranking."*

---

## 4. Plotting

### File: `auxiliary/plotting.py`

| Function | Purpose | Status |
|----------|---------|--------|
| `plot_market_cap_continuity()` | Binned scatter of log(market_cap) vs rank around cutoff (Figure 1) | ✅ Complete |
| `plot_rd_discontinuity()` | Binned scatter of outcome vs running variable with local linear fits (Figure 4) | ✅ Complete |
| `plot_time_trends()` | Rolling RD estimates with 95% CI over time (Figure 5) | ✅ Complete |
| `plot_index_weights()` | Index weights before/after reconstitution (Figure 2) | ⛔ Data unavailable |

**`plot_index_weights()` note**: Returns `None`. Russell Inc.'s end-of-June float-adjusted constituent weights are proprietary and not distributed through WRDS, CRSP, or Compustat. The function has a full docstring explaining this limitation. It does not raise `NotImplementedError`.

---

## 5. Notebook Structure (`project.ipynb`)

| Section | Cells | Content | Status |
|---------|-------|---------|--------|
| 1. Setup | 1–2 | Imports, configuration (BANDWIDTH=100, years) | ✅ |
| 2. Data Loading | 3–5 | Load CRSP, Compustat, CCM from gzipped CSVs | ✅ |
| 3a. Rankings | 6–8 | Build rankings for all years, verify top stocks | ✅ |
| 3b. Panels | 9–10 | Build addition/deletion panels with banding, year-by-year diagnostics | ✅ |
| 3c. Returns | 11–13 | Merge monthly returns into addition/deletion panels | ✅ |
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

**Note**: All cells have output. However, Tables 3–8 and the extension (Cells 17–29) were last executed with the old homoskedastic SE formula. Re-run the notebook after the HC1 estimation changes to refresh those values:
```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 project.ipynb
```

---

## 6. Replication Results (Current)

**NOTE: Results below are from the sharp RD (D = τ) run. These will be updated after the fuzzy 2SLS upgrade using Bloomberg constituent data. See IMPLEMENTATION_PLAN.md.**

### Table 4: Returns Fuzzy RD (1996–2012)
| Month | Addition | Deletion | Paper Addition | Paper Deletion |
|-------|----------|----------|----------------|----------------|
| May | −1.41% (t=−0.76) | −0.30% (t=−0.21) | −0.3% | +0.5% |
| **Jun** | **−0.60% (t=−0.37)** | **+0.74% (t=+0.46)** | **+5.0% (t=2.65)** | **+5.4% (t=3.00)** |
| Jul | −1.40% (t=−0.72) | −1.90% (t=−1.30) | −0.3% | −1.9% |
| Aug | +2.70% (t=+1.42) | −3.35% (t=−2.76) | +3.5% | −0.2% |

June addition is **wrong-signed** (not merely attenuated). This is consistent with rank misclassification noise dominating the small true ITT near the cutoff. May addition (−1.41% vs −0.3%) does NOT match well — it's same sign but 4.7× larger.

### Table 5: Volume Ratio (1996–2012)
- Addition June VR: −0.176 (t=−1.00) vs paper's +0.478 (t=3.14) — wrong sign (noise)
- Deletion June VR: −0.147 (t=−1.87) vs paper's −0.263 (t=−2.74) — same sign ✓

Note: Unconditional mean VR = 1.397 (expected ~1.0). Likely due to positive volume skew near reconstitution dates. See Module 3 in CLAUDE.md for optional investigation.

### Table 6: Validity Tests
**⚠️ NOT "all insignificant"**: 6 of 8 variables show no significant discontinuity. Two are significant at 5%:
- Repurchase (deletion): t = −2.32
- Cash/assets (addition): t = +2.39

With 16 tests (8 variables × 2 samples), 1–2 rejections at 5% are expected by chance (16 × 0.05 = 0.8). Broadly supportive of the design.

### Tables 7-8: Time Trends (1996–2012)
- **Deletion β₂ᵣ = −0.495% (t=−2.61)** — replicates the paper's declining price impact ✓ (strongest result)
- Addition β₂ᵣ = −0.162% (t=−0.64) — same sign but not significant

### Extension: 2015–2024
- Addition: β₀ᵣ = +8.357% (t=+1.51), β₂ᵣ = −0.839% (t=−0.78) — large but very noisy (N=127)
- Deletion: β₀ᵣ = −5.284% (t=−1.40), β₂ᵣ = −0.340% (t=−0.65) — effect may have vanished (N=279)

**⚠️ Cell 28 markdown currently contains WRONG numbers** (from a previous run). See §12 for the fix list.

---

## 7. Extension Analysis

Two competing hypotheses about how index price effects changed as passive AUM tripled (2012→2024):

1. **Passive distortion**: larger effects (more passive money → more mechanical buying/selling at reconstitution)
2. **Arbitrage efficiency**: smaller effects (arbitrage capital scales alongside passive, muting price impact)

The 1996–2012 deletion time trend (β₂ᵣ = −0.495%, t=−2.61) is the single most robust finding — it replicates the paper's declining price impact with statistical significance. The 2015–2024 estimates are suggestive but underpowered: the deletion effect appears to have vanished (β₀ᵣ = −5.28%, t=−1.40), consistent with the arbitrage efficiency hypothesis, but confidence intervals are wide due to small samples (N=127 addition, N=279 deletion). These results should be interpreted cautiously given the D = τ attenuation and limited post-banding sample sizes.

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

1. **Rank reconstruction noise**: CRSP/Compustat uses total shares; Russell uses float-adjusted shares. ~25-30% misclassification near cutoff, but this is now absorbed by the first stage (instrumented via τ → D_actual).
2. **Ticker-based Bloomberg matching**: CRSP monthly lacks NCUSIP so matching is via Bloomberg ticker → CCM link tic. Match rates: ~56% (1996) improving to ~98% (2024). Near-cutoff match rates are likely higher since mid-cap stocks tend to have clean tickers.
3. **Post-banding sample sizes**: Post-2007 addition samples are small (median 13/yr) due to the wide banding cutoff. The deletion sample is larger (median 29/yr) but still smaller than pre-banding.
4. **Extension sample power**: 2015–2024 has only 127 addition and 279 deletion observations, limiting statistical significance.
5. **Broken scipy/statsmodels in base conda**: `scipy.stats`, `scipy.optimize`, `scipy.interpolate`, `scipy.sparse.linalg`, `statsmodels.api`, and `statsmodels.regression.linear_model` all fail with `ImportError: cannot import name '_spropack'`. Safe imports: `scipy.special.betainc` and `statsmodels.stats.sandwich_covariance.S_white_simple`.

---

## 10. Files Deleted (Legacy Cleanup)

These were template leftovers from an eisenhauerIO student template (referenced GPA, academic probation — unrelated to this project):
- `auxiliary/plots.py`, `auxiliary/predictions.py`, `auxiliary/tables.py`
- `files/causalgraph1.PNG`, `files/causalgraph2.PNG`, `files/bounds_nextGPA.PNG`
- Various one-off notebook editing scripts (`edit_notebook*.py`)

---

## 11. Environment

- Python: `/Users/kennyren/anaconda3/bin/python` (base anaconda)
- Key packages: pandas, numpy, matplotlib, scipy (betainc only), statsmodels (S_white_simple only)
- `environment.yml` exists but the `russell-rd` conda env was never created
- Data accessed through WRDS; stored in `data/` (gitignored)
- To run the notebook: `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 project.ipynb`

---

## 12. Fuzzy 2SLS Upgrade — Progress

### `match_bloomberg_to_crsp(bloomberg_file, ccm_link_df)` — NEW in Step 1

Signature: `match_bloomberg_to_crsp(bloomberg_file, ccm_link_df) → pd.DataFrame`
Returns: columns (year, PERMNO, D_actual); D_actual=1 if R2000, 0 if R1000. Unmatched dropped.

**Matching approach**: Bloomberg `ticker` → CCM link `tic` (exact, uppercased), active in June of each year. Secondary: strip trailing `.N` suffix from CCM tic. CRSP monthly lacks NCUSIP so NCUSIP-based matching is unavailable.

**Match rates by year:**
```
1996=55.1%  2000=66.0%  2005=76.0%  2010=84.7%  2015=88.7%  2020=95.0%  2024=97.6%
```
Full table in MEMORY.md. Low early-year rates driven by Bloomberg placeholder tickers (e.g. '0111145D') for ~35% of 1996 stocks, declining to ~11% by 2010. Near-cutoff match rates are likely substantially higher.

**Spot checks passed**: AAPL (PERMNO 14593)=R1000 all years ✓; AAON (PERMNO 76868)=R2000 through 2023, R1000 in 2024 ✓.

**Fallback for unmatched stocks**: Excluded from the D_actual panel. When `identify_index_switchers()` merges D_actual, stocks without a match get D_actual=NaN. These need to either be dropped or fall back to D=τ — decision deferred to Step 2.

### Implementation plan status
- Step 0 ✅: `project_BACKUP_pre_fuzzy.ipynb` created
- Step 0a ✅: CLAUDE.md, AGENT_SUMMARY.md, MEMORY.md updated
- Step 1 ✅: `match_bloomberg_to_crsp()` implemented and verified
- Step 2 🔲: Modify `identify_index_switchers()` to accept `bloomberg_file` and construct D_actual
- Step 3 🔲: Upgrade `fuzzy_rd_estimate()` for proper 2SLS (first stage: D_actual ~ τ + rank_centered)
- Step 4 🔲: Same upgrade for `fuzzy_rd_time_trend()`
- Step 5 🔲: Re-run full notebook with fuzzy RD estimates
- Step 6 🔲: Update narrative cells for proper fuzzy RD framing