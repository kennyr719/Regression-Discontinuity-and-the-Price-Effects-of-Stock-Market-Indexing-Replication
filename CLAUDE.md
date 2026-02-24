# Project Guidance for Claude Code

## Overview

This is a replication project for ECON 481 (Economics Data Science) at the University of Washington. We are replicating the main results from:

> Chang, Y.-C., Hong, H., & Liskovich, I. (2015). "Regression Discontinuity and the Price Effects of Stock Market Indexing." *The Review of Financial Studies*, 28(1), 212–246. DOI: 10.1093/rfs/hhu041

The paper uses a **fuzzy regression discontinuity (RD) design** to estimate the causal price effects of Russell index membership on stock returns around the Russell 1000/2000 cutoff.

## Extension

After replicating the original 1996–2012 results, we extend the sample to 2015–2024 to test whether the index premium has changed as passive investing's market share tripled (~15% to ~50%). This frames a test of two competing hypotheses:
- **Passive distortion hypothesis**: larger price effects due to more passive money
- **Arbitrage efficiency hypothesis**: smaller effects as arbitrage capacity scales alongside passive growth

---

## Current Project State (as of latest session)

### What is COMPLETE and working ✅
- **Data pipeline**: `merge_crsp_compustat()` and `compute_market_cap_rankings()` — fully implemented and verified. Rankings for 1996–2024 are correct (top stocks check out, rank-1000 market caps in expected range).
- **Sample construction**: `identify_index_switchers()` — builds addition/deletion panels using prior-year rank as membership proxy. Sets D = τ (sharp RD approximation).
- **Outcome variables**: `construct_outcome_variables()` (monthly returns) and `construct_volume_ratio()` (VR) are implemented.
- **Estimation**: `fuzzy_rd_estimate()` and `fuzzy_rd_time_trend()` — HC1-robust SEs via `S_white_simple`, optional `poly_degree=2` quadratic robustness check (Chang et al. Section 4.2).
- **Bandwidth**: `optimal_bandwidth()` returns 100 (paper's canonical choice). `bandwidth_sensitivity()` tests h ∈ {50, 100, 150}.
- **Validity tests**: `construct_validity_variables()` merges Compustat annual fundamentals.
- **All notebook sections executed**: Sections 1–10 all have output, including Figure 1, Tables 3–6, Figure 4, Tables 7–8, extension, Figure 5, summary table.
- **`plot_index_weights()`**: Returns `None` with a docstring explaining the data limitation (Russell float-adjusted weights are proprietary and unavailable).

### What STILL NEEDS IMPLEMENTATION 🔲
- Summary table in Cell 30 — outputs reflect old homoskedastic SEs; needs re-running after estimation changes to show HC1-robust values.

### Known Issues / Bugs 🐛
See the "Priority Fixes" section below for detailed instructions.

---

## Priority Fixes (IMPLEMENT THESE)

### FIX 1 ✅ DONE: `compute_banding_cutoffs()` — correct band calculation

**What was fixed**: The band uses *reverse cumulative market cap* as described in footnote 5 of Chang et al. (2015). C_rev%(k) = fraction of total R3000E market cap held by stocks ranked k through N (bottom-up cumulation). C_rev%(1000) ≈ 9–10%. Band: stocks switch only if C_rev%(k) deviates by > 2.5pp from C_rev%(1000).

**Result**: Cutoffs k_add≈1251–1545, k_del≈738–823. Verified against footnote 5 example: stock 1210 in 2007 at C_rev%=7.24% stays in R1000 (band lower limit 6.89%) ✓.

### FIX 2 ✅ DONE: Run unexecuted notebook cells

All notebook cells now have output (Sections 1–10 complete). To regenerate outputs with updated HC1-robust SEs after the estimation changes, run:
```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 project.ipynb
```

### FIX 3 ✅ DONE: Summary Table (Cell 30)

Cell 30 has a completed summary table comparing original vs. replicated values. Note the attenuation caveat — our ITT estimates are smaller than the paper's LATE by factor ~0.785 (D = τ) plus rank noise.

### FIX 4 ✅ DONE: Extension conclusion (Cell 28)

Cell 28 (markdown) contains the extension conclusion, covering: sample size limitations, D = τ attenuation, hypothesis evaluation, and the one robust finding (deletion time trend β₂ᵣ ≈ −0.50%, t ≈ −2.52).

### FIX 5 ✅ DONE: Legacy template files deleted

Deleted: `auxiliary/plots.py`, `auxiliary/predictions.py`, `auxiliary/tables.py`, template images, `edit_notebook*.py` scripts. `auxiliary/__init__.py` updated.

### FIX 6 ✅ DONE: HC1-robust standard errors + robustness tools

- `fuzzy_rd_estimate()` and `fuzzy_rd_time_trend()` now use HC1-robust SEs via `statsmodels.stats.sandwich_covariance.S_white_simple` (the only statsmodels import that avoids the broken scipy chain in base conda).
- Both functions accept `poly_degree=1` (default, local linear) or `poly_degree=2` (quadratic robustness check per Chang et al. Section 4.2).
- `optimal_bandwidth()` returns 100 (paper's choice); no longer raises `NotImplementedError`.
- New `bandwidth_sensitivity(df, outcome, bandwidths=(50,100,150))` for bandwidth robustness.
- `plot_index_weights()` returns `None` with a docstring explaining Russell float-adjusted weights are proprietary and unavailable through WRDS.

---

## Key Methodology

### The RD Design
- Every year on the last trading day of May, stocks are ranked by market capitalization
- Ranks 1–1000 → Russell 1000; Ranks 1001–3000 → Russell 2000
- Because the Russell 2000 is value-weighted, stocks just below rank 1000 receive ~10x higher index weight than stocks just above
- This creates a discontinuity in passive buying pressure at the cutoff
- The paper uses a **fuzzy RD** because predicted rankings don't perfectly match actual Russell assignments
- We use D = τ (sharp RD) because actual Russell constituent lists are unavailable via our WRDS subscription

### Post-2007 Banding Policy
Starting with the 2007 reconstitution, Russell implemented a banding policy:
- Define C_rev%(k) = fraction of total R3000E market cap held by stocks ranked k through N (reverse/bottom-up cumulation)
- C_rev%(1000) ≈ 9–10%; a stock switches only if C_rev%(k) deviates by >2.5 percentage points
- Band: [C_rev%(1000) − 0.025, C_rev%(1000) + 0.025] ≈ [7.5%, 12.5%]
- Cutoff ranges: k_add ≈ 1251–1545, k_del ≈ 738–823
- Verified against footnote 5: stock ranked 1210 in 2007 at C_rev%≈8% stays in R1000 (band ≈ 7.5%) ✓
- The function `compute_banding_cutoffs()` in `auxiliary/data_processing.py` handles this

### Fuzzy RD Specification

**First stage** (Equation 4 in paper):
```
D_it = α_0l + α_1l(r_it - c) + τ_it[α_0r + α_1r(r_it - c)] + ε_it
```
- D_it = actual Russell 2000 membership indicator (we set D = τ)
- r_it = end-of-May market cap rank
- c = cutoff (1000 pre-banding, varies post-banding)
- τ_it = instrument: indicator for predicted rank > cutoff

**Second stage** (Equation 5 in paper):
```
Y_it = β_0l + β_1l(r_it - c) + D_it[β_0r + β_1r(r_it - c)] + ν_it
```
- Y_it = outcome variable (returns, volume ratio, comovement, etc.)
- β_0r = the treatment effect of interest

**Time trend specification** (Equation 7–8 in paper):
```
Y_it = β_0l + β_1l(r_it - c) + β_2l*t + D_it[β_0r + β_1r(r_it - c) + β_2r*t] + ν_it
```
- t = years since base_year (1996 for replication, 2015 for extension)
- β_2r = how the treatment effect changes over time

### Bandwidth
- Default bandwidth: 100 ranks on each side of the cutoff
- Rule-of-thumb (ROT) from Lee and Lemieux (2010) generally gives ~100
- Local linear regression on each side of the cutoff

### Two Separate Samples
- **Addition effect**: Stocks in Russell 1000 in year t-1 (prev_rank ≤ 1000) that are near the cutoff in year t. Comparing those that crossed into Russell 2000 (τ=1) vs. those that just missed (τ=0).
- **Deletion effect**: Stocks in Russell 2000 in year t-1 (prev_rank > 1000) that are near the cutoff in year t. Comparing those that stayed in Russell 2000 (τ=1) vs. those that moved to Russell 1000 (τ=0).

## Data

All datasets are in the `data/` folder (excluded from git via .gitignore):

| File | Source | Description |
|------|--------|-------------|
| `crsp_monthly.csv.gz` | WRDS CRSP | Monthly stock data: PERMNO, date, PRC, RET, SHROUT, VOL, FACSHR, CFACPR, CFACSHR, EXCHCD, SHRCD |
| `crsp_daily.csv.gz` | WRDS CRSP | Daily stock data: same variables. RET column has mixed types — use `pd.to_numeric(errors="coerce")` |
| `compustat_quarterly.csv.gz` | WRDS Compustat | Quarterly: gvkey, datadate, cshoq, rdq, fyearq, fqtr |
| `compustat_annual.csv.gz` | WRDS Compustat | Annual: gvkey, datadate, at, ceq, che, epspx, ni, oibdp, xint, prstkc |
| `crsp_compustat_link.csv.gz` | WRDS CCM | Linking table: gvkey, LPERMNO (=PERMNO), LPERMCO, LINKDT, LINKENDDT, LINKTYPE, LINKPRIM |
| `russell2000_daily.csv.gz` | yfinance (^RUT) | Daily Russell 2000 index returns: date, rut_return |

### Important Data Notes
- **No Russell constituent lists available** — we reconstruct index membership from CRSP/Compustat market cap rankings (this is what the paper does for the running variable)
- CRSP PRC: negative values indicate bid/ask midpoint — take `abs(PRC)` for price
- CRSP SHROUT: in thousands
- CRSP RET: decimal form (0.05 = 5%)
- Compustat columns are lowercase
- CCM link: LINKTYPE in ('LC', 'LU'), filter LINKPRIM in ('P', 'C') for primary links
- Date range: 1995-01 through 2024-12 (CRSP), 1995-01 through 2025-12 (Compustat)

## Constructing End-of-May Rankings (Paper Section 1.1)

This is the most critical data processing step. Follow this procedure for each year:

1. **Get end-of-May prices** from CRSP monthly (last trading day of May)
2. **Get shares outstanding** from Compustat quarterly (CSHOQ):
   - Use RDQ (earnings report date) to determine which quarter's CSHOQ was publicly known before May 31
   - For missing RDQ, apply SEC filing deadline rules:
     - Before 2003: 10-K within 90 days, 10-Q within 45 days
     - 2003–2005: 75 days for 10-K, 40 days for 10-Q (firms > $75M market cap)
     - After 2006: 60 days for 10-K (firms > $700M)
3. **Adjust shares** using CRSP FACSHR for corporate distributions between fiscal quarter-end and May 31
4. **Take the larger** of CRSP shares (SHROUT) and adjusted Compustat shares
5. **Compute market cap** = abs(PRC) × shares
6. **Rank all eligible stocks** by market cap (descending)
7. **Filter eligible stocks**: U.S. common stocks (SHRCD in [10, 11]), closing price ≥ $1.00, listed on NYSE/AMEX/NASDAQ

## Target Results to Replicate

### Table 3: First Stage
| Sample | α_0r | t-stat | R² | F |
|--------|------|--------|-----|---|
| Addition (pre-banding) | 0.785 | 31.50 | 0.863 | 1,876 |
| Addition (post-banding) | 0.820 | 12.98 | 0.845 | 297 |
| Deletion (pre-banding) | 0.705 | 29.15 | 0.817 | 1,799 |
| Deletion (post-banding) | 0.759 | 20.90 | 0.878 | 815 |

Note: Our first stage will show α_0r ≈ 1.0 and F → ∞ because D = τ (no actual Russell lists). This is expected.

### Table 4: Returns Fuzzy RD
| Effect | May | June | July | Aug | Sep |
|--------|-----|------|------|-----|-----|
| Addition | -0.003 | **0.050** (t=2.65) | -0.003 | 0.035 | 0.008 |
| Deletion | 0.005 | **0.054** (t=3.00) | -0.019 | -0.002 | 0.025 |

Note: Our ITT estimates will be attenuated by factor ~0.785 relative to the paper's LATE, plus additional attenuation from rank reconstruction noise (~25-30% misclassification near cutoff).

### Table 5: Volume Ratio and IO
| Effect | VR June | IO |
|--------|---------|-----|
| Addition | **0.478** (t=3.14) | 0.031 (t=0.77, n.s.) |
| Deletion | **-0.263** (t=-2.74) | -0.063 (t=-1.69, n.s.) |

### Table 6: Validity Tests
No significant discontinuities in: Mktcap, Repurchase, ROE, ROA, EPS, Assets, ICR, C/A

### Key Derived Statistics
- Price elasticity of demand: ~-1.5 (using benchmarked assets), ~-0.39 (using passive assets)
- %Demand change at cutoff: 7.3%
- Symmetric addition and deletion effects

## Variable Definitions (Paper Section 3)

- **Returns**: raw monthly stock return (RET from CRSP)
- **VR (Volume Ratio)**: VR_it = (V_it / V̄_i) / (V_mt / V̄_m), where V̄ is 6-month trailing average volume, excluding month t. NASDAQ volume adjusted using Gao and Ritter (2010) procedure (halve pre-2004).
- **SR (Short Ratio)**: shares shorted / shares outstanding (not yet pulled)
- **Comovement**: beta from regressing daily stock returns on Russell 2000 index daily returns within each month
- **IO**: institutional ownership from 13F filings (quarterly, not yet pulled)
- **ROE**: return on equity = NI / CEQ
- **ROA**: return on assets = NI / AT
- **EPS**: earnings per share excluding extraordinary items (EPSPX)
- **Assets**: total assets in millions (AT)
- **C/A**: cash-to-asset ratio = CHE / AT
- **ICR**: interest coverage ratio = OIBDP / XINT
- **Repurchase**: indicator = 1 if PRSTKC > 0
- **Float**: number of floating shares (from Russell, not available in our data)

## Project Structure

```
├── auxiliary/              # Helper functions
│   ├── __init__.py
│   ├── data_processing.py  # Data loading, merging, ranking, banding, VR, validity
│   ├── estimation.py       # Fuzzy RD 2SLS, time trends
│   └── plotting.py         # RD plots, binned scatter, time trend plots
├── data/                   # Raw data files (gitignored)
├── files/                  # Output figures and tables
├── memory/                 # MEMORY.md — persistent notes across sessions
├── tests/                  # Unit tests
├── project.ipynb           # Main notebook — all analysis here
├── CLAUDE.md               # This file — project guidance for Claude Code
├── environment.yml         # Conda environment
└── pyproject.toml          # Project config
```

## Style Guidelines

- Use Ruff for linting (configured in pyproject.toml)
- Helper functions go in `auxiliary/` modules, not inline in the notebook
- Notebook cells should be concise — call functions from auxiliary, don't put 100-line blocks inline
- Save all figures to `files/` directory
- Use descriptive variable names matching the paper's notation where possible