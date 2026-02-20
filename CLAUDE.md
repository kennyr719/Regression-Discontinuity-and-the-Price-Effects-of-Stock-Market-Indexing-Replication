# Project Guidance for Claude Code

## Overview

This is a replication project for ECON 481 (Economics Data Science) at the University of Washington. We are replicating the main results from:

> Chang, Y.-C., Hong, H., & Liskovich, I. (2015). "Regression Discontinuity and the Price Effects of Stock Market Indexing." *The Review of Financial Studies*, 28(1), 212–246. DOI: 10.1093/rfs/hhu041

The paper uses a **fuzzy regression discontinuity (RD) design** to estimate the causal price effects of Russell index membership on stock returns around the Russell 1000/2000 cutoff.

## Extension

After replicating the original 1996–2012 results, we extend the sample to 2015–2024 to test whether the index premium has changed as passive investing's market share tripled (~15% to ~50%). This frames a test of two competing hypotheses:
- **Passive distortion hypothesis**: larger price effects due to more passive money
- **Arbitrage efficiency hypothesis**: smaller effects as arbitrage capacity scales alongside passive growth

## Key Methodology

### The RD Design
- Every year on the last trading day of May, stocks are ranked by market capitalization
- Ranks 1–1000 → Russell 1000; Ranks 1001–3000 → Russell 2000
- Because the Russell 2000 is value-weighted, stocks just below rank 1000 receive ~10x higher index weight than stocks just above
- This creates a discontinuity in passive buying pressure at the cutoff
- The paper uses a **fuzzy RD** because predicted rankings don't perfectly match actual Russell assignments

### Post-2007 Banding Policy
Starting with the 2007 reconstitution, Russell implemented a banding policy:
- A stock only switches indexes if its cumulative market cap deviates more than 2.5% from the 1000th stock's cumulative market cap in the Russell 3000E
- This means the effective cutoff shifts each year after 2007
- The function `compute_banding_cutoffs()` in `auxiliary/data_processing.py` handles this

### Fuzzy RD Specification

**First stage** (Equation 4 in paper):
```
D_it = α_0l + α_1l(r_it - c) + τ_it[α_0r + α_1r(r_it - c)] + ε_it
```
- D_it = actual Russell 2000 membership indicator
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
- t = years since 1996
- β_2r = how the treatment effect changes over time

### Bandwidth
- Default bandwidth: 100 ranks on each side of the cutoff
- Rule-of-thumb (ROT) from Lee and Lemieux (2010) generally gives ~100
- Local linear regression on each side of the cutoff

### Two Separate Samples
- **Addition effect**: Stocks in Russell 1000 in year t-1 that are near the cutoff in year t. Comparing those that just crossed into Russell 2000 vs. those that just missed.
- **Deletion effect**: Stocks in Russell 2000 in year t-1 that are near the cutoff in year t. Comparing those that stayed in Russell 2000 vs. those that moved to Russell 1000.

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

### Table 4: Returns Fuzzy RD
| Effect | May | June | July | Aug | Sep |
|--------|-----|------|------|-----|-----|
| Addition | -0.003 | **0.050** (t=2.65) | -0.003 | 0.035 | 0.008 |
| Deletion | 0.005 | **0.054** (t=3.00) | -0.019 | -0.002 | 0.025 |

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
- **VR (Volume Ratio)**: VR_it = (V_it / V̄_i) / (V_mt / V̄_m), where V̄ is 6-month trailing average volume, excluding month t. NASDAQ volume adjusted using Gao and Ritter (2010) procedure.
- **SR (Short Ratio)**: shares shorted / shares outstanding
- **Comovement**: beta from regressing daily stock returns on Russell 2000 index daily returns within each month
- **IO**: institutional ownership from 13F filings (quarterly)
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
│   ├── data_processing.py  # Data loading, merging, ranking construction
│   ├── estimation.py       # Fuzzy RD estimation, time trends
│   └── plotting.py         # RD plots, binned scatter plots
├── data/                   # Raw data files (gitignored)
├── files/                  # Output figures and tables
├── tests/                  # Unit tests
├── project.ipynb           # Main notebook — all analysis here
├── environment.yml         # Conda environment
└── pyproject.toml          # Project config
```

## Implementation Order

1. Load all datasets (Section 2 of notebook)
2. Merge CRSP and Compustat via CCM link
3. Construct end-of-May market cap rankings for each year 1996–2024
4. Identify addition and deletion samples using prior-year membership
5. Construct outcome variables (returns, VR, comovement)
6. Run first-stage regressions (Table 3)
7. Run fuzzy RD for returns (Table 4) and plot Figure 4
8. Run fuzzy RD for VR and IO (Table 5)
9. Run validity tests (Table 6)
10. Run time trend regressions (Tables 7–8) and plot Figure 5
11. Extension: compare 1996–2012 vs 2015–2024 results

## Style Guidelines

- Use Ruff for linting (configured in pyproject.toml)
- Helper functions go in `auxiliary/` modules, not inline in the notebook
- Notebook cells should be concise — call functions from auxiliary, don't put 100-line blocks inline
- Save all figures to `files/` directory
- Use descriptive variable names matching the paper's notation where possible
