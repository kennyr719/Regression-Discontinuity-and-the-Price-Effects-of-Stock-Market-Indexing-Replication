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
- **Data pipeline**: `merge_crsp_compustat()` and `compute_market_cap_rankings()` — fully implemented and verified. Rankings for 1996–2024 are correct.
- **Sample construction**: `identify_index_switchers()` — builds addition/deletion panels using prior-year rank as membership proxy. Sets D = τ (sharp RD approximation).
- **Outcome variables**: `construct_outcome_variables()` (monthly returns) and `construct_volume_ratio()` (VR) are implemented.
- **Estimation**: `fuzzy_rd_estimate()` and `fuzzy_rd_time_trend()` — HC1-robust SEs via `S_white_simple`, optional `poly_degree=2`.
- **Bandwidth**: `optimal_bandwidth()` returns 100. `bandwidth_sensitivity()` tests h ∈ {50, 100, 150}.
- **Validity tests**: `construct_validity_variables()` merges Compustat annual fundamentals.
- **All notebook sections executed**: Sections 1–10 all have output.
- **Banding**: `compute_banding_cutoffs()` uses reverse cumulative market cap (footnote 5). Verified.
- **HC1 SEs**: Both estimation functions use HC1-robust standard errors.

### What STILL NEEDS FIXING 🔲
The notebook has **internal contradictions, stale numbers, and overclaims** in its markdown narrative cells and print-statement footers. All computation is correct — only the *presentation layer* needs fixing. See "Priority Fixes" below.

---

## Priority Fixes (IMPLEMENT THESE)

**Guiding principle**: Reframe the project from "replication that fell short" → **"methodological implementation with transparent diagnosis of why public-data sharp RD diverges from the original fuzzy RD."**

### Execution order and dependencies

```
Module 0 (backup)
    │
    ▼
Module 1 (markdown fixes) ← HIGHEST VALUE, ZERO RISK — no code changes
    │
    ├──► Module 2 (Table 4 print fix) — depends on M1 for wording
    ├──► Module 4 (validity note) — independent
    └──► Module 5 (extension cleanup) — independent
    │
    ▼
Module 3 (VR investigation) ← OPTIONAL, skip if time-constrained
    │
    ▼
Module 6 (final review) ← ALWAYS DO LAST
```

**Minimum viable delivery**: Modules 0 + 1 + 6.
**Full delivery**: All modules in order.

---

### MODULE 0: Snapshot

`cp project.ipynb project_BACKUP.ipynb`

---

### MODULE 1: Fix All Markdown Narrative Cells (NO code changes)

This module touches ONLY markdown cells. It cannot break any computation. It fixes every internal contradiction and overclaim.

#### Cell 0 (Introduction) — Reframe the project

Replace the current intro. Key points to include:
- We implement the CHL (2015) RD design using publicly available CRSP/Compustat data
- Because actual Russell constituent lists are proprietary and unavailable through our WRDS/FactSet subscriptions, we use a **sharp RD** approximation (D = τ) rather than the paper's fuzzy RD
- This yields Intent-to-Treat (ITT) estimates, mechanically attenuated relative to the paper's LATE
- The project serves as both a replication attempt and a **methodological case study** in how data access constraints propagate through RD estimation
- We extend the sample through 2024 to test whether index reconstitution price effects have grown or declined alongside the rise of passive investing

#### Cell 16 (Section 5 — First Stage) — Add sharp RD context

Add: "Because we set D = τ, the first stage is mechanically 1.0 by construction. We report it for completeness, and to highlight the gap vs. the paper's α₀ᵣ ≈ 0.785, which directly scales the second-stage estimates."

#### Cell 19 (Section 6 — Table 4 intro) — Reframe expectations

Replace generic "second-stage estimates the causal effect" with:
- Under sharp RD, β₀ᵣ recovers the ITT, not the LATE
- Two attenuation sources: (1) missing first-stage scaling (÷ 0.785 ≈ 27% gap), (2) rank reconstruction noise (~25–30% misclassification near cutoff)
- Combined attenuation of 50–70%; when misclassification is severe enough, can flip the sign of small effects toward noise
- Key diagnostic: the deletion time trend (within-estimator, less sensitive to level attenuation) should replicate

#### Cell 22 (Section 7 — VR intro) — Address mean VR = 1.397

Add: "The unconditional mean VR of ~1.4 reflects positive skew in trading volume near index reconstitution dates; the RD estimates are identified from the discontinuity at the cutoff, not the level."

#### Cell 24 (Section 8 — Validity intro) — Do NOT claim "all insignificant"

Current output shows repurchase (deletion) t = −2.32 and cash/assets (addition) t = +2.39 — both significant at 5%. The markdown must say: **"6 of 8 variables show no significant discontinuity. Two — repurchase activity (deletion) and cash-to-assets (addition) — are marginally significant at 5%. With 16 tests (8 variables × 2 samples), 1–2 rejections at 5% are expected by chance (16 × 0.05 = 0.8). We interpret the validity tests as broadly supportive."**

#### Cell 26 (Section 9 — Time Trends intro) — Keep, soften

Don't pre-commit to conclusions before showing estimates.

#### Cell 28 (Extension Conclusions) — FULL REWRITE (most critical)

This cell has the worst contradictions. **Numbers in Cell 28 do NOT match Cell 27 output.** Replace entirely using the ACTUAL numbers from Cell 27:

| Value | Cell 28 claims (WRONG) | Cell 27 output (CORRECT) |
|-------|----------------------|------------------------|
| Deletion 96–12 β₂ᵣ | −0.22% (t=−1.59) | −0.495% (t=−2.61) |
| Addition 15–24 β₀ᵣ | +4.63% (t=2.51) | +8.357% (t=+1.51) |
| Addition 15–24 N | 780 | 127 |
| Deletion 15–24 N | 1,023 | 279 |

New Cell 28 must:
1. Use only numbers from Cell 27 output
2. Acknowledge N=127 addition is very small
3. Use hedged language: "suggestive," "consistent with" — not "evidence leans toward"
4. Frame the deletion time trend 1996–2012 as the single most robust result

#### Cell 30 (Summary Table) — Reconcile all numbers

Every number must match its source cell output:

| Row | Current (wrong) | Correct (from cell output) |
|-----|-----------------|---------------------------|
| Addition June t-stat | −0.39 | −0.37 (Cell 20) |
| Deletion June t-stat | +0.50 | +0.46 (Cell 20) |
| VR addition t-stat | −1.46 | −1.00 (Cell 23) |
| VR deletion t-stat | −1.77 | −1.87 (Cell 23) |
| Validity tests | "All insignificant" | "6/8 insignificant; 2 marginal rejections consistent with multiple testing" |
| Addition June "Match?" | "Attenuated" | "Wrong sign — noise dominates attenuated ITT" |

Add a paragraph below the table: **"The strongest replication successes are: (1) the deletion time trend, which replicates the paper's declining price impact in both sign and significance; (2) the validity tests, which confirm no systematic manipulation at the cutoff; and (3) the deletion VR, which shows the correct sign. The headline June return effects are severely attenuated, as expected given D = τ and rank reconstruction error."**

---

### MODULE 2: Fix Table 4 Print Statements (Cell 20 — code, print only)

The current code prints `"May matches well"` — addition May is −1.41% vs paper's −0.3%, which is NOT matching well.

Replace the trailing print block with:
```python
print()
print("Paper targets (LATE, actual Russell lists):")
print("  Addition: May=-0.3%  Jun=+5.0% (t=2.65)  Jul=-0.3%  Aug=+3.5%  Sep=+0.8%")
print("  Deletion: May=+0.5%  Jun=+5.4% (t=3.00)  Jul=-1.9%  Aug=-0.2%  Sep=+2.5%")
print()
print("Note: Our sharp-RD ITT estimates are expected to be substantially attenuated")
print("(see Section 5 discussion). The addition June estimate is wrong-signed,")
print("consistent with rank misclassification dominating the small true effect")
print("near the cutoff. The deletion August coefficient is the strongest individual")
print("month result, suggesting some index-rebalancing signal survives the noise.")
```

**Acceptance test**: Re-run Cell 20. Table numbers unchanged. Footer text corrected.

---

### MODULE 3: Investigate VR Construction (OPTIONAL — skip if time-constrained)

Mean VR = 1.397; expected ~1.0. Diagnose whether this is a bug or selection effect.

Steps:
1. Print `vr_jun` distribution: median, 25th/75th, min, max
2. Check market-volume denominator normalization
3. If selection effect (reconstitution-adjacent stocks have higher volume) → document and move on
4. If code bug in `construct_volume_ratio()` → fix and re-run

**Damage control**: If rabbit hole, skip. The "positive skew" footnote from Module 1 is sufficient.

---

### MODULE 4: Add Multiple-Testing Note to Validity Tests (Cell 25 — code, print only)

After the table in Cell 25, append:
```python
print()
print("Note: 2 of 16 tests reject at 5% (repurchase-deletion, cash/assets-addition).")
print("Under independent tests, E[rejections] = 16 × 0.05 = 0.8.")
print("Two rejections is within the expected range under the null of no manipulation.")
```

**Acceptance test**: Re-run Cell 25. Table unchanged. Note appended.

---

### MODULE 5: Clean Up Extension Print Statements (Cell 27 — code, print only)

After the main comparison table, add diagnostic print block:
```python
print()
print("Sample sizes:")
print(f"  Addition 1996-2012: N={len(add_rep)}")
print(f"  Addition 2015-2024: N={len(add_ext)}")
print(f"  Deletion 1996-2012: N={len(del_rep)}")
print(f"  Deletion 2015-2024: N={len(del_ext)}")
print()
print("Interpretation caveats:")
print("  - 2015-2024 addition N is very small; estimates are noisy")
print("  - All estimates are ITT (D=tau), attenuated relative to paper's LATE")
print("  - Deletion time trend 1996-2012 is the most robust result")
```

Move the "Passive distortion / Arbitrage efficiency" print block to BEFORE the table (as framing, not conclusion).

**Acceptance test**: Re-run Cell 27. Table numbers unchanged. Diagnostics appended.

### MODULE 5b: Add Deletion Rolling Estimates to Figure 5 (Cell 29 — code)

Cell 29 currently only computes and plots addition rolling estimates. Add a second 
loop over `deletion_df` using the same 3-year rolling window and quality filter 
(N ≥ 15, ≥ 3 obs per side). Create a 2-panel figure (1×2 subplots):
- Left: "Addition Effect on June Returns" (existing)  
- Right: "Deletion Effect on June Returns" (new)

Save as `files/figure5_time_trends.png` (overwrite existing single-panel version).

The deletion panel should visually show a declining trend, reinforcing the 
β₂ᵣ = −0.495% finding from the time trend regression.
---

### MODULE 6: Final Consistency Review

Read the entire notebook top-to-bottom. Verify:
- [ ] Every number in Cell 30 summary matches its source cell output
- [ ] Cell 28 narrative uses only numbers from Cell 27 output
- [ ] No cell claims "all insignificant" when some are significant
- [ ] No cell claims results "match well" when they don't
- [ ] The word "attenuation" is never used to describe a sign flip (sign flips are from noise domination, not attenuation)
- [ ] Extension conclusions use hedged language
- [ ] VR mean is documented
- [ ] Introduction sets up sharp RD framing so weak results aren't surprising

---

### Cells modified by module

| File | Cell | Type | Module |
|------|------|------|--------|
| `project.ipynb` | 0 | markdown | M1 |
| `project.ipynb` | 16 | markdown | M1 |
| `project.ipynb` | 19 | markdown | M1 |
| `project.ipynb` | 20 | code (print only) | M2 |
| `project.ipynb` | 22 | markdown | M1 |
| `project.ipynb` | 24 | markdown | M1 |
| `project.ipynb` | 25 | code (print only) | M4 |
| `project.ipynb` | 26 | markdown | M1 |
| `project.ipynb` | 27 | code (print only) | M5 |
| `project.ipynb` | 28 | markdown | M1 |
| `project.ipynb` | 30 | markdown | M1 |
| `auxiliary/data_processing.py` | `construct_volume_ratio()` | M3 (only if bug found) |

---

### Previously completed fixes (for reference)

These are all done and should NOT be re-implemented:
- ✅ `compute_banding_cutoffs()` — reverse cumulative market cap (footnote 5)
- ✅ All notebook cells executed with output
- ✅ HC1-robust standard errors in both estimation functions
- ✅ `optimal_bandwidth()` returns 100; `bandwidth_sensitivity()` implemented
- ✅ Legacy template files deleted
- ✅ `plot_index_weights()` returns `None` with docstring

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