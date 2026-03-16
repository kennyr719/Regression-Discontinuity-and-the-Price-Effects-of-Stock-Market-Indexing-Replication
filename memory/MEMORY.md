# Project Memory

## Bloomberg Constituent Data and match_bloomberg_to_crsp()

### Data file
- File: data/russell_constituents_clean.csv (86,774 rows, 1996-2024)
- Columns: year, bbg_ticker, ticker, index (R1000/R2000), cusip (9-digit), ncusip (8-digit)
- Coverage: 97.2% of stock-year observations have valid CUSIPs
- Zero overlap between R1000 and R2000 in any year (verified)

### Matching function: `match_bloomberg_to_crsp(bloomberg_file, ccm_link_df)`
- Returns: DataFrame with columns (year, PERMNO, D_actual); D_actual=1 if R2000, 0 if R1000
- **CRITICAL**: CRSP monthly does NOT include NCUSIP (not in WRDS pull). Cannot do NCUSIP matching.
- Primary match: Bloomberg ticker → CCM link tic (exact, uppercased), within links active in June of each year
- Secondary match: Bloomberg ticker → CCM tic with trailing `.N` suffix stripped (e.g. 'AAIC.1' → 'AAIC')
- Unmatched stocks are dropped (not in returned panel); identify_index_switchers() will see D_actual=NaN for these
- Deduplication: keep R2000 (D_actual=1) if same PERMNO matched twice
- Spot checks: AAPL (PERMNO 14593) = R1000 all years ✓; AAON (PERMNO 76868) = R2000 through 2023, R1000 in 2024 ✓

### Match rates by year (full table)
```
1996=55.1%  1997=56.8%  1998=59.6%  1999=62.6%  2000=66.0%
2001=69.1%  2002=71.5%  2003=73.4%  2004=74.5%  2005=76.0%
2006=77.7%  2007=80.2%  2008=82.5%  2009=83.9%  2010=84.7%
2011=85.8%  2012=86.7%  2013=87.5%  2014=88.8%  2015=88.7%
2016=90.2%  2017=91.7%  2018=92.6%  2019=93.7%  2020=95.0%
2021=95.9%  2022=96.6%  2023=97.3%  2024=97.6%
```
- Low early years: ~35% of 1996 Bloomberg stocks have "weird" placeholder tickers (e.g. '0111145D') that never match CCM; this fraction declines to ~11% by 2010
- Among normal-ticker stocks, ~13% additional mismatch from recently formed companies (BNI, COX, CMCSK absent from CCM in Jun 1996) or different ticker formats
- Near-cutoff (rank 900-1100) match rates are likely substantially higher since mid-cap US stocks have stable exchange tickers

### Fuzzy 2SLS upgrade progress (IMPLEMENTATION_PLAN.md)
- Step 0 ✅: project_BACKUP_pre_fuzzy.ipynb created
- Step 0a ✅: CLAUDE.md, AGENT_SUMMARY.md, MEMORY.md updated (sharp RD references removed)
- Step 1 ✅: match_bloomberg_to_crsp() implemented and verified
- Step 2 ✅: identify_index_switchers() now accepts bloomberg_panel=... param. Sets D=D_actual from Bloomberg, fallback D=τ for unmatched. D_actual column preserved.
- Step 3 ✅: fuzzy_rd_estimate() was already proper 2SLS — no code change. Wiring D_actual in was sufficient.
- Step 4 ✅: fuzzy_rd_time_trend() — same, already proper 2SLS.
- Step 5 🔲: Re-run full notebook (all cells need to be re-executed with Bloomberg data)
- Step 6 🔲: Update remaining narrative cells — especially Cell 30 summary table numbers

### First-Stage Results (Steps 2–4 complete, BW=100, 1996–2012)
- Addition pre-banding: α₀r=0.462, t=11.77, F=75, N=856  (paper: 0.785, F=1876)
- Addition post-banding: α₀r=0.215, t=1.19, F=1, N=85   (paper: 0.820, F=297) — unusable
- Deletion pre-banding: α₀r=0.476, t=16.09, F=182, N=1208 (paper: 0.705, F=1799)
- Deletion post-banding: α₀r=0.096, t=1.00, F=13, N=231  (paper: 0.759, F=815) — weak

**Root cause**: Asymmetric rank reconstruction noise. D=1,τ=0 (total share rank says R1000 but Russell says R2000): 9.2% of pre-banding addition obs. D=0,τ=1: only 2.9%. The asymmetry comes from total shares ≥ float shares → our ranks consistently overstate market cap → many stocks we rank ~950 are ranked ~1050 by Russell (genuinely R2000). Cannot fix without float data or NCUSIP for CUSIP-based matching.

## Data Facts
- CRSP monthly dates are the **last trading day** of each month (not necessarily the 31st).
  Filter by `date.dt.month == 5` not `date == f'{year}-05-31'`.
- SHRCD and EXCHCD are **int64** in full file load (no NaN in those columns).
- CRSP SHROUT is in **thousands** of shares. Market cap (millions) = PRC × SHROUT / 1000.
- Compustat CSHOQ is in **millions** of shares. To compare with SHROUT: cshoq × 1000.
- CCM link: LINKENDDT='E' means still active → replace with pd.Timestamp('2099-12-31').
- crsp_daily has 58M rows (679MB compressed). Defer loading until VR/comovement section.

## Ranking Construction Results (verified correct)
- Top 1996 stocks: GE, Coca-Cola, Exxon, AT&T, Philip Morris, Merck, Microsoft ✓
- Rank-1000 at cutoff: $1.07–2.18B for 1996–2012 replication period (71% in $1.3–2.5B)
- Years outside $1.3–2.5B are economically explained: early years pre-bubble, 2002-03 post-crash, 2009 crisis, post-2018 secular growth
- Universe size: 3,400–7,200 eligible stocks per year (larger in dot-com era, declining after)

## merge_crsp_compustat() returns a dict with keys
- 'crsp_monthly', 'compustat_quarterly', 'ccm_link', 'cfacshr_lookup'
- compute_market_cap_rankings(data, year) takes this dict

## Sample Sizes (with correct banding)
- Addition panel: 980 firm-year obs (1997–2024)
- Deletion panel: 1,690 firm-year obs (1997–2024)
- Pre-banding (1997–2006): Addition median 78/yr, Deletion median 113/yr
- Post-banding (2007–2024): Addition median 13/yr, Deletion median 29/yr

## compute_banding_cutoffs() — FIXED (reverse cumulation)
- Uses C_rev%(k) = fraction of total R3000E held by stocks ranked k through N (bottom-up cumulation)
- C_rev%(1000) ≈ 9–10%; band is C_rev%(1000) ± 0.025 (≈ [7.5%, 12.5%])
- Current cutoffs: k_add ≈ 1251–1545, k_del ≈ 738–823
- Verified against footnote 5 of Chang et al. (2015): stock 1210 in 2007 at C_rev%=7.24% stays in R1000 ✓
- Wide bands are intentional — Russell designed banding to significantly reduce turnover

## Replication Results (with correct banding) — STALE: sharp RD, pre-Bloomberg
These are from the D=τ run. Steps 2–4 are done; notebook needs re-execution (Step 5).
- Table 4 June addition: -0.60% (t=-0.37) vs paper's +5.0% (t=2.65) — wrong sign (old sharp RD)
- Table 4 June deletion: +0.74% (t=+0.46) vs paper's +5.4% (t=3.00) — same sign, attenuated
- Deletion time trend 1996-2012: beta_2r = -0.495% (t=-2.61) — replicates paper (STRONGEST RESULT)
- Extension 2015-2024: underpowered (N=127 addition, N=279 deletion)

## Attenuation Diagnosis (OLD sharp RD — now superseded by Bloomberg data)
**UPDATE: Bloomberg constituent data now available. D = τ constraint is resolved. The attenuation diagnosis below describes the OLD sharp RD results; the fuzzy 2SLS should recover estimates close to the paper's LATE.**
1. **D = τ (sharp RD)**: Paper uses actual Russell lists for D, instruments with τ. LATE = ITT/0.785. We now have D_actual from Bloomberg.
2. **Rank reconstruction noise**: ~25–30% of stocks near rank 1000 are misclassified vs Russell's proprietary rankings. This is now absorbed by the first stage.

## Python Environment
- Use /Users/kennyren/anaconda3/bin/python (base anaconda) — has pandas/numpy
- scipy in base anaconda is broken (ImportError on _spropack)
- estimation.py imports `from scipy.special import betainc` which works (stable submodule)
- russell-rd conda env does not exist yet (environment.yml present but not created)

## Files Cleaned Up ✓
- Deleted: auxiliary/plots.py, predictions.py, tables.py (template leftovers)
- Deleted: files/causalgraph1.PNG, causalgraph2.PNG, bounds_nextGPA.PNG (template images)
- Deleted: edit_notebook*.py, fix3_fix4.py (one-off scripts)
- README.md — updated with correct project description and extension info
## Estimation — current state
- fuzzy_rd_estimate() and fuzzy_rd_time_trend() already implement proper 2SLS (first stage: D ~ [1,r,τ,τ*r,FEs] → D_hat; second stage uses D_hat). HC1 SEs via S_white_simple. No changes needed for Steps 3-4.
- __init__.py exports: match_bloomberg_to_crsp, bandwidth_sensitivity, fuzzy_rd_time_trend

## Notebook cell changes (latest session — cells NOT re-executed yet)
- Cell 2: added match_bloomberg_to_crsp import
- Cell 9: rewritten to load bloomberg_panel and pass bloomberg_panel=... to identify_index_switchers(); also prints misclassification diagnostics
- Cell 16 (markdown): removed "Sharp RD note: D=τ by construction" paragraph; replaced with proper fuzzy RD framing
- Cell 17: removed "Note: D=τ here → α_0r≈1.0" footer
- Cell 19 (markdown): removed two-source attenuation / ITT framing; replaced with LATE framing
- Cell 20: removed "Note: Our sharp-RD ITT estimates are expected to be substantially attenuated" footer
- Cells 17, 20, 23, 25, 27, 29 outputs are STALE (show D=τ results); need Step 5 re-run
- Cell 30 (summary table markdown) is also STALE — needs update with new numbers after re-run