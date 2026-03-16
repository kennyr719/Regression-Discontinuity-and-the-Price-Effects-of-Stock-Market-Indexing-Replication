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
- Step 2 🔲: Modify identify_index_switchers() to accept bloomberg_file param and construct D_actual
- Step 3 🔲: Upgrade fuzzy_rd_estimate() for proper 2SLS (first stage: D_actual ~ τ + rank_centered)
- Step 4 🔲: Same for fuzzy_rd_time_trend()
- Step 5 🔲: Re-run full notebook
- Step 6 🔲: Update narrative cells for fuzzy RD framing

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

## Replication Results (with correct banding)
- Table 4 June addition: -0.60% (t=-0.37) vs paper's +5.0% (t=2.65) — WRONG SIGN (noise dominates attenuated ITT)
- Table 4 June deletion: +0.74% (t=+0.46) vs paper's +5.4% (t=3.00) — same sign, heavily attenuated
- May: addition -1.41% (paper -0.3%, NOT a close match), deletion -0.30% (paper +0.5%)
- Table 6 validity: 6/8 insignificant; 2 marginal rejections (repurchase-deletion t=-2.32, cash/assets-addition t=+2.39) consistent with multiple testing (16 tests x 0.05 = 0.8 expected)
- Deletion time trend 1996-2012: beta_2r = -0.495% (t=-2.61) — replicates paper's declining effect (STRONGEST RESULT)
- Extension 2015-2024 Addition: beta_0r = +8.357% (t=+1.51), beta_2r = -0.839% (t=-0.78), N=127 (very small)
- Extension 2015-2024 Deletion: beta_0r = -5.284% (t=-1.40), beta_2r = -0.340% (t=-0.65), N=279

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
## Pending Narrative Fixes
**OBSOLETE: These fixes were for the sharp RD framing. With Bloomberg data, the narrative should instead frame this as a proper fuzzy RD replication. See IMPLEMENTATION_PLAN.md Step 6.**