# Project Memory

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

## Attenuation Diagnosis
Two sources of attenuation vs. paper's results:
1. **D = τ (sharp RD)**: Paper uses actual Russell lists for D, instruments with τ. LATE = ITT/0.785. We set D = τ, so our estimate ≈ ITT, not LATE.
2. **Rank reconstruction noise**: ~25–30% of stocks near rank 1000 are misclassified vs Russell's proprietary rankings (share count methodology, float vs total shares, foreign issuers). Each misclassified stock pulls treatment-group average toward zero.

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
## Pending Narrative Fixes (see CLAUDE.md for full plan)
- Cell 28 (extension conclusions) has WRONG numbers from a previous run — must use Cell 27 output
- Cell 30 (summary table) has stale t-stats — must match actual cell outputs
- Cell 20 footer claims "May matches well" — it doesn't (addition May is 4.7x paper's)
- Cell 24/30 claim "all insignificant" validity — 2 of 16 tests reject at 5%
- Cell 0 intro needs sharp RD / methodological case study framing
- All fixes are markdown or print-statement only — no computation changes needed