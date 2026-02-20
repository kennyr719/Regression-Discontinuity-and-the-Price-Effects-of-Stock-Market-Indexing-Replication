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

## Python Environment
- Use /Users/kennyren/anaconda3/bin/python (base anaconda) — has pandas/numpy
- scipy in base anaconda is broken (ImportError on _spropack)
- russell-rd conda env does not exist yet (environment.yml present but not created)
- To import data_processing without triggering the broken scipy, use importlib.util directly
