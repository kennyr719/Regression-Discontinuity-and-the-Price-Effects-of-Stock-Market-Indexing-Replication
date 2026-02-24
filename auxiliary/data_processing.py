"""Functions for data acquisition, cleaning, and processing."""

import numpy as np
import pandas as pd


def merge_crsp_compustat(crsp_monthly, compustat_quarterly, ccm_link):
    """Pre-process CRSP monthly, Compustat quarterly, and CCM link data.

    Cleans raw files, filters the CCM link to valid primary links,
    pre-computes Compustat filing availability dates, and builds a
    CFACSHR lookup table for the share-split adjustment step.

    Parameters
    ----------
    crsp_monthly : pd.DataFrame
        CRSP monthly stock data with columns including PERMNO, date,
        PRC, SHROUT, SHRCD, EXCHCD, CFACSHR.
    compustat_quarterly : pd.DataFrame
        Compustat quarterly data with gvkey, datadate, cshoq, rdq,
        fqtr, fyearq, and standard-format filter columns.
    ccm_link : pd.DataFrame
        CRSP/Compustat Merged linking table with gvkey, LPERMNO,
        LINKTYPE, LINKPRIM, LINKDT, LINKENDDT.

    Returns
    -------
    dict
        Keys:
        - 'crsp_monthly': cleaned CRSP monthly (dates parsed, PRC abs-valued)
        - 'compustat_quarterly': Compustat with 'available_date' column added
        - 'ccm_link': filtered CCM link (valid types, primary, dates parsed)
        - 'cfacshr_lookup': DataFrame with (PERMNO, ym) → CFACSHR for
          adjusting Compustat shares for splits between quarter-end and May 31
    """
    # --- CCM link: valid primary links only ---
    link = ccm_link[
        ccm_link["LINKTYPE"].isin(["LC", "LU"]) & ccm_link["LINKPRIM"].isin(["P", "C"])
    ].copy()
    link["LINKDT"] = pd.to_datetime(link["LINKDT"], errors="coerce")
    # 'E' means the link is still active (no end date); treat as far future
    link["LINKENDDT"] = link["LINKENDDT"].replace("E", None)
    link["LINKENDDT"] = pd.to_datetime(link["LINKENDDT"], errors="coerce")
    link["LINKENDDT"] = link["LINKENDDT"].fillna(pd.Timestamp("2099-12-31"))

    # --- CRSP monthly: parse dates, abs(PRC) ---
    crsp_m = crsp_monthly.copy()
    crsp_m["date"] = pd.to_datetime(crsp_m["date"])
    crsp_m["PRC"] = crsp_m["PRC"].abs()

    # --- Compustat quarterly: keep standard industrial consolidated USD ---
    cq = compustat_quarterly.copy()
    for col, val in [
        ("datafmt", "STD"),
        ("indfmt", "INDL"),
        ("consol", "C"),
        ("curcdq", "USD"),
    ]:
        if col in cq.columns:
            cq = cq[cq[col] == val]

    cq["datadate"] = pd.to_datetime(cq["datadate"])
    cq["rdq"] = pd.to_datetime(cq["rdq"], errors="coerce")

    # Vectorized SEC filing deadline (days after fiscal quarter-end).
    # If RDQ is missing we estimate when the filing was due:
    #   Annual (fqtr==4): 90 days pre-2003, 75 days 2003-05, 60 days post-2005
    #   Quarterly:        45 days pre-2003, 40 days post-2003
    is_annual = cq["fqtr"] == 4
    pre_2003 = cq["fyearq"] < 2003
    mid_2003_2005 = (cq["fyearq"] >= 2003) & (cq["fyearq"] <= 2005)

    deadline_days = np.select(
        [
            is_annual & pre_2003,
            is_annual & mid_2003_2005,
            is_annual,
            ~is_annual & pre_2003,
            ~is_annual,
        ],
        [90, 75, 60, 45, 40],
        default=45,
    )
    cq["estimated_rdq"] = cq["datadate"] + pd.to_timedelta(deadline_days, unit="D")
    # Use actual RDQ when available; fall back to estimated filing deadline
    cq["available_date"] = cq["rdq"].fillna(cq["estimated_rdq"])

    # --- CFACSHR lookup: (PERMNO, YYYYMM int) → CFACSHR ---
    # Used to adjust Compustat shares for stock splits between fiscal
    # quarter-end and May 31 (ratio = CFACSHR_may / CFACSHR_qtr_end).
    cfacshr_lookup = (
        crsp_m[["PERMNO", "date", "CFACSHR"]]
        .assign(ym=lambda x: x["date"].dt.year * 100 + x["date"].dt.month)
        .drop(columns=["date"])
        .drop_duplicates(["PERMNO", "ym"])
    )

    return {
        "crsp_monthly": crsp_m,
        "compustat_quarterly": cq,
        "ccm_link": link,
        "cfacshr_lookup": cfacshr_lookup,
    }


def compute_market_cap_rankings(data, year):
    """Compute end-of-May market capitalization rankings for a given year.

    Follows Chang et al. (2015, Section 1.1):
    1. End-of-May closing price from CRSP monthly.
    2. Most recent Compustat CSHOQ publicly available before May 31,
       using RDQ or estimated SEC filing deadline.
    3. Adjust Compustat shares for splits/distributions (CFACSHR ratio).
    4. Take the larger of CRSP SHROUT and adjusted Compustat shares.
    5. Market cap = abs(PRC) × shares; rank descending.

    Eligible stocks: SHRCD in {10, 11}, EXCHCD in {1, 2, 3} (NYSE/AMEX/NASDAQ),
    closing price ≥ $1.00.

    Parameters
    ----------
    data : dict
        Cleaned data dict returned by merge_crsp_compustat().
    year : int
        Reconstitution year (Russell ranks end-of-May market caps).

    Returns
    -------
    pd.DataFrame
        Ranked firms sorted by rank (ascending), with columns:
        PERMNO, gvkey, date, PRC, SHROUT, shares, market_cap (millions), rank.
    """
    crsp_monthly = data["crsp_monthly"]
    compustat_quarterly = data["compustat_quarterly"]
    ccm_link = data["ccm_link"]
    cfacshr_lookup = data["cfacshr_lookup"]

    # May 31 used as the cutoff for Compustat availability.
    # CRSP monthly uses the actual last trading day of May (may differ from May 31).
    may_cutoff = pd.Timestamp(f"{year}-05-31")

    # ------------------------------------------------------------------
    # Step 1: Filter CRSP to end-of-May eligible stocks
    # ------------------------------------------------------------------
    may_crsp = crsp_monthly[
        (crsp_monthly["date"].dt.year == year)
        & (crsp_monthly["date"].dt.month == 5)
        & crsp_monthly["SHRCD"].isin([10, 11])
        & crsp_monthly["EXCHCD"].isin([1, 2, 3])
        & (crsp_monthly["PRC"] >= 1.0)
    ].copy()
    # CRSP occasionally has duplicate rows for the same PERMNO in a month
    # (e.g., data entry errors or corporate action timing).  Keep the row
    # with the highest absolute price to prefer the most liquid record.
    may_crsp = (
        may_crsp
        .sort_values("PRC", ascending=False)
        .drop_duplicates("PERMNO", keep="first")
    )

    # ------------------------------------------------------------------
    # Step 2: Attach GVKEY via CCM links active on the reconstitution date
    # ------------------------------------------------------------------
    # A PERMNO can have multiple active CCM links on the same date (e.g.,
    # when a company is involved in a merger and both old and new GVKEYs
    # have overlapping link intervals).  Keep at most one GVKEY per PERMNO
    # by preferring LINKPRIM='P' (primary link) over 'C' (calendar link).
    active_links = ccm_link[
        (ccm_link["LINKDT"] <= may_cutoff) & (ccm_link["LINKENDDT"] >= may_cutoff)
    ][["LPERMNO", "gvkey", "LINKPRIM"]].rename(columns={"LPERMNO": "PERMNO"})

    active_links = (
        active_links
        .sort_values("LINKPRIM", key=lambda s: s.map({"P": 0, "C": 1}))
        .drop_duplicates("PERMNO", keep="first")
        .drop(columns=["LINKPRIM"])
    )

    may_crsp = may_crsp.merge(active_links, on="PERMNO", how="left")

    # ------------------------------------------------------------------
    # Step 3: Get most recent Compustat quarter publicly available by May 31
    # ------------------------------------------------------------------
    cq_avail = compustat_quarterly[
        (compustat_quarterly["datadate"] < may_cutoff)
        & (compustat_quarterly["available_date"] < may_cutoff)
        & compustat_quarterly["cshoq"].notna()
    ]

    # For each gvkey, keep the row with the latest datadate
    cq_latest = cq_avail.loc[
        cq_avail.groupby("gvkey")["datadate"].idxmax()
    ][["gvkey", "datadate", "cshoq"]].rename(
        columns={"datadate": "qtr_end_date", "cshoq": "cshoq_millions"}
    )

    may_crsp = may_crsp.merge(cq_latest, on="gvkey", how="left")

    # ------------------------------------------------------------------
    # Step 4: Adjust Compustat shares for splits/distributions
    #
    # CRSP CFACSHR convention: split-adjusted shares = SHROUT × CFACSHR.
    # The factor is set so that SHROUT × CFACSHR is constant through splits
    # (e.g., a 2:1 forward split doubles SHROUT and halves CFACSHR; a 10:1
    # reverse split reduces SHROUT by 10× and multiplies CFACSHR by 10).
    #
    # To express Compustat CSHOQ (as of fiscal quarter end) on the same
    # share-count basis as May 31 CRSP SHROUT:
    #
    #   compustat_shares_may = CSHOQ_qtr × (CFACSHR_qtr / CFACSHR_may) × 1000
    #
    # This is the ratio CFACSHR_qtr / CFACSHR_may (NOT may/qtr).  Using the
    # inverted ratio would inflate shares by the square of the split factor
    # for reverse splits (e.g., ×100 instead of ×0.1 for a 10:1 reverse).
    # ------------------------------------------------------------------
    # Build year-month integer key for the fiscal quarter-end
    has_qtr = may_crsp["qtr_end_date"].notna()
    may_crsp["qtr_ym"] = 0  # dummy — won't match any real CFACSHR entry
    may_crsp.loc[has_qtr, "qtr_ym"] = (
        may_crsp.loc[has_qtr, "qtr_end_date"].dt.year * 100
        + may_crsp.loc[has_qtr, "qtr_end_date"].dt.month
    ).astype(int)

    cfacshr_at_qtr = cfacshr_lookup.rename(
        columns={"ym": "qtr_ym", "CFACSHR": "cfacshr_qtr"}
    )
    may_crsp = may_crsp.merge(
        cfacshr_at_qtr[["PERMNO", "qtr_ym", "cfacshr_qtr"]],
        on=["PERMNO", "qtr_ym"],
        how="left",
    )

    # Split-adjustment ratio: CFACSHR_qtr / CFACSHR_may
    # Default to 1.0 when either factor is missing or zero.
    cfacshr_may = may_crsp["CFACSHR"].replace(0, np.nan)
    cfacshr_qtr = may_crsp["cfacshr_qtr"].replace(0, np.nan)
    split_adj = (cfacshr_qtr / cfacshr_may).fillna(1.0)

    # Compustat shares in thousands (CSHOQ is in millions)
    compustat_shares_thou = may_crsp["cshoq_millions"] * 1000 * split_adj

    # ------------------------------------------------------------------
    # Step 5: Take the larger of CRSP SHROUT and adjusted Compustat shares
    # ------------------------------------------------------------------
    crsp_shares = may_crsp["SHROUT"].astype(float)
    shares = crsp_shares.copy()
    has_comp = compustat_shares_thou.notna()
    shares[has_comp] = np.maximum(crsp_shares[has_comp], compustat_shares_thou[has_comp])
    may_crsp["shares"] = shares  # thousands of shares

    # ------------------------------------------------------------------
    # Step 6: Market cap in millions USD
    #   PRC ($/share) × shares (thousands) / 1000 = millions USD
    # ------------------------------------------------------------------
    may_crsp["market_cap"] = may_crsp["PRC"] * may_crsp["shares"] / 1000

    # ------------------------------------------------------------------
    # Step 7: Rank by market cap (descending); drop zero/missing
    # ------------------------------------------------------------------
    ranked = may_crsp.dropna(subset=["market_cap"]).query("market_cap > 0").copy()
    ranked["rank"] = (
        ranked["market_cap"].rank(ascending=False, method="first").astype(int)
    )

    out_cols = [
        "PERMNO", "gvkey", "date", "PRC", "SHROUT", "shares",
        "market_cap", "rank", "SHRCD", "EXCHCD",
    ]
    return ranked[out_cols].sort_values("rank").reset_index(drop=True)


def identify_index_switchers(
    all_rankings, year, cutoff=1000, bandwidth=100,
    addition_cutoff=None, deletion_cutoff=None,
):
    """Identify firms switching between Russell 1000 and Russell 2000.

    Separates stocks near the rank cutoff in year t into two samples based
    on their prior-year (t-1) index membership (proxied by reconstructed
    rankings, since actual Russell constituent lists are unavailable):

    - Addition sample: stocks ranked ≤ cutoff in year t-1 (in Russell 1000)
      and within [addition_cutoff-bandwidth, addition_cutoff+bandwidth] in
      year t.  Those with rank > addition_cutoff crossed into Russell 2000.

    - Deletion sample: stocks ranked > cutoff in year t-1 (in Russell 2000)
      and within [deletion_cutoff-bandwidth, deletion_cutoff+bandwidth] in
      year t.  Those with rank ≤ deletion_cutoff crossed into Russell 1000.

    Pre-banding (≤ 2006): addition_cutoff = deletion_cutoff = cutoff = 1000.
    Post-banding (≥ 2007): pass banding-adjusted cutoffs from
    compute_banding_cutoffs(); addition_cutoff > 1000, deletion_cutoff < 1000.

    Because actual Russell constituent lists are not available, prior-year
    index membership is proxied by prior-year reconstructed rankings, and
    current-year actual membership (D) is set equal to the rank-based
    instrument (τ). This yields a reduced-form / sharp-RD estimator; the
    true fuzzy-RD LATE would scale by the inverse first-stage coefficient.

    Parameters
    ----------
    all_rankings : dict
        Mapping {year: pd.DataFrame} from compute_market_cap_rankings().
        Must contain entries for both `year` and `year - 1`.
    year : int
        Reconstitution year t.
    cutoff : int, optional
        Nominal rank cutoff for prior-year membership (always 1000).
    bandwidth : int, optional
        Number of ranks on each side of the effective cutoff (default: 100).
    addition_cutoff : int, optional
        Effective rank cutoff for the addition sample.  If None, defaults to
        `cutoff`.  Post-2007: compute via compute_banding_cutoffs().
    deletion_cutoff : int, optional
        Effective rank cutoff for the deletion sample.  If None, defaults to
        `cutoff`.  Post-2007: compute via compute_banding_cutoffs().

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (addition_sample, deletion_sample).  Each DataFrame contains:
        PERMNO, gvkey, date, PRC, SHROUT, shares, market_cap, rank,
        rank_centered, tau, D, prev_rank, year.
        Returns (None, None) if prior-year rankings are unavailable.
    """
    if year - 1 not in all_rankings:
        return None, None

    _add_c = addition_cutoff if addition_cutoff is not None else cutoff
    _del_c = deletion_cutoff if deletion_cutoff is not None else cutoff

    curr = all_rankings[year]
    prev = all_rankings[year - 1]

    # Prior-year rank: one row per PERMNO, lowest rank number (= largest firm)
    prev_rank = (
        prev.groupby("PERMNO", sort=False)["rank"]
        .min()
        .reset_index()
        .rename(columns={"rank": "prev_rank"})
    )

    # Inner join: keep only stocks present in both years
    merged = curr.merge(prev_rank, on="PERMNO", how="inner")
    merged["year"] = year

    # ── Addition sample ──────────────────────────────────────────────────────
    # Stocks that were in Russell 1000 (prev_rank ≤ cutoff) and are now
    # within bandwidth of the addition cutoff.
    # τ = 1 if rank > _add_c (predicted to cross into R2000).
    add_bw = (
        (merged["rank"] >= _add_c - bandwidth)
        & (merged["rank"] <= _add_c + bandwidth)
    )
    addition = merged[add_bw & (merged["prev_rank"] <= cutoff)].copy()
    addition["rank_centered"] = addition["rank"] - _add_c
    addition["tau"] = (addition["rank"] > _add_c).astype(int)
    addition["D"] = addition["tau"]

    # ── Deletion sample ──────────────────────────────────────────────────────
    # Stocks that were in Russell 2000 (prev_rank > cutoff) and are now
    # within bandwidth of the deletion cutoff.
    # τ = 1 if rank > _del_c (predicted to stay in R2000).
    del_bw = (
        (merged["rank"] >= _del_c - bandwidth)
        & (merged["rank"] <= _del_c + bandwidth)
    )
    deletion = merged[del_bw & (merged["prev_rank"] > cutoff)].copy()
    deletion["rank_centered"] = deletion["rank"] - _del_c
    deletion["tau"] = (deletion["rank"] > _del_c).astype(int)
    deletion["D"] = deletion["tau"]

    return addition, deletion


def compute_banding_cutoffs(rankings_df, year):
    """Compute banding-adjusted cutoffs for post-2007 reconstitutions.

    Starting with the 2007 reconstitution, Russell implemented a banding
    policy to reduce unnecessary index turnover.  A stock only switches
    indexes if its *reverse cumulative market cap percentage* deviates by
    more than 2.5 percentage points from that of the 1000th-ranked stock.

    Define C_rev%(k) = Σ_{i=k}^{N} mktcap_i / total_R3000E, i.e., the
    fraction of total Russell 3000E market cap held by stocks ranked k
    through N (bottom-up cumulation).  Typically C_rev%(1000) ≈ 9–10%.

    Per footnote 5 of Chang et al. (2015):
    - A R1000 incumbent at rank k > 1000 switches to R2000 only if
      C_rev%(k) < C_rev%(1000) − 0.025  (i.e., below ~7.5%)
    - A R2000 incumbent at rank k < 1000 switches to R1000 only if
      C_rev%(k) > C_rev%(1000) + 0.025  (i.e., above ~12.5%)

    Addition cutoff (k_add): last rank > 1000 where C_rev% is still
    within the band; k_add + 1 is the first rank that actually switches.

    Deletion cutoff (k_del): first rank < 1000 where C_rev% is still
    within the band; k_del - 1 is the last rank that actually switches.

    Parameters
    ----------
    rankings_df : pd.DataFrame
        End-of-May rankings from compute_market_cap_rankings().  Must
        include 'rank' (int, 1 = largest market cap) and 'market_cap'
        (float, millions USD) columns.
    year : int
        Reconstitution year (informational; banding applies for year ≥ 2007).

    Returns
    -------
    tuple[int, int]
        (addition_cutoff, deletion_cutoff).  Both equal 1000 if fewer than
        1000 stocks are in the universe (degenerate fallback).
    """
    df = rankings_df.sort_values("rank").copy()

    if not (df["rank"] == 1000).any():
        return 1000, 1000  # degenerate: not enough stocks

    total_mktcap = df["market_cap"].sum()
    if total_mktcap <= 0:
        return 1000, 1000

    # Reverse cumulation: sum market caps from rank k to N (bottom-up)
    df_rev = df.sort_values("rank", ascending=False)
    df_rev["cum_mktcap_rev"] = df_rev["market_cap"].cumsum()
    df["cum_pct_rev"] = df_rev.set_index("rank")["cum_mktcap_rev"].reindex(df["rank"].values).values / total_mktcap

    c_rev_1000 = float(df.loc[df["rank"] == 1000, "cum_pct_rev"].iloc[0])

    # Addition cutoff: last rank > 1000 still within the band
    # (C_rev%(k) >= C_rev%(1000) - 0.025 means protected)
    above = df[df["rank"] > 1000]
    protected = above[above["cum_pct_rev"] >= c_rev_1000 - 0.025]
    addition_cutoff = (
        int(protected["rank"].max()) + 1 if len(protected) > 0
        else 1001
    )

    # Deletion cutoff: first rank < 1000 still within the band
    # (C_rev%(k) <= C_rev%(1000) + 0.025 means protected)
    below = df[df["rank"] < 1000]
    protected_del = below[below["cum_pct_rev"] <= c_rev_1000 + 0.025]
    deletion_cutoff = (
        int(protected_del["rank"].min()) - 1 if len(protected_del) > 0
        else 999
    )

    return addition_cutoff, deletion_cutoff


def construct_validity_variables(compustat_annual, addition_df, deletion_df):
    """Attach prior-year Compustat annual fundamentals to addition/deletion panels.

    For each firm-year (reconstitution year t) in the panels, finds the most
    recent annual Compustat observation whose fiscal year ended before May 31
    of year t and whose 10-K filing was estimated to be available by May 31.

    Filing deadline (estimated as days after fiscal year end):
      - fyear < 2003: 90 days
      - 2003 ≤ fyear ≤ 2005: 75 days
      - fyear > 2005: 60 days

    Computed variables (paper Table 6):
      roe       : NI / CEQ (NaN if CEQ ≤ 0)
      roa       : NI / AT  (NaN if AT ≤ 0)
      eps       : EPSPX (earnings per share excl. extraordinary items)
      assets    : AT (total assets, millions USD)
      icr       : OIBDP / XINT (NaN if XINT ≤ 0)
      ca        : CHE / AT (NaN if AT ≤ 0)
      repurchase: 1 if PRSTKC > 0, else 0

    Market cap (already in both panels as 'market_cap') is tested separately
    in the notebook; it does not require a Compustat merge.

    Parameters
    ----------
    compustat_annual : pd.DataFrame
        Raw Compustat annual data with columns: gvkey, datadate, at, ceq,
        che, epspx, ni, oibdp, xint, prstkc.  Additional format filter
        columns (datafmt, indfmt, consol, curcd) are applied if present.
    addition_df : pd.DataFrame
        Addition panel (output of identify_index_switchers() merged with
        returns).  Must contain 'gvkey' and 'year'.
    deletion_df : pd.DataFrame
        Deletion panel (same structure as addition_df).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (addition_validity, deletion_validity): each panel with validity
        columns appended.  Firms without a Compustat annual match keep their
        row but get NaN for all validity columns.
    """
    VALIDITY_VARS = ["roe", "roa", "eps", "assets", "icr", "ca", "repurchase"]

    # ── Prepare annual data ───────────────────────────────────────────────
    ca = compustat_annual.copy()

    # Apply standard Compustat format filters when columns are present
    for col, val in [
        ("datafmt", "STD"),
        ("indfmt", "INDL"),
        ("consol", "C"),
        ("curcd", "USD"),
    ]:
        if col in ca.columns:
            ca = ca[ca[col] == val]

    ca["datadate"] = pd.to_datetime(ca["datadate"])

    # Estimated 10-K filing deadline: depends on fiscal year
    fy = ca["datadate"].dt.year
    deadline_days = np.where(fy < 2003, 90, np.where(fy <= 2005, 75, 60))
    ca["available_date"] = ca["datadate"] + pd.to_timedelta(deadline_days, unit="D")

    # ── Compute derived validity variables ────────────────────────────────
    at   = ca["at"].where(ca["at"].fillna(0) > 0)    # AT > 0 required for denominator
    ceq  = ca["ceq"].where(ca["ceq"].fillna(0) > 0)  # CEQ > 0 for ROE
    xint = ca["xint"].where(ca["xint"].fillna(0) > 0)

    ca["roe"]       = ca["ni"] / ceq
    ca["roa"]       = ca["ni"] / at
    ca["eps"]       = ca["epspx"]
    ca["assets"]    = ca["at"]
    ca["icr"]       = ca["oibdp"] / xint
    ca["ca"]        = ca["che"] / at
    ca["repurchase"] = (ca["prstkc"].fillna(0) > 0).astype(int)

    keep = ["gvkey", "datadate", "available_date"] + VALIDITY_VARS
    ca = ca[keep].dropna(subset=["gvkey"])

    # ── For each reconstitution year, get most recent annual obs per gvkey ─
    all_years = sorted(set(addition_df["year"]) | set(deletion_df["year"]))
    validity_by_year = {}
    for year in all_years:
        may31 = pd.Timestamp(f"{year}-05-31")
        avail = ca[(ca["datadate"] < may31) & (ca["available_date"] < may31)]
        if len(avail) == 0:
            validity_by_year[year] = pd.DataFrame(columns=["gvkey"] + VALIDITY_VARS)
            continue
        latest_idx = avail.groupby("gvkey")["datadate"].idxmax()
        latest = (
            avail.loc[latest_idx]
            .drop(columns=["datadate", "available_date"])
            .copy()
        )
        validity_by_year[year] = latest

    # ── Merge into each panel ─────────────────────────────────────────────
    def _merge_validity(panel):
        parts = []
        for year, sub in panel.groupby("year", sort=True):
            vdf = validity_by_year.get(year, pd.DataFrame(columns=["gvkey"]))
            parts.append(sub.merge(vdf, on="gvkey", how="left"))
        return pd.concat(parts, ignore_index=True)

    return _merge_validity(addition_df), _merge_validity(deletion_df)


def construct_volume_ratio(data, year, months=(5, 6, 7, 8, 9)):
    """Construct monthly volume ratio (VR) for the fuzzy RD regressions.

    VR_it = (V_it / V̄_i) / (V_mt / V̄_m)

    where:
      V_it  = stock i's adjusted volume in month t
      V̄_i  = mean of stock i's adjusted volume over the 6 months before t
      V_mt  = aggregate market volume (sum over eligible stocks) in month t
      V̄_m  = mean of V_ms over the 6 months before t

    NASDAQ volume is halved pre-2004 to correct for double-counting of
    dealer trades (Gao & Ritter, 2010).  Uses CRSP monthly VOL — daily
    data is not required.  Stocks with zero trailing average volume are
    excluded (VR undefined).

    Parameters
    ----------
    data : dict
        Cleaned data dict from merge_crsp_compustat(), with 'crsp_monthly'.
        Must contain VOL, EXCHCD, SHRCD columns.
    year : int
        Reconstitution year.
    months : tuple of int, optional
        Calendar months to compute VR for (default: May–September).

    Returns
    -------
    pd.DataFrame
        Wide-format: PERMNO, year, vr_may, vr_jun, vr_jul, vr_aug, vr_sep.
        NaN where VR is undefined (no trailing volume or delisted).
    """
    crsp_m = data["crsp_monthly"].copy()
    crsp_m["yr"] = crsp_m["date"].dt.year
    crsp_m["mo"] = crsp_m["date"].dt.month
    crsp_m["VOL"] = pd.to_numeric(crsp_m["VOL"], errors="coerce").fillna(0.0)

    # NASDAQ adjustment (Gao & Ritter 2010): halve volume pre-2004
    crsp_m["vol_adj"] = np.where(
        (crsp_m["EXCHCD"] == 3) & (crsp_m["yr"] < 2004),
        crsp_m["VOL"] / 2.0,
        crsp_m["VOL"],
    )

    # Restrict to eligible U.S. common stocks on NYSE/AMEX/NASDAQ
    eligible = crsp_m[
        crsp_m["SHRCD"].isin([10, 11]) & crsp_m["EXCHCD"].isin([1, 2, 3])
    ].copy()

    # Only keep year-1 and year (at most 13 months of data needed)
    eligible = eligible[eligible["yr"].isin([year - 1, year])].copy()

    # Integer year-month key for fast filtering
    eligible["ym"] = eligible["yr"] * 100 + eligible["mo"]

    # Monthly aggregate market volume indexed by ym integer
    mkt_monthly = eligible.groupby("ym")["vol_adj"].sum()

    month_names = {5: "vr_may", 6: "vr_jun", 7: "vr_jul", 8: "vr_aug", 9: "vr_sep"}
    series_list = []

    for target_month in months:
        col_name = month_names.get(target_month, f"vr_m{target_month:02d}")

        # Build 6 trailing ym integer keys
        trailing_yms = []
        m, y = target_month - 1, year
        for _ in range(6):
            if m == 0:
                m, y = 12, y - 1
            trailing_yms.append(y * 100 + m)
            m -= 1

        # Market VR component
        target_ym = year * 100 + target_month
        mkt_current = float(mkt_monthly.get(target_ym, 0.0))
        mkt_trail_avg = float(np.mean([mkt_monthly.get(ym, 0.0) for ym in trailing_yms]))
        if mkt_trail_avg <= 0:
            series_list.append(pd.Series(name=col_name, dtype=float))
            continue
        mkt_vr = mkt_current / mkt_trail_avg

        # Stock current volume
        cur = (
            eligible[eligible["ym"] == target_ym][["PERMNO", "vol_adj"]]
            .drop_duplicates("PERMNO")
            .rename(columns={"vol_adj": "vol_current"})
        )

        # Stock trailing average (mean across available trailing months)
        trail = (
            eligible[eligible["ym"].isin(trailing_yms)][["PERMNO", "vol_adj"]]
            .groupby("PERMNO")["vol_adj"]
            .mean()
            .rename("vol_trail_avg")
            .reset_index()
        )

        merged = cur.merge(trail, on="PERMNO", how="inner")
        valid = merged[merged["vol_trail_avg"] > 0].copy()
        valid[col_name] = (valid["vol_current"] / valid["vol_trail_avg"]) / mkt_vr

        series_list.append(valid.set_index("PERMNO")[col_name])

    if not series_list:
        return pd.DataFrame(columns=["PERMNO", "year"])

    result = pd.concat(series_list, axis=1).reset_index()
    result["year"] = year
    return result


def construct_outcome_variables(data, year, months=(5, 6, 7, 8, 9)):
    """Construct monthly return outcome variables for the fuzzy RD regressions.

    Extracts CRSP monthly returns for May through September of the
    reconstitution year in wide format (one row per PERMNO).  Designed to
    be merged with addition/deletion panels from identify_index_switchers()
    on ['PERMNO', 'year'].

    Volume ratio (VR) and comovement beta require CRSP daily data and are
    not constructed here; they should be computed separately when crsp_daily
    is loaded.

    Parameters
    ----------
    data : dict
        Cleaned data dict from merge_crsp_compustat(), with 'crsp_monthly'.
    year : int
        Reconstitution year.
    months : tuple of int, optional
        Calendar months to extract (default: May–September, i.e. 5–9).

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame: PERMNO, year, ret_may, ret_jun, ret_jul,
        ret_aug, ret_sep (or ret_mNN for unlisted months).
        NaN where a stock has no return for that month (e.g. delisted).
    """
    crsp_m = data["crsp_monthly"]

    month_cols = {5: "ret_may", 6: "ret_jun", 7: "ret_jul", 8: "ret_aug", 9: "ret_sep"}

    series_list = []
    for m in months:
        col = month_cols.get(m, f"ret_m{m:02d}")
        subset = crsp_m[
            (crsp_m["date"].dt.year == year) & (crsp_m["date"].dt.month == m)
        ][["PERMNO", "RET"]].copy()
        subset["RET"] = pd.to_numeric(subset["RET"], errors="coerce")
        # CRSP monthly has at most one row per PERMNO per month; guard anyway
        s = subset.drop_duplicates("PERMNO").set_index("PERMNO")["RET"].rename(col)
        series_list.append(s)

    result = pd.concat(series_list, axis=1).reset_index()
    result["year"] = year
    return result
