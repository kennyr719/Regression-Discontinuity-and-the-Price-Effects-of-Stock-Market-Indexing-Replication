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

    # ------------------------------------------------------------------
    # Step 2: Attach GVKEY via CCM links active on the reconstitution date
    # ------------------------------------------------------------------
    active_links = ccm_link[
        (ccm_link["LINKDT"] <= may_cutoff) & (ccm_link["LINKENDDT"] >= may_cutoff)
    ][["LPERMNO", "gvkey"]].rename(columns={"LPERMNO": "PERMNO"})

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
    #   adjusted_cshoq = cshoq_millions × 1000 × (CFACSHR_may / CFACSHR_qtr_end)
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

    # Split-adjustment ratio; default to 1.0 when lookup fails
    cfacshr_may = may_crsp["CFACSHR"].replace(0, np.nan)
    cfacshr_qtr = may_crsp["cfacshr_qtr"].replace(0, np.nan)
    split_adj = (cfacshr_may / cfacshr_qtr).fillna(1.0)

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


def identify_index_switchers(rankings_df, constituents_df, year, cutoff=1000):
    """Identify firms that switch between Russell 1000 and Russell 2000.

    Separates the addition effect sample (Russell 1000 firms in year t-1
    that cross below the cutoff in year t) from the deletion effect sample
    (Russell 2000 firms in year t-1 that cross above the cutoff in year t).

    Parameters
    ----------
    rankings_df : pd.DataFrame
        End-of-May market cap rankings for year t.
    constituents_df : pd.DataFrame
        Russell 1000/2000 constituent lists for year t-1.
    year : int
        Reconstitution year.
    cutoff : int, optional
        Rank cutoff between Russell 1000 and 2000 (default: 1000).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (addition_sample, deletion_sample) within the specified bandwidth.
    """
    raise NotImplementedError


def compute_banding_cutoffs(rankings_df, year):
    """Compute banding-adjusted cutoffs for post-2007 reconstitutions.

    After 2007, Russell implemented a banding policy where firms only switch
    indexes if their cumulative market capitalization deviates more than 2.5%
    from the 1000th stock's cumulative market capitalization in the Russell
    3000E.

    Parameters
    ----------
    rankings_df : pd.DataFrame
        End-of-May rankings including cumulative market cap percentiles.
    year : int
        Reconstitution year.

    Returns
    -------
    tuple[int, int]
        (addition_cutoff, deletion_cutoff) adjusted for banding.
    """
    raise NotImplementedError


def construct_outcome_variables(crsp_daily, crsp_monthly, year):
    """Construct dependent variables for the RD regressions.

    Variables constructed:
    - Returns: raw monthly stock returns (May through September)
    - VR: volume ratio relative to 6-month trailing average
    - SR: short interest ratio (shares shorted / shares outstanding)
    - Comovement: monthly beta with Russell 2000 index daily returns

    Parameters
    ----------
    crsp_daily : pd.DataFrame
        CRSP daily stock data.
    crsp_monthly : pd.DataFrame
        CRSP monthly stock data.
    year : int
        Reconstitution year.

    Returns
    -------
    pd.DataFrame
        Panel of outcome variables by firm-month.
    """
    raise NotImplementedError
