"""Functions for data acquisition, cleaning, and processing."""

import os
import re

import numpy as np
import pandas as pd


def load_bloomberg_float(float_file, crsp_monthly_df):
    """Load Bloomberg EQY_FLOAT data and return dict[year, DataFrame].

    Reads float shares from the Bloomberg float CSV file, matches Bloomberg
    tickers to CRSP PERMNOs via NCUSIP (using russell_constituents_clean.csv
    in the same directory), and returns per-year DataFrames with float share
    counts in thousands.

    Coverage: 2004–2024 only.  1996–2003 entries are absent because Bloomberg
    does not carry historical EQY_FLOAT before 2004.  For those years,
    compute_market_cap_rankings() automatically falls back to total shares
    (max of CRSP SHROUT and adjusted Compustat CSHOQ), so no special handling
    is required by the caller.

    Matching uses four stages in order:
      1. Exact (bbg_ticker, year) match via russell_constituents_clean.csv
      2. Exact bbg_ticker match on most-recent constituent year (fallback)
      3. Base-ticker match (strip exchange suffix, e.g. "AAPL UQ" → "AAPL"),
         year-specific then any-year, to handle UQ/UW/UN suffix mismatches
      4. NCUSIP-direct: Bloomberg encodes delisted stocks as "[NCUSIP] [EXCH]"
         (e.g. "9876588D UQ") — extract the 8-char base and match to CRSP

    Parameters
    ----------
    float_file : str
        Path to russell_float_shares.csv.  Columns: bbg_ticker, bbg_security,
        then integer year columns 1996–2024.  Float values are in *millions* of
        shares; NaN / Bloomberg error strings mean no data.
    crsp_monthly_df : pd.DataFrame
        CRSP monthly stock file returned by merge_crsp_compustat().  Must
        contain NCUSIP, PERMNO, date, SHROUT, CFACSHR columns.

    Returns
    -------
    dict[int, pd.DataFrame]
        Keys are calendar years (int).  Each value has columns:
        ['PERMNO', 'float_shares_thou'].  One row per PERMNO.
        Years with no matched data (i.e. 1996–2003) are absent from the dict.
    """
    # --- Read CSV: columns are bbg_ticker, bbg_security, 1996, 1997, …, 2024 ---
    float_raw = pd.read_csv(float_file)

    # Melt from wide to long: (bbg_ticker, year, eqy_float_millions)
    # CSV stores year column names as strings; cast to int where possible.
    year_cols = [
        c for c in float_raw.columns
        if str(c).strip().isdigit() and 1990 <= int(str(c).strip()) <= 2030
    ]
    long_df = float_raw.melt(
        id_vars=["bbg_ticker"],
        value_vars=year_cols,
        var_name="year",
        value_name="eqy_float_millions",
    )
    # Coerce to numeric: Bloomberg exports '#N/A Requesting Data...' and similar
    # error strings for missing cells.  pd.to_numeric converts these to NaN.
    long_df["eqy_float_millions"] = pd.to_numeric(
        long_df["eqy_float_millions"], errors="coerce"
    )
    long_df = long_df.dropna(subset=["eqy_float_millions"])
    long_df["year"] = long_df["year"].astype(int)
    # Bloomberg float is in millions; CRSP SHROUT is in thousands
    long_df["float_shares_thou"] = long_df["eqy_float_millions"] * 1000

    # --- Match bbg_ticker → NCUSIP via russell_constituents_clean.csv ---
    # Use year-specific (bbg_ticker, year) → ncusip mapping where available,
    # then fall back to the most-recent-year ncusip for unmatched rows.
    # Year-specific matching is critical: if a ticker's CUSIP changed over
    # time (merger, spin-off), the most-recent ncusip maps to the wrong
    # PERMNO for earlier years, inflating float shares by an order of magnitude.
    data_dir = os.path.dirname(os.path.abspath(float_file))
    bbg = pd.read_csv(os.path.join(data_dir, "russell_constituents_clean.csv"))
    bbg["ncusip_upper"] = bbg["ncusip"].astype(str).str.strip().str.upper()

    # Stage 1: exact (bbg_ticker, year) match
    yr_map = bbg[["bbg_ticker", "year", "ncusip_upper"]].drop_duplicates(
        ["bbg_ticker", "year"], keep="first"
    )
    long_df = long_df.merge(yr_map, on=["bbg_ticker", "year"], how="left")

    # Stage 2: for unmatched rows, fall back to most-recent-year ncusip
    unmatched = long_df["ncusip_upper"].isna()
    if unmatched.any():
        fallback = (
            bbg.sort_values("year", ascending=False)
            .drop_duplicates("bbg_ticker", keep="first")[["bbg_ticker", "ncusip_upper"]]
            .rename(columns={"ncusip_upper": "ncusip_fallback"})
        )
        fb = long_df.loc[unmatched, ["bbg_ticker"]].merge(
            fallback, on="bbg_ticker", how="left"
        )
        long_df.loc[unmatched, "ncusip_upper"] = fb["ncusip_fallback"].values

    # Stage 3: base-ticker fallback — strip exchange suffix (e.g. "AAPL UQ" → "AAPL")
    # Handles suffix mismatches between the float file and constituents file
    # (Bloomberg uses UQ, UW, UN, UA etc. inconsistently across data products).
    unmatched2 = long_df["ncusip_upper"].isna()
    if unmatched2.any():
        def _base_ticker(t):
            return str(t).strip().rsplit(" ", 1)[0].upper()

        bbg["base_ticker"] = bbg["bbg_ticker"].apply(_base_ticker)
        long_df["base_ticker"] = long_df["bbg_ticker"].apply(_base_ticker)

        # 3a: year-specific base-ticker match
        base_yr_map = (
            bbg.sort_values("year", ascending=False)
            .drop_duplicates(["base_ticker", "year"], keep="first")[
                ["base_ticker", "year", "ncusip_upper"]
            ]
        )
        fb2 = long_df.loc[unmatched2, ["base_ticker", "year"]].merge(
            base_yr_map, on=["base_ticker", "year"], how="left"
        )
        long_df.loc[unmatched2, "ncusip_upper"] = fb2["ncusip_upper"].values

        # 3b: any-year base-ticker fallback (most-recent year wins)
        unmatched3 = long_df["ncusip_upper"].isna()
        if unmatched3.any():
            base_any_map = (
                bbg.sort_values("year", ascending=False)
                .drop_duplicates("base_ticker", keep="first")[
                    ["base_ticker", "ncusip_upper"]
                ]
                .rename(columns={"ncusip_upper": "ncusip_base_fb"})
            )
            fb3 = long_df.loc[unmatched3, ["base_ticker"]].merge(
                base_any_map, on="base_ticker", how="left"
            )
            long_df.loc[unmatched3, "ncusip_upper"] = fb3["ncusip_base_fb"].values

        long_df = long_df.drop(columns=["base_ticker"])

    # Stage 4: NCUSIP-direct fallback for Bloomberg coded tickers.
    # Bloomberg encodes delisted/obscure stocks as "[8-char-NCUSIP] [EXCHANGE]"
    # (e.g. "9876588D UQ").  The base portion IS the NCUSIP — skip constituents
    # and match directly to CRSP.
    _ncusip_pat = re.compile(r"^[A-Z0-9]{8}$")
    unmatched4 = long_df["ncusip_upper"].isna()
    if unmatched4.any():
        candidate = long_df.loc[unmatched4, "bbg_ticker"].apply(
            lambda t: str(t).strip().rsplit(" ", 1)[0].upper()
        )
        is_ncusip_code = candidate.apply(lambda s: bool(_ncusip_pat.match(s)))
        # Use .loc with the subset index to avoid shape mismatch (pandas 3.x)
        ncusip_idx = is_ncusip_code[is_ncusip_code].index
        long_df.loc[ncusip_idx, "ncusip_upper"] = candidate.loc[ncusip_idx].values

    long_df = long_df.dropna(subset=["ncusip_upper"])

    # --- Match NCUSIP → PERMNO via CRSP monthly (June or May rows) ---
    crsp_m = crsp_monthly_df.copy()
    crsp_m["date"] = pd.to_datetime(crsp_m["date"])
    crsp_june = crsp_m[
        crsp_m["date"].dt.month.isin([5, 6]) & crsp_m["NCUSIP"].notna()
    ].copy()
    crsp_june["ncusip_upper"] = (
        crsp_june["NCUSIP"].astype(str).str.strip().str.upper()
    )
    crsp_june["year"] = crsp_june["date"].dt.year

    # One PERMNO per (year, ncusip): largest SHROUT wins
    ncusip_permno = (
        crsp_june.sort_values("SHROUT", ascending=False)
        .drop_duplicates(["year", "ncusip_upper"], keep="first")[
            ["year", "ncusip_upper", "PERMNO"]
        ]
    )

    long_df = long_df.merge(ncusip_permno, on=["year", "ncusip_upper"], how="inner")

    # --- Split-adjust Bloomberg float from current basis to historical basis ---
    # Bloomberg EQY_FLOAT via BDH PULL returns shares on the CURRENT (most recent)
    # split-adjusted basis, not the historical share count at each end-of-May date.
    # Example: Apple had ~910M shares in May 2010, but Bloomberg returns ~25.3B
    # because Apple split 7:1 (2014) and 4:1 (2020) since then (7×4=28× factor).
    #
    # Correction formula: float_hist = float_bloomberg × (CFACSHR_latest / CFACSHR_may)
    # CRSP convention: SHROUT × CFACSHR = constant through splits.
    # So CFACSHR_latest / CFACSHR_may = inverse of the cumulative
    # split factor since May.
    crsp_m2 = crsp_monthly_df.copy()
    crsp_m2["date"] = pd.to_datetime(crsp_m2["date"])

    # CFACSHR at May (or June) of each ranking year
    cfacshr_may_df = (
        crsp_m2[crsp_m2["date"].dt.month.isin([5, 6])].copy()
    )
    cfacshr_may_df["year"] = cfacshr_may_df["date"].dt.year
    cfacshr_may_df["month"] = cfacshr_may_df["date"].dt.month
    cfacshr_may_df = (
        cfacshr_may_df
        .sort_values(
            ["PERMNO", "year", "month", "date"],
            ascending=[True, True, True, False],  # month asc → May(5) before June(6)
        )
        # keeps May; falls back to June only if no May
        .drop_duplicates(["PERMNO", "year"], keep="first")
        [["PERMNO", "year", "CFACSHR"]]
        .rename(columns={"CFACSHR": "cfacshr_may"})
    )

    # Most recent CFACSHR per PERMNO (Bloomberg pull basis)
    cfacshr_latest_df = (
        crsp_m2.sort_values("date", ascending=False)
        .drop_duplicates("PERMNO", keep="first")[["PERMNO", "CFACSHR"]]
        .rename(columns={"CFACSHR": "cfacshr_latest"})
    )

    long_df = long_df.merge(cfacshr_may_df, on=["PERMNO", "year"], how="left")
    long_df = long_df.merge(cfacshr_latest_df, on="PERMNO", how="left")

    # Adjustment factor = CFACSHR_latest / CFACSHR_may; default 1.0 when missing
    adj_factor = (
        long_df["cfacshr_latest"].replace(0, np.nan)
        / long_df["cfacshr_may"].replace(0, np.nan)
    ).fillna(1.0)
    long_df["float_shares_thou"] = long_df["float_shares_thou"] * adj_factor

    # Deduplicate: one row per (PERMNO, year), prefer highest float
    long_df = (
        long_df.sort_values("float_shares_thou", ascending=False)
        .drop_duplicates(["PERMNO", "year"], keep="first")
    )

    # Return as dict[year → DataFrame]
    result = {}
    for year, grp in long_df.groupby("year"):
        result[int(year)] = grp[["PERMNO", "float_shares_thou"]].reset_index(drop=True)

    return result


def match_bloomberg_to_crsp(bloomberg_file, ccm_link_df, crsp_monthly_df=None):
    """Match Bloomberg Russell constituent data to CRSP PERMNOs.

    Three-stage matching for each reconstitution year:

    Stage 1 (NCUSIP — primary): Bloomberg ncusip (8-digit) matched to CRSP
    NCUSIP from June of the reconstitution year.  Requires crsp_monthly_df
    with a NCUSIP column.  When one Bloomberg NCUSIP maps to multiple PERMNOs
    (multiple share classes), the PERMNO with the largest SHROUT is kept.

    Stage 2 (ticker exact — fallback): Bloomberg ticker matched to CCM tic
    (uppercased, active links during June of year t).  LINKPRIM='P' preferred.

    Stage 3 (ticker stripped — fallback): As Stage 2 but with trailing numeric
    suffix removed (e.g. 'AAIC.1' → 'AAIC').

    Unmatched rows are dropped (caller falls back to D = τ for those stocks).

    Parameters
    ----------
    bloomberg_file : str
        Path to russell_constituents_clean.csv.
    ccm_link_df : pd.DataFrame
        CCM linking table — must contain gvkey, tic, LINKPRIM, LINKTYPE,
        LPERMNO, LINKDT, LINKENDDT.
    crsp_monthly_df : pd.DataFrame, optional
        CRSP monthly stock file with NCUSIP, PERMNO, date, SHROUT columns.
        When provided, enables Stage 1 NCUSIP matching.

    Returns
    -------
    pd.DataFrame
        Columns: year, PERMNO (int), D_actual (1 if R2000, 0 if R1000).
        Only matched rows are returned; deduplicated to one row per (year, PERMNO).
    """
    bbg = pd.read_csv(bloomberg_file)
    bbg["ticker_clean"] = bbg["ticker"].str.strip().str.upper()
    bbg["ncusip_clean"] = bbg["ncusip"].astype(str).str.strip().str.upper()

    # --- Prepare CCM ticker maps (Stages 2 & 3) ---
    link = ccm_link_df[
        ccm_link_df["LINKTYPE"].isin(["LC", "LU"])
        & ccm_link_df["LINKPRIM"].isin(["P", "C"])
    ].copy()
    link["LINKDT"] = pd.to_datetime(link["LINKDT"], errors="coerce")
    link["LINKENDDT"] = link["LINKENDDT"].replace("E", None)
    link["LINKENDDT"] = pd.to_datetime(link["LINKENDDT"], errors="coerce")
    link["LINKENDDT"] = link["LINKENDDT"].fillna(pd.Timestamp("2099-12-31"))
    link["tic_clean"] = link["tic"].str.strip().str.upper()
    link["tic_stripped"] = link["tic_clean"].str.replace(r"\.\d*$", "", regex=True)
    _pref = {"P": 0, "C": 1}

    # --- Prepare CRSP NCUSIP lookup (Stage 1) ---
    ncusip_lookup = {}  # year → DataFrame(NCUSIP, PERMNO) deduplicated by SHROUT
    if crsp_monthly_df is not None and "NCUSIP" in crsp_monthly_df.columns:
        crsp_m = crsp_monthly_df.copy()
        crsp_m["date"] = pd.to_datetime(crsp_m["date"])
        crsp_m["ncusip_upper"] = crsp_m["NCUSIP"].astype(str).str.strip().str.upper()
        for year in bbg["year"].unique():
            # Prefer June; fall back to May if June missing for a PERMNO
            june_crsp = crsp_m[
                (crsp_m["date"].dt.year == year)
                & (crsp_m["date"].dt.month.isin([5, 6]))
                & (crsp_m["ncusip_upper"].str.len() > 0)
                & (crsp_m["ncusip_upper"] != "NAN")
            ].copy()
            if june_crsp.empty:
                continue
            # For each NCUSIP, keep the PERMNO with the largest SHROUT
            june_crsp = (
                june_crsp.sort_values("SHROUT", ascending=False)
                .drop_duplicates("ncusip_upper", keep="first")[
                    ["ncusip_upper", "PERMNO"]
                ]
                .rename(columns={"PERMNO": "PERMNO_ncusip"})
            )
            ncusip_lookup[year] = june_crsp

    results = []
    for year, bbg_yr in bbg.groupby("year"):
        june_date = pd.Timestamp(f"{year}-06-15")
        active = link[
            (link["LINKDT"] <= june_date) & (link["LINKENDDT"] >= june_date)
        ]

        # Stage 1: NCUSIP match
        merged = bbg_yr.copy()
        merged["PERMNO"] = np.nan
        if year in ncusip_lookup:
            ncusip_map = ncusip_lookup[year]
            matched = merged.merge(
                ncusip_map, left_on="ncusip_clean", right_on="ncusip_upper", how="left"
            )
            merged["PERMNO"] = matched["PERMNO_ncusip"].values

        # Stage 2: ticker exact match (for unmatched rows)
        unmatched = merged["PERMNO"].isna()
        if unmatched.any():
            exact_map = (
                active.sort_values("LINKPRIM", key=lambda s: s.map(_pref))
                .drop_duplicates("tic_clean", keep="first")[["tic_clean", "LPERMNO"]]
            )
            sec = (
                merged.loc[unmatched, ["ticker_clean"]]
                .merge(
                    exact_map,
                    left_on="ticker_clean",
                    right_on="tic_clean",
                    how="left",
                )
            )
            merged.loc[unmatched, "PERMNO"] = sec["LPERMNO"].values

        # Stage 3: ticker stripped match (for still-unmatched rows)
        unmatched = merged["PERMNO"].isna()
        if unmatched.any():
            strip_map = (
                active.sort_values("LINKPRIM", key=lambda s: s.map(_pref))
                .drop_duplicates("tic_stripped", keep="first")
                [["tic_stripped", "LPERMNO"]]
            )
            sec = (
                merged.loc[unmatched, ["ticker_clean"]]
                .merge(
                    strip_map,
                    left_on="ticker_clean",
                    right_on="tic_stripped",
                    how="left",
                )
            )
            merged.loc[unmatched, "PERMNO"] = sec["LPERMNO"].values

        merged["year"] = year
        results.append(merged[["year", "PERMNO", "index"]])

    panel = pd.concat(results, ignore_index=True)
    panel = panel.dropna(subset=["PERMNO"]).copy()
    panel["PERMNO"] = panel["PERMNO"].astype(int)
    panel["D_actual"] = (panel["index"] == "R2000").astype(int)

    # Deduplicate: one row per (year, PERMNO) — keep R2000 if conflict
    panel = (
        panel.sort_values("D_actual", ascending=False)
        .drop_duplicates(["year", "PERMNO"], keep="first")
        .drop(columns=["index"])
    )

    return panel.reset_index(drop=True)


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


def compute_market_cap_rankings(data, year, float_shares_df=None):
    """Compute end-of-May market capitalization rankings for a given year.

    Follows Chang et al. (2015, Section 1.1):
    1. End-of-May closing price from CRSP monthly.
    2. Most recent Compustat CSHOQ publicly available before May 31,
       using RDQ or estimated SEC filing deadline.
    3. Adjust Compustat shares for splits/distributions (CFACSHR ratio).
    4. Take the larger of CRSP SHROUT and adjusted Compustat shares.
    5. Override with Bloomberg float-adjusted shares when available.
    6. Market cap = abs(PRC) × shares; rank descending.

    Eligible stocks: SHRCD in {10, 11}, EXCHCD in {1, 2, 3} (NYSE/AMEX/NASDAQ),
    closing price ≥ $1.00.

    Parameters
    ----------
    data : dict
        Cleaned data dict returned by merge_crsp_compustat().
    year : int
        Reconstitution year (Russell ranks end-of-May market caps).
    float_shares_df : pd.DataFrame or None, optional
        Per-year float shares from load_bloomberg_float().  Must have columns
        ['PERMNO', 'float_shares_thou'] (thousands of shares).  When provided,
        float shares replace max(CRSP, Compustat) for matched stocks, giving
        rankings closer to Russell's float-adjusted methodology.  Default None
        preserves the original total-shares behavior (backward compatible).

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
    # Step 5: Take the larger of CRSP SHROUT and adjusted Compustat shares;
    #         then override with Bloomberg float-adjusted shares when available.
    #
    # Russell uses float-adjusted shares (excludes insider/government holdings)
    # for its market cap rankings.  Bloomberg EQY_FLOAT provides end-of-May
    # float counts (in millions) for 2004–2024, which are closer to Russell's
    # methodology than total shares outstanding.
    # ------------------------------------------------------------------
    crsp_shares = may_crsp["SHROUT"].astype(float)
    shares = crsp_shares.copy()
    has_comp = compustat_shares_thou.notna()
    shares[has_comp] = np.maximum(
        crsp_shares[has_comp], compustat_shares_thou[has_comp]
    )

    if float_shares_df is not None and len(float_shares_df) > 0:
        may_crsp = may_crsp.merge(
            float_shares_df[["PERMNO", "float_shares_thou"]], on="PERMNO", how="left"
        )
        has_float = (
            may_crsp["float_shares_thou"].notna()
            & (may_crsp["float_shares_thou"] > 0)
        )
        shares[has_float] = may_crsp.loc[has_float, "float_shares_thou"].values
    else:
        may_crsp["float_shares_thou"] = np.nan

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
    bloomberg_panel=None,
):
    """Identify firms switching between Russell 1000 and Russell 2000.

    Separates stocks near the rank cutoff in year t into two samples based
    on their prior-year (t-1) index membership (proxied by reconstructed
    rankings):

    - Addition sample: stocks ranked ≤ cutoff in year t-1 (in Russell 1000)
      and within [addition_cutoff-bandwidth, addition_cutoff+bandwidth] in
      year t.  Those with rank > addition_cutoff crossed into Russell 2000.

    - Deletion sample: stocks ranked > cutoff in year t-1 (in Russell 2000)
      and within [deletion_cutoff-bandwidth, deletion_cutoff+bandwidth] in
      year t.  Those with rank ≤ deletion_cutoff crossed into Russell 1000.

    Pre-banding (≤ 2006): addition_cutoff = deletion_cutoff = cutoff = 1000.
    Post-banding (≥ 2007): pass banding-adjusted cutoffs from
    compute_banding_cutoffs(); addition_cutoff > 1000, deletion_cutoff < 1000.

    When bloomberg_panel is provided, actual index membership (D_actual) from
    Bloomberg constituent lists is used as the treatment indicator D.  The
    rank-based indicator τ remains the instrument, giving a proper fuzzy 2SLS.
    Stocks that cannot be matched in bloomberg_panel fall back to D = τ.

    When bloomberg_panel is None, D = τ (sharp-RD approximation).

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
    bloomberg_panel : pd.DataFrame or None, optional
        Output of match_bloomberg_to_crsp(): columns year, PERMNO, D_actual
        (1=R2000, 0=R1000).  When provided, D is set from Bloomberg data
        (with τ fallback for unmatched stocks).  Default None → sharp RD.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (addition_sample, deletion_sample).  Each DataFrame contains:
        PERMNO, gvkey, date, PRC, SHROUT, shares, market_cap, rank,
        rank_centered, tau, D, D_actual, prev_rank, year.
        D_actual == D when Bloomberg is available; D_actual == tau otherwise.
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

    # ── Overlay Bloomberg D_actual (fuzzy RD) ────────────────────────────────
    # When bloomberg_panel is provided, merge actual index membership and use
    # it as the treatment indicator D.  Unmatched stocks fall back to D = τ.
    if bloomberg_panel is not None:
        bbg = bloomberg_panel[["year", "PERMNO", "D_actual"]]
        addition = addition.merge(bbg, on=["year", "PERMNO"], how="left")
        addition["D_actual"] = addition["D_actual"].fillna(addition["tau"])
        addition["D"] = addition["D_actual"]

        deletion = deletion.merge(bbg, on=["year", "PERMNO"], how="left")
        deletion["D_actual"] = deletion["D_actual"].fillna(deletion["tau"])
        deletion["D"] = deletion["D_actual"]
    else:
        # Sharp RD: D = τ; keep D_actual column for interface consistency
        addition["D_actual"] = addition["tau"]
        deletion["D_actual"] = deletion["tau"]

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
    cum_rev = df_rev.set_index("rank")["cum_mktcap_rev"]
    df["cum_pct_rev"] = (
        cum_rev.reindex(df["rank"].values).values / total_mktcap
    )

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
        mkt_trail_avg = float(np.mean(
            [mkt_monthly.get(ym, 0.0) for ym in trailing_yms]
        ))
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
    return result  # end construct_volume_ratio


def construct_outcome_variables(data, year, months=(5, 6, 7, 8, 9)):
    """Construct monthly return outcome variables for the fuzzy RD regressions.

    Extracts CRSP monthly returns for May through September of the
    reconstitution year in wide format (one row per PERMNO).  Designed to
    be merged with addition/deletion panels from identify_index_switchers()
    on ['PERMNO', 'year'].

    Volume ratio (VR) is constructed by construct_volume_ratio(); comovement
    beta is constructed by construct_comovement() using crsp_daily.

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


def construct_io_variable(panel, io_file):
    """Merge institutional ownership (IO) onto the addition/deletion panel.

    Uses LSEG/Thomson 13F ownership summary file (pre-aggregated by stock
    and quarter).  Matches via 8-digit CUSIP and uses the Q2 observation
    (report date closest to June 30 of each reconstitution year), with a
    Q1 fallback.

    Parameters
    ----------
    panel : pd.DataFrame
        Addition or deletion panel from identify_index_switchers(), must
        contain PERMNO, year, and NCUSIP columns.
    io_file : str
        Path to thomson_13f.csv.gz.  Must contain rdate, cusip (8-digit),
        InstOwn_Perc (decimal 0–1).

    Returns
    -------
    pd.DataFrame
        Input panel with an 'IO' column (institutional ownership, decimal).
        NaN where no 13F record is available.
    """
    io = pd.read_csv(io_file, usecols=["rdate", "cusip", "InstOwn_Perc"])
    io["rdate"] = pd.to_datetime(io["rdate"])
    io["year"] = io["rdate"].dt.year
    io["quarter"] = io["rdate"].dt.quarter
    io["cusip_clean"] = io["cusip"].astype(str).str.strip().str.upper()

    # Keep Q1 and Q2; prefer Q2 (right after reconstitution)
    io_q = io[io["quarter"].isin([1, 2])].copy()
    io_q = (
        io_q.sort_values("quarter", ascending=False)
        .drop_duplicates(["year", "cusip_clean"], keep="first")
        [["year", "cusip_clean", "InstOwn_Perc"]]
        .rename(columns={"InstOwn_Perc": "IO"})
    )

    if "NCUSIP" not in panel.columns:
        panel = panel.copy()
        panel["IO"] = np.nan
        return panel

    panel = panel.copy()
    panel["ncusip_clean"] = panel["NCUSIP"].astype(str).str.strip().str.upper()
    merged = panel.merge(
        io_q, left_on=["year", "ncusip_clean"], right_on=["year", "cusip_clean"],
        how="left"
    )
    merged = merged.drop(columns=["cusip_clean", "ncusip_clean"], errors="ignore")
    return merged


def construct_sr_variable(panel, sr_file, ccm_link_df):
    """Merge short interest ratio (SR) onto the addition/deletion panel.

    Uses Compustat semi-monthly short interest (comp.sec_shortint).  Links
    Compustat GVKEY → PERMNO via CCM, then finds the observation closest to
    June 30 of each reconstitution year.

    SR = shortintadj / (SHROUT × 1000)
    (shortintadj in shares; CRSP SHROUT in thousands)

    Coverage: 2006-07 onward (hard floor in Compustat source data;
    pre-2006 NYSE/AMEX exchange-level data used by CHL 2015 is not in WRDS).
    SR will be NaN for all years before 2007.  Use 2006 as the base year
    for the SR time-trend specification, not 1996.

    Parameters
    ----------
    panel : pd.DataFrame
        Addition or deletion panel; must contain PERMNO, year, SHROUT.
    sr_file : str
        Path to compustat_short_interest.csv.gz.
    ccm_link_df : pd.DataFrame
        CCM linking table as returned by merge_crsp_compustat()['ccm_link'].

    Returns
    -------
    pd.DataFrame
        Input panel with an 'SR' column (short ratio, decimal).
        NaN where no short interest record is available.
    """
    sr = pd.read_csv(sr_file, usecols=["gvkey", "datadate", "shortintadj"])
    sr["datadate"] = pd.to_datetime(sr["datadate"])
    sr["year"] = sr["datadate"].dt.year
    sr["gvkey"] = sr["gvkey"].astype(str).str.zfill(6)

    # Observation closest to June 30 for each (gvkey, year)
    sr["june30"] = pd.to_datetime(sr["year"].astype(str) + "-06-30")
    sr["days_from_june30"] = (sr["datadate"] - sr["june30"]).abs().dt.days
    sr_june = (
        sr.sort_values("days_from_june30")
        .drop_duplicates(["gvkey", "year"], keep="first")
        [["gvkey", "year", "shortintadj"]]
    )

    # Link GVKEY → PERMNO via CCM (prefer P link), date-bounded per year
    link = ccm_link_df.copy()
    link["gvkey"] = link["gvkey"].astype(str).str.zfill(6)
    link = link[link["LINKPRIM"].isin(["P", "C"])].copy()
    link["LINKDT"] = pd.to_datetime(link["LINKDT"])
    link["LINKENDDT"] = pd.to_datetime(link["LINKENDDT"])

    panel = panel.copy()
    parts = []
    for yr, sub in panel.groupby("year"):
        june_date = pd.Timestamp(f"{yr}-06-15")
        active = link[
            (link["LINKDT"] <= june_date) & (link["LINKENDDT"] >= june_date)
        ]
        gvkey_permno_yr = (
            active.sort_values("LINKPRIM", key=lambda s: s.map({"P": 0, "C": 1}))
            .drop_duplicates("gvkey", keep="first")[["gvkey", "LPERMNO"]]
            .rename(columns={"LPERMNO": "PERMNO_sr"})
        )
        sr_yr = sr_june[sr_june["year"] == yr].merge(
            gvkey_permno_yr, on="gvkey", how="inner"
        )
        merged_yr = sub.merge(
            sr_yr[["PERMNO_sr", "year", "shortintadj"]],
            left_on=["PERMNO", "year"],
            right_on=["PERMNO_sr", "year"],
            how="left",
        ).drop(columns=["PERMNO_sr"], errors="ignore")
        parts.append(merged_yr)
    merged = pd.concat(parts, ignore_index=True)
    shrout = merged["SHROUT"].astype(float) * 1000.0
    merged["SR"] = np.where(shrout > 0, merged["shortintadj"] / shrout, np.nan)
    merged = merged.drop(columns=["shortintadj"])
    return merged


def construct_comovement(panel, daily_file, r2000_file, months=(6,), min_days=15):
    """Compute monthly Russell 2000 comovement beta for stocks in the panel.

    For each (PERMNO, year-month), runs OLS of daily stock returns on daily
    Russell 2000 index returns and returns the slope coefficient (beta).

    For Table 5 replication: use months=(6,) for June comovement.
    For time-trend tables (Tables 7–8): use months=(5,6,7,8,9).

    Parameters
    ----------
    panel : pd.DataFrame
        Addition or deletion panel; must contain PERMNO and year.
    daily_file : str
        Path to crsp_daily.csv.gz.  Must contain PERMNO, date, RET.
    r2000_file : str
        Path to russell2000_daily.csv.gz.  Must contain date, rut_return.
    months : tuple of int, optional
        Calendar months to compute beta for (default: (6,) → June only).
    min_days : int, optional
        Minimum trading days required for a valid beta (default: 15).

    Returns
    -------
    pd.DataFrame
        Columns: PERMNO, year, and one 'cov_mNN' column per requested month.
        NaN where fewer than min_days observations are available.
    """
    permnos = panel["PERMNO"].unique()
    years = panel["year"].unique()

    daily = pd.read_csv(daily_file, usecols=["PERMNO", "date", "RET"], low_memory=False)
    daily["date"] = pd.to_datetime(daily["date"])
    daily["RET"] = pd.to_numeric(daily["RET"], errors="coerce")
    daily = daily[
        daily["PERMNO"].isin(permnos)
        & daily["date"].dt.year.isin(years)
        & daily["date"].dt.month.isin(months)
    ].dropna(subset=["RET"])

    rut = pd.read_csv(r2000_file)
    rut["date"] = pd.to_datetime(rut["date"])
    rut = rut.rename(columns={"rut_return": "rut_ret"})

    daily = daily.merge(rut[["date", "rut_ret"]], on="date", how="inner")
    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month

    month_names = {5: "cov_may", 6: "cov_jun", 7: "cov_jul", 8: "cov_aug", 9: "cov_sep"}

    def _beta(grp):
        if len(grp) < min_days:
            return np.nan
        x = grp["rut_ret"].values
        y = grp["RET"].values
        var_x = np.var(x, ddof=1)
        if var_x == 0:
            return np.nan
        return np.cov(x, y, ddof=1)[0, 1] / var_x

    series_list = []
    for m in months:
        col = month_names.get(m, f"cov_m{m:02d}")
        sub = daily[daily["month"] == m]

        betas = (
            sub.groupby(["PERMNO", "year"])
            .apply(_beta, include_groups=False)
            .rename(col)
            .reset_index()
        )
        series_list.append(betas.set_index(["PERMNO", "year"])[col])

    if not series_list:
        return panel[["PERMNO", "year"]].drop_duplicates()

    result = pd.concat(series_list, axis=1).reset_index()
    return result
