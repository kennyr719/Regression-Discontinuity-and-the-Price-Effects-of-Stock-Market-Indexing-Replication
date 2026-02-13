"""Functions for data acquisition, cleaning, and processing."""

import numpy as np
import pandas as pd


def merge_crsp_compustat(crsp_df, compustat_df):
    """Merge CRSP stock data with Compustat fundamentals.

    Parameters
    ----------
    crsp_df : pd.DataFrame
        CRSP daily/monthly stock data with PERMNO identifiers.
    compustat_df : pd.DataFrame
        Compustat quarterly data with GVKEY identifiers.

    Returns
    -------
    pd.DataFrame
        Merged dataset linked via CRSP/Compustat linking table.
    """
    raise NotImplementedError


def compute_market_cap_rankings(df, year):
    """Compute end-of-May market capitalization rankings.

    Follows the methodology in Chang et al. (2015, Section 1.1):
    - Uses end-of-May closing prices from CRSP
    - Uses Compustat quarterly shares outstanding (CSHOQ), selecting the
      most recent quarter publicly available before May 31 based on RDQ
    - Applies CRSP monthly adjustment factor (FACSHR) for corporate
      distributions between fiscal quarter-end and May 31
    - Takes the larger of CRSP shares and adjusted Compustat shares

    Parameters
    ----------
    df : pd.DataFrame
        Merged CRSP/Compustat data.
    year : int
        Reconstitution year.

    Returns
    -------
    pd.DataFrame
        Ranked firms with end-of-May market capitalization and rank.
    """
    raise NotImplementedError


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
