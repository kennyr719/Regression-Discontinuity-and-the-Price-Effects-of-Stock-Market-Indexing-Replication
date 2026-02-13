"""Functions for fuzzy RD estimation and inference."""

import numpy as np
import pandas as pd
from scipy import stats


def fuzzy_rd_estimate(df, outcome, treatment, running, cutoff=0, bandwidth=100):
    """Estimate fuzzy RD treatment effect using 2SLS.

    Implements the estimation procedure from Chang et al. (2015, Section 4.1).

    First stage:
        D_it = α_0l + α_1l(r_it - c) + τ_it[α_0r + α_1r(r_it - c)] + ε_it

    Second stage:
        Y_it = β_0l + β_1l(r_it - c) + D_it[β_0r + β_1r(r_it - c)] + ν_it

    where τ is the instrument (indicator for crossing the cutoff) and D is
    actual index membership.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing outcome, treatment, and running variables.
    outcome : str
        Column name of the outcome variable (e.g., 'returns').
    treatment : str
        Column name of the treatment indicator (actual index membership).
    running : str
        Column name of the running variable (end-of-May rank minus cutoff).
    cutoff : int, optional
        Cutoff value for the running variable (default: 0, i.e., centered).
    bandwidth : int, optional
        Bandwidth around the cutoff (default: 100).

    Returns
    -------
    dict
        Dictionary containing:
        - 'coefficient': estimated treatment effect (β_0r)
        - 'se': standard error
        - 't_stat': t-statistic
        - 'p_value': p-value
        - 'n_obs': number of observations
        - 'first_stage_F': first-stage F-statistic
    """
    raise NotImplementedError


def fuzzy_rd_time_trend(df, outcome, treatment, running, bandwidth=100):
    """Estimate fuzzy RD with linear time trend interaction.

    Implements the time trend specification from Chang et al. (2015, Section 5):

    Second stage:
        Y_it = β_0l + β_1l(r_it - c) + β_2l*t
               + D_it[β_0r + β_1r(r_it - c) + β_2r*t] + ν_it

    where t is years since 1996.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing outcome, treatment, running, and year variables.
    outcome : str
        Column name of the outcome variable.
    treatment : str
        Column name of the treatment indicator.
    running : str
        Column name of the running variable (centered at cutoff).
    bandwidth : int, optional
        Bandwidth around the cutoff (default: 100).

    Returns
    -------
    dict
        Dictionary containing β_0r (base effect) and β_2r (time trend).
    """
    raise NotImplementedError


def optimal_bandwidth(df, outcome, running):
    """Compute rule-of-thumb optimal bandwidth.

    Follows the ROT bandwidth from Lee and Lemieux (2010) as described
    in Chang et al. (2015, Section 4.2).

    Parameters
    ----------
    df : pd.DataFrame
        Data containing outcome and running variables.
    outcome : str
        Column name of the outcome variable.
    running : str
        Column name of the running variable.

    Returns
    -------
    int
        Optimal bandwidth (number of ranks on each side of cutoff).
    """
    raise NotImplementedError
