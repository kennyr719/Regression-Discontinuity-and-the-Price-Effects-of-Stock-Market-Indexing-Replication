"""Functions for generating RD plots and figures."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_market_cap_continuity(df, cutoff=0, title="End-of-May Market Capitalization"):
    """Plot market capitalization against rank around the cutoff.

    Replicates Figure 1 from Chang et al. (2015), showing that market
    capitalization is continuous across the Russell 1000/2000 cutoff.

    Parameters
    ----------
    df : pd.DataFrame
        Data with 'rank_centered' and 'market_cap' columns.
    cutoff : int, optional
        Centered cutoff value (default: 0).
    title : str, optional
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    raise NotImplementedError


def plot_rd_discontinuity(
    df, outcome, running, bin_width=2, title="June Returns", ylabel="Returns"
):
    """Plot outcome variable against running variable with RD fit lines.

    Replicates Figure 4 from Chang et al. (2015), showing the discontinuity
    in June returns at the Russell 1000/2000 cutoff with binned scatter
    plots and local linear fit lines on each side.

    Parameters
    ----------
    df : pd.DataFrame
        Data with running variable and outcome columns.
    outcome : str
        Column name of the outcome variable.
    running : str
        Column name of the running variable (centered at cutoff).
    bin_width : int, optional
        Number of ranks per bin for scatter plot (default: 2).
    title : str, optional
        Plot title.
    ylabel : str, optional
        Y-axis label.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    raise NotImplementedError


def plot_index_weights(df, title="Index Weights Around Upper Cutoff"):
    """Plot index weights before and after reconstitution.

    Replicates Figure 2 from Chang et al. (2015), showing the jump in
    index weights at the 1000 cutoff after June reconstitution.

    Parameters
    ----------
    df : pd.DataFrame
        Data with rank, May weights, and June weights.
    title : str, optional
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    raise NotImplementedError


def plot_time_trends(estimates_df, outcome="price_impact", title="RD Estimates Over Time"):
    """Plot rolling RD estimates over time with confidence intervals.

    Replicates Figure 5 from Chang et al. (2015), showing how the price
    impact and volume ratio effects evolve from 1996 to 2012.

    Parameters
    ----------
    estimates_df : pd.DataFrame
        DataFrame with columns 'year', 'estimate', 'ci_lower', 'ci_upper'.
    outcome : str, optional
        Which outcome is being plotted.
    title : str, optional
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    raise NotImplementedError
