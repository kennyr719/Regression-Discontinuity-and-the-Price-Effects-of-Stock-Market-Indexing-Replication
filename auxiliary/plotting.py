"""Functions for generating RD plots and figures."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_market_cap_continuity(df, cutoff=0, title="End-of-May Market Capitalization"):
    """Plot market capitalization against rank around the cutoff.

    Replicates Figure 1 from Chang et al. (2015), showing that market
    capitalization is continuous across the Russell 1000/2000 cutoff.
    A discontinuity here would indicate manipulation of the running variable
    and invalidate the RD design.

    Parameters
    ----------
    df : pd.DataFrame
        Data with 'rank_centered' and 'market_cap' columns.
        Typically a pooled cross-section of end-of-May rankings restricted
        to a bandwidth around the cutoff (e.g. ±300 ranks).
    cutoff : int, optional
        Centered cutoff value (default: 0).
    title : str, optional
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    sample = df.dropna(subset=["rank_centered", "market_cap"]).copy()
    sample = sample[sample["market_cap"] > 0]
    sample["log_mktcap"] = np.log(sample["market_cap"])

    # Bin by rank_centered and take mean log-market-cap per bin
    bin_width = 5
    sample["bin"] = (
        np.floor(sample["rank_centered"] / bin_width) * bin_width + bin_width / 2
    )
    binned = sample.groupby("bin")["log_mktcap"].mean().reset_index()

    left_b  = binned[binned["bin"] < cutoff]
    right_b = binned[binned["bin"] >= cutoff]
    left_r  = sample[sample["rank_centered"] < cutoff]
    right_r = sample[sample["rank_centered"] >= cutoff]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        left_b["bin"], left_b["log_mktcap"],
        color="black", s=20, alpha=0.6, label="Bins below cutoff",
    )
    ax.scatter(
        right_b["bin"], right_b["log_mktcap"],
        edgecolor="black", facecolor="white", s=20, label="Bins above cutoff",
    )

    # Local linear fit on each side
    for side_r, x_end, step in [(left_r, -0.1, -1), (right_r, None, 1)]:
        if len(side_r) < 2:
            continue
        coef = np.polyfit(side_r["rank_centered"], side_r["log_mktcap"], 1)
        x_min = side_r["rank_centered"].min()
        x_max = side_r["rank_centered"].max()
        x_fit = np.linspace(x_min, x_max, 200)
        ax.plot(x_fit, np.polyval(coef, x_fit), "k-", linewidth=1.5)

    ax.axvline(x=cutoff, color="gray", linestyle="--", alpha=0.7, label="Cutoff")
    ax.set_title(title)
    ax.set_xlabel("Rank Relative to Cutoff (Russell 1000/2000 boundary = 0)")
    ax.set_ylabel("Log Market Capitalization (millions USD)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


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
    sample = df.dropna(subset=[outcome, running]).copy()
    
    # Create bin assignments
    sample['bin'] = np.floor(sample[running] / bin_width) * bin_width + (bin_width / 2)
    binned = sample.groupby('bin')[outcome].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    left_binned = binned[binned['bin'] < 0]
    right_binned = binned[binned['bin'] >= 0]
    
    ax.scatter(left_binned['bin'], left_binned[outcome], color='black', alpha=0.5, s=20, label='Bins < Cutoff')
    ax.scatter(right_binned['bin'], right_binned[outcome], edgecolor='black', facecolor='white', s=20, label='Bins >= Cutoff')
    
    # Fit lines on the unbinned underlying data
    left_raw = sample[sample[running] < 0]
    right_raw = sample[sample[running] >= 0]
    
    if len(left_raw) > 1:
        coef_l = np.polyfit(left_raw[running], left_raw[outcome], 1)
        x_l = np.array([left_raw[running].min(), -0.01])
        ax.plot(x_l, np.polyval(coef_l, x_l), 'k-', linewidth=2)
        
    if len(right_raw) > 1:
        coef_r = np.polyfit(right_raw[running], right_raw[outcome], 1)
        x_r = np.array([0, right_raw[running].max()])
        ax.plot(x_r, np.polyval(coef_r, x_r), 'k-', linewidth=2)
        
        # Calculate discontinuity gap
        if len(left_raw) > 1:
            gap = np.polyval(coef_r, 0) - np.polyval(coef_l, 0)
            ax.text(0.05, 0.95, f"Discontinuity gap: {gap:.4f}", transform=ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Market Capitalization Rank (centered at cutoff)")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    
    return fig


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
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Optional styling for different outcome types
    if outcome == "price_impact":
        ylabel = "Treatment Effect (June Return)"
        color = "blue"
    elif outcome == "vr_impact":
        ylabel = "Treatment Effect (Volume Ratio)"
        color = "red"
    else:
        ylabel = "RD Estimate"
        color = "black"

    years = estimates_df["year"]
    estimates = estimates_df["estimate"]
    ci_lower = estimates_df["ci_lower"]
    ci_upper = estimates_df["ci_upper"]

    # Fill between the confidence intervals
    ax.fill_between(years, ci_lower, ci_upper, color=color, alpha=0.2, label="95% CI")
    
    # Plot the estimates line
    ax.plot(years, estimates, color=color, marker="o", linestyle="-", linewidth=2, label="RD Estimate")
    
    # Add a horizontal line at 0 for reference
    ax.axhline(0, color="k", linestyle="--", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_xticks(years)
    ax.legend(loc="best")
    
    fig.tight_layout()
    return fig
