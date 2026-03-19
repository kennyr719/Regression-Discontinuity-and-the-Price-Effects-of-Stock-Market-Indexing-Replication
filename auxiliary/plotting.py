"""Functions for generating RD plots and figures."""

import matplotlib.pyplot as plt
import numpy as np


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

    ax.scatter(
        left_binned['bin'], left_binned[outcome],
        color='black', alpha=0.5, s=20, label='Bins < Cutoff',
    )
    ax.scatter(
        right_binned['bin'], right_binned[outcome],
        edgecolor='black', facecolor='white', s=20,
        label='Bins >= Cutoff',
    )

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
            ax.text(
                0.05, 0.95,
                f"Discontinuity gap: {gap:.4f}",
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(
                    boxstyle="round",
                    facecolor="white",
                    alpha=0.8,
                ),
            )

    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Market Capitalization Rank (centered at cutoff)")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()

    return fig


def plot_index_weights(df=None, title="Index Weights Around Upper Cutoff"):
    """Plot index weights before and after reconstitution.

    Replicates Figure 2 from Chang et al. (2015), showing the jump in
    Russell 2000 index weights at the rank-1000 cutoff after June
    reconstitution.  Stocks just above rank 1000 receive index weights
    approximately 10× larger than stocks just below rank 1000, creating
    the discontinuous passive buying pressure that identifies the RD
    treatment effect.

    NOTE — Data Unavailable: This function cannot be implemented with the
    data accessible through WRDS.  Russell Inc.'s end-of-June float-adjusted
    index constituent weights are proprietary and are not distributed through
    WRDS, CRSP, or Compustat.  The CRSP SHROUT field captures total shares
    outstanding, not the float-adjusted shares that Russell uses to compute
    value weights.  Obtaining Figure 2 requires the annual reconstitution
    weight files published by FTSE Russell, which are not part of this
    project's data subscription.

    Parameters
    ----------
    df : pd.DataFrame or None
        Unused.  Kept for interface compatibility.
    title : str, optional
        Unused.  Kept for interface compatibility.

    Returns
    -------
    None
    """
    return None


def plot_first_stage(
    df, running="rank_centered", treatment="D", bin_width=5,
    title="First Stage: P(Russell 2000) vs. Rank", cutoff=0,
    y_label="P(D = 1 | rank)",
):
    """Plot P(D=1 | rank) binned by bin_width intervals around the cutoff.

    Visualises the first-stage discontinuity: the fraction of stocks actually
    assigned to the Russell 2000 (D_actual = 1) as a function of the centered
    market-cap rank.  A clean jump at rank 0 confirms instrument relevance;
    the magnitude of the jump is approximately α_0r from Table 3.

    Parameters
    ----------
    df : pd.DataFrame
        Data with `running` and `treatment` columns.
    running : str, optional
        Column name of the centered running variable (default: 'rank_centered').
    treatment : str, optional
        Column name of the treatment indicator (default: 'D').
    bin_width : int, optional
        Number of ranks per bin (default: 5).
    title : str, optional
        Plot title.
    cutoff : float, optional
        Centered cutoff value (default: 0).

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    sample = df.dropna(subset=[running, treatment]).copy()
    sample["_bin"] = (
        np.floor((sample[running] - cutoff) / bin_width) * bin_width
        + bin_width / 2 + cutoff
    )
    binned = sample.groupby("_bin")[treatment].mean().reset_index()
    binned.columns = ["bin", "mean_d"]

    left_b = binned[binned["bin"] < cutoff]
    right_b = binned[binned["bin"] >= cutoff]
    left_raw = sample[sample[running] < cutoff]
    right_raw = sample[sample[running] >= cutoff]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        left_b["bin"], left_b["mean_d"],
        color="black", s=25, alpha=0.7, label="Bins below cutoff",
    )
    ax.scatter(
        right_b["bin"], right_b["mean_d"],
        edgecolor="black", facecolor="white", s=25, label="Bins above cutoff",
    )

    for side in [left_raw, right_raw]:
        if len(side) < 2:
            continue
        coef = np.polyfit(
            side[running].values,
            side[treatment].values.astype(float), 1,
        )
        x_fit = np.linspace(side[running].min(), side[running].max(), 200)
        ax.plot(x_fit, np.polyval(coef, x_fit), "k-", linewidth=1.5)

    ax.axvline(x=cutoff, color="gray", linestyle="--", alpha=0.7, label="Cutoff")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.set_xlabel("Rank Relative to Cutoff")
    ax.set_ylabel(y_label)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_time_trends(
    add_df, del_df,
    title="Rolling RD Estimates Over Time",
    ylabel="Treatment Effect",
    add_label="Addition",
    del_label="Deletion",
    add_color="blue",
    del_color="red",
    suptitle=None,
):
    """Plot rolling RD estimates over time for addition and deletion samples.

    Produces a 2-panel figure (addition left, deletion right) showing point
    estimates and 95% confidence intervals from rolling RD windows.  Replicates
    the style of Figure 5 in Chang et al. (2015).

    Parameters
    ----------
    add_df : pd.DataFrame
        Addition-sample rolling estimates.  Must have columns:
        'year', 'estimate', 'ci_lower', 'ci_upper'.
    del_df : pd.DataFrame
        Deletion-sample rolling estimates.  Same required columns.
    title : str, optional
        Per-panel title suffix appended after "Addition Effect on ..." /
        "Deletion Effect on ..." (default: "Rolling RD Estimates Over Time").
    ylabel : str, optional
        Y-axis label for both panels (default: "Treatment Effect").
    add_label : str, optional
        Legend label for the addition estimate line (default: "Addition").
    del_label : str, optional
        Legend label for the deletion estimate line (default: "Deletion").
    add_color : str, optional
        Colour for the addition panel (default: "blue").
    del_color : str, optional
        Colour for the deletion panel (default: "red").
    suptitle : str or None, optional
        Overall figure title.  If None, a default is generated from `title`.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure with two side-by-side subplots.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    panels = [
        (axes[0], add_df, add_color, f"Addition Effect on {title}", add_label),
        (axes[1], del_df, del_color, f"Deletion Effect on {title}", del_label),
    ]

    for ax, df, color, panel_title, label in panels:
        if df is None or df.empty:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(panel_title)
            continue

        years     = df["year"]
        estimates = df["estimate"]
        ci_lower  = df["ci_lower"]
        ci_upper  = df["ci_upper"]

        ax.fill_between(
            years, ci_lower, ci_upper,
            color=color, alpha=0.2, label="95% CI",
        )
        ax.plot(years, estimates, color=color, marker="o", linestyle="-",
                linewidth=2, label=f"RD Estimate ({label})")
        ax.axhline(0, color="k", linestyle="--", linewidth=1)
        ax.set_title(panel_title)
        ax.set_xlabel("Year (end of rolling window)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(years)
        ax.tick_params(axis="x", rotation=45)
        ax.legend(loc="best")

    fig.suptitle(suptitle or f"Figure 5: {title}", fontsize=13)
    fig.tight_layout()
    return fig
