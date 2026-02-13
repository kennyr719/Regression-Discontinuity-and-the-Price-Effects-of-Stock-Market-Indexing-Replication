"""Auxiliary functions for plotting which are used in the main notebook."""

import matplotlib.pyplot as plt
import pandas as pd


def plot_RDD_curve(df, running_variable, outcome, cutoff):
    """Plot RDD curves.

    Function splits dataset into treated and untreated group based on running variable
    and plots outcome (group below cutoff is treated, group above cutoff is untreated).

    Args:
    -------
        df(DataFrame): Dataframe containing the data to be plotted.
        running_variable(column): DataFrame column name of the running variable.
        outome(column): DataFrame column name of the outcome variable.
        cutoff(numeric): Value of cutoff.

    Returns:
    ---------
        matplotlib.pyplpt.plot
    """
    plt.grid(True)
    df_treat = df[df[running_variable] < cutoff]
    df_untreat = df[df[running_variable] >= cutoff]
    plt.plot(df_treat[outcome])
    plt.plot(df_untreat[outcome])

    return


def plot_RDD_curve_colored(df, running_variable, outcome, cutoff, color):
    """Plot RDD curves.

    Function splits dataset into treated and untreated group based on running variable
    and plots outcome (group below cutoff is treated, group above cutoff is untreated).

    Args:
    -------
        df(DataFrame): Dataframe containing the data to be plotted.
        running_variable(column): DataFrame column name of the running variable.
        outome(column): DataFrame column name of the outcome variable.
        cutoff(numeric): Value of cutoff.

    Returns:
    ---------
        matplotlib.pyplpt.plot

    """
    plt.grid(True)
    df_treat = df[df[running_variable] < cutoff]
    df_untreat = df[df[running_variable] >= cutoff]
    plt.plot(df_treat[outcome], color=color, label="_nolegend_")
    plt.plot(df_untreat[outcome], color=color, label="_nolegend_")


def plot_RDD_curve_CI(
    df, running_variable, outcome, cutoff, lbound, ubound, CI_color, linecolor
):
    """Plot RDD curves with confidence intervals.

    Function splits dataset into treated and untreated group based on running variable
    and plots outcome (group below cutoff is treated, group above cutoff is untreated).

    Args:
    ------
        df(DataFrame): Dataframe containing the data to be plotted.
        running_variable(column): DataFrame column name of the running variable.
        outome(column): DataFrame column name of the outcome variable.
        cutoff(numeric): Value of cutoff.
        lbound(column): Lower bound of confidence interval.
        ubound(column): Upper bound of confidence interval.


    Returns:
    ----------
        matplotlib.pyplpt.plot

    """
    plt.grid(True)
    df_treat = df[df[running_variable] < cutoff]
    df_untreat = df[df[running_variable] >= cutoff]

    # Plot confidence Intervals.
    plt.plot(df_treat[lbound], color=CI_color, alpha=0.3)
    plt.plot(df_treat[ubound], color=CI_color, alpha=0.3)
    plt.plot(df_untreat[lbound], color=CI_color, alpha=0.3)
    plt.plot(df_untreat[ubound], color=CI_color, alpha=0.3)
    plt.fill_between(
        df_treat[running_variable],
        y1=df_treat[lbound],
        y2=df_treat[ubound],
        facecolor=CI_color,
        alpha=0.3,
    )
    plt.fill_between(
        df_untreat[running_variable],
        y1=df_untreat[lbound],
        y2=df_untreat[ubound],
        facecolor=CI_color,
        alpha=0.3,
    )

    # Plot estimated lines.
    plt.plot(df_untreat[outcome], color=linecolor, label="_nolegend_")
    plt.plot(df_treat[outcome], color=linecolor, label="_nolegend_")


def plot_hist_GPA(data):
    """
    Plot historgram showing the distribution of students according to distance
    from fist year cutoff.
    """
    plt.xlim(-1.8, 3)
    plt.ylim(0, 3500)
    plt.xticks([-1.2, -0.6, 0, 0.6, 1.2, 1.8, 2.4, 3])
    plt.hist(data["dist_from_cut"], bins=30, color="orange", alpha=0.7)
    plt.axvline(x=-1.2, color="c", alpha=0.8)
    plt.axvline(x=1.2, color="c", alpha=0.8)
    plt.axvline(x=0.6, color="c", alpha=0.3)
    plt.axvline(x=-0.6, color="c", alpha=0.3)
    plt.axvline(x=0, color="r")
    plt.fill_betweenx(y=range(3500), x1=-1.8, x2=-1.2, alpha=0.8, facecolor="c")
    plt.fill_betweenx(y=range(3500), x1=-1.2, x2=-0.6, alpha=0.3, facecolor="c")
    plt.fill_betweenx(y=range(3500), x1=1.2, x2=0.6, alpha=0.3, facecolor="c")
    plt.fill_betweenx(y=range(3500), x1=3, x2=1.2, alpha=0.8, facecolor="c")
    plt.xlabel("First year GPA minus probation cutoff")
    plt.ylabel("Freq.")
    plt.title("Distribution of student GPAs distance from the cutoff")


def plot_covariates(data, descriptive_table, bins):
    """
    Plots covariates with bins of size 0.5 grade points.
    """
    plt.figure(figsize=(13, 10), dpi=70, facecolor="w", edgecolor="k")
    plt.subplots_adjust(wspace=0.2, hspace=0.4)

    for idx, var in enumerate(descriptive_table.index):
        plt.subplot(3, 3, idx + 1)
        plt.axvline(x=0, color="r")
        plt.grid(True)
        plt.plot(
            data[var].groupby(data["dist_from_cut_med05"]).mean(),
            "o",
            color="c",
            alpha=0.5,
        )
        plt.xlabel("Distance from cutoff")
        plt.ylabel("Mean")
        plt.title(descriptive_table.iloc[idx, 4])


def plot_figure1(data, bins, pred):
    """
    Plots Figure 1.

    Args:
    ------
        data(pd.DataFrame): Dataframe containing the frequency of each bin.
        bins(list): List of bins.
        pred(pd.DataFrame): Predicted frequency of each bin.

    Returns:
    ---------
        matplotlib.pyplpt.plot
    """
    plt.xlim(-1.5, 1.5)
    plt.ylim(0, 2100.5)
    plt.axvline(x=0, color="r")
    plt.xlabel("First year GPA minus probation cutoff")
    plt.ylabel("Frequency count")
    plt.plot(data.bins, data.freq, "o")
    plot_RDD_curve(df=pred, running_variable="bins", outcome="prediction", cutoff=0)
    plt.title("Figure 1. Distribution of Student Grades Relative to their Cutoff")


def plot_figure2(data, pred):
    """
    Plots Figure 2.
    """
    plt.xlim(-1.5, 1.5)
    plt.plot(data["dist_from_cut_med10"], data["gpalscutoff"], "o")
    plot_RDD_curve(
        df=pred, running_variable="dist_from_cut", outcome="prediction", cutoff=0
    )
    plt.axvline(x=0, color="r")
    plt.title("Figure 2: Porbation Status at the end of first year")
    plt.xlabel("First year GPA minus probation cutoff")
    plt.ylabel("Probation Status")


def plot_figure3(inputs_dict, outputs_dict, keys):
    """Plot results from RD analysis for six subgroups of students in Figure3.

    Args:
    -------
        inputs_dict(dict): Dictionary containing all dataframes for each subgroup, used
        for plotting the bins (dots).
        outputs_dict(dict): Dictionary containing the results from RD analysis for each
        subgroup, used for plotting the lines.
        keys(list): List of keys of the dictionaries, both dictionaries must have the
        same keys.

    Returns:
    ----------
        matplotlib.pyplpt.plot: Figure 3 from the paper (figure consists of 6 subplots,
        one for each subgroup of students)
    """
    # Frame for entire figure.
    plt.figure(figsize=(10, 13), dpi=70, facecolor="w", edgecolor="k")
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    # Remove dataframe 'All' because I only want to plot the results for the
    # subgroups of students.
    keys = keys.copy()
    keys.remove("All")

    # Create plots for all subgroups.
    for idx, key in enumerate(keys):
        # Define position of subplot.
        plt.subplot(3, 2, idx + 1)
        # Create frame for subplot.
        plt.xlim(-1.5, 1.5)
        plt.ylim(0, 0.22)
        plt.axvline(x=0, color="r")
        plt.xlabel("First year GPA minus probation cutoff")
        plt.ylabel("Left university voluntarily")
        # Calculate bin means.
        bin_means = (
            inputs_dict[key]
            .left_school.groupby(inputs_dict[key]["dist_from_cut_med10"])
            .mean()
        )
        bin_means = pd.Series.to_frame(bin_means)
        # Plot subplot.
        plt.plot(list(bin_means.index), list(bin_means.left_school), "o")
        plot_RDD_curve(
            df=outputs_dict[key],
            running_variable="dist_from_cut",
            outcome="prediction",
            cutoff=0,
        )
        plt.title(key)


def plot_figure4(data, pred):
    """
    Plots Figure 4.
    """
    plt.figure(figsize=(8, 5))
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1, 1.5)
    plt.axvline(x=0, color="r")
    plt.xlabel("First year GPA minus probation cutoff")
    plt.ylabel("Subsequent GPA minus Cutoff")
    plt.plot(data.nextGPA.groupby(data["dist_from_cut_med10"]).mean(), "o")
    plot_RDD_curve(
        df=pred, running_variable="dist_from_cut", outcome="prediction", cutoff=0
    )
    plt.title("Figure 4 - GPA in the next enrolled term")


def plot_figure5(data, pred_1, pred_2, pred_3):
    """
    Plots Figure 5.
    """
    plt.figure(figsize=(8, 5))
    plt.xlim(-1.5, 1.5)
    plt.ylim(0, 1)
    plt.axvline(x=0, color="r")
    plt.xlabel("First year GPA minus probation cutoff")
    plt.ylabel("Has Graduated")

    plt.plot(
        data.gradin4.groupby(data["dist_from_cut_med10"]).mean(),
        "o",
        color="k",
        label="Within 4 years",
    )
    plot_RDD_curve_colored(
        df=pred_1,
        running_variable="dist_from_cut",
        outcome="prediction",
        cutoff=0,
        color="k",
    )

    plt.plot(
        data.gradin5.groupby(data["dist_from_cut_med10"]).mean(),
        "x",
        color="C0",
        label="Within 5 years",
    )
    plot_RDD_curve_colored(
        df=pred_2,
        running_variable="dist_from_cut",
        outcome="prediction",
        cutoff=0,
        color="C0",
    )

    plt.plot(
        data.gradin6.groupby(data["dist_from_cut_med10"]).mean(),
        "^",
        color="g",
        label="Within 6 years",
    )
    plot_RDD_curve_colored(
        df=pred_3,
        running_variable="dist_from_cut",
        outcome="prediction",
        cutoff=0,
        color="g",
    )

    plt.legend()
    plt.title("Figure 5 - Graduation Rates")


def plot_figure4_with_CI(data, pred):
    """
    Plots Figure 4 with confidence intervals.
    """
    plt.figure(figsize=(8, 6))
    plt.xlim(-1.5, 1.5)
    plt.ylim(-0.5, 1.2)
    plt.axvline(x=0, color="r")
    plt.xlabel("First year GPA minus probation cutoff")
    plt.ylabel("Subsequent GPA minus Cutoff")
    plt.plot(data.nextGPA.groupby(data["dist_from_cut_med10"]).mean(), "o")
    plot_RDD_curve_CI(
        df=pred,
        running_variable="dist_from_cut",
        outcome="prediction",
        cutoff=0,
        lbound="lower_bound",
        ubound="upper_bound",
        CI_color="c",
        linecolor="orange",
    )

    plt.title("GPA in the next enrolled term with CI")


def plot_figure_credits_year2(data, pred):
    plt.figure(figsize=(8, 5))
    plt.xlim(-1.5, 1.5)
    plt.ylim(2.5, 5)
    plt.axvline(x=0, color="r")
    plt.xlabel("First year GPA minus probation cutoff")
    plt.ylabel("Total credits in year 2")
    plt.plot(data.total_credits_year2.groupby(data["dist_from_cut_med10"]).mean(), "o")
    plot_RDD_curve(
        df=pred, running_variable="dist_from_cut", outcome="prediction", cutoff=0
    )
    plt.title("Total credits in Second Year")


def plot_left_school_all(data, pred):
    plt.xlim(-1.5, 1.5)
    plt.ylim(0, 0.22)
    plt.axvline(x=0, color="r")
    plt.xlabel("First year GPA minus probation cutoff")
    plt.ylabel("Left university voluntarily")

    bin_means = data.left_school.groupby(data["dist_from_cut_med10"]).mean()
    bin_means = pd.Series.to_frame(bin_means)
    plt.plot(list(bin_means.index), list(bin_means.left_school), "o")

    plot_RDD_curve(
        df=pred, running_variable="dist_from_cut", outcome="prediction", cutoff=0
    )
    plt.title("Left university voluntarily")


def plot_nextCGPA(data, pred):
    plt.figure(figsize=(8, 5))
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1, 1.5)
    plt.axvline(x=0, color="r")
    plt.xlabel("First year GPA minus probation cutoff")
    plt.ylabel("Subsequent CGPA minus cutoff")
    plt.plot(data.nextCGPA.groupby(data["dist_from_cut_med10"]).mean(), "o")
    plot_RDD_curve(
        df=pred, running_variable="dist_from_cut", outcome="prediction", cutoff=0
    )
    plt.title("CGPA in the next enrolled term")
