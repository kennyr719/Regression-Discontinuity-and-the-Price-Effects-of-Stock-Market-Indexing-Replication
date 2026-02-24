"""Functions for fuzzy RD estimation and inference."""

import numpy as np
import pandas as pd
from scipy.special import betainc  # scipy.stats is broken in base env; betainc is stable


def fuzzy_rd_estimate(
    df, outcome, treatment="D", running="rank_centered",
    instrument="tau", cutoff=0, bandwidth=100, year_fe=True,
):
    """Estimate fuzzy RD treatment effect using 2SLS.

    Implements Chang et al. (2015, Section 4.1).

    First stage:
        D_it = α_0l + α_1l(r_it) + τ_it[α_0r + α_1r(r_it)] + [γ_t] + ε_it

    Second stage (2SLS):
        Y_it = β_0l + β_1l(r_it) + D̂_it[β_0r + β_1r(r_it)] + [δ_t] + ν_it

    where r_it = running variable centered at the cutoff, τ is the instrument
    (indicator for predicted index crossing), D̂ is the first-stage fitted
    value, and γ_t / δ_t are optional year fixed effects (year_fe=True).

    Standard errors use the 2SLS formula:
        Var(β̂) = σ² (X̂'X̂)⁻¹,  σ² from residuals Y - X·β̂ (original X).

    Note: When D = τ (no actual Russell constituent lists available), the
    first stage is near-perfect (α_0r ≈ 1) and 2SLS ≈ sharp-RD OLS.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing outcome, treatment, running variable, instrument,
        and (if year_fe=True) a 'year' column.
    outcome : str
        Column name of the outcome variable.
    treatment : str, optional
        Column name of the treatment indicator D (default: 'D').
    running : str, optional
        Column name of the centered running variable (default: 'rank_centered').
    instrument : str, optional
        Column name of the instrument τ (default: 'tau').
    cutoff : float, optional
        Value to subtract from `running` before regression (default: 0).
    bandwidth : int, optional
        Half-width of the estimation window (default: 100).
    year_fe : bool, optional
        Include year fixed effects (default: True, matches Chang et al.).

    Returns
    -------
    dict or None
        None if the sample is too small.  Otherwise a dict with keys:

        Second stage:
          'coef'        β_0r — treatment effect at the cutoff
          'se'          standard error of β_0r
          't_stat'      t-statistic
          'p_value'     two-sided p-value
          'r_squared'   R² of the second stage
          'n_obs'       number of observations used

        First stage:
          'fs_alpha_0r'   α_0r — first-stage jump at the cutoff
          'fs_alpha_0r_t' t-statistic on α_0r
          'fs_r2'         first-stage R²
          'fs_F'          F-statistic (joint test: α_0r = α_1r = 0)
    """
    # ── 0. Restrict to bandwidth window and drop missing ──────────────────
    in_bw = (df[running] >= cutoff - bandwidth) & (df[running] <= cutoff + bandwidth)
    cols_needed = [outcome, treatment, running, instrument]
    sample = df[in_bw].dropna(subset=cols_needed).copy()

    n = len(sample)

    # ── Build year fixed-effect dummies ───────────────────────────────────
    use_fe = year_fe and "year" in sample.columns
    if use_fe:
        unique_years = np.sort(sample["year"].unique())
        T = len(unique_years)
        if T > 1:
            year_map = {y: i for i, y in enumerate(unique_years)}
            year_idx = np.array([year_map[y] for y in sample["year"].values])
            # T-1 dummies: drop first year to avoid collinearity with intercept
            fe = np.zeros((n, T - 1))
            for j in range(1, T):
                fe[:, j - 1] = (year_idx == j).astype(float)
        else:
            fe = np.empty((n, 0))
    else:
        fe = np.empty((n, 0))

    k = 4 + fe.shape[1]  # intercept, slope, jump, jump×slope + year dummies
    if n <= k:
        return None

    df_resid = n - k
    r = sample[running].values.astype(float) - cutoff
    D = sample[treatment].values.astype(float)
    tau = sample[instrument].values.astype(float)
    Y = sample[outcome].values.astype(float)

    def _hstack(base, extra):
        return np.column_stack([base, extra]) if extra.shape[1] > 0 else base

    # ── 1. First stage: D ~ [1, r, τ, τ·r, year_fe] ──────────────────────
    Z = _hstack(np.column_stack([np.ones(n), r, tau, tau * r]), fe)
    alpha = np.linalg.lstsq(Z, D, rcond=None)[0]
    D_hat = Z @ alpha

    fs_resid = D - D_hat
    RSS_u = float(fs_resid @ fs_resid)
    fs_sigma2 = RSS_u / df_resid

    # F-statistic: restricted model drops τ and τ·r (keeps intercept, r, FEs)
    Z_r = _hstack(Z[:, :2], fe)
    alpha_r = np.linalg.lstsq(Z_r, D, rcond=None)[0]
    RSS_r = float(np.sum((D - Z_r @ alpha_r) ** 2))
    q = 2  # two restrictions: α_0r = α_1r = 0
    if RSS_u <= 0:
        fs_F = np.nan
    else:
        fs_F = ((RSS_r - RSS_u) / q) / (RSS_u / df_resid)
        # Near-perfect first stage (D = τ exactly): RSS_u ≈ machine epsilon
        # and the ratio overflows to astronomically large finite numbers.
        # Return np.inf so callers can format it as ">999" or similar.
        if RSS_u / max(RSS_r, 1e-15) < 1e-6:
            fs_F = np.inf

    try:
        ZtZ_inv = np.linalg.inv(Z.T @ Z)
    except np.linalg.LinAlgError:
        return None
    fs_alpha_0r = float(alpha[2])
    fs_alpha_0r_se = float(np.sqrt(max(fs_sigma2 * ZtZ_inv[2, 2], 0)))
    fs_alpha_0r_t = fs_alpha_0r / fs_alpha_0r_se if fs_alpha_0r_se > 0 else np.nan

    SS_D = float(np.sum((D - D.mean()) ** 2))
    fs_r2 = 1.0 - RSS_u / SS_D if SS_D > 0 else np.nan

    # ── 2. Second stage: Y ~ [1, r, D̂, D̂·r, year_fe] ───────────────────
    X_hat = _hstack(np.column_stack([np.ones(n), r, D_hat, D_hat * r]), fe)
    X_orig = _hstack(np.column_stack([np.ones(n), r, D, D * r]), fe)

    try:
        XhXh_inv = np.linalg.inv(X_hat.T @ X_hat)
    except np.linalg.LinAlgError:
        return None

    beta = XhXh_inv @ (X_hat.T @ Y)

    # 2SLS SEs: residuals from original X (not X̂)
    resid = Y - X_orig @ beta
    sigma2 = float(resid @ resid) / df_resid
    var_beta = sigma2 * XhXh_inv

    beta_0r = float(beta[2])
    beta_0r_se = float(np.sqrt(max(var_beta[2, 2], 0)))
    t_stat = beta_0r / beta_0r_se if beta_0r_se > 0 else np.nan
    # Two-sided p-value: I_{df/(df+t²)}(df/2, 0.5)  ≡  2*(1 - t_df.CDF(|t|))
    p_val = (
        float(betainc(df_resid / 2, 0.5, df_resid / (df_resid + t_stat**2)))
        if not np.isnan(t_stat) else np.nan
    )

    SS_Y = float(np.sum((Y - Y.mean()) ** 2))
    r2 = 1.0 - float(resid @ resid) / SS_Y if SS_Y > 0 else np.nan

    return {
        # Second stage
        "coef": beta_0r,
        "se": beta_0r_se,
        "t_stat": t_stat,
        "p_value": p_val,
        "r_squared": r2,
        "n_obs": n,
        # First stage
        "fs_alpha_0r": fs_alpha_0r,
        "fs_alpha_0r_t": fs_alpha_0r_t,
        "fs_r2": fs_r2,
        "fs_F": fs_F,
    }


def fuzzy_rd_time_trend(
    df, outcome, treatment="D", running="rank_centered",
    instrument="tau", cutoff=0, bandwidth=100, base_year=1996,
):
    """Estimate fuzzy RD with linear time trend interaction.

    Implements the time trend specification from Chang et al. (2015, Section 5,
    Equations 7–8):

    First stage:
        D_it = α_0l + α_1l(r_it) + α_2l*t
               + τ_it[α_0r + α_1r(r_it) + α_2r*t] + ε_it

    Second stage (2SLS):
        Y_it = β_0l + β_1l(r_it) + β_2l*t
               + D̂_it[β_0r + β_1r(r_it) + β_2r*t] + ν_it

    where t = year - base_year.  No year fixed effects are included; the
    continuous time trend replaces them.

    β_0r is the base treatment effect at t = 0 (= base_year).
    β_2r is the annual change in the treatment effect.

    Parameters
    ----------
    df : pd.DataFrame
        Data with outcome, treatment, running, instrument, and 'year' columns.
    outcome : str
        Column name of the outcome variable.
    treatment : str, optional
        Column name of the treatment indicator D (default: 'D').
    running : str, optional
        Column name of the centered running variable (default: 'rank_centered').
    instrument : str, optional
        Column name of the instrument τ (default: 'tau').
    cutoff : float, optional
        Value to subtract from `running` before regression (default: 0).
    bandwidth : int, optional
        Half-width of the estimation window (default: 100).
    base_year : int, optional
        Year corresponding to t = 0 (default: 1996).

    Returns
    -------
    dict or None
        None if the sample is too small.  Otherwise a dict with keys:

        Second stage:
          'coef'       β_0r — base treatment effect at cutoff (t = base_year)
          'coef_t'     β_2r — annual change in treatment effect
          'se'         standard error of β_0r
          'se_t'       standard error of β_2r
          't_stat'     t-statistic on β_0r
          't_stat_t'   t-statistic on β_2r
          'p_value'    two-sided p-value for β_0r
          'p_value_t'  two-sided p-value for β_2r
          'r_squared'  R² of the second stage
          'n_obs'      number of observations

        First stage:
          'fs_alpha_0r'    α_0r — first-stage jump at cutoff
          'fs_alpha_0r_t'  t-statistic on α_0r
          'fs_r2'          first-stage R²
          'fs_F'           F-statistic (joint test: α_0r = α_1r = α_2r = 0)
    """
    # ── 0. Restrict to bandwidth window and drop missing ──────────────────
    in_bw = (df[running] >= cutoff - bandwidth) & (df[running] <= cutoff + bandwidth)
    cols_needed = [outcome, treatment, running, instrument, "year"]
    sample = df[in_bw].dropna(subset=cols_needed).copy()

    n = len(sample)
    k = 6  # columns: [1, r, t, D/τ, D·r/τ·r, D·t/τ·t]
    if n <= k:
        return None

    df_resid = n - k

    r   = sample[running].values.astype(float) - cutoff
    D   = sample[treatment].values.astype(float)
    tau = sample[instrument].values.astype(float)
    Y   = sample[outcome].values.astype(float)
    t   = (sample["year"].values - base_year).astype(float)

    # ── 1. First stage: D ~ [1, r, t, τ, τ·r, τ·t] ──────────────────────
    Z = np.column_stack([np.ones(n), r, t, tau, tau * r, tau * t])
    alpha = np.linalg.lstsq(Z, D, rcond=None)[0]
    D_hat = Z @ alpha

    fs_resid = D - D_hat
    RSS_u = float(fs_resid @ fs_resid)

    # F-statistic: restricted model drops τ, τ·r, τ·t (keeps 1, r, t)
    Z_r = Z[:, :3]
    alpha_r = np.linalg.lstsq(Z_r, D, rcond=None)[0]
    RSS_r = float(np.sum((D - Z_r @ alpha_r) ** 2))
    q = 3  # three restrictions: α_0r = α_1r = α_2r = 0
    if RSS_u <= 0:
        fs_F = np.nan
    else:
        fs_F = ((RSS_r - RSS_u) / q) / (RSS_u / df_resid)
        if RSS_u / max(RSS_r, 1e-15) < 1e-6:
            fs_F = np.inf

    try:
        ZtZ_inv = np.linalg.inv(Z.T @ Z)
    except np.linalg.LinAlgError:
        return None

    fs_sigma2 = RSS_u / df_resid
    fs_alpha_0r = float(alpha[3])
    fs_alpha_0r_se = float(np.sqrt(max(fs_sigma2 * ZtZ_inv[3, 3], 0)))
    fs_alpha_0r_t = fs_alpha_0r / fs_alpha_0r_se if fs_alpha_0r_se > 0 else np.nan

    SS_D = float(np.sum((D - D.mean()) ** 2))
    fs_r2 = 1.0 - RSS_u / SS_D if SS_D > 0 else np.nan

    # ── 2. Second stage: Y ~ [1, r, t, D̂, D̂·r, D̂·t] ──────────────────
    X_hat  = np.column_stack([np.ones(n), r, t, D_hat, D_hat * r, D_hat * t])
    X_orig = np.column_stack([np.ones(n), r, t, D,     D * r,     D * t    ])

    try:
        XhXh_inv = np.linalg.inv(X_hat.T @ X_hat)
    except np.linalg.LinAlgError:
        return None

    beta = XhXh_inv @ (X_hat.T @ Y)

    # 2SLS SEs: residuals from original X (not X̂)
    resid  = Y - X_orig @ beta
    sigma2 = float(resid @ resid) / df_resid
    var_beta = sigma2 * XhXh_inv

    def _extract(idx):
        b   = float(beta[idx])
        se  = float(np.sqrt(max(var_beta[idx, idx], 0)))
        tv  = b / se if se > 0 else np.nan
        pv  = (
            float(betainc(df_resid / 2, 0.5, df_resid / (df_resid + tv**2)))
            if not np.isnan(tv) else np.nan
        )
        return b, se, tv, pv

    beta_0r, se_0r, t_0r, p_0r = _extract(3)  # base treatment effect
    beta_2r, se_2r, t_2r, p_2r = _extract(5)  # time trend in effect

    SS_Y = float(np.sum((Y - Y.mean()) ** 2))
    r2 = 1.0 - float(resid @ resid) / SS_Y if SS_Y > 0 else np.nan

    return {
        # Second stage
        "coef":       beta_0r,
        "coef_t":     beta_2r,
        "se":         se_0r,
        "se_t":       se_2r,
        "t_stat":     t_0r,
        "t_stat_t":   t_2r,
        "p_value":    p_0r,
        "p_value_t":  p_2r,
        "r_squared":  r2,
        "n_obs":      n,
        # First stage
        "fs_alpha_0r":   fs_alpha_0r,
        "fs_alpha_0r_t": fs_alpha_0r_t,
        "fs_r2":         fs_r2,
        "fs_F":          fs_F,
    }


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
