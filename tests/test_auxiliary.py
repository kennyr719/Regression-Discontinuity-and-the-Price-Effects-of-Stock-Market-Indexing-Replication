"""Tests for auxiliary functions."""

import os
import tempfile

import numpy as np
import pandas as pd

from auxiliary.data_processing import (
    compute_banding_cutoffs,
    compute_market_cap_rankings,
    construct_comovement,
    construct_io_variable,
    construct_sr_variable,
    construct_volume_ratio,
    identify_index_switchers,
)
from auxiliary.estimation import (
    bandwidth_sensitivity,
    fuzzy_rd_estimate,
    fuzzy_rd_time_trend,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rankings(n=1500, seed=0):
    """Synthetic end-of-May rankings DataFrame for one year.

    Returns a DataFrame with columns PERMNO, rank, market_cap, gvkey, date,
    PRC, SHROUT, shares, SHRCD, EXCHCD — matching the output schema of
    compute_market_cap_rankings().
    """
    rng = np.random.default_rng(seed)
    permnos = np.arange(10000, 10000 + n)
    # Assign market caps so that rank order is deterministic
    market_caps = np.linspace(5000, 10, n)  # largest → smallest
    return pd.DataFrame({
        "PERMNO":     permnos,
        "rank":       np.arange(1, n + 1),
        "market_cap": market_caps,
        "gvkey":      np.arange(1, n + 1).astype(str),
        "date":       pd.Timestamp("2000-05-31"),
        "PRC":        rng.uniform(5, 200, n),
        "SHROUT":     rng.uniform(1000, 50000, n),
        "shares":     rng.uniform(1000, 50000, n),
        "SHRCD":      10,
        "EXCHCD":     1,
    })


def _make_all_rankings(year=2000, n=1500, seed=0):
    """Synthetic all_rankings dict for year t and year t-1."""
    return {
        year - 1: _make_rankings(n=n, seed=seed),
        year:     _make_rankings(n=n, seed=seed + 1),
    }


# ---------------------------------------------------------------------------
# TestMarketCapRankings
# ---------------------------------------------------------------------------

class TestMarketCapRankings:
    """Tests for ranking construction properties."""

    def test_rankings_are_descending(self):
        """Rank 1 should have the highest market cap; ranks are descending."""
        df = _make_rankings()
        assert df["rank"].is_monotonic_increasing
        # Market cap should be weakly decreasing with rank
        assert (df["market_cap"].diff().dropna() <= 0).all()

    def test_rankings_cover_full_universe(self):
        """Every stock in the synthetic universe should receive a rank."""
        n = 1500
        df = _make_rankings(n=n)
        assert len(df) == n
        assert df["rank"].nunique() == n

    def test_rank_1000_separates_indexes(self):
        """Stocks ranked ≤ 1000 belong to Russell 1000; > 1000 to Russell 2000."""
        df = _make_rankings(n=1500)
        r1000 = df[df["rank"] <= 1000]
        r2000 = df[df["rank"] > 1000]
        assert len(r1000) == 1000
        assert len(r2000) == 500
        # All R1000 stocks have larger market caps than all R2000 stocks
        assert r1000["market_cap"].min() > r2000["market_cap"].max()


# ---------------------------------------------------------------------------
# TestIndexSwitchers
# ---------------------------------------------------------------------------

class TestIndexSwitchers:
    """Tests for identify_index_switchers() sample construction."""

    def test_addition_sample_from_russell_1000(self):
        """Addition sample should only contain prior Russell 1000 members."""
        all_rankings = _make_all_rankings(year=2000)
        addition, _ = identify_index_switchers(all_rankings, year=2000, bandwidth=100)
        assert addition is not None
        # prev_rank ≤ 1000 means the stock was in Russell 1000 last year
        assert (addition["prev_rank"] <= 1000).all(), (
            "Addition sample contains a stock with prev_rank > 1000"
        )

    def test_deletion_sample_from_russell_2000(self):
        """Deletion sample should only contain prior Russell 2000 members."""
        all_rankings = _make_all_rankings(year=2000)
        _, deletion = identify_index_switchers(all_rankings, year=2000, bandwidth=100)
        assert deletion is not None
        # prev_rank > 1000 means the stock was in Russell 2000 last year
        assert (deletion["prev_rank"] > 1000).all(), (
            "Deletion sample contains a stock with prev_rank ≤ 1000"
        )

    def test_addition_within_bandwidth(self):
        """Addition sample should be within ±bandwidth of the cutoff."""
        bw = 100
        all_rankings = _make_all_rankings(year=2000)
        addition, _ = identify_index_switchers(all_rankings, year=2000, bandwidth=bw)
        assert addition is not None
        assert (addition["rank_centered"].abs() <= bw).all()

    def test_deletion_within_bandwidth(self):
        """Deletion sample should be within ±bandwidth of the cutoff."""
        bw = 100
        all_rankings = _make_all_rankings(year=2000)
        _, deletion = identify_index_switchers(all_rankings, year=2000, bandwidth=bw)
        assert deletion is not None
        assert (deletion["rank_centered"].abs() <= bw).all()

    def test_missing_prior_year_returns_none(self):
        """Returns (None, None) when the prior year is absent from all_rankings."""
        all_rankings = {2000: _make_rankings()}  # no 1999
        addition, deletion = identify_index_switchers(all_rankings, year=2000)
        assert addition is None
        assert deletion is None

    def test_tau_equals_d_without_bloomberg(self):
        """Without bloomberg_panel, D should equal τ for every observation."""
        all_rankings = _make_all_rankings(year=2000)
        addition, deletion = identify_index_switchers(all_rankings, year=2000)
        for df in (addition, deletion):
            if df is not None and len(df) > 0:
                assert (df["D"] == df["tau"]).all()


# ---------------------------------------------------------------------------
# TestFuzzyRD
# ---------------------------------------------------------------------------

class TestFuzzyRD:
    """Tests for the fuzzy RD 2SLS estimator."""

    @staticmethod
    def _sharp_rd_df(n=1000, true_effect=0.05, noise=0.1, seed=42):
        """Synthetic sharp-RD dataset (D = τ, perfect compliance)."""
        rng = np.random.default_rng(seed)
        running = rng.uniform(-100, 100, n)
        treatment = (running > 0).astype(int)
        outcome = 0.01 + true_effect * treatment + rng.normal(0, noise, n)
        years = rng.choice([1998, 1999, 2000, 2001, 2002], n)
        return pd.DataFrame({
            "outcome":      outcome,
            "D":            treatment,
            "rank_centered": running,
            "tau":          treatment,   # sharp RD: instrument = treatment
            "year":         years,
        })

    def test_sharp_rd_recovers_known_effect(self):
        """With perfect compliance, fuzzy 2SLS should recover the true effect."""
        true_effect = 0.05
        df = self._sharp_rd_df(n=1000, true_effect=true_effect, noise=0.1)
        result = fuzzy_rd_estimate(df, "outcome", year_fe=True)
        assert result is not None
        assert abs(result["coef"] - true_effect) < 0.03, (
            f"Estimated {result['coef']:.4f}, expected ~{true_effect}"
        )

    def test_returns_none_when_sample_too_small(self):
        """Should return None when the sample is smaller than the parameter count."""
        df = self._sharp_rd_df(n=5)  # tiny sample — fewer obs than parameters
        result = fuzzy_rd_estimate(df, "outcome", bandwidth=1, year_fe=False)
        assert result is None

    def test_bandwidth_restricts_sample(self):
        """n_obs with narrow bandwidth should be fewer than with wide bandwidth."""
        df = self._sharp_rd_df(n=2000)
        result_narrow = fuzzy_rd_estimate(df, "outcome", bandwidth=20, year_fe=True)
        result_wide   = fuzzy_rd_estimate(df, "outcome", bandwidth=80, year_fe=True)
        assert result_narrow is not None and result_wide is not None
        assert result_narrow["n_obs"] < result_wide["n_obs"]

    def test_result_keys_present(self):
        """Result dict should contain all documented keys."""
        df = self._sharp_rd_df()
        result = fuzzy_rd_estimate(df, "outcome")
        expected_keys = {
            "coef", "se", "t_stat", "p_value", "r_squared", "n_obs",
            "fs_alpha_0r", "fs_alpha_0r_t", "fs_r2", "fs_F",
        }
        assert expected_keys.issubset(result.keys())

    def test_first_stage_jump_near_one_for_sharp_rd(self):
        """Under perfect compliance (D = τ), fs_alpha_0r should be ≈ 1."""
        df = self._sharp_rd_df(n=2000)
        result = fuzzy_rd_estimate(df, "outcome", year_fe=True)
        assert result is not None
        assert abs(result["fs_alpha_0r"] - 1.0) < 0.05, (
            f"First-stage jump = {result['fs_alpha_0r']:.4f}, expected ≈ 1.0"
        )

    def test_zero_effect_not_significant(self):
        """With no treatment effect, estimate should not be significant at 1%."""
        df = self._sharp_rd_df(n=1000, true_effect=0.0, noise=0.15)
        result = fuzzy_rd_estimate(df, "outcome")
        assert result is not None
        assert result["p_value"] > 0.01, (
            f"p_value = {result['p_value']:.4f}"
            " — spurious significance with zero effect"
        )

    def test_quadratic_poly_runs_without_error(self):
        """poly_degree=2 robustness check should run and return valid output."""
        df = self._sharp_rd_df(n=1000)
        result = fuzzy_rd_estimate(df, "outcome", poly_degree=2, year_fe=True)
        assert result is not None
        assert np.isfinite(result["coef"])


# ---------------------------------------------------------------------------
# TestFloatShareRankings
# ---------------------------------------------------------------------------

class TestFloatShareRankings:
    """Tests for Bloomberg float-adjusted shares in compute_market_cap_rankings()."""

    YEAR = 2010
    PERMNOS = [1001, 1002, 1003]
    # CRSP shares outstanding (thousands)
    SHROUT = {1001: 10_000, 1002: 20_000, 1003: 30_000}

    @staticmethod
    def _make_data_dict(year=2010, permnos=None, shrout=None):
        """Minimal synthetic data dict matching merge_crsp_compustat() output."""
        if permnos is None:
            permnos = [1001, 1002, 1003]
        if shrout is None:
            shrout = {1001: 10_000, 1002: 20_000, 1003: 30_000}

        may_date = pd.Timestamp(f"{year}-05-31")

        crsp_monthly = pd.DataFrame({
            "PERMNO":  permnos,
            "date":    [may_date] * len(permnos),
            "PRC":     [50.0] * len(permnos),
            "SHROUT":  [shrout[p] for p in permnos],
            "SHRCD":   [10] * len(permnos),
            "EXCHCD":  [1] * len(permnos),
            "CFACSHR": [1.0] * len(permnos),
        })

        # Empty Compustat: no CSHOQ data so CRSP SHROUT is always used.
        # Explicit dtypes avoid an AttributeError when the .dt accessor is
        # called on an empty object-dtype qtr_end_date column after the merge.
        compustat_quarterly = pd.DataFrame({
            "gvkey":          pd.Series(dtype="object"),
            "datadate":       pd.Series(dtype="datetime64[ns]"),
            "cshoq":          pd.Series(dtype="float64"),
            "available_date": pd.Series(dtype="datetime64[ns]"),
            "rdq":            pd.Series(dtype="datetime64[ns]"),
            "fqtr":           pd.Series(dtype="float64"),
            "fyearq":         pd.Series(dtype="float64"),
        })

        gvkeys = [f"G{p}" for p in permnos]
        ccm_link = pd.DataFrame({
            "LPERMNO":   permnos,
            "gvkey":     gvkeys,
            "LINKPRIM":  ["P"] * len(permnos),
            "LINKDT":    [pd.Timestamp(f"{year-5}-01-01")] * len(permnos),
            "LINKENDDT": [pd.Timestamp("2099-12-31")] * len(permnos),
        })

        cfacshr_lookup = pd.DataFrame({
            "PERMNO":  permnos,
            "ym":      [year * 100 + 5] * len(permnos),
            "CFACSHR": [1.0] * len(permnos),
        })

        return {
            "crsp_monthly":        crsp_monthly,
            "compustat_quarterly": compustat_quarterly,
            "ccm_link":            ccm_link,
            "cfacshr_lookup":      cfacshr_lookup,
        }

    def test_float_shares_used_when_provided(self):
        """Stock with float < SHROUT should use float shares in rankings."""
        data = self._make_data_dict()
        # PERMNO 1001: float=5000 < SHROUT=10000 → float wins
        float_df = pd.DataFrame({
            "PERMNO":           [1001],
            "float_shares_thou": [5_000.0],
        })
        result = compute_market_cap_rankings(data, self.YEAR, float_shares_df=float_df)
        row = result[result["PERMNO"] == 1001].iloc[0]
        assert row["shares"] == 5_000.0, (
            f"Expected float shares=5000, got {row['shares']}"
        )

    def test_fallback_when_float_missing(self):
        """Stocks absent from float_shares_df should use max(CRSP, Compustat)."""
        data = self._make_data_dict()
        # Only provide float for PERMNO 1001; 1002 and 1003 should fall back
        float_df = pd.DataFrame({
            "PERMNO":           [1001],
            "float_shares_thou": [5_000.0],
        })
        result = compute_market_cap_rankings(data, self.YEAR, float_shares_df=float_df)
        for permno, expected in [(1002, 20_000), (1003, 30_000)]:
            row = result[result["PERMNO"] == permno].iloc[0]
            assert row["shares"] == expected, (
                f"PERMNO {permno}: expected fallback "
                f"shares={expected}, got {row['shares']}"
            )

    def test_none_float_df_backward_compatible(self):
        """float_shares_df=None should produce identical output to old behavior."""
        data = self._make_data_dict()
        result_none = compute_market_cap_rankings(data, self.YEAR, float_shares_df=None)
        result_old  = compute_market_cap_rankings(data, self.YEAR)
        pd.testing.assert_frame_equal(
            result_none.reset_index(drop=True),
            result_old.reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# TestBandingCutoffs
# ---------------------------------------------------------------------------

class TestBandingCutoffs:
    """Tests for compute_banding_cutoffs()."""

    @staticmethod
    def _make_banding_rankings(n=2000):
        """Synthetic rankings with known market caps for banding tests.

        Market caps decline linearly: rank 1 = 10000, rank n = 1.
        This allows predictable reverse cumulative percentages.
        """
        ranks = np.arange(1, n + 1)
        market_caps = np.linspace(10000, 1, n)
        return pd.DataFrame({"rank": ranks, "market_cap": market_caps})

    def test_addition_cutoff_above_1000(self):
        """Addition cutoff should be > 1000 (stocks below rank 1000 are banded)."""
        df = self._make_banding_rankings()
        k_add, k_del = compute_banding_cutoffs(df, year=2010)
        assert k_add > 1000, f"Addition cutoff {k_add} should be > 1000"

    def test_deletion_cutoff_below_1000(self):
        """Deletion cutoff should be < 1000 (stocks above rank 1000 are banded)."""
        df = self._make_banding_rankings()
        k_add, k_del = compute_banding_cutoffs(df, year=2010)
        assert k_del < 1000, f"Deletion cutoff {k_del} should be < 1000"

    def test_degenerate_returns_1000(self):
        """Fewer than 1000 stocks should return (1000, 1000)."""
        df = pd.DataFrame({"rank": [1, 2, 3], "market_cap": [100, 50, 10]})
        assert compute_banding_cutoffs(df, year=2010) == (1000, 1000)

    def test_band_contains_rank_1000(self):
        """The band [k_del, k_add] should contain rank 1000."""
        df = self._make_banding_rankings()
        k_add, k_del = compute_banding_cutoffs(df, year=2010)
        assert k_del < 1000 < k_add


# ---------------------------------------------------------------------------
# TestVolumeRatio
# ---------------------------------------------------------------------------

class TestVolumeRatio:
    """Tests for construct_volume_ratio()."""

    @staticmethod
    def _make_vr_data(year=2005):
        """Minimal CRSP monthly data with known volumes for VR verification.

        Trailing 6 months for June = Dec(year-1), Jan, Feb, Mar, Apr, May.
        """
        rows = []
        permnos = [100, 200]
        for p in permnos:
            # Dec of prior year
            rows.append({
                "PERMNO": p, "date": pd.Timestamp(f"{year-1}-12-28"),
                "PRC": 50.0, "VOL": 1000.0, "EXCHCD": 1, "SHRCD": 10,
                "RET": 0.01,
            })
            # Jan–May of year: trailing months
            for m in range(1, 6):
                rows.append({
                    "PERMNO": p, "date": pd.Timestamp(f"{year}-{m:02d}-28"),
                    "PRC": 50.0, "VOL": 1000.0, "EXCHCD": 1, "SHRCD": 10,
                    "RET": 0.01,
                })
            # June of year: stock 100 volume = 2000, stock 200 volume = 500
            vol_june = 2000.0 if p == 100 else 500.0
            rows.append({
                "PERMNO": p, "date": pd.Timestamp(f"{year}-06-28"),
                "PRC": 50.0, "VOL": vol_june, "EXCHCD": 1, "SHRCD": 10,
                "RET": 0.01,
            })
        crsp = pd.DataFrame(rows)
        crsp["date"] = pd.to_datetime(crsp["date"])
        return {"crsp_monthly": crsp}

    def test_vr_formula(self):
        """VR = (V_stock / V̄_stock) / (V_market / V̄_market)."""
        data = self._make_vr_data(year=2005)
        result = construct_volume_ratio(data, year=2005, months=(6,))
        assert "vr_jun" in result.columns
        assert len(result) == 2
        # Both stocks have trailing avg volume = 1000
        # Market trailing avg = 2 * 1000 = 2000 per month
        # June market volume = 2000 + 500 = 2500
        # Market VR = 2500 / 2000 = 1.25
        # Stock 100: stock VR = 2000/1000 = 2.0, final VR = 2.0 / 1.25 = 1.6
        row100 = result[result["PERMNO"] == 100].iloc[0]
        assert abs(row100["vr_jun"] - 1.6) < 0.01, (
            f"Stock 100 VR = {row100['vr_jun']:.4f}, expected 1.6"
        )
        # Stock 200: stock VR = 500/1000 = 0.5, final VR = 0.5 / 1.25 = 0.4
        row200 = result[result["PERMNO"] == 200].iloc[0]
        assert abs(row200["vr_jun"] - 0.4) < 0.01, (
            f"Stock 200 VR = {row200['vr_jun']:.4f}, expected 0.4"
        )

    def test_vr_has_year_column(self):
        """Result should include a 'year' column."""
        data = self._make_vr_data(year=2005)
        result = construct_volume_ratio(data, year=2005, months=(6,))
        assert "year" in result.columns
        assert (result["year"] == 2005).all()

    def test_nasdaq_adjustment_pre_2004(self):
        """NASDAQ stocks (EXCHCD=3) before 2004 should have halved volume."""
        data = self._make_vr_data(year=2003)
        # Switch to NASDAQ exchange — all volumes get halved
        data["crsp_monthly"]["EXCHCD"] = 3
        result = construct_volume_ratio(data, year=2003, months=(6,))
        assert len(result) > 0
        # After halving: trailing avg per stock = 500, June stock 100 = 1000
        # Market trailing avg = 2*500 = 1000, June market = 1000 + 250 = 1250
        # Stock 100: 1000/500 = 2.0, mkt = 1250/1000 = 1.25, VR = 2.0/1.25 = 1.6
        row100 = result[result["PERMNO"] == 100].iloc[0]
        assert abs(row100["vr_jun"] - 1.6) < 0.01


# ---------------------------------------------------------------------------
# TestFuzzyRDTimeTrend
# ---------------------------------------------------------------------------

class TestFuzzyRDTimeTrend:
    """Tests for fuzzy_rd_time_trend()."""

    @staticmethod
    def _trend_df(n=2000, base_effect=0.05, trend=0.01, noise=0.08, seed=99):
        """Synthetic sharp-RD dataset with a known time trend."""
        rng = np.random.default_rng(seed)
        running = rng.uniform(-100, 100, n)
        treatment = (running > 0).astype(int)
        years = rng.choice(np.arange(2000, 2010), n)
        t = years - 2000
        outcome = (
            0.01
            + (base_effect + trend * t) * treatment
            + rng.normal(0, noise, n)
        )
        return pd.DataFrame({
            "outcome":       outcome,
            "D":             treatment,
            "rank_centered": running,
            "tau":           treatment,
            "year":          years,
        })

    def test_recovers_base_effect(self):
        """Should recover the base treatment effect at t=0."""
        df = self._trend_df(n=3000, base_effect=0.05, trend=0.01)
        res = fuzzy_rd_time_trend(df, "outcome", base_year=2000)
        assert res is not None
        assert abs(res["coef"] - 0.05) < 0.03, (
            f"Base effect = {res['coef']:.4f}, expected ~0.05"
        )

    def test_recovers_time_trend(self):
        """Should recover the annual trend coefficient."""
        df = self._trend_df(n=3000, base_effect=0.05, trend=0.01)
        res = fuzzy_rd_time_trend(df, "outcome", base_year=2000)
        assert res is not None
        assert abs(res["coef_t"] - 0.01) < 0.008, (
            f"Trend = {res['coef_t']:.4f}, expected ~0.01"
        )

    def test_result_keys(self):
        """Result should contain all documented keys."""
        df = self._trend_df()
        res = fuzzy_rd_time_trend(df, "outcome", base_year=2000)
        assert res is not None
        expected = {"coef", "coef_t", "se", "se_t", "t_stat", "t_stat_t",
                    "p_value", "p_value_t", "r_squared", "n_obs",
                    "fs_alpha_0r", "fs_alpha_0r_t", "fs_r2", "fs_F"}
        assert expected.issubset(res.keys())

    def test_returns_none_tiny_sample(self):
        """Should return None for a too-small sample."""
        df = self._trend_df(n=5)
        res = fuzzy_rd_time_trend(df, "outcome", bandwidth=1, base_year=2000)
        assert res is None


# ---------------------------------------------------------------------------
# TestBandwidthSensitivity
# ---------------------------------------------------------------------------

class TestBandwidthSensitivity:
    """Tests for bandwidth_sensitivity()."""

    @staticmethod
    def _rd_df(n=1000, seed=42):
        rng = np.random.default_rng(seed)
        running = rng.uniform(-100, 100, n)
        treatment = (running > 0).astype(int)
        return pd.DataFrame({
            "outcome":       0.01 + 0.05 * treatment + rng.normal(0, 0.1, n),
            "D":             treatment,
            "rank_centered": running,
            "tau":           treatment,
            "year":          rng.choice([2000, 2001, 2002], n),
        })

    def test_returns_correct_number_of_rows(self):
        """Should return one row per bandwidth."""
        df = self._rd_df(n=2000)
        result = bandwidth_sensitivity(df, "outcome", bandwidths=(50, 100, 150))
        assert len(result) == 3

    def test_bandwidth_column_matches(self):
        """Each row should have the correct bandwidth value."""
        df = self._rd_df(n=2000)
        result = bandwidth_sensitivity(df, "outcome", bandwidths=(30, 60))
        assert list(result["bandwidth"]) == [30, 60]

    def test_n_obs_increases_with_bandwidth(self):
        """Wider bandwidths should have more observations."""
        df = self._rd_df(n=2000)
        result = bandwidth_sensitivity(df, "outcome", bandwidths=(30, 60, 90))
        n_obs = result["n_obs"].tolist()
        assert n_obs == sorted(n_obs), "n_obs should increase with bandwidth"


# ---------------------------------------------------------------------------
# TestConstructIO
# ---------------------------------------------------------------------------

class TestConstructIO:
    """Tests for construct_io_variable()."""

    def test_io_merges_via_ncusip(self):
        """IO should merge using 8-digit NCUSIP → Thomson cusip."""
        panel = pd.DataFrame({
            "PERMNO": [1, 2],
            "year":   [2010, 2010],
            "NCUSIP": ["12345678", "87654321"],
        })
        io_csv = (
            "rdate,cusip,InstOwn_Perc\n"
            "2010-06-30,12345678,0.45\n"
            "2010-06-30,87654321,0.60\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(io_csv)
            f.flush()
            result = construct_io_variable(panel, f.name)
        os.unlink(f.name)

        assert "IO" in result.columns
        assert abs(result.loc[result["PERMNO"] == 1, "IO"].iloc[0] - 0.45) < 1e-6
        assert abs(result.loc[result["PERMNO"] == 2, "IO"].iloc[0] - 0.60) < 1e-6

    def test_q2_preferred_over_q1(self):
        """Q2 observation should be preferred when both Q1 and Q2 exist."""
        panel = pd.DataFrame({
            "PERMNO": [1],
            "year":   [2010],
            "NCUSIP": ["12345678"],
        })
        io_csv = (
            "rdate,cusip,InstOwn_Perc\n"
            "2010-03-31,12345678,0.30\n"
            "2010-06-30,12345678,0.55\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(io_csv)
            f.flush()
            result = construct_io_variable(panel, f.name)
        os.unlink(f.name)

        assert abs(result["IO"].iloc[0] - 0.55) < 1e-6

    def test_missing_ncusip_returns_nan(self):
        """Panel without NCUSIP column should get NaN IO."""
        panel = pd.DataFrame({"PERMNO": [1], "year": [2010]})
        io_csv = "rdate,cusip,InstOwn_Perc\n2010-06-30,12345678,0.50\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(io_csv)
            f.flush()
            result = construct_io_variable(panel, f.name)
        os.unlink(f.name)

        assert result["IO"].isna().all()


# ---------------------------------------------------------------------------
# TestConstructSR
# ---------------------------------------------------------------------------

class TestConstructSR:
    """Tests for construct_sr_variable()."""

    @staticmethod
    def _make_sr_data():
        """Synthetic short interest CSV, CCM link, and panel."""
        panel = pd.DataFrame({
            "PERMNO": [1001, 1002],
            "year":   [2010, 2010],
            "SHROUT": [10000.0, 20000.0],  # thousands
        })
        sr_csv = (
            "gvkey,datadate,shortintadj\n"
            "000001,2010-06-15,500000\n"  # 500K shares short
            "000002,2010-06-20,1000000\n"  # 1M shares short
        )
        ccm_link = pd.DataFrame({
            "gvkey":     ["000001", "000002"],
            "LPERMNO":   [1001, 1002],
            "LINKPRIM":  ["P", "P"],
            "LINKDT":    [pd.Timestamp("2005-01-01"), pd.Timestamp("2005-01-01")],
            "LINKENDDT": [pd.Timestamp("2099-12-31"), pd.Timestamp("2099-12-31")],
        })
        return panel, sr_csv, ccm_link

    def test_sr_formula(self):
        """SR = shortintadj / (SHROUT * 1000)."""
        panel, sr_csv, ccm_link = self._make_sr_data()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(sr_csv)
            f.flush()
            result = construct_sr_variable(panel, f.name, ccm_link)
        os.unlink(f.name)

        assert "SR" in result.columns
        # PERMNO 1001: 500000 / (10000 * 1000) = 0.05
        sr1 = result.loc[result["PERMNO"] == 1001, "SR"].iloc[0]
        assert abs(sr1 - 0.05) < 1e-6, f"Expected SR=0.05, got {sr1}"
        # PERMNO 1002: 1000000 / (20000 * 1000) = 0.05
        sr2 = result.loc[result["PERMNO"] == 1002, "SR"].iloc[0]
        assert abs(sr2 - 0.05) < 1e-6, f"Expected SR=0.05, got {sr2}"

    def test_sr_nan_when_no_data(self):
        """SR should be NaN for PERMNOs not in short interest file."""
        panel = pd.DataFrame({
            "PERMNO": [9999], "year": [2010], "SHROUT": [5000.0],
        })
        sr_csv = "gvkey,datadate,shortintadj\n000001,2010-06-15,100000\n"
        ccm_link = pd.DataFrame({
            "gvkey":     ["000001"],
            "LPERMNO":   [1001],
            "LINKPRIM":  ["P"],
            "LINKDT":    [pd.Timestamp("2005-01-01")],
            "LINKENDDT": [pd.Timestamp("2099-12-31")],
        })
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(sr_csv)
            f.flush()
            result = construct_sr_variable(panel, f.name, ccm_link)
        os.unlink(f.name)

        assert result["SR"].isna().all()

    def test_date_bounded_linking(self):
        """CCM links should be filtered by date — expired links ignored."""
        panel = pd.DataFrame({
            "PERMNO": [1001], "year": [2015], "SHROUT": [10000.0],
        })
        sr_csv = "gvkey,datadate,shortintadj\n000001,2015-06-15,500000\n"
        # Link expired in 2012 — should NOT match for year 2015
        ccm_link = pd.DataFrame({
            "gvkey":     ["000001"],
            "LPERMNO":   [1001],
            "LINKPRIM":  ["P"],
            "LINKDT":    [pd.Timestamp("2005-01-01")],
            "LINKENDDT": [pd.Timestamp("2012-12-31")],
        })
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(sr_csv)
            f.flush()
            result = construct_sr_variable(panel, f.name, ccm_link)
        os.unlink(f.name)

        assert result["SR"].isna().all(), "Expired CCM link should not match"


# ---------------------------------------------------------------------------
# TestConstructComovement
# ---------------------------------------------------------------------------

class TestConstructComovement:
    """Tests for construct_comovement()."""

    @staticmethod
    def _make_comovement_data(seed=42):
        """Create synthetic daily returns for 1 stock × 1 month (20 days).

        Stock return = 1.5 × market return + noise → beta ≈ 1.5.
        """
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2010-06-01", periods=20)
        rut_ret = rng.normal(0, 0.01, 20)
        stock_ret = 1.5 * rut_ret + rng.normal(0, 0.002, 20)

        daily_df = pd.DataFrame({
            "PERMNO": 1001,
            "date": dates,
            "RET": stock_ret,
        })
        rut_df = pd.DataFrame({
            "date": dates,
            "rut_return": rut_ret,
        })
        panel = pd.DataFrame({"PERMNO": [1001], "year": [2010]})
        return panel, daily_df, rut_df

    def test_beta_near_true_value(self):
        """Estimated beta should be close to the true slope (1.5)."""
        panel, daily_df, rut_df = self._make_comovement_data()
        # Write to temp files
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f_d:
            daily_df.to_csv(f_d.name, index=False)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f_r:
            rut_df.to_csv(f_r.name, index=False)

        result = construct_comovement(
            panel, f_d.name, f_r.name, months=(6,), min_days=15,
        )
        os.unlink(f_d.name)
        os.unlink(f_r.name)

        assert "cov_jun" in result.columns
        beta = result["cov_jun"].iloc[0]
        assert abs(beta - 1.5) < 0.3, f"Beta = {beta:.3f}, expected ~1.5"

    def test_too_few_days_returns_nan(self):
        """Beta should be NaN if fewer than min_days observations."""
        panel, daily_df, rut_df = self._make_comovement_data()
        # Keep only 5 days
        daily_df = daily_df.head(5)
        rut_df = rut_df.head(5)

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f_d:
            daily_df.to_csv(f_d.name, index=False)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f_r:
            rut_df.to_csv(f_r.name, index=False)

        result = construct_comovement(
            panel, f_d.name, f_r.name, months=(6,), min_days=15,
        )
        os.unlink(f_d.name)
        os.unlink(f_r.name)

        assert result["cov_jun"].isna().all()
