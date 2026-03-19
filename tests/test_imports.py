"""Smoke tests to verify all modules can be imported and key symbols exist."""


def test_import_data_processing():
    """Test that data_processing module exports all expected functions."""
    from auxiliary import (
        compute_banding_cutoffs,
        compute_market_cap_rankings,
        construct_comovement,
        construct_io_variable,
        construct_outcome_variables,
        construct_sr_variable,
        construct_validity_variables,
        construct_volume_ratio,
        identify_index_switchers,
        match_bloomberg_to_crsp,
        merge_crsp_compustat,
    )

    assert callable(merge_crsp_compustat)
    assert callable(compute_market_cap_rankings)
    assert callable(compute_banding_cutoffs)
    assert callable(match_bloomberg_to_crsp)
    assert callable(identify_index_switchers)
    assert callable(construct_outcome_variables)
    assert callable(construct_volume_ratio)
    assert callable(construct_io_variable)
    assert callable(construct_sr_variable)
    assert callable(construct_comovement)
    assert callable(construct_validity_variables)


def test_import_estimation():
    """Test that estimation module exports all expected functions."""
    from auxiliary import (
        bandwidth_sensitivity,
        fuzzy_rd_estimate,
        fuzzy_rd_time_trend,
        optimal_bandwidth,
    )

    assert callable(fuzzy_rd_estimate)
    assert callable(fuzzy_rd_time_trend)
    assert callable(optimal_bandwidth)
    assert callable(bandwidth_sensitivity)


def test_import_plotting():
    """Test that plotting module exports all expected functions."""
    from auxiliary import (
        plot_first_stage,
        plot_market_cap_continuity,
        plot_rd_discontinuity,
        plot_time_trends,
    )

    assert callable(plot_first_stage)
    assert callable(plot_market_cap_continuity)
    assert callable(plot_rd_discontinuity)
    assert callable(plot_time_trends)
