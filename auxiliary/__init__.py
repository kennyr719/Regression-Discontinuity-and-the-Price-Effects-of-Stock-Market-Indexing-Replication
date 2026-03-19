"""Auxiliary functions for the Russell RD replication project."""

from auxiliary.data_processing import (  # noqa: F401
    compute_banding_cutoffs,
    compute_market_cap_rankings,
    construct_comovement,
    construct_io_variable,
    construct_outcome_variables,
    construct_sr_variable,
    construct_validity_variables,
    construct_volume_ratio,
    identify_index_switchers,
    load_bloomberg_float,
    match_bloomberg_to_crsp,
    merge_crsp_compustat,
)
from auxiliary.estimation import (  # noqa: F401
    bandwidth_sensitivity,
    fuzzy_rd_estimate,
    fuzzy_rd_time_trend,
    optimal_bandwidth,
    reduced_form_estimate,
    reduced_form_time_trend,
)
from auxiliary.plotting import (  # noqa: F401
    plot_first_stage,
    plot_market_cap_continuity,
    plot_rd_discontinuity,
    plot_time_trends,
)
