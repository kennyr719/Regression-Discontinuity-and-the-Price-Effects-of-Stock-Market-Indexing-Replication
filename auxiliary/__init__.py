"""Auxiliary functions for the Russell RD replication project."""

from auxiliary.data_processing import (
    compute_banding_cutoffs,
    compute_market_cap_rankings,
    construct_outcome_variables,
    construct_validity_variables,
    construct_volume_ratio,
    identify_index_switchers,
    merge_crsp_compustat,
)
from auxiliary.estimation import (
    bandwidth_sensitivity,
    fuzzy_rd_estimate,
    fuzzy_rd_time_trend,
    optimal_bandwidth,
)
from auxiliary.plotting import (
    plot_rd_discontinuity,
    plot_market_cap_continuity,
)
