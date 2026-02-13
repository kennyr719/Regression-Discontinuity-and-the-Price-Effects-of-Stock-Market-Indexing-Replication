"""Auxiliary functions for the Russell RD replication project."""

from auxiliary.data_processing import (
    compute_market_cap_rankings,
    identify_index_switchers,
    merge_crsp_compustat,
)
from auxiliary.estimation import (
    fuzzy_rd_estimate,
    optimal_bandwidth,
)
from auxiliary.plotting import (
    plot_rd_discontinuity,
    plot_market_cap_continuity,
)
