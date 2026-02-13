"""Tests for auxiliary functions."""

import numpy as np
import pandas as pd
import pytest

from auxiliary.data_processing import compute_market_cap_rankings, identify_index_switchers
from auxiliary.estimation import fuzzy_rd_estimate


class TestMarketCapRankings:
    """Tests for market capitalization ranking construction."""

    def test_rankings_are_descending(self):
        """Market cap rankings should be in descending order of market cap."""
        # TODO: Implement with sample data
        pass

    def test_rankings_cover_full_universe(self):
        """All eligible firms should receive a ranking."""
        # TODO: Implement with sample data
        pass

    def test_rank_1000_separates_indexes(self):
        """Rank 1000 should be the cutoff between Russell 1000 and 2000."""
        # TODO: Implement with sample data
        pass


class TestIndexSwitchers:
    """Tests for identifying firms that switch indexes."""

    def test_addition_sample_from_russell_1000(self):
        """Addition sample should only contain prior Russell 1000 members."""
        # TODO: Implement with sample data
        pass

    def test_deletion_sample_from_russell_2000(self):
        """Deletion sample should only contain prior Russell 2000 members."""
        # TODO: Implement with sample data
        pass


class TestFuzzyRD:
    """Tests for the fuzzy RD estimator."""

    def test_sharp_rd_recovers_known_effect(self):
        """With perfect compliance, fuzzy RD should reduce to sharp RD."""
        np.random.seed(42)
        n = 500
        running = np.random.uniform(-100, 100, n)
        treatment = (running > 0).astype(int)
        true_effect = 0.05
        outcome = 0.01 + true_effect * treatment + np.random.normal(0, 0.1, n)

        df = pd.DataFrame({
            "outcome": outcome,
            "treatment": treatment,
            "running": running,
        })

        # TODO: Uncomment when fuzzy_rd_estimate is implemented
        # result = fuzzy_rd_estimate(df, "outcome", "treatment", "running")
        # assert abs(result["coefficient"] - true_effect) < 0.03
        pass

    def test_bandwidth_restriction(self):
        """Only observations within bandwidth should be used."""
        # TODO: Implement
        pass
