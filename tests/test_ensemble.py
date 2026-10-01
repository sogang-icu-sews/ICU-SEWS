"""Tests for model-independent soft voting. Owner: 이택훈."""

import numpy as np

from icu_sews.models.ensemble import soft_vote


def test_soft_vote_uses_configured_weight() -> None:
    tree = np.array([0.8, 0.2])
    sequence = np.array([0.4, 0.6])
    result = soft_vote(tree, sequence, tree_weight=0.75)
    np.testing.assert_allclose(result, np.array([0.7, 0.3]))
