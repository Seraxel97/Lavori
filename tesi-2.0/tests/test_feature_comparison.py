"""Test feature_comparison: sanity checks strutturali."""

from __future__ import annotations

import numpy as np


def test_build_X_graphtheory_shape():
    from ml_training.feature_comparison import build_X_graphtheory

    X_gt = build_X_graphtheory()
    assert X_gt.ndim == 2
    assert X_gt.shape[1] == 8
    assert X_gt.shape[0] > 0


def test_no_nan_in_X_gt():
    from ml_training.feature_comparison import build_X_graphtheory

    X_gt = build_X_graphtheory()
    assert not np.any(np.isnan(X_gt)), "X_gt contiene NaN dopo imputazione"
    assert not np.any(np.isinf(X_gt)), "X_gt contiene inf dopo imputazione"


def test_feature_names_order():
    from ml_training.feature_comparison import GT_FEATURE_ORDER

    assert len(GT_FEATURE_ORDER) == 8
    assert "clustering_coeff" in GT_FEATURE_ORDER
    assert "global_efficiency" in GT_FEATURE_ORDER
