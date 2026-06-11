"""Test per common.dispatcher_base — validazione chiave Literal."""

import pytest

from common.dispatcher_base import BaseDispatcher, DispatcherProtocol, validate_dispatch_key

_METRICS = ("coh", "imcoh", "plv", "ciplv", "pli", "wpli", "wpli2_debiased")
_ALGORITHMS = ("logreg", "svm", "mlp", "rf", "gb")


def test_validate_valid_keys():
    for m in _METRICS:
        validate_dispatch_key(m, _METRICS, "metric")
    for a in _ALGORITHMS:
        validate_dispatch_key(a, _ALGORITHMS, "algorithm")


def test_validate_invalid_raises():
    with pytest.raises(ValueError, match="Unknown metric='bad_metric'"):
        validate_dispatch_key("bad_metric", _METRICS, "metric")

    with pytest.raises(ValueError, match="Unknown algorithm='xgboost'"):
        validate_dispatch_key("xgboost", _ALGORITHMS, "algorithm")


def test_validate_error_message_lists_valid():
    with pytest.raises(ValueError, match="ciplv"):
        validate_dispatch_key("invalid", _METRICS, "metric")


def test_fc_dispatcher_rejects_invalid_metric(synthetic_label_tc):
    """Usa shared fixture: (10, 20, 1000) sintetico da tests/fixtures/synthetic.py."""
    from connectivity.fc_dispatcher import compute_fc

    with pytest.raises(ValueError, match="Unknown metric"):
        compute_fc(synthetic_label_tc, sfreq=250.0, metric="bad_metric")  # type: ignore[arg-type]


def test_ml_dispatcher_rejects_invalid_algorithm(synthetic_X_y_groups):
    """Usa shared fixture: X(50,100), y(50,), groups da tests/fixtures/synthetic.py."""
    from ml_training.ml_dispatcher import run_cv

    X, y, _ = synthetic_X_y_groups
    with pytest.raises(ValueError, match="Unknown algorithm"):
        run_cv(X, y, algorithm="xgboost")  # type: ignore[arg-type]


def test_base_dispatcher_validate():
    class MyDispatcher(BaseDispatcher):
        _valid_keys = ("a", "b", "c")
        _key_type = "mode"

    d = MyDispatcher()
    d.validate_key("a")
    with pytest.raises(ValueError, match="Unknown mode='z'"):
        d.validate_key("z")


def test_dispatcher_protocol_is_runtime_checkable():
    assert isinstance(BaseDispatcher(), DispatcherProtocol)
