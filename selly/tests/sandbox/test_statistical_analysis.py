"""Test bootstrap CI determinismo e sign test."""
import pytest

from selly.sandbox.feedback_simulator import FeedbackResult
from selly.sandbox.statistical_analysis import analyze, _bootstrap_mean, _sign_test_p


def _make_fb(predicted: float, actual: float) -> FeedbackResult:
    return FeedbackResult(
        trade_id="t",
        predicted_roi=predicted,
        actual_roi=actual,
        prediction_error=actual - predicted,
        actual_price_low=10.0,
        actual_price_high=30.0,
    )


def test_analyze_empty_returns_zero():
    r = analyze([])
    assert r.n_trades == 0
    assert r.calibration_pass is False


def test_accuracy_rate_perfect():
    # tutti predicted > 0 e actual > 0 → accuracy = 1.0
    fbs = [_make_fb(1.5, 1.0), _make_fb(1.2, 0.8), _make_fb(0.5, 0.6)]
    r = analyze(fbs)
    assert r.prediction_accuracy_rate == pytest.approx(1.0)


def test_accuracy_rate_zero():
    # tutti predicted > 0 ma actual < 0
    fbs = [_make_fb(1.5, -0.5), _make_fb(1.2, -0.1)]
    r = analyze(fbs)
    assert r.prediction_accuracy_rate == pytest.approx(0.0)


def test_mae_positive():
    fbs = [_make_fb(1.0, 0.5), _make_fb(0.5, 1.0)]
    r = analyze(fbs)
    assert r.mean_abs_error == pytest.approx(0.5)


def test_bootstrap_determinism():
    errors = [0.1, -0.2, 0.3, -0.4, 0.5]
    ci1 = _bootstrap_mean(errors, seed=42)
    ci2 = _bootstrap_mean(errors, seed=42)
    assert ci1 == ci2


def test_bootstrap_ci_ordered():
    errors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    lo, hi = _bootstrap_mean(errors, seed=42)
    assert lo <= hi


def test_sign_test_p_random_baseline():
    # 5 su 10 corretti → p vicino a 1.0 (non significativo)
    p = _sign_test_p(n_correct=5, n_total=10)
    assert p > 0.05


def test_sign_test_p_high_accuracy():
    # 9 su 10 corretti → p < 0.05
    p = _sign_test_p(n_correct=9, n_total=10)
    assert p < 0.05


def test_calibration_pass_requires_both_conditions():
    # accuracy alta + p basso → pass
    fbs = [_make_fb(1.0, 1.0)] * 10  # tutti corretti
    r = analyze(fbs)
    # 10/10 accuracy → p molto basso
    assert r.prediction_accuracy_rate == pytest.approx(1.0)
    assert r.calibration_pass is True
