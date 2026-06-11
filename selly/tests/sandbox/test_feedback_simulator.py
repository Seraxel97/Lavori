"""Test calcolo prediction_error nel feedback simulator."""
import pytest

from selly.sandbox.feedback_simulator import FeedbackResult


def test_feedback_result_error_computation():
    fb = FeedbackResult(
        trade_id="abc",
        predicted_roi=1.5,
        actual_roi=1.2,
        prediction_error=-0.3,
        actual_price_low=10.0,
        actual_price_high=30.0,
    )
    assert fb.prediction_error == pytest.approx(-0.3)


def test_feedback_result_positive_error():
    fb = FeedbackResult(
        trade_id="xyz",
        predicted_roi=0.8,
        actual_roi=1.1,
        prediction_error=0.3,
        actual_price_low=5.0,
        actual_price_high=15.0,
    )
    assert fb.prediction_error > 0  # modello sottostimava il ROI


def test_feedback_result_zero_error():
    fb = FeedbackResult(
        trade_id="zzz",
        predicted_roi=1.0,
        actual_roi=1.0,
        prediction_error=0.0,
        actual_price_low=10.0,
        actual_price_high=20.0,
    )
    assert fb.prediction_error == pytest.approx(0.0)
