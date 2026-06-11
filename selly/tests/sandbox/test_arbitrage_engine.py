"""Test puri per la formula arbitrage — nessuna I/O."""
import pytest
from selly.sandbox.arbitrage_engine import compute_arbitrage, VAT_IT, FEES_MARKETPLACE, SHIPPING_EUR
from selly.sandbox.price_discovery import PriceAggregate


def _make_agg(low: float, high: float) -> PriceAggregate:
    return PriceAggregate(product_query="test_product", avg_price_low=low, avg_price_high=high)


def test_score_formula_basic():
    agg = _make_agg(low=10.0, high=30.0)
    opp = compute_arbitrage(agg)
    gross = 30.0 - 10.0
    fees = 30.0 * FEES_MARKETPLACE
    vat = 10.0 * VAT_IT
    net = gross - fees - SHIPPING_EUR - vat
    expected_score = net / 10.0
    assert abs(opp.score - expected_score) < 1e-6


def test_passes_threshold_when_roi_above_1():
    # low=5, high=20 → gross=15, fees=2.6, ship=5, vat=1.1 → net≈6.3 → score≈1.26
    agg = _make_agg(low=5.0, high=20.0)
    opp = compute_arbitrage(agg, threshold=1.0)
    assert opp.passes_threshold is True
    assert opp.score >= 1.0


def test_fails_threshold_when_roi_below():
    # low=10, high=11 → quasi nulla dopo fee
    agg = _make_agg(low=10.0, high=11.0)
    opp = compute_arbitrage(agg, threshold=1.0)
    assert opp.passes_threshold is False
    assert opp.score < 1.0


def test_zero_prices_returns_score_zero():
    agg = _make_agg(low=0.0, high=0.0)
    opp = compute_arbitrage(agg)
    assert opp.score == 0.0
    assert opp.passes_threshold is False
    assert "insufficient_price_data" in opp.metadata.get("reason", "")


def test_gross_margin_computed_correctly():
    agg = _make_agg(low=8.0, high=20.0)
    opp = compute_arbitrage(agg)
    assert opp.gross_margin == pytest.approx(12.0, abs=0.01)


def test_net_margin_less_than_gross():
    agg = _make_agg(low=10.0, high=25.0)
    opp = compute_arbitrage(agg)
    assert opp.net_margin < opp.gross_margin


def test_custom_threshold_respected():
    agg = _make_agg(low=5.0, high=20.0)
    opp_strict = compute_arbitrage(agg, threshold=2.0)
    opp_lenient = compute_arbitrage(agg, threshold=0.5)
    # stesso score, diverso passes_threshold
    assert opp_strict.score == opp_lenient.score
    assert opp_lenient.passes_threshold is True
