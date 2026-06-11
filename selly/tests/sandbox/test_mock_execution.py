"""Test log format e idempotenza MockTrade."""
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone

import pytest

from selly.sandbox.arbitrage_engine import ArbitrageOpportunity
from selly.sandbox.mock_execution import MockTrade, TradeLog, create_mock_trade


def _make_opp(score: float = 1.5) -> ArbitrageOpportunity:
    return ArbitrageOpportunity(
        product_query="collagene polvere",
        score=score,
        avg_price_low=10.0,
        avg_price_high=35.0,
        gross_margin=25.0,
        net_margin=15.0,
        passes_threshold=score >= 1.0,
        metadata={},
    )


def test_create_mock_trade_fields():
    opp = _make_opp()
    trade = create_mock_trade(opp)
    assert trade.product == "collagene polvere"
    assert trade.predicted_roi == 1.5
    assert trade.actual_roi is None
    assert trade.prediction_error is None
    assert len(trade.trade_id) == 36  # UUID4


def test_mock_trade_buy_sell_ts_order():
    from datetime import timedelta
    opp = _make_opp()
    trade = create_mock_trade(opp)
    buy = datetime.fromisoformat(trade.mock_buy_ts)
    sell = datetime.fromisoformat(trade.mock_sell_ts_estimated)
    assert sell > buy
    assert (sell - buy).days == 14


def test_trade_log_append_and_len(tmp_path):
    log_path = tmp_path / "trades.jsonl"
    log = TradeLog(log_path)
    assert len(log) == 0

    opp = _make_opp()
    trade = create_mock_trade(opp)
    log.append(trade)
    assert len(log) == 1


def test_trade_log_persisted_to_file(tmp_path):
    log_path = tmp_path / "trades.jsonl"
    log = TradeLog(log_path)
    trade = create_mock_trade(_make_opp())
    log.append(trade)

    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 1
    d = json.loads(lines[0])
    assert d["product"] == "collagene polvere"
    assert d["predicted_roi"] == pytest.approx(1.5)


def test_trade_log_reload_idempotent(tmp_path):
    log_path = tmp_path / "trades.jsonl"
    log1 = TradeLog(log_path)
    log1.append(create_mock_trade(_make_opp(1.2)))
    log1.append(create_mock_trade(_make_opp(1.8)))

    log2 = TradeLog(log_path)
    assert len(log2) == 2
    assert log2.trades[0].product == "collagene polvere"
