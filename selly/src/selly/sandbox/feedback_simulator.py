"""Feedback simulator — confronta predicted_roi vs actual market price a T+14d."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .mock_execution import MockTrade
from .price_discovery import PriceDiscovery
from .arbitrage_engine import compute_arbitrage


@dataclass
class FeedbackResult:
    trade_id: str
    predicted_roi: float
    actual_roi: float
    prediction_error: float  # actual - predicted
    actual_price_low: float
    actual_price_high: float
    metadata: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class FeedbackSimulator:
    """
    Simula T+14d: rifetch prezzi di mercato al mock_sell_ts_estimated
    e calcola prediction_error = actual_roi - predicted_roi.

    Nella sandbox F1 il "T+14d" viene calcolato NOW perché i prezzi Vinted/eBay
    sono snapshot correnti (non abbiamo storico). Questo misura comunque la
    calibration del modello: quanto è stabile il ROI previsto vs reale.
    """

    def __init__(self) -> None:
        self._discovery = PriceDiscovery()

    async def evaluate(self, trade: MockTrade) -> FeedbackResult:
        """Rifetch prezzi per il prodotto e calcola actual_roi."""
        agg = await self._discovery.discover(trade.product)
        opp = compute_arbitrage(agg)

        actual_roi = opp.score
        error = actual_roi - trade.predicted_roi

        return FeedbackResult(
            trade_id=trade.trade_id,
            predicted_roi=trade.predicted_roi,
            actual_roi=actual_roi,
            prediction_error=error,
            actual_price_low=agg.avg_price_low,
            actual_price_high=agg.avg_price_high,
            metadata={
                "product": trade.product,
                "n_price_sources": agg.n_sources,
                "passes_threshold": opp.passes_threshold,
            },
        )

    async def evaluate_batch(self, trades: list[MockTrade]) -> list[FeedbackResult]:
        results: list[FeedbackResult] = []
        for trade in trades:
            result = await self.evaluate(trade)
            results.append(result)
        return results
