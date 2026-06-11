"""Mock execution — log trade simulato senza alcun acquisto/listing reale."""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .arbitrage_engine import ArbitrageOpportunity

_DEFAULT_LOG_DIR = Path("data/sandbox_results")

# T+14 giorni: stima sell timestamp per feedback simulator
MOCK_HOLD_DAYS = 14


@dataclass
class MockTrade:
    """Entry di trade simulato — nessun denaro mosso."""
    trade_id: str
    product: str
    source_low: str
    source_high: str
    predicted_roi: float
    avg_price_low: float
    avg_price_high: float
    mock_buy_ts: str          # ISO UTC
    mock_sell_ts_estimated: str  # ISO UTC = buy_ts + 14d
    actual_roi: float | None = None     # popolato da feedback_simulator
    prediction_error: float | None = None  # actual_roi - predicted_roi
    metadata: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def create_mock_trade(
    opp: ArbitrageOpportunity,
    source_low: str = "bigbuy",
    source_high: str = "ebay_it",
    buy_ts: datetime | None = None,
) -> MockTrade:
    """Crea entry di trade simulato da un'opportunità di arbitrage."""
    now = buy_ts or datetime.now(tz=timezone.utc)
    sell_est = now + timedelta(days=MOCK_HOLD_DAYS)
    return MockTrade(
        trade_id=str(uuid.uuid4()),
        product=opp.product_query,
        source_low=source_low,
        source_high=source_high,
        predicted_roi=opp.score,
        avg_price_low=opp.avg_price_low,
        avg_price_high=opp.avg_price_high,
        mock_buy_ts=now.isoformat(),
        mock_sell_ts_estimated=sell_est.isoformat(),
        metadata={**opp.metadata, "gross_margin": opp.gross_margin, "net_margin": opp.net_margin},
    )


class TradeLog:
    """Log persistente di trade simulati (append-only JSONL)."""

    def __init__(self, log_path: Path | None = None) -> None:
        self._path = log_path or (_DEFAULT_LOG_DIR / "trades.jsonl")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._trades: list[MockTrade] = self._load()

    def _load(self) -> list[MockTrade]:
        if not self._path.exists():
            return []
        trades: list[MockTrade] = []
        with self._path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    trades.append(MockTrade(**d))
        return trades

    def append(self, trade: MockTrade) -> None:
        self._trades.append(trade)
        with self._path.open("a") as f:
            f.write(json.dumps(trade.to_dict()) + "\n")

    @property
    def trades(self) -> list[MockTrade]:
        return list(self._trades)

    def __len__(self) -> int:
        return len(self._trades)
