"""ABC Scraper — interfaccia comune per tutti i source."""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrendSignal:
    """Segnale di trend estratto da una fonte."""
    source: str
    keyword: str
    score: float  # 0-100 normalizzato
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PricePoint:
    """Prezzo rilevato da un marketplace."""
    source: str
    product_query: str
    price_eur: float
    currency: str = "EUR"
    url: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseScraper(ABC):
    """Interfaccia comune per scraper trend e price."""

    # Rate limit conservativo: rispetta robots.txt + throttling umano
    DEFAULT_DELAY_S: float = 2.0

    def __init__(self, rate_limit_s: float | None = None) -> None:
        self.rate_limit_s = rate_limit_s or self.DEFAULT_DELAY_S
        self._last_call_ts: float = 0.0

    async def _throttle(self) -> None:
        import time
        elapsed = time.monotonic() - self._last_call_ts
        if elapsed < self.rate_limit_s:
            await asyncio.sleep(self.rate_limit_s - elapsed)
        self._last_call_ts = time.monotonic()

    @property
    @abstractmethod
    def source_name(self) -> str: ...

    @abstractmethod
    async def fetch_trends(self, keywords: list[str]) -> list[TrendSignal]:
        """Recupera segnali di trend per le keyword fornite."""
        ...

    @abstractmethod
    async def fetch_prices(self, query: str, max_results: int = 5) -> list[PricePoint]:
        """Recupera prezzi per un prodotto dato."""
        ...

    async def is_available(self) -> bool:
        """Verifica availability del source (network/API)."""
        return True
