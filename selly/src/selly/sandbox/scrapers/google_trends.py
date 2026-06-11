"""Google Trends scraper via pytrends (nessuna API key richiesta)."""
from __future__ import annotations

from .base import BaseScraper, PricePoint, TrendSignal

try:
    from pytrends.request import TrendReq
    _PYTRENDS_AVAILABLE = True
except ImportError:
    _PYTRENDS_AVAILABLE = False


class GoogleTrendsScraper(BaseScraper):
    """Recupera interesse di ricerca da Google Trends (pytrends)."""

    DEFAULT_DELAY_S = 5.0  # Google Trends throttle aggressivo

    @property
    def source_name(self) -> str:
        return "google_trends"

    async def is_available(self) -> bool:
        return _PYTRENDS_AVAILABLE

    async def fetch_trends(self, keywords: list[str]) -> list[TrendSignal]:
        if not _PYTRENDS_AVAILABLE:
            return []
        import asyncio
        return await asyncio.to_thread(self._sync_fetch_trends, keywords)

    def _sync_fetch_trends(self, keywords: list[str]) -> list[TrendSignal]:
        signals: list[TrendSignal] = []
        # pytrends accetta max 5 keyword per volta
        for chunk_start in range(0, len(keywords), 5):
            chunk = keywords[chunk_start:chunk_start + 5]
            try:
                pt = TrendReq(hl="it-IT", tz=60)
                pt.build_payload(chunk, timeframe="now 7-d", geo="IT")
                df = pt.interest_over_time()
                if df.empty:
                    for kw in chunk:
                        signals.append(TrendSignal(source=self.source_name, keyword=kw, score=0.0))
                    continue
                for kw in chunk:
                    if kw in df.columns:
                        score = float(df[kw].mean())
                    else:
                        score = 0.0
                    signals.append(TrendSignal(
                        source=self.source_name,
                        keyword=kw,
                        score=score,
                        metadata={"timeframe": "7d", "geo": "IT"},
                    ))
            except Exception:
                for kw in chunk:
                    signals.append(TrendSignal(source=self.source_name, keyword=kw, score=0.0))
        return signals

    async def fetch_prices(self, query: str, max_results: int = 5) -> list[PricePoint]:
        # Google Trends non ha price data — metodo no-op per contratto ABC
        return []
