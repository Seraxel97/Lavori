"""Price discovery — aggrega prezzi low-cost e high-margin per un prodotto."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .scrapers.base import BaseScraper, PricePoint
from .scrapers.bigbuy_api import BigBuyApiScraper
from .scrapers.ebay_search import EbaySearchScraper
from .scrapers.vinted import VintedScraper, _VINTED_CATALOG_URL, _VINTED_BASE
from .scrapers.subito import SubitoScraper
from .scrapers.aliexpress import AliExpressScraper


@dataclass
class PriceAggregate:
    """Prezzi aggregati per un prodotto: low (acquisto) e high (vendita)."""
    product_query: str
    avg_price_low: float   # media source economiche (wholesale/Vinted prezzi bassi)
    avg_price_high: float  # media source alto margine (Vinted prezzi medi, eBay)
    price_points_low: list[PricePoint] = field(default_factory=list)
    price_points_high: list[PricePoint] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def spread(self) -> float:
        return self.avg_price_high - self.avg_price_low

    @property
    def n_sources(self) -> int:
        return (1 if self.price_points_low else 0) + (1 if self.price_points_high else 0)


def _safe_avg(prices: list[PricePoint]) -> float:
    vals = [p.price_eur for p in prices if p.price_eur > 0]
    return sum(vals) / len(vals) if vals else 0.0


class _VintedHighScraper(VintedScraper):
    """Vinted con ordering per rilevanza (prezzo medio/alto di mercato) — source HIGH."""

    @property
    def source_name(self) -> str:
        return "vinted_it_high"

    async def fetch_prices(self, query: str, max_results: int = 5) -> list[PricePoint]:
        await self._throttle()
        prices: list[PricePoint] = []
        text = await self._get_api_response(
            _VINTED_CATALOG_URL,
            params={"search_text": query, "per_page": str(max_results), "order": "relevance"},
        )
        if not text:
            return prices
        try:
            data = json.loads(text)
            items = data.get("items", [])
            for item in items[:max_results]:
                price_raw = item.get("price", {})
                price_val = float(price_raw.get("amount", 0))
                if price_val > 0:
                    prices.append(PricePoint(
                        source=self.source_name,
                        product_query=query,
                        price_eur=price_val,
                        url=f"https://www.vinted.it/items/{item.get('id', '')}",
                        metadata={"title": item.get("title", ""), "order": "relevance"},
                    ))
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        # HIGH source: escludi i prezzi più bassi (bottom 20%) per catturare il range medio-alto
        if prices:
            prices_sorted = sorted(prices, key=lambda p: p.price_eur)
            cut = max(1, len(prices_sorted) // 5)
            prices = prices_sorted[cut:]
        return prices


class PriceDiscovery:
    """
    Aggrega prezzi da source low-cost e high-margin.

    Strategia sandbox F1 (quando eBay/BigBuy non disponibili):
    - LOW: Vinted prezzi_low_to_high (articoli sottovalutati) + AliExpress/BigBuy
    - HIGH: Vinted rilevanza top-range + eBay
    Questo cattura l'asimmetria di pricing intra-Vinted (opportunità reale di arbitrage).
    """

    def __init__(self) -> None:
        self._low_sources: list = [BigBuyApiScraper(), AliExpressScraper(), SubitoScraper(), VintedScraper()]
        self._high_sources: list = [EbaySearchScraper(), _VintedHighScraper()]

    async def discover(self, product_query: str, max_results_per_source: int = 5) -> PriceAggregate:
        low_prices: list[PricePoint] = []
        high_prices: list[PricePoint] = []

        for scraper in self._low_sources:
            if await scraper.is_available():
                pts = await scraper.fetch_prices(product_query, max_results_per_source)
                low_prices.extend(pts)

        for scraper in self._high_sources:
            if await scraper.is_available():
                pts = await scraper.fetch_prices(product_query, max_results_per_source)
                high_prices.extend(pts)

        return PriceAggregate(
            product_query=product_query,
            avg_price_low=_safe_avg(low_prices),
            avg_price_high=_safe_avg(high_prices),
            price_points_low=low_prices,
            price_points_high=high_prices,
            metadata={
                "n_low": len(low_prices),
                "n_high": len(high_prices),
                "sources_low": [s.source_name for s in self._low_sources],
                "sources_high": [s.source_name for s in self._high_sources],
            },
        )
