"""BigBuy B2B catalog scraper via REST API (richiede API key in .env)."""
from __future__ import annotations

import os
import httpx
from .base import BaseScraper, PricePoint, TrendSignal

_BIGBUY_API_URL = "https://api.bigbuy.eu/rest"


class BigBuyApiScraper(BaseScraper):
    """Recupera prezzi wholesale da BigBuy (source low-cost)."""

    DEFAULT_DELAY_S = 1.5

    def __init__(self, api_key: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("BIGBUY_API_KEY", "")

    @property
    def source_name(self) -> str:
        return "bigbuy"

    async def is_available(self) -> bool:
        return bool(self.api_key)

    async def fetch_trends(self, keywords: list[str]) -> list[TrendSignal]:
        return []

    async def fetch_prices(self, query: str, max_results: int = 5) -> list[PricePoint]:
        if not self.api_key:
            return []
        await self._throttle()
        prices: list[PricePoint] = []
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
            try:
                r = await client.get(
                    f"{_BIGBUY_API_URL}/catalog/productsinfo.json",
                    params={"lang": "it", "currency": "EUR"},
                )
                r.raise_for_status()
                products = r.json()
                for p in products[:max_results]:
                    name = p.get("description", [{}])[0].get("name", "")
                    if query.lower() not in name.lower():
                        continue
                    price_val = float(p.get("retailPrice", 0))
                    if price_val > 0:
                        prices.append(PricePoint(
                            source=self.source_name,
                            product_query=query,
                            price_eur=price_val,
                            metadata={"name": name, "sku": p.get("sku", "")},
                        ))
            except (httpx.HTTPError, KeyError, ValueError):
                pass
        return prices
