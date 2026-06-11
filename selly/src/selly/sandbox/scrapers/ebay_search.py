"""eBay IT price scraper via Finding API (search-only, no auth per public search)."""
from __future__ import annotations

import os
import httpx
from .base import BaseScraper, PricePoint, TrendSignal

# eBay Finding API — endpoint pubblico, usa APPID da .env se disponibile
_EBAY_FIND_URL = "https://svcs.ebay.com/services/search/FindingService/v1"


class EbaySearchScraper(BaseScraper):
    """Recupera prezzi di vendita su eBay IT via Finding API."""

    DEFAULT_DELAY_S = 2.0

    def __init__(self, app_id: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.app_id = app_id or os.getenv("EBAY_APP_ID", "")

    @property
    def source_name(self) -> str:
        return "ebay_it"

    async def is_available(self) -> bool:
        return bool(self.app_id)

    async def fetch_trends(self, keywords: list[str]) -> list[TrendSignal]:
        # eBay è source price, non trend
        return []

    async def fetch_prices(self, query: str, max_results: int = 5) -> list[PricePoint]:
        if not self.app_id:
            return []
        await self._throttle()
        prices: list[PricePoint] = []
        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                r = await client.get(
                    _EBAY_FIND_URL,
                    params={
                        "OPERATION-NAME": "findItemsByKeywords",
                        "SERVICE-VERSION": "1.0.0",
                        "SECURITY-APPNAME": self.app_id,
                        "RESPONSE-DATA-FORMAT": "JSON",
                        "REST-PAYLOAD": "",
                        "keywords": query,
                        "paginationInput.entriesPerPage": max_results,
                        "itemFilter(0).name": "ListingType",
                        "itemFilter(0).value": "FixedPrice",
                        "sortOrder": "PricePlusShippingLowest",
                        "outputSelector": "SellerInfo",
                    },
                )
                r.raise_for_status()
                data = r.json()
                items = (
                    data.get("findItemsByKeywordsResponse", [{}])[0]
                    .get("searchResult", [{}])[0]
                    .get("item", [])
                )
                for item in items[:max_results]:
                    price_val = float(
                        item.get("sellingStatus", [{}])[0]
                        .get("currentPrice", [{"__value__": "0"}])[0]
                        .get("__value__", 0)
                    )
                    url = item.get("viewItemURL", [""])[0]
                    if price_val > 0:
                        prices.append(PricePoint(
                            source=self.source_name,
                            product_query=query,
                            price_eur=price_val,
                            url=url,
                            metadata={"title": item.get("title", [""])[0]},
                        ))
            except (httpx.HTTPError, KeyError, ValueError):
                pass
        return prices
