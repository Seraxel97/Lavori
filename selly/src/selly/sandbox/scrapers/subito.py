"""Subito.it scraper — prezzi annunci seconda mano (source LOW-cost, no auth)."""
from __future__ import annotations

import re
import httpx
from .base import BaseScraper, PricePoint, TrendSignal

_SUBITO_API_URL = "https://api.subito.it/v1/search/items/?shp=0&q={query}&sort=price_asc&lim={limit}"


class SubitoScraper(BaseScraper):
    """Scrape prezzi da Subito.it (annunci privati = prezzi bassi = source LOW)."""

    DEFAULT_DELAY_S = 2.0

    @property
    def source_name(self) -> str:
        return "subito_it"

    async def fetch_trends(self, keywords: list[str]) -> list[TrendSignal]:
        signals: list[TrendSignal] = []
        async with httpx.AsyncClient(timeout=15.0, headers=self._headers()) as client:
            for kw in keywords:
                await self._throttle()
                try:
                    url = _SUBITO_API_URL.format(query=kw.replace(" ", "+"), limit=20)
                    r = await client.get(url)
                    r.raise_for_status()
                    data = r.json()
                    total = data.get("total_count", 0)
                    score = min(total / 10.0, 100.0)
                    signals.append(TrendSignal(
                        source=self.source_name,
                        keyword=kw,
                        score=score,
                        metadata={"total_count": total},
                    ))
                except httpx.HTTPError:
                    signals.append(TrendSignal(source=self.source_name, keyword=kw, score=0.0))
        return signals

    async def fetch_prices(self, query: str, max_results: int = 5) -> list[PricePoint]:
        await self._throttle()
        prices: list[PricePoint] = []
        async with httpx.AsyncClient(timeout=15.0, headers=self._headers()) as client:
            try:
                url = _SUBITO_API_URL.format(query=query.replace(" ", "+"), limit=max_results * 2)
                r = await client.get(url)
                r.raise_for_status()
                data = r.json()
                ads = data.get("ads", [])
                for ad in ads[:max_results]:
                    features = {f["uri"]: f.get("values", [{}])[0] for f in ad.get("features", [])}
                    price_raw = features.get("/price", {}).get("value", "")
                    if not price_raw:
                        continue
                    price_str = re.sub(r"[^\d,.]", "", str(price_raw)).replace(",", ".")
                    try:
                        price_val = float(price_str)
                    except ValueError:
                        continue
                    if price_val > 0:
                        ad_id = ad.get("urn", "").split(":")[-1]
                        prices.append(PricePoint(
                            source=self.source_name,
                            product_query=query,
                            price_eur=price_val,
                            url=f"https://www.subito.it/annunci/{ad_id}",
                            metadata={"title": ad.get("subject", "")},
                        ))
            except (httpx.HTTPError, KeyError, ValueError):
                pass
        return prices

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=8.0, headers=self._headers()) as client:
                r = await client.get("https://www.subito.it", follow_redirects=True)
                return r.status_code < 500
        except httpx.HTTPError:
            return False

    @staticmethod
    def _headers() -> dict[str, str]:
        return {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
            "Accept-Language": "it-IT,it;q=0.9",
            "Accept": "application/json, text/html",
        }
