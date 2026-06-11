"""Vinted scraper via Playwright headless (richiede sessione browser per public API)."""
from __future__ import annotations

import json
from .base import BaseScraper, PricePoint, TrendSignal

_VINTED_CATALOG_URL = "https://www.vinted.it/api/v2/catalog/items"
_VINTED_BASE = "https://www.vinted.it"


class VintedScraper(BaseScraper):
    """Scrape prezzi e volume da Vinted IT via Playwright (evita 403 API diretta)."""

    DEFAULT_DELAY_S = 4.0

    @property
    def source_name(self) -> str:
        return "vinted_it"

    async def _get_api_response(self, url: str, params: dict | None = None) -> str | None:
        """Fetch URL con Playwright headless per ottenere cookie di sessione Vinted."""
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return None

        query_str = ""
        if params:
            query_str = "?" + "&".join(f"{k}={v}" for k, v in params.items())

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            ctx = await browser.new_context(
                user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/125.0 Safari/537.36",
                locale="it-IT",
            )
            page = await ctx.new_page()
            try:
                # Prima visita homepage per ottenere cookie di sessione
                await page.goto(_VINTED_BASE, wait_until="domcontentloaded", timeout=20_000)
                await self._throttle()
                response = await page.request.get(
                    url + query_str,
                    headers={"Accept": "application/json"},
                )
                if response.status == 200:
                    return await response.text()
            except Exception:
                pass
            finally:
                await browser.close()
        return None

    async def fetch_trends(self, keywords: list[str]) -> list[TrendSignal]:
        signals: list[TrendSignal] = []
        for kw in keywords:
            await self._throttle()
            text = await self._get_api_response(
                _VINTED_CATALOG_URL,
                params={"search_text": kw, "per_page": "20", "order": "relevance"},
            )
            if text:
                try:
                    data = json.loads(text)
                    total = data.get("pagination", {}).get("total_count", 0)
                    score = min(total / 100.0, 100.0)
                    signals.append(TrendSignal(
                        source=self.source_name,
                        keyword=kw,
                        score=score,
                        metadata={"total_items": total},
                    ))
                    continue
                except (json.JSONDecodeError, KeyError):
                    pass
            signals.append(TrendSignal(source=self.source_name, keyword=kw, score=0.0))
        return signals

    async def fetch_prices(self, query: str, max_results: int = 5) -> list[PricePoint]:
        await self._throttle()
        prices: list[PricePoint] = []
        text = await self._get_api_response(
            _VINTED_CATALOG_URL,
            params={"search_text": query, "per_page": str(max_results), "order": "price_low_to_high"},
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
                        metadata={"title": item.get("title", "")},
                    ))
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        return prices

    async def is_available(self) -> bool:
        try:
            from playwright.async_api import async_playwright  # noqa: F401
            return True
        except ImportError:
            return False
