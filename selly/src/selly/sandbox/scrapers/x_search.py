"""X (Twitter) commercial search scraper via Playwright headless (nitter public)."""
from __future__ import annotations

import re
from .base import BaseScraper, PricePoint, TrendSignal

# Usa nitter public instance (scraping permesso per ricerca pubblica)
_NITTER_SEARCH_URL = "https://nitter.net/search?q={query}&f=tweets"


class XSearchScraper(BaseScraper):
    """Scrape volume tweet commerciali su X tramite istanza nitter pubblica."""

    DEFAULT_DELAY_S = 3.0

    @property
    def source_name(self) -> str:
        return "x_search"

    async def fetch_trends(self, keywords: list[str]) -> list[TrendSignal]:
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return []

        signals: list[TrendSignal] = []
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()
            for kw in keywords:
                await self._throttle()
                query = f"{kw} comprare prezzo"
                url = _NITTER_SEARCH_URL.format(query=query.replace(" ", "+"))
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=20_000)
                    content = await page.content()
                    # conta tweet nel risultato
                    tweet_count = len(re.findall(r'class="timeline-item"', content))
                    score = min(tweet_count * 5.0, 100.0)
                    signals.append(TrendSignal(
                        source=self.source_name,
                        keyword=kw,
                        score=score,
                        metadata={"tweet_count": tweet_count, "query": query},
                    ))
                except Exception:
                    signals.append(TrendSignal(source=self.source_name, keyword=kw, score=0.0))
            await browser.close()
        return signals

    async def fetch_prices(self, query: str, max_results: int = 5) -> list[PricePoint]:
        return []

    async def is_available(self) -> bool:
        import httpx
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                r = await client.get("https://nitter.net", follow_redirects=True)
                return r.status_code < 500
        except httpx.HTTPError:
            return False
