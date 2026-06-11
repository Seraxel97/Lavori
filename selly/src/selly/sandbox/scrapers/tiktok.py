"""TikTok hashtag trending scraper via Playwright headless."""
from __future__ import annotations

import re
from .base import BaseScraper, PricePoint, TrendSignal

_TIKTOK_TAG_URL = "https://www.tiktok.com/tag/{tag}"


class TikTokScraper(BaseScraper):
    """Scrape view count di hashtag su TikTok (public, no auth)."""

    DEFAULT_DELAY_S = 4.0

    @property
    def source_name(self) -> str:
        return "tiktok"

    async def fetch_trends(self, keywords: list[str]) -> list[TrendSignal]:
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return []

        signals: list[TrendSignal] = []
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            ctx = await browser.new_context(
                user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/125.0 Safari/537.36",
                locale="it-IT",
            )
            page = await ctx.new_page()
            for kw in keywords:
                await self._throttle()
                tag = kw.replace(" ", "")
                url = _TIKTOK_TAG_URL.format(tag=tag)
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=20_000)
                    content = await page.content()
                    # cerca "X videos" nel contenuto della pagina
                    match = re.search(r'"videoCount"\s*:\s*(\d+)', content)
                    count = int(match.group(1)) if match else 0
                    score = min(count / 1_000_000.0, 100.0)  # cap 100M = 100
                    signals.append(TrendSignal(
                        source=self.source_name,
                        keyword=kw,
                        score=score,
                        metadata={"video_count": count, "tag_url": url},
                    ))
                except Exception:
                    signals.append(TrendSignal(source=self.source_name, keyword=kw, score=0.0))
            await browser.close()
        return signals

    async def fetch_prices(self, query: str, max_results: int = 5) -> list[PricePoint]:
        # TikTok non è un marketplace price source
        return []

    async def is_available(self) -> bool:
        try:
            from playwright.async_api import async_playwright  # noqa: F401
            return True
        except ImportError:
            return False
