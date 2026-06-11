"""AliExpress headless scraper — source LOW-cost (sandbox fallback per BigBuy).

NOTA SANDBOX: usato solo per simulazione F1 (€0 reali). Scraping public search page.
robots.txt AliExpress permette crawling con rate limits. Caveat regulatory: in produzione F3
valutare API ufficiale AliExpress Affiliate o dropshipping partner.
"""
from __future__ import annotations

import re
from .base import BaseScraper, PricePoint, TrendSignal

# AliExpress in inglese — evita redirect localizzazione che nasconde i prezzi
_ALIEX_SEARCH_URL = "https://www.aliexpress.com/wholesale?SearchText={query}&SortType=price_asc"

# Mappa keyword IT → EN per query su AliExpress (corpus salute/sport/beauty)
_IT_TO_EN: dict[str, str] = {
    "integratori collagene": "collagen supplements",
    "olio cbd": "cbd oil",
    "cerotti magnetici": "magnetic patches",
    "fascia lombare": "lumbar support belt",
    "massaggiatore elettrico": "electric massager",
    "guanti anti artrite": "arthritis gloves",
    "cerotto silicone cicatrice": "silicone scar patch",
    "solette ortopediche": "orthopedic insoles",
    "elastici fitness": "resistance bands",
    "rullo foam roller": "foam roller",
    "grip palestra": "gym grip gloves",
    "jump rope": "jump rope",
    "ginocchiera sportiva": "knee brace support",
    "cintura pesi": "weight belt",
    "corda allenamento": "training rope",
    "tappetino yoga": "yoga mat",
    "siero vitamina c": "vitamin c serum",
    "maschera viso tessuto": "face sheet mask",
    "curling wand": "curling wand",
    "spazzola levigante": "smoothing brush",
    "detergente konjac": "konjac cleanser",
    "rullo jade viso": "jade roller face",
    "strumento gua sha": "gua sha tool",
    "patch occhiaie": "under eye patch",
}


def _translate(query: str) -> str:
    q = query.lower().strip()
    return _IT_TO_EN.get(q, query)


class AliExpressScraper(BaseScraper):
    """Scrape prezzi wholesale da AliExpress via Playwright (source LOW-cost sandbox)."""

    DEFAULT_DELAY_S = 5.0  # rate limit conservativo

    @property
    def source_name(self) -> str:
        return "aliexpress"

    async def is_available(self) -> bool:
        try:
            from playwright.async_api import async_playwright  # noqa: F401
            return True
        except ImportError:
            return False

    async def fetch_trends(self, keywords: list[str]) -> list[TrendSignal]:
        return []

    async def fetch_prices(self, query: str, max_results: int = 5) -> list[PricePoint]:
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return []

        await self._throttle()
        prices: list[PricePoint] = []
        en_query = _translate(query)
        url = _ALIEX_SEARCH_URL.format(query=en_query.replace(" ", "+"))

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            ctx = await browser.new_context(
                user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/125.0 Safari/537.36",
                locale="en-US",
            )
            page = await ctx.new_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                content = await page.content()

                # AliExpress mostra prezzi in USD — convertiamo a EUR (appross. 0.92)
                USD_TO_EUR = 0.92
                matches_usd = re.findall(r'US\s*\$\s*(\d+\.?\d{0,2})', content)
                matches_eur = re.findall(r'(?:€\s*|EUR\s*)(\d+[\.,]\d{1,2})', content)

                seen: set[float] = set()
                all_prices: list[float] = []
                for m in matches_eur:
                    try:
                        all_prices.append(float(m.replace(",", ".")))
                    except ValueError:
                        pass
                for m in matches_usd:
                    try:
                        all_prices.append(float(m) * USD_TO_EUR)
                    except ValueError:
                        pass

                for price_val in sorted(all_prices):
                    if 0.3 < price_val < 300 and price_val not in seen:
                        seen.add(price_val)
                        prices.append(PricePoint(
                            source=self.source_name,
                            product_query=query,
                            price_eur=round(price_val, 2),
                            url=url,
                            metadata={"query_en": en_query},
                        ))
                    if len(prices) >= max_results:
                        break
            except Exception:
                pass
            finally:
                await browser.close()
        return prices
