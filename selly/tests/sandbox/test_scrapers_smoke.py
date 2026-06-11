"""Smoke test scrapers — 1 call reale per source. Marcati slow + sandbox."""
import pytest

from selly.sandbox.scrapers.vinted import VintedScraper
from selly.sandbox.scrapers.google_trends import GoogleTrendsScraper, _PYTRENDS_AVAILABLE
from selly.sandbox.scrapers.ebay_search import EbaySearchScraper
from selly.sandbox.scrapers.bigbuy_api import BigBuyApiScraper


@pytest.mark.sandbox
@pytest.mark.slow
@pytest.mark.asyncio
async def test_vinted_availability():
    scraper = VintedScraper()
    available = await scraper.is_available()
    # non falliamo se Vinted è down — ma logghiamo
    assert isinstance(available, bool)


@pytest.mark.sandbox
@pytest.mark.slow
@pytest.mark.asyncio
async def test_vinted_fetch_trends_returns_signals():
    scraper = VintedScraper()
    if not await scraper.is_available():
        pytest.skip("Vinted non raggiungibile")
    signals = await scraper.fetch_trends(["collagene"])
    assert len(signals) == 1
    assert signals[0].source == "vinted_it"
    assert 0.0 <= signals[0].score <= 100.0


@pytest.mark.sandbox
@pytest.mark.slow
@pytest.mark.asyncio
async def test_vinted_fetch_prices_returns_list():
    scraper = VintedScraper()
    if not await scraper.is_available():
        pytest.skip("Vinted non raggiungibile")
    prices = await scraper.fetch_prices("collagene", max_results=3)
    assert isinstance(prices, list)
    for p in prices:
        assert p.price_eur > 0
        assert p.source == "vinted_it"


@pytest.mark.sandbox
@pytest.mark.slow
@pytest.mark.asyncio
async def test_google_trends_availability():
    scraper = GoogleTrendsScraper()
    available = await scraper.is_available()
    assert available == _PYTRENDS_AVAILABLE


@pytest.mark.sandbox
@pytest.mark.slow
@pytest.mark.asyncio
async def test_google_trends_fetch_trends():
    scraper = GoogleTrendsScraper()
    if not await scraper.is_available():
        pytest.skip("pytrends non installato")
    signals = await scraper.fetch_trends(["collagene"])
    assert len(signals) == 1
    assert 0.0 <= signals[0].score <= 100.0


@pytest.mark.sandbox
@pytest.mark.asyncio
async def test_ebay_skips_without_api_key():
    scraper = EbaySearchScraper(app_id="")
    assert not await scraper.is_available()
    prices = await scraper.fetch_prices("collagene")
    assert prices == []


@pytest.mark.sandbox
@pytest.mark.asyncio
async def test_bigbuy_skips_without_api_key():
    scraper = BigBuyApiScraper(api_key="")
    assert not await scraper.is_available()
    prices = await scraper.fetch_prices("collagene")
    assert prices == []
