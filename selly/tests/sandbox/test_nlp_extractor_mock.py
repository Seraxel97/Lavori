"""Test NLP extractor — verifica Vesta MCP + fallback euristico."""
import pytest

from selly.sandbox.nlp_extractor import NlpExtractor


@pytest.mark.asyncio
async def test_heuristic_fallback_returns_products():
    """Fallback euristico deve sempre restituire qualcosa."""
    extractor = NlpExtractor()
    extractor._vesta_available = False  # forza fallback
    products = extractor._extract_heuristic(["collagene", "elastici fitness", "trend"])
    assert len(products) == 3
    names = [p.name for p in products]
    assert "collagene" in names


@pytest.mark.asyncio
async def test_heuristic_confidence_lower_for_stop_words():
    extractor = NlpExtractor()
    products = extractor._extract_heuristic(["trend", "collagene polvere"])
    trend_p = next(p for p in products if p.name == "trend")
    collagene_p = next(p for p in products if p.name == "collagene polvere")
    assert trend_p.confidence < collagene_p.confidence


@pytest.mark.asyncio
async def test_vesta_availability_check_no_crash():
    """_check_vesta non deve mai crashare."""
    extractor = NlpExtractor()
    result = await extractor._check_vesta()
    assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_extract_returns_list():
    extractor = NlpExtractor()
    products = await extractor.extract(["siero vitamina C", "rullo jade"])
    assert isinstance(products, list)
    assert all(hasattr(p, "name") for p in products)
