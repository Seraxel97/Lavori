"""Test enforcement regulatory hardcoded (design v2 §11)."""

from selly.compliance.regulatory_check import (
    is_keyword_regulated,
    is_source_allowed,
    validate_trade,
)


def test_blacklist_blocks_aliexpress() -> None:
    assert is_source_allowed("https://www.aliexpress.com/item/1234") is False
    assert is_source_allowed("https://temu.com/anything") is False


def test_whitelist_allows_bigbuy() -> None:
    assert is_source_allowed("https://www.bigbuy.eu/product/9999") is True
    assert is_source_allowed("https://ebay.it/itm/abc") is True


def test_unknown_source_blocked_default_deny() -> None:
    assert is_source_allowed("https://random-shop.tld/x") is False
    assert is_source_allowed("") is False


def test_keyword_cbd_regulated() -> None:
    regulated, reason = is_keyword_regulated("Olio CBD 10% biologico premium")
    assert regulated is True
    assert reason is not None and "cannabinoide" in reason.lower()


def test_keyword_normal_not_regulated() -> None:
    regulated, reason = is_keyword_regulated("Maglietta cotone taglia M")
    assert regulated is False
    assert reason is None


def test_validate_trade_aggregates_errors() -> None:
    bad = {
        "source_url": "https://aliexpress.com/item/x",
        "product_name": "Cerotti magnetici antidolore",
    }
    valid, errors = bad_validate_unpack(validate_trade(bad))
    assert valid is False
    assert len(errors) == 2
    assert any("blacklisted" in e for e in errors)
    assert any("regulated keyword" in e for e in errors)

    good = {
        "source_url": "https://bigbuy.eu/product/123",
        "product_name": "Tazza ceramica bianca",
    }
    valid, errors = bad_validate_unpack(validate_trade(good))
    assert valid is True
    assert errors == []


def bad_validate_unpack(result: tuple[bool, list[str]]) -> tuple[bool, list[str]]:
    return result
