"""Regulatory enforcement hardcoded — design v2 §11 "anti-pattern NO".

Implementa whitelist sources CE-UE + blacklist sources extra-UE + keyword
filtering per dispositivi medici/cosmetici regolati (MDR Reg UE 2017/745,
Reg CE 1924/2006, D.Lgs 169/2004). Default-deny per sources sconosciute.

Riferimento: SELLY_DEEP_REVIEW_2026-06-04.md §"Regulatory enforcement gap".
"""

from __future__ import annotations

from urllib.parse import urlparse

SOURCES_BLACKLIST: frozenset[str] = frozenset({
    "aliexpress.com",
    "wish.com",
    "temu.com",
    "alibaba.com",
})

SOURCES_WHITELIST: frozenset[str] = frozenset({
    "bigbuy.eu",
    "brandsdistribution.com",
    "vinted.it",
    "subito.it",
    "ebay.it",
    "tiktok.com",
})

REGULATED_KEYWORDS: dict[str, str] = {
    "olio cbd": "cannabinoide D.Lgs 5/8/2023 autorizzazione Min.Salute richiesta",
    "integratori collagene": (
        "claim salutistici Reg. CE 1924/2006 + notifica Min.Salute D.Lgs 169/2004"
    ),
    "cerotti magnetici": (
        "dispositivo medico classe I MDR Reg UE 2017/745 se claim terapeutico"
    ),
    "massaggiatore elettrico": "dispositivo medico classe I MDR se uso terapeutico",
    "guanti anti artrite": "dispositivo medico classe I MDR potenziale",
    "solette ortopediche": "dispositivo medico classe I MDR potenziale",
}


def _extract_host(url: str) -> str:
    """Estrae host normalizzato (lowercase, no leading www.) da un URL."""
    if "://" not in url:
        url = "http://" + url
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def is_source_allowed(url: str) -> bool:
    """True solo se host esplicitamente in WHITELIST. Default-deny."""
    host = _extract_host(url)
    if not host:
        return False
    if host in SOURCES_BLACKLIST:
        return False
    return host in SOURCES_WHITELIST


def is_keyword_regulated(text: str) -> tuple[bool, str | None]:
    """Case-insensitive substring match. Ritorna (True, motivo) o (False, None)."""
    lower = (text or "").lower()
    for keyword, reason in REGULATED_KEYWORDS.items():
        if keyword in lower:
            return True, reason
    return False, None


def validate_trade(trade: dict) -> tuple[bool, list[str]]:
    """Aggrega check source + keyword su un trade dict.

    Trade richiesto: {'source_url': str, 'product_name': str}.
    Ritorna (valid, errors). valid=False se any check fail; errors lista
    descrizioni ordinata: source error first, then keyword errors.
    """
    errors: list[str] = []

    source_url = trade.get("source_url", "")
    if not is_source_allowed(source_url):
        host = _extract_host(source_url) or "<no-host>"
        if host in SOURCES_BLACKLIST:
            errors.append(
                f"source blacklisted: {host} (extra-UE non CE-tracciabile)"
            )
        else:
            errors.append(
                f"source not whitelisted: {host} (default-deny, design v2 §7.5)"
            )

    product_name = trade.get("product_name", "")
    regulated, reason = is_keyword_regulated(product_name)
    if regulated:
        errors.append(f"regulated keyword in product_name: {reason}")

    return (not errors), errors
