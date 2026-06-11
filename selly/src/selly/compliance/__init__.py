"""Compliance module — regulatory hardcoded enforcement per design v2 §11."""

from selly.compliance.regulatory_check import (
    REGULATED_KEYWORDS,
    SOURCES_BLACKLIST,
    SOURCES_WHITELIST,
    is_keyword_regulated,
    is_source_allowed,
    validate_trade,
)

__all__ = [
    "SOURCES_BLACKLIST",
    "SOURCES_WHITELIST",
    "REGULATED_KEYWORDS",
    "is_source_allowed",
    "is_keyword_regulated",
    "validate_trade",
]
