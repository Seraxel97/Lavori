"""Arbitrage engine — formula ROI e filtro threshold."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .price_discovery import PriceAggregate

# Costanti fiscali e operative Italia
VAT_IT = 0.22          # IVA standard 22%
FEES_MARKETPLACE = 0.13  # ~13% fee media eBay/Vinted
SHIPPING_EUR = 5.0     # stima spedizione media Italia €5


@dataclass
class ArbitrageOpportunity:
    """Opportunità di arbitrage calcolata."""
    product_query: str
    score: float          # ROI normalizzato: (high - low - fees - shipping - VAT) / low
    avg_price_low: float
    avg_price_high: float
    gross_margin: float   # high - low
    net_margin: float     # gross - fees - shipping - VAT
    passes_threshold: bool
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_arbitrage(
    agg: PriceAggregate,
    threshold: float = 1.0,
    fees_rate: float = FEES_MARKETPLACE,
    shipping_eur: float = SHIPPING_EUR,
    vat_rate: float = VAT_IT,
) -> ArbitrageOpportunity:
    """
    score = (avg_high - avg_low - fees - shipping - VAT_amount) / avg_low
    Threshold default = 1.0 (ROI ≥ 100%).
    """
    low = agg.avg_price_low
    high = agg.avg_price_high

    if low <= 0 or high <= 0:
        return ArbitrageOpportunity(
            product_query=agg.product_query,
            score=0.0,
            avg_price_low=low,
            avg_price_high=high,
            gross_margin=0.0,
            net_margin=0.0,
            passes_threshold=False,
            metadata={"reason": "insufficient_price_data"},
        )

    gross = high - low
    fees_abs = high * fees_rate
    vat_abs = low * vat_rate     # IVA sull'acquisto
    net = gross - fees_abs - shipping_eur - vat_abs
    score = net / low

    return ArbitrageOpportunity(
        product_query=agg.product_query,
        score=round(score, 4),
        avg_price_low=round(low, 2),
        avg_price_high=round(high, 2),
        gross_margin=round(gross, 2),
        net_margin=round(net, 2),
        passes_threshold=score >= threshold,
        metadata={
            "fees_abs": round(fees_abs, 2),
            "shipping_eur": shipping_eur,
            "vat_abs": round(vat_abs, 2),
            "n_price_sources": agg.n_sources,
            "threshold_used": threshold,
        },
    )
