"""NLP product extractor via Vesta router → Groq llama-3.3-70b."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractedProduct:
    """Prodotto estratto da contenuto virale."""
    name: str
    confidence: float  # 0-1
    source_context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


_EXTRACT_PROMPT = """Sei un analista e-commerce. Dalla lista di keyword di trend italiane fornita,
estrai i nomi di prodotti fisici che potrebbero essere rivenduti su marketplace (eBay, Vinted).
Ignora brand astratti, eventi, persone. Ritorna JSON con chiave "products": lista di oggetti
{"name": string, "confidence": float 0-1, "category": string}.
Keyword input: {keywords}"""


class NlpExtractor:
    """Estrae nomi prodotti virali dalle keyword di trend via Vesta LLM router."""

    def __init__(self) -> None:
        self._vesta_available: bool | None = None

    async def _check_vesta(self) -> bool:
        if self._vesta_available is not None:
            return self._vesta_available
        try:
            from vesta.router import LLMRouter  # type: ignore[import]
            self._vesta_available = True
        except ImportError:
            self._vesta_available = False
        return self._vesta_available

    async def extract(self, keywords: list[str]) -> list[ExtractedProduct]:
        """Estrae prodotti dalle keyword. Fallback euristico se Vesta non disponibile."""
        if await self._check_vesta():
            return await self._extract_via_vesta(keywords)
        return self._extract_heuristic(keywords)

    async def _extract_via_vesta(self, keywords: list[str]) -> list[ExtractedProduct]:
        from vesta.router import LLMRouter  # type: ignore[import]

        router = LLMRouter()
        prompt = _EXTRACT_PROMPT.format(keywords=", ".join(keywords))
        try:
            response = await router.chat(
                prompt=prompt,
                model="llama-3.3-70b",
                provider="groq",
                max_tokens=512,
            )
            raw = response.get("content", "{}")
            data = json.loads(raw)
            return [
                ExtractedProduct(
                    name=p["name"],
                    confidence=float(p.get("confidence", 0.5)),
                    source_context="vesta_groq",
                    metadata={"category": p.get("category", "")},
                )
                for p in data.get("products", [])
            ]
        except Exception:
            return self._extract_heuristic(keywords)

    def _extract_heuristic(self, keywords: list[str]) -> list[ExtractedProduct]:
        """Euristico semplice: filtra keyword che sembrano prodotti fisici."""
        # stop-list generica per keyword non-product
        stop = {"tendenze", "moda", "trend", "viral", "tiktok", "youtube", "instagram",
                 "skincare", "beauty", "fitness", "salute", "sport"}
        products: list[ExtractedProduct] = []
        for kw in keywords:
            kw_lower = kw.lower()
            if any(s in kw_lower for s in stop):
                # keyword categoria, non prodotto specifico — abbassa confidence
                products.append(ExtractedProduct(
                    name=kw, confidence=0.3, source_context="heuristic"
                ))
            else:
                products.append(ExtractedProduct(
                    name=kw, confidence=0.6, source_context="heuristic"
                ))
        return products

    async def store_in_vesta_belief(self, product: ExtractedProduct) -> bool:
        """Persiste confidence in Vesta belief API."""
        try:
            from vesta.belief.api import BeliefStore  # type: ignore[import]
            store = BeliefStore()
            await store.upsert(
                entity=f"product:{product.name}",
                fact="trend_confidence",
                value=product.confidence,
                source="selly_sandbox_f1",
            )
            return True
        except Exception:
            return False
