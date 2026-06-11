"""Base protocol e helper per il dispatcher pattern — Tesi_2.0.

I tre dispatcher (fc, ml, features) condividono un pattern comune:
ricevono una chiave Literal (metric/algorithm), la validano, e delegano
all'implementazione. Questo modulo centralizza la validazione e il
messaggio di errore.

Uso:
    from common.dispatcher_base import validate_dispatch_key

    validate_dispatch_key(metric, VALID_METRICS, "metric")
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DispatcherProtocol(Protocol):
    """Interfaccia comune per i dispatcher Tesi_2.0.

    Un dispatcher valida una chiave stringa e delega l'esecuzione.
    I dispatcher esistenti sono funzioni di modulo, non classi: questo
    Protocol serve come contratto per type checking, non come base class.
    """

    def validate_key(self, key: str, valid_keys: tuple[str, ...], key_type: str) -> None:
        """Verifica che key sia in valid_keys; lancia ValueError altrimenti."""
        ...

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Esegue il dispatch principale."""
        ...


def validate_dispatch_key(key: str, valid_keys: tuple[str, ...], key_type: str) -> None:
    """Valida che key sia una delle valid_keys; lancia ValueError con messaggio uniforme.

    Parameters
    ----------
    key:
        Valore da validare (es. "wpli", "logreg").
    valid_keys:
        Insieme dei valori accettati.
    key_type:
        Nome del tipo per il messaggio di errore (es. "metric", "algorithm").

    Raises
    ------
    ValueError
        Se key non e' in valid_keys.

    Examples
    --------
    >>> validate_dispatch_key("wpli", ("wpli", "plv", "coh"), "metric")
    >>> validate_dispatch_key("bad", ("wpli", "plv"), "metric")
    ValueError: Unknown metric='bad'. Valid: coh, plv, wpli
    """
    if key not in valid_keys:
        valid_sorted = ", ".join(sorted(valid_keys))
        raise ValueError(f"Unknown {key_type}={key!r}. Valid: {valid_sorted}")


class BaseDispatcher:
    """Classe base opzionale per dispatcher che preferiscono lo stile OOP.

    I dispatcher correnti (fc, ml, features) sono funzioni di modulo e non
    ereditano da questa classe. BaseDispatcher e' disponibile per dispatcher
    futuri che vogliono lo stile class-based.
    """

    _valid_keys: tuple[str, ...] = ()
    _key_type: str = "key"

    def validate_key(self, key: str) -> None:
        validate_dispatch_key(key, self._valid_keys, self._key_type)

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
