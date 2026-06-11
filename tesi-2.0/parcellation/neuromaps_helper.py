"""
STEP 3b — Associazione mappe neuroscientifiche a ROI (opzionale).

Usa la libreria `neuromaps` (se disponibile) per scaricare annotazioni
cerebrali (es. gradient evolutivo, mielinizzazione) e parcellizzarle
secondo gli atlanti supportati da extract_label_tc.

Se `neuromaps` non è installato, le funzioni ritornano dict vuoti con warning.

Variabile d'ambiente:
  NEUROMAPS_DATA_DIR : directory cache per dati neuromaps (default: ~/neuromaps-data)
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from parcellation.extract_label_tc import AtlasName, get_labels

if TYPE_CHECKING:
    pass

_NEUROMAPS_AVAILABLE: bool
try:
    import neuromaps  # noqa: F401
    _NEUROMAPS_AVAILABLE = True
except ImportError:
    _NEUROMAPS_AVAILABLE = False

_NEUROMAPS_DATA_DIR = Path(
    os.environ.get("NEUROMAPS_DATA_DIR", str(Path.home() / "neuromaps-data"))
)

# Annotazioni di default da scaricare se neuromaps disponibile
# Formato: (source, desc, space, den)
_DEFAULT_ANNOTATIONS: list[tuple[str, str, str, str]] = [
    ("margulies2016", "fcgradient01", "fsLR", "32k"),
]


def _warn_no_neuromaps() -> None:
    warnings.warn(
        "neuromaps non installato — fetch annotazioni disabilitato. "
        "Installa con: pip install neuromaps",
        ImportWarning,
        stacklevel=3,
    )


def fetch_annotation_for_atlas(
    atlas: AtlasName,
    *,
    subject: str = "fsaverage",
    annotations: list[tuple[str, str, str, str]] | None = None,
) -> dict[str, np.ndarray]:
    """Scarica annotazioni neuromaps e le parcellizza per l'atlante richiesto.

    Parameters
    ----------
    atlas:
        Chiave atlante (aparc, destrieux, schaefer100, schaefer200, schaefer400).
    subject:
        Subject fsaverage per le label (default: "fsaverage").
    annotations:
        Lista di tuple (source, desc, space, den) da neuromaps.datasets.fetch_annotation.
        Default: gradient evolutivo Margulies 2016.

    Returns
    -------
    dict[str, np.ndarray]
        Chiave: "<source>_<desc>", valore: array shape (n_roi,) con valore
        medio per ciascuna ROI. Ritorna dict vuoto se neuromaps non disponibile.
    """
    if not _NEUROMAPS_AVAILABLE:
        _warn_no_neuromaps()
        return {}

    from neuromaps import datasets, transforms  # type: ignore[import]

    if annotations is None:
        annotations = _DEFAULT_ANNOTATIONS

    labels = get_labels(atlas, subject=subject)
    result: dict[str, np.ndarray] = {}

    for source, desc, space, den in annotations:
        key = f"{source}_{desc}"
        try:
            img = datasets.fetch_annotation(source=source, desc=desc, space=space, den=den)
            # Resample su fsaverage se necessario
            if space != "fsaverage":
                img = transforms.mni152_to_fsaverage(img, fsavg_density="10k")

            # Parcellizza: media per ROI
            roi_values = _parcellate_annotation(img, labels)
            result[key] = roi_values

        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Fetch annotazione {key} fallita: {exc}", RuntimeWarning, stacklevel=2)

    return result


def _parcellate_annotation(
    img: object,
    labels: list,
) -> np.ndarray:
    """Calcola il valore medio dell'annotazione per ciascuna label.

    Strategia: estrae i vertici di ciascuna label e ne fa la media
    sul dato dell'annotazione (array 1D per emisfero, concatenati lh+rh).

    Parameters
    ----------
    img:
        Immagine neuromaps (niimg o tuple (lh_data, rh_data) già su fsaverage).
    labels:
        Lista mne.Label da parcellation.extract_label_tc.get_labels.

    Returns
    -------
    np.ndarray, shape (n_roi,)
    """
    import nibabel as nib  # type: ignore[import]

    # Gestisci tuple (lh_arr, rh_arr) o niimg
    if isinstance(img, tuple) and len(img) == 2:
        lh_data, rh_data = img
        if hasattr(lh_data, "agg_data"):
            lh_arr = np.asarray(lh_data.agg_data()).ravel()
            rh_arr = np.asarray(rh_data.agg_data()).ravel()
        else:
            lh_arr = np.asarray(lh_data).ravel()
            rh_arr = np.asarray(rh_data).ravel()
    else:
        arr = np.asarray(nib.load(img).get_fdata()).ravel()
        n = len(arr) // 2
        lh_arr, rh_arr = arr[:n], arr[n:]

    roi_values = np.zeros(len(labels))
    for i, lbl in enumerate(labels):
        hemi_arr = lh_arr if lbl.hemi == "lh" else rh_arr
        verts = lbl.vertices
        if len(verts) == 0 or np.max(verts) >= len(hemi_arr):
            roi_values[i] = np.nan
        else:
            roi_values[i] = float(np.nanmean(hemi_arr[verts]))

    return roi_values
