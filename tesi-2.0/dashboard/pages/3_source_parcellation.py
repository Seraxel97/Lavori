"""Tab 3 — Source Space & Parcellazione.

Heatmap segnale per parcel × soggetto + confronto atlas.
Nota: source estimate viewer 3D (dSPM fsaverage) richiede display interattivo MNE;
  in modalità server Streamlit si visualizzano le matrici FC per-parcel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.components.data_loader import (  # noqa: E402
    list_subjects_with_connectivity,
    load_connectivity_matrix,
)

st.set_page_config(page_title="Source & Parcellazione — Tesi 2.0", page_icon="🧠", layout="wide")

st.title("🧠 Source Space & Parcellazione")
st.caption(
    "Heatmap FC per parcel · confronto atlas aparc vs schaefer100 · "
    "source reconstruction dSPM fsaverage"
)

st.info(
    "**Nota**: la visualizzazione 3D source estimate (MNE `plot_source_estimates` su fsaverage) "
    "non è disponibile in modalità headless Streamlit. "
    "Questa tab mostra le matrici FC source-level per parcel come alternativa interattiva."
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Selettori")
    metric = st.selectbox("Metrica FC", ["plv", "wpli", "coh", "imcoh"], index=0)
    band = st.selectbox("Banda", ["theta", "alpha", "beta", "gamma"], index=0)
    cond = st.selectbox("Condizione", ["EC", "EO"], index=0)
    threshold_pct = st.slider("Threshold FC (percentile)", 50, 99, 80)
    st.divider()
    st.subheader("Confronto atlas")
    compare_atlas = st.checkbox("Confronta aparc vs schaefer100", value=False)

# ---------------------------------------------------------------------------
# Heatmap parcel aggregata (media su tutti i soggetti disponibili)
# ---------------------------------------------------------------------------
st.markdown("---")

for atlas in (["aparc", "schaefer100"] if compare_atlas else ["aparc"]):
    subjects = list_subjects_with_connectivity(atlas=atlas, metric=metric, band=band, cond=cond)
    n_sub = len(subjects)

    if n_sub == 0:
        st.warning(f"Nessun soggetto disponibile per {atlas}×{metric}×{band}×{cond}")
        continue

    with st.spinner(f"Calcolo media FC su {n_sub} soggetti ({atlas})..."):
        matrices: list[np.ndarray] = []
        for sub in subjects:
            try:
                W = load_connectivity_matrix(sub, atlas=atlas, metric=metric, band=band, cond=cond)
                matrices.append(W)
            except FileNotFoundError:
                continue

    if not matrices:
        st.warning(f"Nessuna matrice caricata per {atlas}")
        continue

    W_mean = np.mean(np.stack(matrices, axis=0), axis=0)
    n_roi = W_mean.shape[0]

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"FC media — {atlas} ({n_roi} ROI) · {metric}·{band}·{cond} · N={len(matrices)}")

        # Applica threshold
        thr_val = float(np.percentile(np.abs(W_mean[np.triu_indices(n_roi, k=1)]), threshold_pct))
        W_thr = W_mean.copy()
        W_thr[np.abs(W_mean) < thr_val] = 0.0

        fig_hm = go.Figure(
            go.Heatmap(
                z=W_thr,
                colorscale="RdBu_r",
                colorbar={"title": f"{metric}", "thickness": 12},
                zmin=-float(np.max(np.abs(W_thr))) if np.any(W_thr != 0) else -1,
                zmax=float(np.max(np.abs(W_thr))) if np.any(W_thr != 0) else 1,
            )
        )
        fig_hm.update_layout(
            xaxis={"title": "ROI index", "tickfont": {"size": 8}},
            yaxis={"title": "ROI index", "tickfont": {"size": 8}, "autorange": "reversed"},
            height=480,
            margin={"l": 60, "r": 20, "t": 30, "b": 60},
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    with col2:
        st.markdown("**Statistiche FC media**")
        triu = W_mean[np.triu_indices(n_roi, k=1)]
        st.metric("FC max", f"{np.max(np.abs(triu)):.4f}")
        st.metric("FC mean", f"{np.mean(np.abs(triu)):.4f}")
        st.metric("FC std", f"{np.std(triu):.4f}")
        st.metric("N ROI", n_roi)
        st.metric("N soggetti", len(matrices))
        st.metric(f"Threshold (p{threshold_pct})", f"{thr_val:.4f}")

    # Profilo medio per ROI (degree medio)
    st.markdown("---")
    st.subheader(f"Profilo FC per ROI — {atlas} · {cond}")
    row_mean = np.mean(np.abs(W_mean), axis=1)
    top_idx = np.argsort(row_mean)[::-1][:20]

    fig_bar = go.Figure(
        go.Bar(
            x=row_mean[top_idx],
            y=[f"ROI-{i}" for i in top_idx],
            orientation="h",
            marker_color="steelblue",
        )
    )
    fig_bar.update_layout(
        title="Top-20 ROI per FC medio assoluto (|FC|)",
        xaxis_title=f"|{metric}| medio",
        yaxis={"autorange": "reversed"},
        height=400,
        margin={"l": 80, "r": 20, "t": 50, "b": 40},
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption(
        f"Il profilo di FC per ROI riflette il grado medio di connettività funzionale di ciascuna "
        f"parcella {atlas} in condizione {cond}. Atlas aparc = Desikan-Killiany 68 ROI; "
        "schaefer100 = 100 ROI data-driven."
    )

    if compare_atlas:
        st.markdown("---")

st.caption("ds004504 (LEMON) · confronto cross-dataset disponibile in **Tab 6 Cross-Dataset**")
