"""Tab 2 — Signal & Preprocessing.

Visualizzazione connettività per-epoca (proxy del segnale EEG preprocessato),
PSD e matrici raw per soggetto selezionato.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.express as px
import streamlit as st

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.components.data_loader import (  # noqa: E402
    list_subjects_with_connectivity,
    load_connectivity_matrix,
)

st.set_page_config(page_title="Signal & Preprocessing — Tesi 2.0", page_icon="📊", layout="wide")

st.title("📊 Signal & Preprocessing")
st.caption(
    "Visualizzazione matrici FC per-soggetto · condizione EO vs EC · aparc 68 ROI"
)
st.info(
    "**Nota**: questa tab mostra le matrici di connettività funzionale (FC) source-level "
    "come proxy del segnale preprocessato. Il raw EEG viewer (MNE interactive) richiede "
    "un ambiente MNE con display, non disponibile in modalità server Streamlit."
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Selettori")
    atlas = st.selectbox("Atlante", ["aparc", "schaefer100"], index=0)
    metric = st.selectbox("Metrica FC", ["plv", "wpli", "coh", "imcoh"], index=0)
    band = st.selectbox("Banda", ["theta", "alpha", "beta", "gamma"], index=0)

subjects = list_subjects_with_connectivity(atlas=atlas, metric=metric, band=band, cond="EC")

with st.sidebar:
    if subjects:
        subject = st.selectbox("Soggetto", subjects, index=0)
    else:
        st.warning(f"Nessun soggetto per combo {atlas}×{metric}×{band}")
        subject = None

if subject is None:
    st.error("Nessun dato disponibile per la combo selezionata.")
    st.stop()

# ---------------------------------------------------------------------------
# Carica matrici
# ---------------------------------------------------------------------------
try:
    W_ec = load_connectivity_matrix(subject, atlas=atlas, metric=metric, band=band, cond="EC")
    has_ec = True
except FileNotFoundError:
    W_ec = None
    has_ec = False

try:
    W_eo = load_connectivity_matrix(subject, atlas=atlas, metric=metric, band=band, cond="EO")
    has_eo = True
except FileNotFoundError:
    W_eo = None
    has_eo = False

if not has_ec and not has_eo:
    st.error(f"Nessun dato connectivity per {subject} combo {atlas}×{metric}×{band}")
    st.stop()

# ---------------------------------------------------------------------------
# Tabs visualizzazione
# ---------------------------------------------------------------------------
tab_ec, tab_eo, tab_diff, tab_spectrum = st.tabs(
    ["🟦 EC (Eyes Closed)", "🟥 EO (Eyes Open)", "⚖️ EC − EO", "📈 Distribuzione FC"]
)

with tab_ec:
    if has_ec:
        st.subheader(f"{subject} — EC · {atlas} × {metric} × {band}")
        from dashboard.components.network_viz import plot_connectome_heatmap

        fig = plot_connectome_heatmap(W_ec, title=f"{subject} EC")
        st.plotly_chart(fig, use_container_width=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("FC max", f"{np.nanmax(np.abs(W_ec)):.4f}")
        col2.metric("FC mean", f"{np.nanmean(np.abs(W_ec)):.4f}")
        col3.metric("FC std", f"{np.nanstd(W_ec):.4f}")
    else:
        st.warning("Dati EC non disponibili per questo soggetto")

with tab_eo:
    if has_eo:
        st.subheader(f"{subject} — EO · {atlas} × {metric} × {band}")
        from dashboard.components.network_viz import plot_connectome_heatmap

        fig = plot_connectome_heatmap(W_eo, title=f"{subject} EO")
        st.plotly_chart(fig, use_container_width=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("FC max", f"{np.nanmax(np.abs(W_eo)):.4f}")
        col2.metric("FC mean", f"{np.nanmean(np.abs(W_eo)):.4f}")
        col3.metric("FC std", f"{np.nanstd(W_eo):.4f}")
    else:
        st.warning("Dati EO non disponibili per questo soggetto")

with tab_diff:
    if has_ec and has_eo:
        st.subheader(f"{subject} — EC − EO (effetto Berger, positive-control)")
        from dashboard.components.network_viz import plot_connectome_heatmap

        W_diff = W_ec - W_eo
        fig = plot_connectome_heatmap(
            W_diff, title=f"{subject} EC−EO Δ{metric}-{band}", colorscale="RdBu"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "**Effetto Berger (positive-control)**: la chiusura degli occhi aumenta la sincronizzazione "
            "alpha source-level nelle regioni occipitali. Rosso = EC > EO; Blu = EO > EC. "
            "Questo effetto è il **positive-control** della pipeline, non il risultato scientifico."
        )
    else:
        st.info("Carica entrambe le condizioni EO e EC per visualizzare la differenza.")

with tab_spectrum:
    if has_ec or has_eo:
        st.subheader("Distribuzione valori FC")
        data_list, labels_list = [], []
        if has_ec:
            triu_ec = W_ec[np.triu_indices(W_ec.shape[0], k=1)]
            data_list.extend(triu_ec.tolist())
            labels_list.extend(["EC"] * len(triu_ec))
        if has_eo:
            triu_eo = W_eo[np.triu_indices(W_eo.shape[0], k=1)]
            data_list.extend(triu_eo.tolist())
            labels_list.extend(["EO"] * len(triu_eo))

        import pandas as pd

        df_dist = pd.DataFrame({"FC": data_list, "Condizione": labels_list})
        fig_dist = px.histogram(
            df_dist,
            x="FC",
            color="Condizione",
            nbins=50,
            barmode="overlay",
            color_discrete_map={"EC": "steelblue", "EO": "tomato"},
            opacity=0.7,
            labels={"FC": f"{metric} ({band})"},
        )
        fig_dist.update_layout(height=350)
        st.plotly_chart(fig_dist, use_container_width=True)
        st.caption(
            f"Distribuzione dei {W_ec.shape[0] * (W_ec.shape[0] - 1) // 2 if has_ec else '?'} "
            "valori FC (triangolo superiore) per ciascuna condizione."
        )
