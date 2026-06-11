"""Tab 4 — Connectivity & Graph Theory.

Viz PyVis network + metriche graph theory + hub identification.
Usa dati reali data/connectivity/ds005385/ e features/graph_theory.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.components.data_loader import (  # noqa: E402
    list_subjects_with_connectivity,
    load_connectivity_matrix,
)
from dashboard.components.network_viz import (  # noqa: E402
    compute_hub_metrics,
    plot_connectome_heatmap,
    plot_hub_bar,
    plot_network_plotly,
)

st.set_page_config(
    page_title="Connectivity & Graph — Tesi 2.0", page_icon="🌐", layout="wide"
)

st.title("🌐 Connectivity & Graph Theory")
st.caption(
    "Matrice connectome · rete interattiva · hub detection · metriche small-world"
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Selettori")
    atlas = st.selectbox("Atlante", ["aparc", "schaefer100"], index=0)
    metric = st.selectbox(
        "Metrica FC",
        ["plv", "wpli", "coh", "imcoh", "ciplv", "pli", "wpli2_debiased"],
        index=0,
        help="wpli e imcoh sono robusti al volume conduction",
    )
    band = st.selectbox(
        "Banda",
        ["theta", "alpha", "beta", "gamma"],
        index=0,
    )
    cond = st.selectbox("Condizione", ["EC", "EO"], index=0)

    subjects = list_subjects_with_connectivity(atlas=atlas, metric=metric, band=band, cond=cond)
    if subjects:
        subject = st.selectbox("Soggetto", ["[media gruppo]"] + subjects, index=0)
    else:
        st.warning(f"Nessun dato per {atlas}×{metric}×{band}×{cond}")
        subject = None

    st.divider()
    st.subheader("Parametri rete")
    threshold_pct = st.slider(
        "Threshold archi (percentile)",
        50,
        99,
        80,
        help="Mantieni archi con FC > percentile X",
    )
    hub_metric_sel = st.selectbox(
        "Centralità hub",
        ["eigenvector_centrality", "betweenness_centrality", "degree_centrality", "pagerank"],
        index=0,
    )

if subject is None:
    st.error("Nessun dato disponibile per la combo selezionata.")
    st.stop()

# ---------------------------------------------------------------------------
# Carica matrice FC
# ---------------------------------------------------------------------------
with st.spinner("Carico matrice FC..."):
    if subject == "[media gruppo]":
        matrices = []
        for sub in subjects[:20]:  # limite 20 per velocità
            try:
                W = load_connectivity_matrix(sub, atlas=atlas, metric=metric, band=band, cond=cond)
                matrices.append(W)
            except FileNotFoundError:
                continue
        if not matrices:
            st.error("Nessuna matrice disponibile.")
            st.stop()
        W_fc = np.mean(np.stack(matrices, axis=0), axis=0)
        label = f"Media gruppo (N={len(matrices)}) · {atlas}×{metric}×{band}×{cond}"
    else:
        try:
            W_fc = load_connectivity_matrix(subject, atlas=atlas, metric=metric, band=band, cond=cond)
        except FileNotFoundError:
            st.error(f"File non trovato per {subject}")
            st.stop()
        label = f"{subject} · {atlas}×{metric}×{band}×{cond}"

# Threshold
n_roi = W_fc.shape[0]
triu_vals = np.abs(W_fc[np.triu_indices(n_roi, k=1)])
threshold_val = float(np.percentile(triu_vals, threshold_pct))
threshold_frac = 1.0 - threshold_pct / 100.0

# ---------------------------------------------------------------------------
# Layout 2-colonne: matrice + rete
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader(f"Connectome — {label}")

col_hm, col_net = st.columns([1, 1])

with col_hm:
    fig_hm = plot_connectome_heatmap(W_fc, title="Matrice FC")
    st.plotly_chart(fig_hm, use_container_width=True)

with col_net:
    with st.spinner("Calcolo rete..."):
        fig_net = plot_network_plotly(
            W_fc,
            threshold=threshold_frac,
            title=f"Rete FC (thr={threshold_val:.3f})",
        )
    st.plotly_chart(fig_net, use_container_width=True)

# ---------------------------------------------------------------------------
# Graph metrics
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Metriche Graph Theory")

with st.spinner("Calcolo metriche graph..."):
    from features.graph_theory import compute_graph_metrics

    gm = compute_graph_metrics(W_fc, threshold=threshold_frac)
    hub_data = compute_hub_metrics(W_fc, threshold=threshold_frac)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Clustering coeff (C)", f"{gm.get('clustering_coeff', float('nan')):.4f}")
c2.metric("Global efficiency (E_glob)", f"{gm.get('global_efficiency', float('nan')):.4f}")
c3.metric("Local efficiency (E_loc)", f"{gm.get('local_efficiency', float('nan')):.4f}")
c4.metric("Modularity (Q)", f"{gm.get('modularity_q', float('nan')):.4f}")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Path length (L)", f"{gm.get('path_length', float('nan')):.4f}")
c6.metric("Small-worldness (σ)", f"{gm.get('small_worldness', float('nan')):.4f}")
c7.metric("Mean degree", f"{gm.get('mean_degree', float('nan')):.2f}")
c8.metric("Mean strength", f"{gm.get('mean_strength', float('nan')):.4f}")

st.caption(
    "Metriche calcolate su grafo binario/pesato sogliato al percentile selezionato. "
    "σ > 1 → small-world (C_real > C_rand e L_real ≈ L_rand). "
    "Q > 0.3 → struttura modulare significativa."
)

# ---------------------------------------------------------------------------
# Hub identification
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Hub Identification — Top-15 regioni")

col_hub1, col_hub2 = st.columns([2, 1])

with col_hub1:
    fig_hub = plot_hub_bar(hub_data, metric=hub_metric_sel, top_n=15)
    st.plotly_chart(fig_hub, use_container_width=True)

with col_hub2:
    st.markdown(f"**Top-10 hub ({hub_metric_sel})**")
    top10 = hub_data.get("top10_hubs", [])
    if top10:
        for rank, (roi, val) in enumerate(top10, 1):
            st.markdown(f"{rank}. `{roi}` — {val:.4f}")
    st.caption(
        "Eigenvector centrality: rileva hub che si connettono ad altri hub. "
        "Betweenness: hub ponte tra comunità. "
        "PageRank: influenza globale nella rete."
    )

# ---------------------------------------------------------------------------
# Comparazione EO vs EC graph metrics
# ---------------------------------------------------------------------------
st.markdown("---")
with st.expander("Confronto EO vs EC — graph metrics", expanded=False):
    if subject != "[media gruppo]":
        try:
            W_alt_cond = load_connectivity_matrix(
                subject,
                atlas=atlas,
                metric=metric,
                band=band,
                cond="EO" if cond == "EC" else "EC",
            )
            gm_alt = compute_graph_metrics(W_alt_cond, threshold=threshold_frac)
            alt_cond = "EO" if cond == "EC" else "EC"

            metrics_labels = [
                ("clustering_coeff", "Clustering C"),
                ("global_efficiency", "E_glob"),
                ("local_efficiency", "E_loc"),
                ("modularity_q", "Modularity Q"),
                ("small_worldness", "Small-worldness σ"),
                ("mean_degree", "Mean degree"),
            ]

            comp_rows = []
            for key, name in metrics_labels:
                v1 = gm.get(key, float("nan"))
                v2 = gm_alt.get(key, float("nan"))
                comp_rows.append(
                    {"Metrica": name, cond: f"{v1:.4f}", alt_cond: f"{v2:.4f}"}
                )

            import pandas as pd

            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)
        except FileNotFoundError:
            st.info(f"Dati condizione alternativa non disponibili per {subject}")
    else:
        st.info("Confronto EO/EC disponibile solo per singolo soggetto")

st.caption("ds004504 (LEMON) · confronto cross-dataset disponibile in **Tab 6 Cross-Dataset**")
