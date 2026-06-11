"""Network visualization component — Plotly-based connectome and graph viz."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import networkx as nx

    _HAS_NX = True
except ImportError:
    _HAS_NX = False


def plot_connectome_heatmap(
    W: np.ndarray,
    roi_names: list[str] | None = None,
    title: str = "Connectome Matrix",
    colorscale: str = "RdBu_r",
) -> go.Figure:
    """Plotly heatmap interattiva della matrice di connettività FC."""
    n = W.shape[0]
    labels = roi_names if roi_names and len(roi_names) == n else [str(i) for i in range(n)]

    # Simmetrizza e azzera diagonale per visualizzazione
    W_disp = (W + W.T) / 2.0
    np.fill_diagonal(W_disp, 0.0)

    vlim = float(np.percentile(np.abs(W_disp[W_disp != 0]), 95)) if np.any(W_disp != 0) else 1.0

    fig = go.Figure(
        go.Heatmap(
            z=W_disp,
            x=labels,
            y=labels,
            colorscale=colorscale,
            zmin=-vlim,
            zmax=vlim,
            colorbar={"title": "FC", "thickness": 12},
        )
    )
    fig.update_layout(
        title=title,
        xaxis={"tickfont": {"size": 7}, "tickangle": 45},
        yaxis={"tickfont": {"size": 7}, "autorange": "reversed"},
        height=550,
        margin={"l": 80, "r": 20, "t": 50, "b": 80},
    )
    return fig


def compute_hub_metrics(
    W: np.ndarray,
    threshold: float = 0.20,
    roi_names: list[str] | None = None,
) -> dict[str, object]:
    """Calcola hub centrality su grafo sogliato. Richiede networkx."""
    if not _HAS_NX:
        n = W.shape[0]
        labels = roi_names if roi_names and len(roi_names) == n else [str(i) for i in range(n)]
        # Fallback: degree basato su forza (row sum assoluto)
        strength = np.sum(np.abs(W), axis=1)
        strength_norm = strength / (strength.max() + 1e-10)
        deg = {labels[i]: float(strength_norm[i]) for i in range(n)}
        top10 = sorted(deg.items(), key=lambda x: x[1], reverse=True)[:10]
        return {
            "degree_centrality": deg,
            "betweenness_centrality": deg,
            "eigenvector_centrality": deg,
            "pagerank": deg,
            "top10_hubs": top10,
            "graph": None,
        }

    from features.graph_theory import _threshold_proportional  # noqa: PLC0415

    W_thr = _threshold_proportional(W, threshold)
    n = W_thr.shape[0]
    labels = roi_names if roi_names and len(roi_names) == n else [str(i) for i in range(n)]

    G = nx.from_numpy_array(np.abs(W_thr))
    mapping = {i: labels[i] for i in range(n)}
    G = nx.relabel_nodes(G, mapping)

    deg = nx.degree_centrality(G)
    try:
        between = nx.betweenness_centrality(G, weight="weight", normalized=True)
    except Exception:
        between = {node: 0.0 for node in G.nodes()}
    try:
        eig = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
    except Exception:
        eig = {node: 0.0 for node in G.nodes()}
    pr = nx.pagerank(G, weight="weight")

    top10 = sorted(eig.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "degree_centrality": deg,
        "betweenness_centrality": between,
        "eigenvector_centrality": eig,
        "pagerank": pr,
        "top10_hubs": top10,
        "graph": G,
    }


def plot_network_plotly(
    W: np.ndarray,
    roi_names: list[str] | None = None,
    threshold: float = 0.15,
    top_n_hubs: int = 10,
    title: str = "Connectivity Network",
) -> go.Figure:
    """Visualizzazione network Plotly con layout spring. Richiede networkx."""
    if not _HAS_NX:
        fig = go.Figure()
        fig.update_layout(
            title=f"{title} — networkx non installato (pip install networkx)",
            annotations=[{"text": "Installa networkx per la visualizzazione rete", "showarrow": False}],
        )
        return fig

    from features.graph_theory import _threshold_proportional  # noqa: PLC0415

    W_thr = _threshold_proportional(W, threshold)
    n = W_thr.shape[0]
    labels = roi_names if roi_names and len(roi_names) == n else [str(i) for i in range(n)]

    G = nx.from_numpy_array(np.abs(W_thr))
    mapping = {i: labels[i] for i in range(n)}
    G = nx.relabel_nodes(G, mapping)

    if G.number_of_edges() == 0:
        fig = go.Figure()
        fig.update_layout(title=f"{title} — nessun arco sopra threshold={threshold:.2f}")
        return fig

    pos = nx.spring_layout(G, seed=42, weight="weight")

    # Edge traces
    edge_x, edge_y, edge_w = [], [], []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_w.append(d.get("weight", 0))

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line={"width": 0.5, "color": "#888"},
        hoverinfo="none",
    )

    # Node traces con eigenvector centrality come colore
    try:
        eig = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
    except Exception:
        eig = {node: 0.0 for node in G.nodes()}

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_color = [eig[n] for n in G.nodes()]
    node_size = [8 + 30 * eig[n] for n in G.nodes()]
    node_text = [
        f"{n}<br>Eig: {eig[n]:.3f}<br>Degree: {G.degree(n)}"
        for n in G.nodes()
    ]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=list(G.nodes()),
        textposition="top center",
        textfont={"size": 6},
        hovertext=node_text,
        marker={
            "size": node_size,
            "color": node_color,
            "colorscale": "Viridis",
            "showscale": True,
            "colorbar": {"title": "Eigenvec<br>Centrality", "thickness": 12},
            "line": {"width": 1, "color": "white"},
        },
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"{title} (threshold={threshold:.2f}, n_edges={G.number_of_edges()})",
            showlegend=False,
            hovermode="closest",
            xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            height=550,
            margin={"l": 10, "r": 10, "t": 50, "b": 10},
        ),
    )
    return fig


def plot_hub_bar(
    hub_metrics: dict[str, object],
    metric: str = "eigenvector_centrality",
    top_n: int = 15,
    title: str | None = None,
) -> go.Figure:
    """Bar chart top-N hub regions per una metrica di centralità."""
    data: dict[str, float] = hub_metrics.get(metric, {})  # type: ignore[assignment]
    top = sorted(data.items(), key=lambda x: x[1], reverse=True)[:top_n]
    if not top:
        fig = go.Figure()
        fig.update_layout(title="Nessun hub disponibile")
        return fig

    names = [t[0] for t in top]
    vals = [t[1] for t in top]
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker_color="steelblue",
        )
    )
    fig.update_layout(
        title=title or f"Top-{top_n} Hub — {metric}",
        xaxis_title=metric,
        yaxis={"autorange": "reversed"},
        height=450,
        margin={"l": 150, "r": 20, "t": 50, "b": 40},
    )
    return fig
