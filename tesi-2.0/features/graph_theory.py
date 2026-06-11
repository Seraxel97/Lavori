"""Graph-theory features su matrice di connettività EEG.

Riduce la matrice FC (n×n) a ~8 scalari:
  mean_degree, mean_strength, clustering_coeff, path_length,
  global_efficiency, local_efficiency, modularity_q, small_worldness
"""

from __future__ import annotations

import warnings

import networkx as nx
import numpy as np

_N_RAND = 10  # random graph per small-worldness baseline


def _threshold_proportional(W: np.ndarray, threshold: float) -> np.ndarray:
    """Mantieni il top `threshold` fraction di archi per peso assoluto.

    Il ranking usa np.abs(W) per gestire correttamente matrici con segno
    (wPLI, imCoh). Gli archi mantenuti conservano il loro valore originale.
    """
    W = W.copy()
    np.fill_diagonal(W, 0.0)
    W = (W + W.T) / 2.0  # forza simmetria
    n = W.shape[0]
    n_edges_total = n * (n - 1) // 2
    n_keep = max(1, int(np.round(n_edges_total * threshold)))
    abs_vals = np.abs(W)[np.triu_indices(n, k=1)]
    if n_keep < len(abs_vals):
        cutoff = np.partition(abs_vals, -n_keep)[-n_keep]
        W[np.abs(W) < cutoff] = 0.0
    return W


def compute_graph_metrics(
    W: np.ndarray,
    threshold: float = 0.20,
) -> dict[str, float]:
    """Calcola metriche graph-theory su matrice di connettività.

    Parameters
    ----------
    W:
        Matrice simmetrica (n_nodes × n_nodes), valori ≥ 0.
    threshold:
        Frazione top archi da mantenere (default 0.20 = top 20%).

    Returns
    -------
    dict con chiavi: mean_degree, mean_strength, clustering_coeff,
    path_length, global_efficiency, local_efficiency,
    modularity_q, small_worldness
    """
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W deve essere quadrata, got shape {W.shape}")

    W_thr = _threshold_proportional(W, threshold)
    G = nx.from_numpy_array(W_thr)

    # Rimuovi self-loops residui
    G.remove_edges_from(nx.selfloop_edges(G))

    n = G.number_of_nodes()
    k = np.array([d for _, d in G.degree()])
    mean_degree = float(k.mean())

    strengths = np.array([sum(d for _, d in G[v].items() for d in [d.get("weight", 1.0)]) for v in G.nodes()])
    mean_strength = float(strengths.mean())

    # Clustering NON pesato — coerente con path_length/efficiency (anche unweighted)
    # e con C_rand dei grafi casuali in small_worldness. Scelta: unweighted per sigma.
    clustering_coeff = nx.average_clustering(G)

    # Path length: su componente più grande
    if nx.is_connected(G):
        path_length = nx.average_shortest_path_length(G)
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        subG = G.subgraph(largest_cc)
        if len(largest_cc) > 1:
            path_length = nx.average_shortest_path_length(subG)
        else:
            path_length = float("inf")

    global_efficiency = nx.global_efficiency(G)
    local_efficiency = nx.local_efficiency(G)

    # Modularità Louvain
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        communities = nx.community.louvain_communities(G, seed=42)
    modularity_q = nx.community.modularity(G, communities, weight="weight")

    # Small-worldness σ = (C/C_rand) / (L/L_rand)
    m = G.number_of_edges()
    if m > 0 and np.isfinite(path_length):
        rand_c_list = []
        rand_l_list = []
        rng = np.random.default_rng(42)
        for _ in range(_N_RAND):
            seed_i = int(rng.integers(0, 2**31))
            Gr = nx.gnm_random_graph(n, m, seed=seed_i)
            rand_c_list.append(nx.average_clustering(Gr))
            if nx.is_connected(Gr):
                rand_l_list.append(nx.average_shortest_path_length(Gr))
        c_rand = float(np.mean(rand_c_list)) if rand_c_list else np.nan
        l_rand = float(np.mean(rand_l_list)) if rand_l_list else np.nan
        if c_rand > 0 and l_rand > 0:
            small_worldness = (clustering_coeff / c_rand) / (path_length / l_rand)
        else:
            small_worldness = np.nan
    else:
        small_worldness = np.nan

    return {
        "mean_degree": mean_degree,
        "mean_strength": mean_strength,
        "clustering_coeff": clustering_coeff,
        "path_length": path_length,
        "global_efficiency": global_efficiency,
        "local_efficiency": local_efficiency,
        "modularity_q": modularity_q,
        "small_worldness": small_worldness,
    }
