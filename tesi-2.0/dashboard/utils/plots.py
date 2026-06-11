"""Plotly figure generators for the ML dashboard."""
from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go


def plot_confusion_matrix(
    cm: list[list[int]],
    labels: list[str] | None = None,
    title: str = "Confusion Matrix (LOSO aggregate)",
) -> go.Figure:
    """Annotated confusion matrix heatmap."""
    labels = labels or ["EO (0)", "EC (1)"]
    cm_arr = np.array(cm)
    total = cm_arr.sum()
    pct = [[f"{v/total*100:.1f}%" for v in row] for row in cm_arr]
    annotations_text = [
        [f"{cm_arr[i,j]}<br><sub>{pct[i][j]}</sub>" for j in range(2)]
        for i in range(2)
    ]

    fig = go.Figure(go.Heatmap(
        z=cm_arr,
        x=labels,
        y=labels,
        colorscale="Blues",
        showscale=False,
        text=annotations_text,
        texttemplate="%{text}",
        textfont={"size": 18},
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="Predicted",
        yaxis_title="True",
        xaxis=dict(side="top"),
        width=400, height=380,
        margin=dict(l=60, r=20, t=80, b=40),
    )
    return fig


def plot_roc_curve(
    fpr: list[float],
    tpr: list[float],
    auc_val: float,
    title: str = "ROC Curve (LOSO aggregate)",
) -> go.Figure:
    """ROC curve with AUC annotation."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        name=f"ROC (AUC={auc_val:.3f})",
        line=dict(color="#2166ac", width=2.5),
        fill="tozeroy", fillcolor="rgba(33,102,172,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Chance",
        line=dict(color="gray", width=1.2, dash="dash"),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.6, y=0.05),
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.02]),
        width=480, height=380,
        margin=dict(l=60, r=20, t=60, b=50),
    )
    return fig


def plot_learning_curve(
    ba_per_fold: list[float],
    title: str = "BA per LOSO fold",
) -> go.Figure:
    """Balanced accuracy per fold (learning curve proxy)."""
    x = list(range(1, len(ba_per_fold) + 1))
    mean_ba = float(np.mean(ba_per_fold))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=ba_per_fold,
        mode="markers+lines",
        name="BA per fold",
        marker=dict(size=5, color="#4dac26"),
        line=dict(color="#4dac26", width=1.2),
    ))
    fig.add_hline(
        y=mean_ba,
        line=dict(color="#d73027", width=1.5, dash="dash"),
        annotation_text=f"Mean BA={mean_ba:.3f}",
        annotation_position="top right",
    )
    fig.add_hline(
        y=0.5,
        line=dict(color="gray", width=1, dash="dot"),
        annotation_text="Chance",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="Fold (subject withheld)",
        yaxis_title="Balanced Accuracy",
        yaxis=dict(range=[0, 1.05]),
        width=600, height=340,
        margin=dict(l=60, r=20, t=60, b=50),
    )
    return fig


def plot_permutation_null(
    null_distribution: list[float],
    observed_ba: float,
    p_perm: float | None = None,
    n_perm: int | None = None,
    title: str = "Permutation Null Distribution",
) -> go.Figure:
    """Histogram of null distribution with observed BA highlighted."""
    from analysis.stats_utility import format_pvalue

    null = np.array(null_distribution)
    p_label = format_pvalue(p_perm, n_perm or 1) if p_perm is not None else "p=N/A"
    n_label = f" (n={n_perm})" if n_perm else ""

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=null,
        nbinsx=40,
        name=f"Null distribution{n_label}",
        marker=dict(color="#4575b4", opacity=0.75),
    ))
    fig.add_vline(
        x=observed_ba,
        line=dict(color="#d73027", width=2.5),
        annotation_text=f"Observed BA={observed_ba:.3f}<br>{p_label}",
        annotation_position="top right",
        annotation=dict(font=dict(color="#d73027", size=12)),
    )
    fig.add_vline(
        x=float(null.mean()),
        line=dict(color="gray", width=1.2, dash="dash"),
        annotation_text=f"Null mean={null.mean():.3f}",
        annotation_position="top left",
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="Balanced Accuracy (permuted labels)",
        yaxis_title="Count",
        showlegend=True,
        width=580, height=360,
        margin=dict(l=60, r=20, t=60, b=50),
    )
    return fig


def plot_feature_importance(
    coef: np.ndarray,
    roi_names: list[str] | None,
    n_top: int = 20,
    title: str = "Top FC Feature Importance (|logreg coef|)",
) -> go.Figure:
    """Bar chart of top FC features by absolute logistic regression coefficient."""
    n_feat = len(coef)
    if roi_names is not None:
        # Reconstruct edge labels from upper-triangle
        n_roi = int(round((1 + (1 + 8 * n_feat) ** 0.5) / 2))
        labels = []
        for i in range(n_roi):
            for j in range(i + 1, n_roi):
                r_i = roi_names[i] if i < len(roi_names) else f"ROI-{i}"
                r_j = roi_names[j] if j < len(roi_names) else f"ROI-{j}"
                labels.append(f"{r_i}↔{r_j}")
    else:
        labels = [f"feat-{i}" for i in range(n_feat)]

    idx_sorted = np.argsort(np.abs(coef))[::-1][:n_top]
    top_coef = coef[idx_sorted]
    top_labels = [labels[i] if i < len(labels) else f"feat-{i}" for i in idx_sorted]
    colors = ["#d73027" if v >= 0 else "#4575b4" for v in top_coef]

    fig = go.Figure(go.Bar(
        x=top_coef,
        y=top_labels,
        orientation="h",
        marker_color=colors,
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="Coefficient (pos=EC, neg=EO)",
        yaxis=dict(autorange="reversed"),
        width=640, height=max(300, n_top * 20 + 80),
        margin=dict(l=220, r=20, t=60, b=50),
    )
    return fig


def plot_results_comparison(results: list[dict[str, Any]]) -> go.Figure:
    """Grouped bar chart comparing BA across loaded results."""
    if not results:
        return go.Figure()

    labels = [
        f"{r.get('atlas','?')}×{r.get('metric','?')}×{r.get('band','?')}×{r.get('clf',r.get('algorithm','?'))}"
        for r in results
    ]
    bas = [r.get("ba", r.get("agg", {}).get("balanced_accuracy", 0)) for r in results]
    colors = ["#d73027" if b == max(bas) else "#4575b4" for b in bas]

    fig = go.Figure(go.Bar(
        x=bas,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{b:.3f}" for b in bas],
        textposition="outside",
    ))
    fig.add_vline(x=0.5, line=dict(color="gray", dash="dash", width=1))
    fig.update_layout(
        title=dict(text="Balanced Accuracy — tutte le combinazioni caricate", font=dict(size=14)),
        xaxis=dict(title="Balanced Accuracy", range=[0, 1.05]),
        height=max(300, len(results) * 28 + 100),
        margin=dict(l=300, r=60, t=60, b=50),
    )
    return fig
