"""ML Scientific Dashboard — Tesi_2.0 (ds005385).

Dashboard home page + ML results viewer.
Le pagine complete sono in dashboard/pages/ (navigazione sidebar Streamlit):
  1_dataset_cohort · 2_signal_preprocessing · 3_source_parcellation
  4_connectivity_graph · 5_ml_predictions

Nota scientifica: EO vs EC (Eyes Open / Eyes Closed) è usato come **positive-control**
(validazione pipeline) — l'effetto Berger è atteso dal 1929 (BA≈0.92 non è una scoperta).
I target scientifici principali sono età e sesso (FASE 1+).

Run: streamlit run dashboard/app.py --server.port 8501
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root on path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from analysis.stats_utility import format_pvalue  # noqa: E402
from dashboard.utils.data_loader import (  # noqa: E402
    list_available_combos,
    list_available_datasets,
    load_existing_results,
    n50_winner_full_result,
    run_combo,
)
from dashboard.utils.export import export_all, fig_to_bytes  # noqa: E402
from dashboard.utils.plots import (  # noqa: E402
    plot_confusion_matrix,
    plot_feature_importance,
    plot_learning_curve,
    plot_permutation_null,
    plot_results_comparison,
    plot_roc_curve,
)


# ---------------------------------------------------------------------------
# Helpers (defined early to avoid forward-reference NameError in Streamlit)
# ---------------------------------------------------------------------------
def _load_for_importance(result: dict, dataset_id: str = "ds005385"):
    """Load feature matrix for a result dict, return (X, y, groups, roi_names) or (None,...)."""
    from dashboard.utils.data_loader import load_feature_matrix

    atlas = result.get("atlas", "?")
    metric = result.get("metric", "?")
    band = result.get("band", "?")
    try:
        X, y, groups, roi_names = load_feature_matrix(atlas, metric, band, dataset_id=dataset_id)
        return X, y, groups, roi_names
    except FileNotFoundError:
        return None, None, None, None


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ML Dashboard — Tesi 2.0",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Tooltips scientifici
# ---------------------------------------------------------------------------
TOOLTIPS = {
    "plv": "**PLV (Phase Locking Value)**: misura la sincronizzazione di fase inter-regionale. "
    "Cattura coupling genuino a livello sorgente dopo ricostruzione dSPM. "
    "Sensibile a volume conduction residuo — mitigato da parcellazione aparc.",
    "wpli": "**wPLI (weighted Phase Lag Index)**: metrica robusta al volume conduction. "
    "Scarta componenti di fase in-phase. Più conservativa di PLV.",
    "coh": "**Coherence**: coerenza spettrale (ampiezza + fase). Include informazione di ampiezza "
    "che può introdurre bias da volume conduction.",
    "imcoh": "**imCoh (imaginary Coherence)**: parte immaginaria della coerenza. "
    "Robusta al volume conduction, meno sensibile di wPLI alle interazioni a lungo range.",
    "theta": "**Theta (4–8 Hz)**: bande associate a network ippocampo-neocorticali a riposo "
    "e default-mode network. EC favorisce sincronizzazione theta; EO la interrompe.",
    "alpha": "**Alpha (8–13 Hz)**: banda del ritmo di Berger. Alpha blocking: l'apertura degli occhi "
    "sopprime la potenza alpha posteriore. Riflette processi attentivi e inibizione corticale.",
    "beta": "**Beta (13–30 Hz)**: legata a mantenimento dello stato cognitivo corrente, "
    "controllo motorio e processing sensoriale attivo.",
    "gamma": "**Gamma (30–80 Hz)**: associata a binding percettivo e processing visivo ad alta frequenza. "
    "Più suscettibile ad artefatti muscolari.",
    "logreg": "**Logistic Regression (L2, C=1.0)**: classificatore lineare con regolarizzazione ridge. "
    "Spesso ottimale per EEG con N piccolo e alta dimensione (Varoquaux et al., 2017).",
    "svm_rbf": "**SVM RBF**: Support Vector Machine con kernel radiale. Efficace per classi "
    "non linearmente separabili, ma richiede tuning di γ.",
    "lda": "**LDA (Linear Discriminant Analysis)**: assume gaussianità delle classi. "
    "Computazionalmente efficiente; sensibile a dimensionalità alta senza regolarizzazione.",
    "aparc": "**aparc (Desikan-Killiany, 68 ROI)**: atlante standard FreeSurfer. "
    "2278 feature FC (triangolo superiore 68×68). Risoluzione bassa, stime stabili.",
    "schaefer100": "**Schaefer-100 (100 ROI)**: atlante data-driven con 100 parcelle. "
    "4950 feature FC. Bilancia specificità anatomica e stabilità della stima.",
    "loso_cv": "**LOSO GroupKFold (N=50 fold)**: Leave-One-Subject-Out. Ogni fold esclude "
    "tutti i trial di un soggetto. Elimina subject leakage (+25 pp con k-fold naïve, "
    "Gemein et al. 2023). Gold standard per EEG multi-soggetto.",
    "perm_test": "**Permutation test**: 1000 permutazioni delle etichette. p-value empirico = "
    "#{null ≥ obs} / (N_perm+1). Con 1000 perm, risoluzione p=0.001 "
    "(Combrisson & Jerbi 2015; Phipson & Smyth 2010).",
}

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "all_results" not in st.session_state:
    st.session_state.all_results = []


# ---------------------------------------------------------------------------
# Button/selector initialization (avoid NameError for other targets)
# ---------------------------------------------------------------------------
run_btn = False
load_btn = False
winner_btn = False
sel_atlas = None
sel_metric = None
sel_band = None
sel_clf = None
sel_nperm = None
combo_exists = False

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🧠 ML Dashboard")
    st.caption("ds005385 · N=50 · Positive-control: EO vs EC (validazione pipeline) · LOSO")
    st.markdown("---")

    # ── Dataset selector ─────────────────────────────────────────────────────
    available_datasets = list_available_datasets()
    sel_dataset = st.selectbox(
        "Dataset",
        available_datasets,
        index=0,
        key="sel_dataset",
    )
    if "last_dataset" not in st.session_state:
        st.session_state.last_dataset = sel_dataset
    if sel_dataset != st.session_state.last_dataset:
        st.session_state.current_result = None
        st.session_state.all_results = []
        st.session_state.last_dataset = sel_dataset
        st.rerun()
    st.divider()

    # ── Target selector ──────────────────────────────────────────────────────
    sel_target = st.selectbox(
        "Target ML",
        ["EO/EC (positive-control)", "Sesso", "Età"],
        key="sel_target",
        help="EO/EC = validazione pipeline (effetto Berger). Sesso/Età = target scientifici.",
    )
    st.divider()

    if sel_target == "EO/EC (positive-control)":
        st.subheader("Selettori combo")

        available = list_available_combos(dataset_id=sel_dataset)
        atlases = sorted({c["atlas"] for c in available})
        metrics = sorted({c["metric"] for c in available})
        bands = sorted({c["band"] for c in available})

        sel_atlas = st.selectbox(
            "Parcellation", atlases, index=atlases.index("aparc") if "aparc" in atlases else 0
        )
        sel_metric = st.selectbox(
            "Connectivity metric", metrics, index=metrics.index("plv") if "plv" in metrics else 0
        )
        sel_band = st.selectbox(
            "Frequency band", bands, index=bands.index("theta") if "theta" in bands else 0
        )
        sel_clf = st.selectbox("Classifier", ["logreg", "svm", "mlp", "rf", "gb"], index=0)
        sel_nperm = st.select_slider(
            "Permutations (0=skip, lento)",
            [0, 50, 100, 200],
            value=0,
            help="0 = skip. 50 perm ≈ 25s, 100 perm ≈ 50s. Full 1000 perm ≈ 8h (usa winner pre-calcolato).",
        )

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            run_btn = st.button("▶ Run combo", use_container_width=True, type="primary")
        with col2:
            load_btn = st.button("📂 Load reports/", use_container_width=True)

        winner_btn = st.button(
            "🏆 Load N=50 winner",
            use_container_width=True,
            help="Carica aparc×plv×theta×logreg BA=0.920 (pre-calcolato)",
        )

        st.markdown("---")
        # Check if selected combo has feature file
        combo_exists = any(
            c["atlas"] == sel_atlas and c["metric"] == sel_metric and c["band"] == sel_band
            for c in available
        )
        if not combo_exists:
            st.warning(f"⚠ Feature file mancante:\nX_{sel_atlas}_{sel_metric}_{sel_band}.npz")

        # Tooltip scientifici
        st.markdown("---")
        st.subheader("📖 Info scientifica")
        for key in [sel_metric, sel_band, sel_clf, sel_atlas]:
            if key in TOOLTIPS:
                with st.expander(key.upper(), expanded=False):
                    st.markdown(TOOLTIPS[key])
        with st.expander("LOSO CV", expanded=False):
            st.markdown(TOOLTIPS["loso_cv"])
        with st.expander("Permutation test", expanded=False):
            st.markdown(TOOLTIPS["perm_test"])

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
if winner_btn:
    with st.spinner("Carico N=50 winner (aparc×plv×theta×logreg)..."):
        result = n50_winner_full_result()
    st.session_state.current_result = result
    if result not in st.session_state.all_results:
        st.session_state.all_results.append(result)
    st.success("✅ Winner N=50 caricato (BA=0.920, p_perm<0.001)")

if run_btn:
    if not combo_exists:
        st.error(f"Feature file non trovato: X_{sel_atlas}_{sel_metric}_{sel_band}.npz")
    else:
        with st.spinner(
            f"Running {sel_atlas}×{sel_metric}×{sel_band}×{sel_clf} (perm={sel_nperm})..."
        ):
            t0 = time.time()
            result = run_combo(sel_atlas, sel_metric, sel_band, sel_clf, n_perm=sel_nperm)
        st.session_state.current_result = result
        if result not in st.session_state.all_results:
            st.session_state.all_results.append(result)
        elapsed = time.time() - t0
        st.success(
            f"✅ Done in {elapsed:.1f}s — BA={result['ba']:.3f} "
            f"CI=[{result['ci_lo']:.3f},{result['ci_hi']:.3f}]"
        )

if load_btn:
    existing = load_existing_results()
    for r in existing:
        # Normalise schema
        ba = r.get("ba", r.get("agg", {}).get("balanced_accuracy", 0))
        norm = {
            "atlas": r.get("atlas", "?"),
            "metric": r.get("metric", "?"),
            "band": r.get("band", "?"),
            "clf": r.get("clf", r.get("algorithm", "?")),
            "ba": ba,
            "ci_lo": None,
            "ci_hi": None,
            "p_perm": None,
            "confusion": None,
            "ba_per_fold": [],
            "null_distribution": [],
            "roc_fpr": [],
            "roc_tpr": [],
            "roc_auc": None,
            "n_features": r.get("n_features", "?"),
            "source": "ml_results.json",
        }
        st.session_state.all_results.append(norm)
    st.success(f"📂 Caricati {len(existing)} risultati da reports/")


# ---------------------------------------------------------------------------
# Viste risultati Sesso / Età
# ---------------------------------------------------------------------------
if st.session_state.get("sel_target", "EO/EC (positive-control)") != "EO/EC (positive-control)":
    target_key = st.session_state["sel_target"]
    is_sex = target_key == "Sesso"
    results_path = _ROOT / "reports" / (
        "ml_sex_results.json" if is_sex else "ml_age_results.json"
    )

    st.title(f"{'🚻 Classificazione Sesso' if is_sex else '🎂 Predizione Età'}")
    st.caption(
        "Positive-control EO/EC: seleziona 'EO/EC (positive-control)' dalla sidebar."
    )

    if not results_path.exists():
        st.warning(
            f"Risultati non ancora disponibili. Eseguire:\n\n"
            f"```\npython -m ml_training.{'ml_sex' if is_sex else 'ml_age'}\n```"
        )
    else:
        res = json.loads(results_path.read_text())
        st.success(f"✅ Risultati caricati da `{results_path.name}`")
        st.markdown(
            f"**N soggetti**: {res.get('n_subjects', '?')} | "
            f"**N campioni**: {res.get('n_samples', '?')}"
        )

        clf_key = "classifiers" if is_sex else "regressors"
        clfs = res.get(clf_key, {})
        if clfs:
            rows = []
            for name, r in clfs.items():
                if is_sex:
                    rows.append(
                        {
                            "Modello": name,
                            "BA (mean)": f"{r.get('ba_mean', 0):.3f}",
                            "BA (std)": f"{r.get('ba_std', 0):.3f}",
                            "AUC": f"{r.get('auc_mean', float('nan')):.3f}",
                            "p-value": format_pvalue(r.get("p_value", 1.0)),
                            "n_perm": r.get("n_perm", "?"),
                        }
                    )
                else:
                    rows.append(
                        {
                            "Modello": name,
                            "MAE": f"{r.get('mae', 0):.2f}",
                            "R²": f"{r.get('r2', 0):.3f}",
                            "Brain-age gap": f"{r.get('brain_age_gap', 0):.2f}",
                            "p-value": format_pvalue(r.get("p_value", 1.0)),
                        }
                    )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("Nessun risultato classificatore nel file JSON.")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
result = st.session_state.current_result

st.title("EEG Source-Level FC — ML Classification Dashboard")
st.caption(
    "ds005385 · Dortmund Vital EEG · N=50 soggetti · "
    "**Positive-control: EO vs EC (validazione pipeline)** · LOSO GroupKFold"
)

if result is None:
    st.info(
        "👈 Seleziona una combo e premi **Run combo**, oppure carica il **N=50 winner** dalla sidebar."
    )
    st.markdown("---")
    # Show comparison of all loaded results if any
    if st.session_state.all_results:
        st.subheader("Confronto combinazioni caricate")
        fig_cmp = plot_results_comparison(st.session_state.all_results)
        st.plotly_chart(fig_cmp, use_container_width=False)
else:
    # --- Statistics panel ---
    st.markdown("---")
    st.subheader("📊 Statistics Panel")
    c1, c2, c3, c4, c5 = st.columns(5)
    ba_val = result.get("ba", 0)
    ci_lo = result.get("ci_lo")
    ci_hi = result.get("ci_hi")
    p_perm = result.get("p_perm")
    n_perm = result.get("n_perm", 0)
    n_feat = result.get("n_features", "?")
    wall = result.get("wall_clock_s")

    c1.metric("Balanced Accuracy", f"{ba_val:.3f}", help="Mean LOSO BA across 50 folds")
    c2.metric(
        "CI 95% (BCa)",
        f"[{ci_lo:.3f}, {ci_hi:.3f}]" if ci_lo is not None else "N/A",
        help="Bootstrap BCa 2000 resamples su ba_per_fold",
    )
    c3.metric(
        "p_perm",
        format_pvalue(p_perm, n_perm) if p_perm is not None else "skip",
        help=f"Permutation test (n_perm={n_perm})",
    )
    c4.metric("n_features", str(n_feat), help="Numero feature FC (upper-triangle ROI pairs)")
    c5.metric(
        "Wall-clock", f"{wall:.1f}s" if wall else "N/A", help="Tempo esecuzione combo completo"
    )

    st.caption(
        f"**Combo**: `{result.get('atlas', '?')}` × `{result.get('metric', '?')}` × "
        f"`{result.get('band', '?')}` × `{result.get('clf', '?')}` | "
        f"N=50 soggetti, LOSO CV (50 fold) | "
        f"Significatività Bonferroni-corrected: α=0.05/N_combos"
    )
    if ba_val >= 0.90:
        st.success(
            "✅ **Positive-control PASS** — BA ≥ 0.90: PLV-theta phase synchronization "
            "distinguishes EC from EO at source level (effetto Berger atteso — "
            "valida la pipeline end-to-end, non è il risultato scientifico principale)."
        )
    elif ba_val >= 0.75:
        st.warning("⚠ Moderate discriminability — BA ≥ 0.75")
    else:
        st.error("❌ Low discriminability — BA < 0.75 (near chance level)")

    st.markdown("---")

    # --- Plots tabs ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "🔲 Confusion Matrix",
            "📈 ROC Curve",
            "📉 Learning Curve",
            "🎲 Permutation Null",
            "🏷 Feature Importance",
            "📊 Comparison",
        ]
    )

    with tab1:
        cm = result.get("confusion")
        if cm:
            fig_cm = plot_confusion_matrix(cm)
            st.plotly_chart(fig_cm, use_container_width=False)
            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            st.caption(
                f"Sensitivity (EC recall): **{sens:.3f}** | Specificity (EO recall): **{spec:.3f}** | "
                f"N={sum(cm[0]) + sum(cm[1])} trial totali (2 per soggetto: EO+EC)"
            )
        else:
            st.info("Confusion matrix non disponibile per questa combo.")

    with tab2:
        fpr = result.get("roc_fpr", [])
        tpr = result.get("roc_tpr", [])
        auc_val = result.get("roc_auc")
        if fpr and auc_val is not None:
            fig_roc = plot_roc_curve(fpr, tpr, auc_val)
            st.plotly_chart(fig_roc, use_container_width=False)
            st.caption(
                f"AUC = **{auc_val:.3f}** | Curva aggregata su tutti i fold LOSO. "
                "AUC=1.0 indica separabilità perfetta; AUC=0.5 = chance level."
            )
        else:
            st.info("ROC non disponibile. Esegui 'Run combo' per calcolarla.")

    with tab3:
        ba_folds = result.get("ba_per_fold", [])
        if ba_folds:
            fig_lc = plot_learning_curve(ba_folds)
            st.plotly_chart(fig_lc, use_container_width=False)
            st.caption(
                f"{len(ba_folds)} fold LOSO. BA std={float(np.std(ba_folds)):.3f}. "
                "Variabilità alta tra fold indica soggetti outlier o eterogeneità del campione."
            )
        else:
            st.info("Dati per fold non disponibili.")

    with tab4:
        null_dist = result.get("null_distribution", [])
        if null_dist:
            fig_perm = plot_permutation_null(
                null_dist,
                ba_val,
                p_perm=result.get("p_perm"),
                n_perm=len(null_dist),
            )
            st.plotly_chart(fig_perm, use_container_width=False)
            st.caption(
                "Distribuzione nulla generata permutando le etichette y mantenendo la struttura LOSO. "
                "p_perm = #{null ≥ obs} / (N_perm+1). Risoluzione: 1/1001 con 1000 permutazioni."
            )
        else:
            perm_msg = "Permutation test non eseguito (n_perm=0)."
            if result.get("source") == "PERM_1000_WINNER_LOG.md":
                perm_msg += (
                    " Null distribution approssimata da Gaussiana (μ=0.493, σ=0.062) per display."
                )
            st.info(perm_msg)

    with tab5:
        # Feature importance via logistic regression coefficients
        if result.get("clf") in ("logreg", None) or result.get("algorithm") == "logreg":
            try:
                from sklearn.linear_model import LogisticRegression
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import StandardScaler

                X_feat, y_feat, g_feat, roi_names = _load_for_importance(
                    result, dataset_id=sel_dataset
                )
                if X_feat is not None:
                    pipe = Pipeline(
                        [
                            ("sc", StandardScaler()),
                            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
                        ]
                    )
                    pipe.fit(X_feat, y_feat)
                    coef = pipe.named_steps["clf"].coef_[0]
                    fig_fi = plot_feature_importance(coef, roi_names, n_top=20)
                    st.plotly_chart(fig_fi, use_container_width=False)
                    st.caption(
                        "Coefficienti logreg (fit su intero dataset). Positivi → favoriscono EC; "
                        "Negativi → favoriscono EO. Le connessioni theta PLV con coefficiente alto "
                        "identificano network che si sincronizzano durante il riposo ad occhi chiusi."
                    )
                else:
                    st.info("Feature file non caricabile per questa combo.")
            except Exception as e:
                st.warning(f"Feature importance non disponibile: {e}")
        else:
            st.info("Feature importance tramite coefficienti disponibile solo per logreg.")

    with tab6:
        all_res = st.session_state.all_results
        if all_res:
            fig_cmp = plot_results_comparison(all_res)
            st.plotly_chart(fig_cmp, use_container_width=True)
            st.caption(f"{len(all_res)} combo caricate. Rosso = massima BA.")
        else:
            st.info("Nessun risultato caricato. Usa 'Load reports/' o 'Load N=50 winner'.")

    # --- Export section ---
    st.markdown("---")
    st.subheader("📤 Export figure pub-quality")
    exp_col1, exp_col2, exp_col3 = st.columns(3)
    with exp_col1:
        exp_fmt = st.selectbox("Formato", ["png", "svg", "pdf"], index=0)
    with exp_col2:
        exp_dir = st.text_input("Output dir", value="reports/figures/export/")
    with exp_col3:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        export_btn = st.button("💾 Esporta tutte", use_container_width=True)

    if export_btn:
        out_dir = _ROOT / exp_dir
        with st.spinner(f"Esportazione in {out_dir} ({exp_fmt})..."):
            paths = export_all(result, out_dir, fmt=exp_fmt)
        if paths:
            st.success(f"✅ {len(paths)} figure esportate in `{out_dir}`")
            for p in paths:
                st.code(str(p))
        else:
            st.warning("Nessuna figura esportata (dati mancanti).")

    # Quick download: ROC PNG
    if result.get("roc_fpr") and result.get("roc_auc") is not None:
        import matplotlib.pyplot as plt

        _fig_tmp, _ax_tmp = plt.subplots(figsize=(4, 3.5))
        _ax_tmp.plot(result["roc_fpr"], result["roc_tpr"])
        _ax_tmp.plot([0, 1], [0, 1], "k--")
        _ax_tmp.set_title(f"ROC AUC={result.get('roc_auc', 0):.3f}")
        _buf = fig_to_bytes(_fig_tmp, fmt="png")
        plt.close(_fig_tmp)
        st.download_button(
            "⬇ Download ROC (PNG)",
            data=_buf,
            file_name=f"roc_{result.get('atlas', '?')}_{result.get('metric', '?')}_{result.get('band', '?')}.png",
            mime="image/png",
        )
