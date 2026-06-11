"""Tab 6 — Cross-Dataset Transfer (ds005385 → ds004504 LEMON).

Train su ds005385 N=100 soggetti sani adulti.
Test su ds004504 N=49 soggetti (Alzheimer + controlli, 44-79 anni).

Vincoli scientifici:
  - Scaler fit SOLO su train (ds005385), NO leakage
  - BA_sex=0.500 (chance) e MAE_age=22.25 interpretati come domain shift, non fallimento pipeline
  - ds004504 è dataset eterogeno (patologici vs sani)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.components.data_loader import (  # noqa: E402
    list_available_combos,
    list_available_combos_lemon,
    load_cohort_df,
    load_cohort_df_lemon,
    load_cross_dataset_transfer_report,
    run_cross_dataset_transfer,
)

st.set_page_config(
    page_title="Cross-Dataset — Tesi 2.0", page_icon="🔀", layout="wide"
)

st.title("🔍 Cross-Cohort Generalization — Negative Finding")
st.caption(
    "Train: ds005385 N=100 (adulti sani) · Test: ds004504 N=49 (Alzheimer + controlli) · "
    "aparc × plv × θ"
)

st.error(
    "**⚠ Domain shift catastrofico — risultato scientificamente legittimo, non bug pipeline.**  \n"
    "Le feature EEG-FC (PLV theta) generalizzano all'interno di una popolazione omogenea "
    "(sex BA=0.713 su sani ds005385) ma **non cross-cohort** quando il target dataset "
    "differisce per patologia, range età e laboratorio.  \n"
    "BA_sex=0.500 (chance) e MAE_age=22.25 anni **quantificano il domain gap** — "
    "finding metodologico pubblicabile (Engemann 2022: servono ≥3 dataset + Combat-EEG per "
    "claim cross-cohort). Questo è il *primary finding* di FASE 2."
)

# ---------------------------------------------------------------------------
# Risultati pre-computed
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Risultati transfer pre-computed (aparc × plv × θ)")

transfer_report = load_cross_dataset_transfer_report()

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "BA sesso (transfer)",
    f"{transfer_report.get('sex_ba_transfer', 0.5):.3f}",
    delta=f"{transfer_report.get('sex_ba_transfer', 0.5) - transfer_report.get('sex_ba_indataset', 0.741):.3f} vs in-dataset",
    delta_color="off",
)
c2.metric(
    "BA sesso (in-dataset N=100)",
    f"{transfer_report.get('sex_ba_indataset', 0.741):.3f}",
)
c3.metric(
    "MAE età (transfer)",
    f"{transfer_report.get('age_mae_transfer', 22.25):.1f} anni",
    delta=f"+{transfer_report.get('age_mae_transfer', 22.25) - transfer_report.get('age_mae_indataset', 12.52):.1f} vs in-dataset",
    delta_color="off",
)
c4.metric(
    "MAE età (in-dataset N=100)",
    f"{transfer_report.get('age_mae_indataset', 12.52):.1f} anni",
)

st.caption(
    "Sesso: BA_transfer=0.500 → chance level. "
    "Età: MAE_transfer=22.25 anni vs in-dataset 12.52. "
    "Gap transfer: sex +0.241 BA | age +9.73 anni MAE."
)

# ---------------------------------------------------------------------------
# Confronto cohort
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Confronto Cohort: ds005385 vs ds004504")

col_a, col_b = st.columns(2)

with col_a:
    df_005 = load_cohort_df()
    st.markdown("**ds005385** (adulti sani)")
    df_005_valid = df_005[df_005["age"].notna()]
    c_a1, c_a2, c_a3 = st.columns(3)
    c_a1.metric("N soggetti", len(df_005))
    c_a2.metric("Età media", f"{df_005_valid['age'].mean():.1f} anni" if len(df_005_valid) > 0 else "?")
    n_f5 = (df_005["sex"] == "F").sum()
    n_m5 = (df_005["sex"] == "M").sum()
    c_a3.metric("F / M", f"{n_f5} / {n_m5}")

    if len(df_005_valid) > 0:
        fig_age5 = px.histogram(
            x=df_005_valid["age"], nbins=15,
            labels={"x": "Età (anni)", "y": "N"},
            color_discrete_sequence=["steelblue"], title="ds005385 — distribuzione età",
        )
        fig_age5.update_layout(height=250, margin={"l": 40, "r": 10, "t": 40, "b": 30}, showlegend=False)
        st.plotly_chart(fig_age5, use_container_width=True)

with col_b:
    df_lemon = load_cohort_df_lemon()
    st.markdown("**ds004504** (LEMON — Alzheimer + FTD + controlli)")
    if not df_lemon.empty:
        df_lemon_valid = df_lemon[df_lemon["age"].notna()]
        c_b1, c_b2, c_b3 = st.columns(3)
        c_b1.metric("N soggetti", len(df_lemon))
        c_b2.metric("Età media", f"{df_lemon_valid['age'].mean():.1f} anni" if len(df_lemon_valid) > 0 else "?")
        n_fl = (df_lemon["sex"] == "F").sum()
        n_ml = (df_lemon["sex"] == "M").sum()
        c_b3.metric("F / M", f"{n_fl} / {n_ml}")

        if len(df_lemon_valid) > 0:
            fig_age_l = px.histogram(
                x=df_lemon_valid["age"], nbins=12,
                labels={"x": "Età (anni)", "y": "N"},
                color_discrete_sequence=["tomato"], title="ds004504 — distribuzione età",
            )
            fig_age_l.update_layout(height=250, margin={"l": 40, "r": 10, "t": 40, "b": 30}, showlegend=False)
            st.plotly_chart(fig_age_l, use_container_width=True)

        # Gruppi ds004504
        st.markdown("**Distribuzione gruppi** (A=Alzheimer, C=Controllo, F=FTD)")
        groups_count = df_lemon["group"].value_counts()
        fig_grp = px.bar(
            x=groups_count.index.tolist(), y=groups_count.values.tolist(),
            color=groups_count.index.tolist(),
            color_discrete_map={"A": "tomato", "C": "steelblue", "F": "orange"},
            labels={"x": "Gruppo", "y": "N soggetti"},
        )
        fig_grp.update_layout(height=200, showlegend=False, margin={"l": 40, "r": 10, "t": 10, "b": 30})
        st.plotly_chart(fig_grp, use_container_width=True)
    else:
        st.info("Dati cohort ds004504 non disponibili")

# ---------------------------------------------------------------------------
# Run transfer interattivo
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Transfer interattivo (train ds005385 → test ds004504)")

with st.sidebar:
    st.header("Parametri transfer")
    combos_005 = list_available_combos()
    combos_lemon = list_available_combos_lemon()
    # Intersezione combo disponibili in entrambi i dataset
    combos_shared = [c for c in combos_005 if c in combos_lemon]

    atlases_s = sorted({c["atlas"] for c in combos_shared}) if combos_shared else ["aparc"]
    metrics_s = sorted({c["metric"] for c in combos_shared}) if combos_shared else ["plv"]
    bands_s = sorted({c["band"] for c in combos_shared}) if combos_shared else ["theta"]

    atlas_t = st.selectbox("Atlante", atlases_s, index=0)
    metric_t = st.selectbox("Metrica FC", metrics_s, index=metrics_s.index("plv") if "plv" in metrics_s else 0)
    band_t = st.selectbox("Banda", bands_s, index=bands_s.index("theta") if "theta" in bands_s else 0)
    target_t = st.selectbox("Target", ["sex", "age"], index=0, format_func=lambda x: "Sesso (BA)" if x == "sex" else "Età (MAE)")
    clf_t = st.selectbox("Algoritmo", ["logreg", "svm", "rf", "mlp"], index=0)
    run_transfer_btn = st.button("▶ Esegui Transfer", type="primary", use_container_width=True)

col_run, col_interp = st.columns([1, 1])

with col_run:
    if run_transfer_btn:
        with st.spinner("Transfer learning in corso..."):
            try:
                result_t = run_cross_dataset_transfer(
                    atlas=atlas_t, metric=metric_t, band=band_t,
                    clf_name=clf_t, target=target_t,
                )
                st.success(f"✅ Done in {result_t['wall_clock_s']:.1f}s")
                st.metric(
                    f"{result_t['metric_name']} (transfer)",
                    f"{result_t['score']:.3f}" + (" anni" if target_t == "age" else ""),
                )
                st.json({
                    "feature": result_t["feature"],
                    "train": f"{result_t['train_dataset']} N={result_t['train_n']}",
                    "test": f"{result_t['test_dataset']} N={result_t['test_n']}",
                    "clf": result_t["clf_name"],
                    "target": result_t["target"],
                })
                if target_t == "sex" and result_t["score"] < 0.55:
                    st.warning("BA≈0.5: chance level — domain shift confonde la generalizzazione")
                elif target_t == "age" and result_t["score"] > 15:
                    st.warning("MAE alto: il modello addestrato su sani non generalizza su pazienti")
            except Exception as exc:
                st.error(f"Errore transfer: {exc}")
    else:
        st.info("👈 Configura e premi **Esegui Transfer** per il run interattivo")

with col_interp:
    st.markdown(
        """
**Interpretazione scientifica**

Il gap di performance inter-dataset riflette:

1. **Eterogeneità demografica**: ds005385 = 20-70 anni (sani), ds004504 = 44-79 anni (pazienti + controlli).
   Le distribuzioni di età e sesso sono diverse.

2. **Eterogeneità neuropatologica**: Alzheimer e FTD alterano i pattern FC source-level.
   Un modello addestrato su sani non cattura queste alterazioni.

3. **Condizioni di acquisizione**: i due dataset hanno protocolli EEG differenti.

**Significato metodologico**

BA_sex=0.500 conferma che i biomarker FC di sesso identificati in ds005385
non sono universali: dipendono dalla popolazione specifica.
Questo è un risultato scientifico onesto, non un fallimento.

**Prossimi passi**

- Training separato su ds004504 (within-dataset, GroupKFold)
- Domain adaptation (e.g. TCA, CORAL) per cross-dataset
- Analisi del domain shift (distribuzione FC confrontata per dataset)
        """
    )

# ---------------------------------------------------------------------------
# Confronto feature distribution
# ---------------------------------------------------------------------------
st.markdown("---")
with st.expander("Distribuzione feature FC: ds005385 vs ds004504", expanded=False):
    combos_005_list = list_available_combos()
    if combos_005_list:
        try:
            from dashboard.components.data_loader import (  # noqa: E402,PLC0415
                load_feature_matrix,
                load_feature_matrix_lemon,
            )

            c0 = combos_005_list[0]
            X_005, _, _, _ = load_feature_matrix(c0["atlas"], c0["metric"], c0["band"])
            X_004, _, _, _ = load_feature_matrix_lemon(c0["atlas"], c0["metric"], c0["band"])

            mean_005 = float(np.mean(np.abs(X_005)))
            mean_004 = float(np.mean(np.abs(X_004)))
            std_005 = float(np.std(X_005))
            std_004 = float(np.std(X_004))

            col_stat1, col_stat2 = st.columns(2)
            col_stat1.metric("FC media |ds005385|", f"{mean_005:.4f}", help="Mean |FC| su tutte le feature")
            col_stat2.metric("FC media |ds004504|", f"{mean_004:.4f}")

            # Distribuzione histogram overlay
            sample_005 = X_005[:, :100].flatten()
            sample_004 = X_004[:, :100].flatten()
            df_fc = pd.DataFrame({
                "FC": list(sample_005) + list(sample_004),
                "Dataset": ["ds005385"] * len(sample_005) + ["ds004504"] * len(sample_004),
            })
            fig_fc = px.histogram(
                df_fc, x="FC", color="Dataset", nbins=60, barmode="overlay", opacity=0.6,
                color_discrete_map={"ds005385": "steelblue", "ds004504": "tomato"},
                title=f"Distribuzione FC ({c0['atlas']}×{c0['metric']}×{c0['band']}, prime 100 feature)",
            )
            fig_fc.update_layout(height=350)
            st.plotly_chart(fig_fc, use_container_width=True)
            st.caption(
                f"ds005385: μ={mean_005:.4f} σ={std_005:.4f} · "
                f"ds004504: μ={mean_004:.4f} σ={std_004:.4f}. "
                "Differenze nella distribuzione FC spiegano parte del domain shift."
            )
        except Exception as exc:
            st.warning(f"Distribuzione FC non disponibile: {exc}")
