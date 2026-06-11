"""Tab 1 — Dataset & Cohort.

Selettori N-soggetti, filtri età/sesso, preview tabellare.
Dati: data/raw/ds005385/participants.tsv + data/features/ds005385/metadata.json
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

from dashboard.components.data_loader import load_cohort_df, load_metadata  # noqa: E402

st.set_page_config(page_title="Dataset & Cohort — Tesi 2.0", page_icon="🎛️", layout="wide")

st.title("🎛️ Dataset & Cohort — ds005385")
st.caption(
    "Dortmund Vital EEG · N=100 soggetti whitelist · condizioni: EO (Eyes Open) / EC (Eyes Closed)"
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Filtri cohort")
    n_sub = st.slider("N soggetti", min_value=5, max_value=100, value=100, step=5)
    st.divider()
    sex_filter = st.multiselect("Sesso", ["F", "M", "?"], default=["F", "M", "?"])
    age_min, age_max = st.slider("Range età", 18, 90, (18, 90))
    st.divider()
    show_full = st.checkbox("Mostra tabella completa", value=False)

# ---------------------------------------------------------------------------
# Carica dati
# ---------------------------------------------------------------------------
meta = load_metadata()

with st.spinner("Carico metadata cohort..."):
    df_full = load_cohort_df()

df = df_full.copy()
if n_sub < len(df):
    df = df.iloc[:n_sub]

# Applica filtri
if sex_filter:
    df = df[df["sex"].isin(sex_filter)]
df = df[
    (df["age"].isna()) | ((df["age"] >= age_min) & (df["age"] <= age_max))
]

# ---------------------------------------------------------------------------
# Metriche top
# ---------------------------------------------------------------------------
st.markdown("---")
c1, c2, c3, c4, c5 = st.columns(5)
n_valid_age = int(df["age"].notna().sum())
n_female = int((df["sex"] == "F").sum())
n_male = int((df["sex"] == "M").sum())
mean_age = float(df["age"].mean()) if n_valid_age > 0 else float("nan")
n_conn = int(df["has_EC"].sum())

c1.metric("N soggetti filtrati", len(df))
c2.metric("Età media", f"{mean_age:.1f}" if not np.isnan(mean_age) else "N/A")
c3.metric("Femmine / Maschi", f"{n_female} / {n_male}")
c4.metric("Con dati connectivity", n_conn)
c5.metric("Combos feature", meta.get("n_combos", "?"))

st.markdown("---")

# ---------------------------------------------------------------------------
# Visualizzazioni
# ---------------------------------------------------------------------------
col_age, col_sex = st.columns(2)

with col_age:
    st.subheader("Distribuzione età")
    age_data = df["age"].dropna()
    if len(age_data) > 0:
        fig_age = px.histogram(
            x=age_data,
            nbins=20,
            labels={"x": "Età (anni)", "y": "N soggetti"},
            color_discrete_sequence=["steelblue"],
        )
        fig_age.update_layout(
            height=300,
            margin={"l": 40, "r": 10, "t": 10, "b": 40},
            showlegend=False,
        )
        st.plotly_chart(fig_age, use_container_width=True)
    else:
        st.info("Dati età non disponibili (TSV non trovato o filtri troppo restrittivi)")

with col_sex:
    st.subheader("Distribuzione sesso")
    sex_counts = df["sex"].value_counts()
    if len(sex_counts) > 0:
        fig_sex = px.bar(
            x=sex_counts.index.tolist(),
            y=sex_counts.values.tolist(),
            labels={"x": "Sesso", "y": "N soggetti"},
            color=sex_counts.index.tolist(),
            color_discrete_map={"F": "#e377c2", "M": "#1f77b4", "?": "#aaa"},
        )
        fig_sex.update_layout(
            height=300,
            margin={"l": 40, "r": 10, "t": 10, "b": 40},
            showlegend=False,
        )
        st.plotly_chart(fig_sex, use_container_width=True)
    else:
        st.info("Dati sesso non disponibili")

# ---------------------------------------------------------------------------
# Preview tabellare
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Preview soggetti")

display_cols = ["participant_id", "age", "sex", "handedness", "has_EO", "has_EC"]
df_display = df[display_cols].copy()
df_display["age"] = df_display["age"].apply(lambda x: f"{x:.0f}" if not np.isnan(x) else "?")

if show_full:
    st.dataframe(df_display, use_container_width=True, height=600)
else:
    st.dataframe(df_display.head(20), use_container_width=True)
    if len(df_display) > 20:
        st.caption(f"Mostrando 20 / {len(df_display)} soggetti. Abilita 'tabella completa' in sidebar.")

# ---------------------------------------------------------------------------
# Info dataset
# ---------------------------------------------------------------------------
st.markdown("---")
with st.expander("Info dataset ds005385", expanded=False):
    st.markdown(
        """
**ds005385 — Dortmund Vital EEG**
- Soggetti: 608 totali, whitelist N=100 processati
- Condizioni: **EO** (Eyes Open, 1 min) / **EC** (Eyes Closed, 1 min)
- EO/EC usato come **positive-control** (effetto Berger, 1929) — non è il risultato scientifico
- Target scientifici: **età** e **sesso** (FASE 1+)
- Pipeline: MNE-BIDS → dSPM source reconstruction → parcellazione aparc/schaefer100
- Metriche FC: wpli, coh, plv, imcoh · Bande: θ α β γ
        """
    )

    if meta:
        st.json(
            {
                "atlases": meta.get("atlases", []),
                "metrics": meta.get("metrics", []),
                "bands": meta.get("bands", []),
                "band_hz": meta.get("band_hz", {}),
                "n_subjects": meta.get("n_subjects"),
                "n_samples": meta.get("n_samples"),
                "n_combos": meta.get("n_combos"),
            }
        )

st.caption("ds004504 (LEMON) disponibile in **Tab 6 Cross-Dataset** · BA_sex=0.500 domain shift sano→clinico")
