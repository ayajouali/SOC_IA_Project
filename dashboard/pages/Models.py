"""
SOC-IA | Informations Modèles
Affiche uniquement LightGBM + Autoencoder
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.api_client import client
from utils.styles import CUSTOM_CSS

st.set_page_config(
    page_title="Modèles - SOC-IA",
    page_icon="🤖",
    layout="wide",
)

st.markdown(CUSTOM_CSS,
            unsafe_allow_html=True)

st.markdown(
    "<div class='main-title'>"
    "🤖 Modèles IA</div>",
    unsafe_allow_html=True)

st.markdown(
    "<div class='subtitle'>"
    "Détails techniques des modèles "
    "déployés</div>",
    unsafe_allow_html=True)

# ============================================================
# MODÈLES (Filtrer RandomForest)
# ============================================================

st.markdown("## 🥇 Modèles Déployés")

models = client.get_models_info()

if "models" in models:
    # ✅ Filtrer
    models_visible = [
        m for m in models["models"]
        if m["name"] != "randomforest"
    ]

    for i, m in enumerate(models_visible):
        medal = ["🥇", "🥈"][i]

        with st.container(border=True):
            col1, col2, col3 = st.columns(
                [2, 2, 1])

            with col1:
                st.markdown(
                    f"### {medal} "
                    f"{m['name'].title()}")
                st.caption(
                    f"📋 {m['role']}")
                st.caption(
                    f"🔧 Type : "
                    f"{m.get('type', 'N/A')}")

            with col2:
                perf = m.get(
                    "performance", {})
                if "f1_weighted" in perf:
                    st.metric(
                        "F1-Weighted",
                        f"{perf['f1_weighted']*100:.2f}%")
                if "accuracy" in perf:
                    st.metric(
                        "Accuracy",
                        f"{perf['accuracy']*100:.2f}%")
                if "f1_score" in perf:
                    st.metric(
                        "F1-Score",
                        f"{perf['f1_score']*100:.2f}%")

            with col3:
                st.metric(
                    "Taille",
                    f"{m.get('file_size_mb', 0)} MB")

st.markdown("---")

# ============================================================
# FEATURES
# ============================================================

st.markdown(
    "## 🔬 Top 10 Features Importantes")

features = client.get_features()

if "features" in features:
    df_f = pd.DataFrame(
        features["features"][:10])

    fig = px.bar(
        df_f,
        x='importance_pct',
        y='name',
        orientation='h',
        color='importance_pct',
        color_continuous_scale='Blues',
        labels={
            'importance_pct':
                'Importance (%)',
            'name': 'Feature'})
    fig.update_layout(
        height=500,
        showlegend=False,
        yaxis={
            'categoryorder': 'total ascending'})
    st.plotly_chart(
        fig, use_container_width=True)

st.markdown("---")

# ============================================================
# CLASSES
# ============================================================

st.markdown(
    "## 🎯 Classes Détectables")

classes = client.get_classes()

if "classes" in classes:
    cols = st.columns(3)
    for i, c in enumerate(
            classes["classes"]):
        col = cols[i % 3]
        with col:
            level_colors = {
                "CRITICAL": "#C62828",
                "HIGH": "#F57C00",
                "MEDIUM": "#FFA000",
                "INFO": "#1976D2",
                "LOW": "#1976D2",
            }
            color = level_colors.get(
                c["alert_level"],
                "#546E7A")

            st.markdown(f"""
            <div style='background:white;
                    padding:1rem;
                    border-radius:8px;
                    border-left:4px solid
                    {color};
                    margin-bottom:0.5rem;
                    box-shadow:
                    0 2px 4px
                    rgba(0,0,0,0.05);'>
                <p style='font-weight:bold;
                    margin:0;
                    color:#1A237E;'>
                    {c['name']}
                </p>
                <p style='color:{color};
                    font-size:0.85rem;
                    margin:0.3rem 0 0 0;
                    font-weight:bold;'>
                    {c['alert_level']}
                </p>
            </div>
            """, unsafe_allow_html=True)