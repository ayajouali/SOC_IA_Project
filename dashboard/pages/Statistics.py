"""
SOC-IA | Statistiques
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.api_client import client
from utils.styles import CUSTOM_CSS

st.set_page_config(
    page_title="Statistiques - SOC-IA",
    page_icon="📊",
    layout="wide",
)

st.markdown(CUSTOM_CSS,
            unsafe_allow_html=True)

st.markdown(
    "<div class='main-title'>"
    "📊 Statistiques</div>",
    unsafe_allow_html=True)

st.markdown(
    "<div class='subtitle'>"
    "Activité et performances "
    "de l'API en temps réel</div>",
    unsafe_allow_html=True)

# Bouton refresh
if st.button("🔄 Rafraîchir"):
    st.rerun()

# Récupérer stats
stats = client.get_stats()

if "error" in stats:
    st.error(f"❌ {stats['error']}")
    st.stop()

# ============================================================
# KPI PRINCIPAUX
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "📈 Total Prédictions",
        stats.get(
            "total_predictions", 0))

with col2:
    avg_conf = stats.get(
        "avg_confidence", 0) * 100
    st.metric(
        "🎯 Confiance Moyenne",
        f"{avg_conf:.1f}%")

with col3:
    st.metric(
        "⚠️ Anomalies (Zero-Day)",
        stats.get(
            "anomalies_detected", 0))

with col4:
    st.metric(
        "🔍 Désaccords Modèles",
        stats.get("disagreements", 0))

st.markdown("---")

# ============================================================
# DISTRIBUTION PAR ALERTE
# ============================================================

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        "### 🎯 Distribution par Alerte")

    per_alert = stats.get(
        "predictions_per_alert", {})

    if per_alert and sum(
            per_alert.values()) > 0:
        df_alert = pd.DataFrame(
            list(per_alert.items()),
            columns=['Alerte', 'Nombre'])

        colors_map = {
            "CRITICAL": "#C62828",
            "HIGH": "#F57C00",
            "MEDIUM": "#FFA000",
            "INFO": "#1976D2",
            "REVIEW": "#7B1FA2",
        }

        fig = px.pie(
            df_alert,
            values='Nombre',
            names='Alerte',
            color='Alerte',
            color_discrete_map=colors_map,
            hole=0.4)
        fig.update_layout(height=400)
        st.plotly_chart(
            fig,
            use_container_width=True)
    else:
        st.info(
            "🔄 Aucune prédiction "
            "encore effectuée")

# ============================================================
# DISTRIBUTION PAR CLASSE
# ============================================================

with col2:
    st.markdown(
        "### 🎯 Distribution par Classe")

    per_class = stats.get(
        "predictions_per_class", {})
    per_class = {
        k: v for k, v in per_class.items()
        if v > 0
    }

    if per_class:
        df_class = pd.DataFrame(
            list(per_class.items()),
            columns=['Classe', 'Nombre'])
        df_class = df_class.sort_values(
            'Nombre', ascending=False)

        fig = px.bar(
            df_class,
            x='Classe',
            y='Nombre',
            color='Nombre',
            color_continuous_scale='Blues')
        fig.update_layout(
            height=400,
            showlegend=False)
        st.plotly_chart(
            fig,
            use_container_width=True)
    else:
        st.info(
            "🔄 Aucune prédiction "
            "encore effectuée")

st.markdown("---")

# ============================================================
# UPTIME
# ============================================================

uptime = stats.get(
    "uptime_seconds", 0)
hours = int(uptime // 3600)
minutes = int((uptime % 3600) // 60)
seconds = int(uptime % 60)

st.markdown(
    f"⏱ **API en marche depuis :** "
    f"{hours}h {minutes}m {seconds}s")