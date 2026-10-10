"""
SOC-IA | Dashboard Principal
============================
Affiche uniquement LightGBM + Autoencoder
"""

import streamlit as st
import pandas as pd
from datetime import datetime

from utils.api_client import client
from utils.styles import (
    CUSTOM_CSS, get_alert_color)

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="SOC-IA Dashboard",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS,
            unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        "<div class='sidebar-title'>"
        "🛡 SOC-IA</div>",
        unsafe_allow_html=True)

    st.markdown("---")

    health = client.health()

    if health.get("status") == "ok":
        st.success("✅ API connectée")
        st.caption(
            f"Version : "
            f"{health.get('version', 'N/A')}")
        st.caption(
            f"Uptime : "
            f"{health.get('uptime_seconds', 0):.0f}s")
    else:
        st.error("❌ API déconnectée")
        st.warning(
            "Lancez l'API :\n"
            "`python -m uvicorn "
            "app.main:app --reload`")

    st.markdown("---")

    st.markdown("### 📍 Navigation")
    st.info(
        "Utilisez le menu de gauche pour "
        "naviguer entre les pages.")

    st.markdown("---")

    st.markdown("### ℹ️ À propos")
    st.caption(
        "Système de détection automatique "
        "de cyberattaques basé sur "
        "2 modèles IA complémentaires.")
    st.caption(
        f"📅 {datetime.now().strftime('%Y-%m-%d')}")

# ============================================================
# CONTENU PRINCIPAL
# ============================================================

st.markdown(
    "<div class='main-title'>"
    "🛡 SOC-IA Dashboard</div>",
    unsafe_allow_html=True)

st.markdown(
    "<div class='subtitle'>"
    "Détection Automatique de "
    "Cyberattaques par Intelligence "
    "Artificielle</div>",
    unsafe_allow_html=True)

if health.get("status") != "ok":
    st.error(
        "⚠️ L'API n'est pas accessible.")
    st.code(
        "python -m uvicorn "
        "app.main:app --reload")
    st.stop()

# ============================================================
# SECTION 1 : KPI PRINCIPAUX
# ============================================================

st.markdown("## 📊 Vue d'Ensemble")

stats = client.get_stats()
models = client.get_models_info()
classes = client.get_classes()
features = client.get_features()

# ✅ Modifié : 2 modèles au lieu de 3
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class='kpi-card'>
        <p class='kpi-label'>
            🤖 Modèles Actifs
        </p>
        <p class='kpi-value'>
            2
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='kpi-card'>
        <p class='kpi-label'>
            🎯 Classes Détectables
        </p>
        <p class='kpi-value'>
            {classes.get('total_classes', 9)}
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='kpi-card'>
        <p class='kpi-label'>
            🔬 Features
        </p>
        <p class='kpi-value'>
            {features.get('total_features', 24)}
        </p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class='kpi-card'>
        <p class='kpi-label'>
            📈 Prédictions Faites
        </p>
        <p class='kpi-value'>
            {stats.get('total_predictions', 0)}
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# SECTION 2 : MODÈLES (LightGBM + Autoencoder)
# ============================================================

st.markdown("## 🤖 Modèles IA Déployés")

if "models" in models:
    # ✅ Filtrer : enlever RandomForest
    models_filtered = [
        m for m in models["models"]
        if m["name"] != "randomforest"
    ]

    cols = st.columns(2)

    for i, model in enumerate(
            models_filtered[:2]):
        with cols[i]:
            with st.container(border=True):
                medal = ["🥇", "🥈"][i]
                st.markdown(
                    f"### {medal} "
                    f"{model['name'].title()}")

                st.caption(
                    f"📋 {model['role']}")

                perf = model.get(
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

                size = model.get(
                    "file_size_mb", 0)
                st.caption(
                    f"💾 Taille : {size} MB")

st.markdown("---")

# ============================================================
# SECTION 3 : ARCHITECTURE
# ============================================================

st.markdown("## 🏗 Architecture du Système")

with st.container(border=True):
    st.markdown("""
    ```
    Flux Réseau Entrant
            │
            ▼
    ┌──────────────────────┐
    │   Prétraitement      │
    │   (Scalers + Norm)   │
    └──────────┬───────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
    LightGBM      Autoencoder
    (99.78%)      (Zero-Day)
        │             │
        └──────┬──────┘
               ▼
    ┌──────────────────────┐
    │  Moteur de Décision  │
    │  + Système d'Alerte  │
    └──────────┬───────────┘
               ▼
        Notifications SOC
    ```
    """)

st.markdown("---")

# ============================================================
# SECTION 4 : GUIDE
# ============================================================

st.markdown("## 🚀 Comment Utiliser")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown("""
        ### 🎯 Prédiction Manuelle

        1. Allez sur la page **Predict**
        2. Entrez 24 valeurs de features
        3. Cliquez sur **Analyser**
        4. Obtenez la classe détectée
        """)

with col2:
    with st.container(border=True):
        st.markdown("""
        ### 📁 Analyse en Masse

        1. Allez sur la page **Analyze CSV**
        2. Uploadez un fichier CSV
        3. Cliquez sur **Analyser Tout**
        4. Visualisez les résultats
        """)

st.markdown("---")

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div style='text-align: center;
            color: #546E7A;
            font-size: 0.85rem;
            padding: 1rem;'>
    SOC-IA v1.0.0 | Dataset:
    CIC-IDS2017 | 
</div>
""", unsafe_allow_html=True)