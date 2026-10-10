"""
SOC-IA | Page AI Analyst (LLM)
"""

import streamlit as st
import numpy as np
import json

from utils.api_client import client
from utils.styles import CUSTOM_CSS

st.set_page_config(
    page_title="AI Analyst - SOC-IA",
    page_icon="🤖",
    layout="wide",
)

st.markdown(CUSTOM_CSS,
            unsafe_allow_html=True)

# ============================================================
# TITRE
# ============================================================

st.markdown(
    "<div class='main-title'>"
    "🤖 AI Analyst</div>",
    unsafe_allow_html=True)

st.markdown(
    "<div class='subtitle'>"
    "Analyse intelligente par "
    "Llama 3.3 (Groq)</div>",
    unsafe_allow_html=True)

# ============================================================
# VÉRIFIER STATUTS
# ============================================================

health = client.health()
llm_status = client.llm_status()

if health.get("status") != "ok":
    st.error("❌ API non disponible")
    st.code(
        "python -m uvicorn "
        "app.main:app --reload")
    st.stop()

if not llm_status.get(
        "available", False):
    st.error(
        "❌ LLM non disponible. "
        "Vérifiez GROQ_API_KEY "
        "dans le fichier .env")
    st.code(
        "GROQ_API_KEY="
        "gsk_votre_cle_ici\n"
        "GROQ_MODEL="
        "llama-3.3-70b-versatile")
    st.info(
        "📝 Inscrivez-vous gratuitement "
        "sur https://console.groq.com/")
    st.stop()

st.success(
    f"✅ LLM connecté : "
    f"`{llm_status.get('model', 'N/A')}` "
    f"via {llm_status.get('provider', 'N/A')}")

# ============================================================
# DEUX MODES (TABS)
# ============================================================

tab1, tab2 = st.tabs([
    "🔍 Analyse d'Attaque",
    "💬 Chat Expert"
])

# ============================================================
# TAB 1 : ANALYSE D'ATTAQUE
# ============================================================

with tab1:
    st.markdown(
        "### 🎯 Analyser un Flux Réseau")
    st.caption(
        "Le LLM va analyser la prédiction "
        "et fournir un rapport SOC complet")

    col1, col2 = st.columns([1, 1])

    with col1:
        mode = st.radio(
            "Mode :",
            ["🎲 Générer aléatoirement",
             "📝 Saisir manuellement"],
            key="llm_mode")

        features = []

        if mode == "🎲 Générer aléatoirement":
            if st.button(
                    "🎲 Générer",
                    type="primary",
                    key="gen_btn"):
                np.random.seed(
                    np.random.randint(
                        0, 9999))
                features = (
                    np.random.randn(24)
                    .tolist())
                st.session_state.llm_features = (
                    features)

            if "llm_features" in \
                    st.session_state:
                features = (
                    st.session_state
                    .llm_features)
                st.success(
                    "✅ Features générées")
        else:
            text = st.text_area(
                "24 valeurs séparées "
                "par virgules :",
                value=", ".join(
                    ["0.5"] * 24),
                height=100,
                key="manual_text")
            try:
                features = [
                    float(x.strip())
                    for x in text.split(",")]
                if len(features) != 24:
                    st.error(
                        f"⚠️ {len(features)}"
                        " valeurs au lieu "
                        "de 24")
                    features = []
            except Exception:
                st.error("Format invalide")
                features = []

    with col2:
        if features and \
                len(features) == 24:
            st.info(
                "📊 24 features prêtes")
            with st.expander(
                    "Voir les valeurs"):
                st.json(
                    {f"feature_{i+1}":
                     round(v, 4)
                     for i, v in
                     enumerate(features)})

    st.markdown("---")

    if features and len(features) == 24:
        if st.button(
                "🚀 ANALYSER AVEC LE LLM",
                type="primary",
                use_container_width=True,
                key="analyze_llm_btn"):

            with st.spinner(
                    "🧠 Le LLM analyse... "
                    "(15-30 secondes)"):
                result = client\
                    .explain_with_llm(
                    features)

            if "error" in result:
                st.error(
                    f"❌ Erreur : "
                    f"{result['error']}")
            else:
                # ============================
                # AFFICHER PRÉDICTION
                # ============================
                pred = result["prediction"]
                analysis = result[
                    "llm_analysis"]

                st.markdown(
                    "## 📊 Prédiction du Modèle")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "🎯 Classe",
                        pred["prediction"])
                with col2:
                    st.metric(
                        "📈 Confiance",
                        f"{pred['confidence']*100:.1f}%")
                with col3:
                    st.metric(
                        "⚠️ Alerte",
                        pred["alert_level"])

                st.markdown("---")

                # ============================
                # ANALYSE LLM
                # ============================
                if "error" in analysis:
                    st.error(
                        f"⚠️ Erreur LLM : "
                        f"{analysis['error']}")
                else:
                    st.markdown(
                        "## 🤖 Analyse par "
                        "Intelligence Artificielle")

                    # Résumé
                    if "summary" in analysis:
                        st.markdown(
                            "### 📝 Résumé")
                        st.info(
                            analysis["summary"])

                    # Sévérité + Risk Score
                    col1, col2 = st.columns(2)

                    with col1:
                        sev = analysis.get(
                            "severity", "N/A")
                        sev_colors = {
                            "CRITICAL": "🔴",
                            "HIGH": "🟠",
                            "MEDIUM": "🟡",
                            "LOW": "🟢"}
                        icon = sev_colors.get(
                            sev, "⚪")

                        with st.container(
                                border=True):
                            st.markdown(
                                f"### {icon} "
                                "Sévérité")
                            st.markdown(
                                f"# **{sev}**")
                            if "severity_reason" in \
                                    analysis:
                                st.caption(
                                    analysis[
                                        "severity_reason"])

                    with col2:
                        risk = analysis.get(
                            "risk_score", 0)
                        try:
                            risk = float(risk)
                        except:
                            risk = 0
                        with st.container(
                                border=True):
                            st.markdown(
                                "### 📊 Score "
                                "de Risque")
                            st.markdown(
                                f"# **{risk}/10**")
                            st.progress(
                                min(risk/10, 1.0))

                    # Détails attaque
                    st.markdown(
                        "### 🎯 Détails de "
                        "l'Attaque")

                    with st.container(
                            border=True):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown(
                                f"**Type :** "
                                f"`{analysis.get('attack_type', 'N/A')}`")
                            st.markdown(
                                f"**Vecteur :** "
                                f"{analysis.get('attack_vector', 'N/A')}")

                        with col2:
                            fp = analysis.get(
                                "false_positive_likelihood",
                                "N/A")
                            st.markdown(
                                f"**Faux Positif :** "
                                f"{fp}")
                            st.markdown(
                                f"**Modèle :** "
                                f"`{analysis.get('llm_model', 'N/A')}`")

                        if "attack_description" in \
                                analysis:
                            st.markdown(
                                f"**Description :** "
                                f"{analysis['attack_description']}")

                        if "false_positive_reason" in \
                                analysis:
                            st.caption(
                                f"💡 {analysis['false_positive_reason']}")

                    # Actions immédiates
                    if "immediate_actions" in \
                            analysis:
                        st.markdown(
                            "### ✅ Actions "
                            "Immédiates")
                        actions = analysis[
                            "immediate_actions"]
                        if isinstance(
                                actions, list):
                            with st.container(
                                    border=True):
                                for i, action in \
                                        enumerate(
                                        actions, 1):
                                    st.markdown(
                                        f"**{i}.** "
                                        f"{action}")

                    # Investigation
                    if "investigation_steps" in \
                            analysis:
                        st.markdown(
                            "### 🔍 Investigation "
                            "Suggérée")
                        steps = analysis[
                            "investigation_steps"]
                        if isinstance(
                                steps, list):
                            with st.container(
                                    border=True):
                                for step in steps:
                                    st.markdown(
                                        f"- {step}")

                    # Mesures préventives
                    if "preventive_measures" in \
                            analysis:
                        st.markdown(
                            "### 🛡 Mesures "
                            "Préventives")
                        measures = analysis[
                            "preventive_measures"]
                        if isinstance(
                                measures, list):
                            with st.container(
                                    border=True):
                                for m in measures:
                                    st.markdown(
                                        f"- {m}")

                    # Contexte technique
                    if "technical_context" in \
                            analysis:
                        with st.expander(
                                "📚 Contexte "
                                "Technique"):
                            st.markdown(
                                analysis[
                                    "technical_context"])

                # JSON complet
                with st.expander(
                        "🔍 Voir JSON complet"):
                    st.json(result)
    else:
        st.info(
            "👆 Générez ou saisissez "
            "des features pour lancer "
            "l'analyse")

# ============================================================
# TAB 2 : CHAT EXPERT
# ============================================================

with tab2:
    st.markdown(
        "### 💬 Chat avec l'Expert SOC")
    st.caption(
        "Posez vos questions sur la "
        "cybersécurité, les attaques, "
        "les bonnes pratiques...")

    # Initialiser historique
    if "chat_history" not in \
            st.session_state:
        st.session_state.chat_history = []

    # Boutons d'exemples
    st.markdown(
        "**Questions exemples :**")
    col1, col2, col3 = st.columns(3)

    examples = {
        "🛡 Qu'est-ce qu'un DDoS ?":
            "Qu'est-ce qu'une attaque DDoS "
            "et comment s'en protéger ?",
        "🔍 Détecter un Zero-Day":
            "Comment détecter une attaque "
            "Zero-Day dans un SOC ?",
        "📋 Bonnes pratiques":
            "Quelles sont les bonnes "
            "pratiques pour sécuriser "
            "un réseau d'entreprise ?",
    }

    selected_example = None
    with col1:
        if st.button(
                "🛡 Qu'est-ce qu'un DDoS ?",
                use_container_width=True):
            selected_example = examples[
                "🛡 Qu'est-ce qu'un DDoS ?"]
    with col2:
        if st.button(
                "🔍 Détecter un Zero-Day",
                use_container_width=True):
            selected_example = examples[
                "🔍 Détecter un Zero-Day"]
    with col3:
        if st.button(
                "📋 Bonnes pratiques",
                use_container_width=True):
            selected_example = examples[
                "📋 Bonnes pratiques"]

    st.markdown("---")

    # Afficher historique
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])

    # Input
    prompt = st.chat_input(
        "Posez votre question...")

    # Si exemple sélectionné, l'utiliser
    if selected_example and not prompt:
        prompt = selected_example

    if prompt:
        # Ajouter à l'historique
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt
        })

        # Afficher
        with st.chat_message("user"):
            st.write(prompt)

        # Réponse LLM
        with st.chat_message("assistant"):
            with st.spinner(
                    "🧠 Réflexion..."):
                result = client.chat_llm(
                    question=prompt)

            if "error" in result:
                st.error(result["error"])
                response = (
                    f"Erreur : "
                    f"{result['error']}")
            else:
                response = result.get(
                    "response", "")
                st.write(response)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response
            })

        st.rerun()

    # Bouton effacer
    if st.session_state.chat_history:
        st.markdown("---")
        if st.button(
                "🗑 Effacer historique"):
            st.session_state.chat_history = []
            st.rerun()

            