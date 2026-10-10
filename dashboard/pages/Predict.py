"""
SOC-IA | Page Prediction Unifiee
ML + LLM + PDF
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time

from utils.api_client import client
from utils.styles import (
    CUSTOM_CSS, get_alert_color)

st.set_page_config(
    page_title="Prediction - SOC-IA",
    page_icon="🎯",
    layout="wide",
)

st.markdown(CUSTOM_CSS,
            unsafe_allow_html=True)

# ============================================================
# TITRE
# ============================================================

st.markdown(
    "<div class='main-title'>"
    "🎯 Detection & Analyse Intelligente"
    "</div>",
    unsafe_allow_html=True)

st.markdown(
    "<div class='subtitle'>"
    "Les modeles ML detectent — "
    "Le LLM explique et recommande"
    "</div>",
    unsafe_allow_html=True)

# Verifier API
health = client.health()
if health.get("status") != "ok":
    st.error("API non disponible")
    st.stop()

# Verifier LLM
llm_status = client.llm_status()
llm_ok = llm_status.get(
    "available", False)

if llm_ok:
    st.success(
        f"Modeles ML + LLM "
        f"({llm_status.get('model','')}) "
        f"connectes")
else:
    st.warning(
        "LLM non disponible. "
        "Verifiez GROQ_API_KEY")

# Charger features
features_data = client.get_features()
feature_names = [
    f["name"]
    for f in features_data.get(
        "features", [])]

# ============================================================
# SAISIE DES FEATURES
# ============================================================

st.markdown("---")
st.markdown("## Donnees du Flux Reseau")

mode = st.radio(
    "Mode :",
    ["Aleatoire",
     "Slider",
     "Manuel"],
    horizontal=True,
    label_visibility="collapsed")

features = []

if mode == "Aleatoire":
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button(
                "Generer",
                type="primary",
                use_container_width=True):
            np.random.seed(
                np.random.randint(0, 9999))
            features = np.random.randn(
                24).tolist()
            st.session_state.features = (
                features)
            # Reset anciens resultats
            if "prediction" in (
                    st.session_state):
                del st.session_state[
                    "prediction"]
            if "analysis" in (
                    st.session_state):
                del st.session_state[
                    "analysis"]

    if "features" in st.session_state:
        features = (
            st.session_state.features)
        with col2:
            st.success("24 features generees")

        with st.expander("Voir les valeurs"):
            df = pd.DataFrame({
                "Feature": (
                    feature_names
                    or [f"F{i+1}"
                        for i in range(24)]),
                "Valeur": [
                    round(v, 4)
                    for v in features]
            })
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True)

elif mode == "Slider":
    cols = st.columns(4)
    for i in range(24):
        col = cols[i % 4]
        with col:
            name = (
                feature_names[i][:15]
                if i < len(feature_names)
                else f"Feature {i+1}")
            val = st.slider(
                name,
                min_value=-3.0,
                max_value=3.0,
                value=0.0,
                step=0.1,
                key=f"slider_{i}")
            features.append(val)

else:
    text_input = st.text_area(
        "24 valeurs separees par virgules :",
        value=(
            "0.5, 1.2, 0.8, 0.3, 0.1, "
            "0.9, 0.4, 0.7, 0.2, 0.6, "
            "0.5, 0.8, 0.3, 0.1, 0.9, "
            "0.4, 0.7, 0.2, 0.6, 0.5, "
            "0.8, 0.3, 0.1, 0.9"),
        height=80)
    try:
        features = [
            float(x.strip())
            for x in text_input.split(",")]
        if len(features) != 24:
            st.error(
                f"Attendu 24, recu "
                f"{len(features)}")
            features = []
        else:
            st.success("24 valeurs OK")
    except Exception:
        st.error("Format invalide")
        features = []

# ============================================================
# BOUTON ANALYSE
# ============================================================

st.markdown("---")

if features and len(features) == 24:

    btn_text = (
        "DETECTER + ANALYSER (ML + LLM)"
        if llm_ok
        else "DETECTER (ML)")

    if st.button(
            btn_text,
            type="primary",
            use_container_width=True):

        # PREDICTION ML
        with st.spinner(
                "Etape 1/2 : Detection ML..."):
            t1 = time.time()
            prediction = (
                client.predict_single(
                    features))
            t_ml = time.time() - t1

        if "error" in prediction:
            st.error(
                f"Erreur : "
                f"{prediction['error']}")
            st.stop()

        # ✅ SAUVEGARDER dans session_state
        st.session_state.prediction = (
            prediction)
        st.session_state.t_ml = t_ml
        st.session_state.current_features = (
            features)

        # ANALYSE LLM
        analysis = {}
        t_llm = 0
        if llm_ok:
            with st.spinner(
                    "Etape 2/2 : "
                    "Analyse LLM..."):
                t2 = time.time()
                llm_result = (
                    client.explain_with_llm(
                        features))
                t_llm = time.time() - t2
                analysis = llm_result.get(
                    "llm_analysis", {})

        # ✅ SAUVEGARDER dans session_state
        st.session_state.analysis = analysis
        st.session_state.t_llm = t_llm

# ============================================================
# AFFICHAGE DES RÉSULTATS (depuis session_state)
# ============================================================

if "prediction" in st.session_state:
    prediction = st.session_state.prediction
    t_ml = st.session_state.get("t_ml", 0)
    analysis = st.session_state.get(
        "analysis", {})
    t_llm = st.session_state.get("t_llm", 0)

    pred_class = prediction.get(
        "prediction", "N/A")
    confidence = prediction.get(
        "confidence", 0)
    alert_level = prediction.get(
        "alert_level", "N/A")
    details = prediction.get("details", {})
    alert_color = get_alert_color(
        alert_level)

    # ============================
    # RÉSULTAT ML
    # ============================
    st.markdown("## Resultat de Detection")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div style='background:white;
                padding:1.2rem;
                border-radius:10px;
                border-left:5px solid
                {alert_color};
                box-shadow:0 2px 8px
                rgba(0,0,0,0.05);'>
            <p style='color:#546E7A;
                font-size:0.8rem;
                margin:0;'>
                Classe Detectee</p>
            <p style='font-size:1.8rem;
                font-weight:bold;
                color:{alert_color};
                margin:0.3rem 0 0 0;'>
                {pred_class}</p>
        </div>""",
        unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style='background:white;
                padding:1.2rem;
                border-radius:10px;
                border-left:5px solid
                #1565C0;
                box-shadow:0 2px 8px
                rgba(0,0,0,0.05);'>
            <p style='color:#546E7A;
                font-size:0.8rem;
                margin:0;'>
                Confiance</p>
            <p style='font-size:1.8rem;
                font-weight:bold;
                color:#1565C0;
                margin:0.3rem 0 0 0;'>
                {confidence*100:.1f}%</p>
        </div>""",
        unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style='background:white;
                padding:1.2rem;
                border-radius:10px;
                border-left:5px solid
                {alert_color};
                box-shadow:0 2px 8px
                rgba(0,0,0,0.05);'>
            <p style='color:#546E7A;
                font-size:0.8rem;
                margin:0;'>
                Niveau Alerte</p>
            <p style='font-size:1.8rem;
                font-weight:bold;
                color:{alert_color};
                margin:0.3rem 0 0 0;'>
                {alert_level}</p>
        </div>""",
        unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div style='background:white;
                padding:1.2rem;
                border-radius:10px;
                border-left:5px solid
                #546E7A;
                box-shadow:0 2px 8px
                rgba(0,0,0,0.05);'>
            <p style='color:#546E7A;
                font-size:0.8rem;
                margin:0;'>
                Temps ML</p>
            <p style='font-size:1.8rem;
                font-weight:bold;
                color:#546E7A;
                margin:0.3rem 0 0 0;'>
                {t_ml*1000:.0f}ms</p>
        </div>""",
        unsafe_allow_html=True)

    st.markdown("---")

    # ============================
    # DÉTAILS ML
    # ============================
    st.markdown("### Details des Modeles")

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("#### LightGBM")
            st.caption(
                "Classification principale")
            lgbm_d = details.get(
                "lightgbm", {})
            st.markdown(
                f"**Prediction :** "
                f"`{lgbm_d.get('predicted_class', 'N/A')}`")
            conf_v = lgbm_d.get(
                'confidence', 0)
            if conf_v:
                st.progress(
                    conf_v,
                    text=f"Confiance: "
                         f"{conf_v*100:.1f}%")

    with col2:
        with st.container(border=True):
            st.markdown("#### Autoencoder")
            st.caption("Detection Zero-Day")
            ae_d = details.get(
                "autoencoder", {})
            is_anom = ae_d.get(
                "is_anomaly", False)
            if is_anom:
                st.error(
                    "ANOMALIE DETECTEE")
            else:
                st.success(
                    "Pas d'anomalie")
            err = ae_d.get(
                "reconstruction_error", 0)
            thr = ae_d.get("threshold", 0)
            st.caption(
                f"Erreur: {err:.6f} | "
                f"Seuil: {thr:.6f}")

            if thr > 0:
                ratio = min(err/thr, 2.0)
                fig_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=ratio,
                        title={"text":
                               "Erreur/Seuil"},
                        gauge={
                            "axis": {
                                "range": [0, 2]},
                            "bar": {
                                "color":
                                ("#C62828"
                                 if ratio > 1
                                 else
                                 "#1565C0")},
                            "threshold": {
                                "line": {
                                    "color":
                                    "#F57C00",
                                    "width": 3},
                                "thickness":
                                0.8,
                                "value": 1},
                        }))
                fig_gauge.update_layout(
                    height=200,
                    margin=dict(
                        t=50, b=10,
                        l=30, r=30))
                st.plotly_chart(
                    fig_gauge,
                    use_container_width=True)

    st.markdown("---")

    # ============================
    # ANALYSE LLM
    # ============================
    if analysis and "error" not in analysis:
        st.markdown(
            "## Analyse Intelligente (LLM)")
        st.caption(
            f"Modele: "
            f"{analysis.get('llm_model','N/A')} | "
            f"Temps: {t_llm:.1f}s")

        if "summary" in analysis:
            st.info(
                f"**Resume:** "
                f"{analysis['summary']}")

        col1, col2 = st.columns(2)
        with col1:
            sev = analysis.get(
                "severity", "N/A")
            sev_icons = {
                "CRITICAL": "🔴",
                "HIGH": "🟠",
                "MEDIUM": "🟡",
                "LOW": "🟢"}
            icon = sev_icons.get(sev, "")
            with st.container(border=True):
                st.markdown(
                    f"### {icon} Severite: "
                    f"**{sev}**")
                reason = analysis.get(
                    "severity_reason", "")
                if reason:
                    st.caption(reason)

        with col2:
            risk = analysis.get(
                "risk_score", 0)
            try:
                risk = float(risk)
            except:
                risk = 0
            with st.container(border=True):
                st.markdown(
                    f"### Score Risque: "
                    f"**{risk}/10**")
                st.progress(
                    min(risk/10, 1.0))

        with st.container(border=True):
            st.markdown(
                "### Analyse de l'Attaque")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"**Type:** "
                    f"`{analysis.get('attack_type','N/A')}`")
                st.markdown(
                    f"**Vecteur:** "
                    f"{analysis.get('attack_vector','N/A')}")
                st.markdown(
                    f"**Faux Positif:** "
                    f"{analysis.get('false_positive_likelihood','N/A')}")
            with col2:
                desc = analysis.get(
                    "attack_description", "")
                if desc:
                    st.markdown(
                        f"**Description:** "
                        f"{desc}")

        actions = analysis.get(
            "immediate_actions", [])
        if actions and isinstance(
                actions, list):
            with st.container(border=True):
                st.markdown(
                    "### Actions Recommandees")
                for i, a in enumerate(
                        actions, 1):
                    st.markdown(
                        f"**{i}.** {a}")

        steps = analysis.get(
            "investigation_steps", [])
        if steps and isinstance(
                steps, list):
            with st.container(border=True):
                st.markdown(
                    "### Investigation")
                for s in steps:
                    st.markdown(f"- {s}")

        measures = analysis.get(
            "preventive_measures", [])
        if measures and isinstance(
                measures, list):
            with st.container(border=True):
                st.markdown(
                    "### Mesures Preventives")
                for m in measures:
                    st.markdown(f"- {m}")

        context = analysis.get(
            "technical_context", "")
        if context:
            with st.expander(
                    "Contexte Technique"):
                st.markdown(context)

    st.markdown("---")

    # ============================
    # TEMPS
    # ============================
    cols = st.columns(3)
    with cols[0]:
        st.metric("Temps ML",
                   f"{t_ml*1000:.0f} ms")
    with cols[1]:
        st.metric("Temps LLM",
                   f"{t_llm:.1f} s"
                   if t_llm > 0
                   else "N/A")
    with cols[2]:
        t_total = t_ml + t_llm
        st.metric("Temps Total",
                   f"{t_total:.1f} s")

    # JSON
    with st.expander("Voir JSON"):
        st.json(prediction)
        if analysis and \
                "error" not in analysis:
            st.json(analysis)

    # ============================
    # ✅ RAPPORT PDF (CORRIGÉ)
    # ============================
    st.markdown("---")
    st.markdown("### Rapport PDF")

    # ✅ Utiliser st.download_button
    # directement avec les données
    # en session_state
    try:
        from utils.pdf_generator import (
            generate_report)

        llm_data = None
        if analysis and \
                "error" not in analysis:
            llm_data = analysis

        pdf_path = generate_report(
            prediction=prediction,
            analysis=llm_data,
            features=(
                st.session_state.get(
                    "current_features",
                    None)))

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        st.download_button(
            label=(
                "Telecharger le "
                "Rapport PDF"),
            data=pdf_bytes,
            file_name=(
                "SOC_IA_Rapport.pdf"),
            mime="application/pdf",
            use_container_width=True,
            type="primary")

    except ImportError:
        st.warning(
            "Installez fpdf2 : "
            "pip install fpdf2")
    except Exception as e:
        st.error(
            f"Erreur PDF : {e}")

else:
    st.info(
        "Entrez 24 features et "
        "cliquez sur le bouton pour "
        "lancer la detection")

# ============================================================
# CHAT EXPERT
# ============================================================

st.markdown("---")
st.markdown("## Poser une Question")

if not llm_ok:
    st.info(
        "Activez le LLM pour "
        "utiliser le chat")
else:
    st.caption(
        "Questions sur la cybersecurite")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button(
                "DDoS ?",
                use_container_width=True,
                key="q1"):
            st.session_state.chat_q = (
                "Qu'est-ce qu'une attaque "
                "DDoS et comment s'en "
                "proteger ?")
    with col2:
        if st.button(
                "Zero-Day",
                use_container_width=True,
                key="q2"):
            st.session_state.chat_q = (
                "Comment detecter une "
                "attaque Zero-Day ?")
    with col3:
        if st.button(
                "Bonnes pratiques",
                use_container_width=True,
                key="q3"):
            st.session_state.chat_q = (
                "Bonnes pratiques SOC ?")

    chat_input = st.text_input(
        "Question :",
        value=st.session_state.get(
            "chat_q", ""),
        key="chat_main")

    if chat_input:
        if st.button("Envoyer",
                      type="primary"):
            with st.spinner("Reflexion..."):
                result = client.chat_llm(
                    question=chat_input)
            if "error" in result:
                st.error(result["error"])
            else:
                with st.container(
                        border=True):
                    st.markdown(
                        f"**Reponse :**")
                    st.markdown(
                        result.get(
                            "response", ""))
            if "chat_q" in (
                    st.session_state):
                del st.session_state.chat_q