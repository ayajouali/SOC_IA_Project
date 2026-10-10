"""
SOC-IA | Analyse CSV + LLM Global
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import tempfile
import io

from utils.api_client import client
from utils.styles import (
    CUSTOM_CSS, get_alert_color)

st.set_page_config(
    page_title="Analyse CSV - SOC-IA",
    page_icon="📁",
    layout="wide",
)

st.markdown(CUSTOM_CSS,
            unsafe_allow_html=True)

st.markdown(
    "<div class='main-title'>"
    "Analyse en Masse</div>",
    unsafe_allow_html=True)

st.markdown(
    "<div class='subtitle'>"
    "Uploadez un CSV pour analyser "
    "plusieurs flux</div>",
    unsafe_allow_html=True)

health = client.health()
if health.get("status") != "ok":
    st.error("API non disponible")
    st.stop()

# ============================================================
# UPLOAD
# ============================================================

st.markdown("## Upload du Fichier")

with st.container(border=True):
    st.markdown("""
    **Format attendu :**
    - Fichier CSV
    - 24 colonnes (features)
    - Valeurs numeriques
    """)

    uploaded_file = st.file_uploader(
        "Choisissez un CSV",
        type=["csv"])

if uploaded_file is None:
    st.markdown("---")
    st.markdown(
        "### Ou donnees de demo")

    if st.button(
            "Generer 100 flux demo",
            type="primary"):
        np.random.seed(42)
        demo_data = np.random.randn(
            100, 24)
        demo_df = pd.DataFrame(
            demo_data,
            columns=[
                f"feature_{i+1}"
                for i in range(24)])
        csv_buffer = io.StringIO()
        demo_df.to_csv(
            csv_buffer, index=False)
        csv_bytes = csv_buffer\
            .getvalue().encode()
        st.session_state.demo_csv = (
            csv_bytes)
        st.session_state.demo_df = demo_df
        st.success("100 flux generes")
        st.rerun()

if uploaded_file is None and \
        "demo_csv" in st.session_state:
    st.info("Donnees demo chargees")
    df_preview = st.session_state.demo_df
    csv_bytes = st.session_state.demo_csv
else:
    csv_bytes = None
    if uploaded_file:
        df_preview = pd.read_csv(
            uploaded_file)
        uploaded_file.seek(0)
        csv_bytes = uploaded_file.read()

# ============================================================
# ANALYSE
# ============================================================

if csv_bytes:
    st.markdown("---")
    st.markdown("## Apercu des Donnees")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Lignes",
                   len(df_preview))
    with col2:
        st.metric("Colonnes",
                   len(df_preview.columns))
    with col3:
        st.metric(
            "Taille",
            f"{len(csv_bytes)/1024:.1f} KB")

    with st.expander("Premieres lignes"):
        st.dataframe(
            df_preview.head(10),
            use_container_width=True)

    st.markdown("---")

    if st.button(
            f"ANALYSER LES "
            f"{len(df_preview)} FLUX",
            type="primary",
            use_container_width=True):

        import time
        with st.spinner(
                "Analyse en cours..."):
            t_start = time.time()

            with tempfile.NamedTemporaryFile(
                    mode='wb',
                    suffix='.csv',
                    delete=False) as tmp:
                tmp.write(csv_bytes)
                tmp_path = tmp.name

            result = client.predict_csv(
                tmp_path)
            t_elapsed = time.time() - t_start

        if "error" in result:
            st.error(result['error'])
        else:
            # ✅ Sauvegarder resultats
            st.session_state.csv_result = (
                result)
            st.session_state.csv_time = (
                t_elapsed)
            # Reset LLM analysis
            if "csv_llm_analysis" in (
                    st.session_state):
                del st.session_state[
                    "csv_llm_analysis"]

# ============================================================
# AFFICHAGE RÉSULTATS
# ============================================================

if "csv_result" in st.session_state:
    result = st.session_state.csv_result
    t_elapsed = st.session_state.get(
        "csv_time", 0)

    total = result.get("total", 0)
    time_ms = result.get(
        "processing_time_ms", 0)
    summary = result.get("summary", {})
    predictions = result.get(
        "predictions", [])

    st.markdown("---")
    st.markdown("## Resultats Globaux")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Analyse", total)
    with col2:
        critical = summary.get(
            "CRITICAL", 0) + \
            summary.get("HIGH", 0)
        st.metric("Alertes Hautes",
                   critical)
    with col3:
        info = summary.get("INFO", 0)
        st.metric("Trafic Normal", info)
    with col4:
        st.metric("Temps",
                   f"{t_elapsed:.1f}s")

    st.markdown("---")

    if predictions:
        df_pred = pd.DataFrame(predictions)

        col1, col2 = st.columns(2)

        # Pie chart alertes
        with col1:
            st.markdown(
                "### Distribution Alertes")
            alert_counts = pd.Series(
                summary).reset_index()
            alert_counts.columns = [
                'Alerte', 'Nombre']

            colors_map = {
                "CRITICAL": "#C62828",
                "HIGH": "#F57C00",
                "MEDIUM": "#FFA000",
                "INFO": "#1976D2",
                "REVIEW": "#7B1FA2",
            }

            fig = px.pie(
                alert_counts,
                values='Nombre',
                names='Alerte',
                color='Alerte',
                color_discrete_map=\
                    colors_map,
                hole=0.4)
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label')
            fig.update_layout(height=400)
            st.plotly_chart(
                fig,
                use_container_width=True)

        # Bar chart classes
        with col2:
            st.markdown(
                "### Classes Detectees")
            class_counts = df_pred[
                'prediction']\
                .value_counts()\
                .reset_index()
            class_counts.columns = [
                'Classe', 'Nombre']

            fig2 = px.bar(
                class_counts,
                x='Classe',
                y='Nombre',
                color='Nombre',
                color_continuous_scale=\
                    'Blues')
            fig2.update_layout(
                height=400,
                showlegend=False)
            st.plotly_chart(
                fig2,
                use_container_width=True)

        # Timeline
        st.markdown("---")
        st.markdown("### Timeline")

        df_pred['idx'] = range(
            len(df_pred))
        fig3 = px.scatter(
            df_pred,
            x='idx',
            y='confidence',
            color='alert_level',
            color_discrete_map=\
                colors_map,
            hover_data=['prediction'],
            labels={
                'idx': 'Flux N',
                'confidence': 'Confiance'})
        fig3.update_layout(height=400)
        st.plotly_chart(
            fig3,
            use_container_width=True)

        # Tableau
        st.markdown("---")
        st.markdown("### Tableau Detaille")

        col1, col2 = st.columns(2)
        with col1:
            filter_alert = st.multiselect(
                "Filtrer alertes:",
                options=df_pred[
                    'alert_level'].unique(),
                default=df_pred[
                    'alert_level'].unique())
        with col2:
            filter_class = st.multiselect(
                "Filtrer classes:",
                options=df_pred[
                    'prediction'].unique(),
                default=df_pred[
                    'prediction'].unique())

        df_filtered = df_pred[
            df_pred['alert_level']
                .isin(filter_alert)
            & df_pred['prediction']
                .isin(filter_class)]

        df_display = df_filtered[[
            'idx', 'prediction',
            'confidence',
            'alert_level']].copy()
        df_display.columns = [
            'Flux', 'Classe',
            'Confiance', 'Alerte']
        df_display['Confiance'] = \
            df_display['Confiance'].apply(
                lambda x: f"{x*100:.2f}%")

        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            height=400)

        # ============================
        # ✅ ANALYSE LLM GLOBALE
        # ============================
        st.markdown("---")

        llm_csv_status = (
            client.llm_status())
        llm_csv_ok = (
            llm_csv_status.get(
                "available", False))

        if llm_csv_ok:
            st.markdown(
                "### Analyse Globale "
                "par LLM")

            # Construire le resume
            class_dist = df_pred[
                'prediction']\
                .value_counts().to_dict()
            n_high = summary.get(
                "CRITICAL", 0) + \
                summary.get("HIGH", 0)

            summary_text = (
                f"{total} flux reseau "
                f"analyses.\n"
                f"Alertes : {summary}\n"
                f"Classes : {class_dist}\n"
                f"Alertes hautes : "
                f"{n_high} "
                f"({n_high/total*100:.1f}%)")

            # Afficher si deja fait
            if "csv_llm_analysis" in (
                    st.session_state):
                with st.container(
                        border=True):
                    st.markdown(
                        "**Analyse LLM :**")
                    st.markdown(
                        st.session_state[
                            "csv_llm_analysis"
                        ])
            else:
                if st.button(
                        "Analyser avec LLM",
                        type="primary",
                        use_container_width=
                            True,
                        key="llm_csv"):

                    question = (
                        f"En tant qu'analyste "
                        f"SOC expert, analyse "
                        f"ce rapport de "
                        f"detection :\n\n"
                        f"{summary_text}\n\n"
                        f"Donne :\n"
                        f"1. Un resume de "
                        f"la situation\n"
                        f"2. Les risques "
                        f"principaux\n"
                        f"3. Les actions "
                        f"recommandees\n"
                        f"4. Le niveau de "
                        f"menace global")

                    with st.spinner(
                            "LLM analyse..."):
                        llm_resp = (
                            client.chat_llm(
                                question=
                                question))

                    if "error" in llm_resp:
                        st.error(
                            llm_resp["error"])
                    else:
                        response = (
                            llm_resp.get(
                                "response",
                                ""))
                        st.session_state[
                            "csv_llm_analysis"
                        ] = response

                        with st.container(
                                border=True):
                            st.markdown(
                                "**Analyse "
                                "LLM :**")
                            st.markdown(
                                response)

        st.markdown("---")

        # Telecharger CSV
        csv_out = df_pred.to_csv(
            index=False)
        st.download_button(
            "Telecharger resultats CSV",
            data=csv_out,
            file_name=
                "predictions_soc_ia.csv",
            mime="text/csv",
            use_container_width=True)