"""
SOC-IA | Générateur de Rapports PDF
"""

import os
import tempfile
from datetime import datetime
from fpdf import FPDF


class SOC_IA_PDF(FPDF):
    """Rapport PDF professionnel SOC-IA."""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(
            auto=True, margin=25)

    def header(self):
        """En-tête de chaque page."""
        # Bande bleue
        self.set_fill_color(13, 71, 161)
        self.rect(0, 0, 210, 18, 'F')

        # Titre
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 4)
        self.cell(0, 10,
                  'SOC-IA | Rapport de Detection',
                  align='L')

        # Date
        self.set_font('Helvetica', '', 9)
        self.set_xy(10, 11)
        self.cell(
            0, 5,
            datetime.now().strftime(
                '%d/%m/%Y %H:%M:%S'),
            align='R')

        self.ln(15)

    def footer(self):
        """Pied de page."""
        self.set_y(-15)
        self.set_fill_color(13, 71, 161)
        self.rect(0, 282, 210, 15, 'F')

        self.set_font('Helvetica', '', 8)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 283)
        self.cell(
            0, 10,
            f'SOC-IA v1.0.0 | '
            f'CIC-IDS2017 | '
            f'Page {self.page_no()}',
            align='C')

    def add_section_title(self, title,
                           icon=""):
        """Ajoute un titre de section."""
        self.set_font('Helvetica', 'B', 13)
        self.set_text_color(13, 71, 161)
        self.ln(5)
        self.cell(
            0, 8,
            f'{icon} {title}',
            new_x="LMARGIN", new_y="NEXT")
        # Ligne sous le titre
        self.set_draw_color(13, 71, 161)
        self.set_line_width(0.5)
        self.line(10, self.get_y(),
                  200, self.get_y())
        self.ln(4)

    def add_kpi_box(self, label, value,
                     color=(13, 71, 161)):
        """Ajoute une box KPI."""
        x = self.get_x()
        y = self.get_y()

        # Bordure gauche colorée
        self.set_fill_color(*color)
        self.rect(x, y, 2, 15, 'F')

        # Fond clair
        self.set_fill_color(248, 250, 254)
        self.rect(x + 2, y, 40, 15, 'F')

        # Label
        self.set_font('Helvetica', '', 8)
        self.set_text_color(84, 110, 122)
        self.set_xy(x + 5, y + 1)
        self.cell(35, 5, label)

        # Valeur
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(*color)
        self.set_xy(x + 5, y + 7)
        self.cell(35, 6, str(value))

        self.set_xy(x + 45, y)

    def add_text(self, text,
                  bold=False, size=10,
                  color=(33, 33, 33)):
        """Ajoute du texte."""
        style = 'B' if bold else ''
        self.set_font(
            'Helvetica', style, size)
        self.set_text_color(*color)
        self.multi_cell(
            0, 6, text,
            new_x="LMARGIN", new_y="NEXT")

    def add_bullet(self, text,
                    number=None):
        """Ajoute un point ou numéro."""
        self.set_font('Helvetica', '', 10)
        self.set_text_color(33, 33, 33)
        prefix = (f"{number}. "
                   if number else "- ")
        self.cell(8, 6, prefix)
        self.multi_cell(
            0, 6, text,
            new_x="LMARGIN", new_y="NEXT")

    def add_alert_badge(self, level):
        """Ajoute un badge d'alerte."""
        colors = {
            "CRITICAL": (198, 40, 40),
            "HIGH": (245, 124, 0),
            "MEDIUM": (255, 160, 0),
            "LOW": (25, 118, 210),
            "INFO": (25, 118, 210),
            "REVIEW": (123, 31, 162),
        }
        color = colors.get(
            level, (97, 97, 97))

        self.set_fill_color(*color)
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(
            255, 255, 255)

        w = self.get_string_width(level) + 8
        x = self.get_x()
        y = self.get_y()
        self.rect(x, y, w, 8, 'F')
        self.set_xy(x + 4, y + 1)
        self.cell(w - 8, 6, level)
        self.set_xy(x + w + 5, y)

    def add_separator(self):
        """Ligne de séparation."""
        self.set_draw_color(207, 216, 220)
        self.set_line_width(0.3)
        self.line(10, self.get_y(),
                  200, self.get_y())
        self.ln(4)


def generate_report(
        prediction,
        analysis=None,
        features=None):
    """
    Génère un rapport PDF complet.

    Args:
        prediction: dict de la prédiction ML
        analysis: dict de l'analyse LLM
        features: list des 24 features

    Returns:
        str: chemin du fichier PDF
    """
    pdf = SOC_IA_PDF()
    pdf.add_page()

    # ==========================
    # TITRE DU RAPPORT
    # ==========================
    pdf.set_font('Helvetica', 'B', 20)
    pdf.set_text_color(10, 36, 114)
    pdf.cell(
        0, 12,
        'Rapport d\'Incident de Securite',
        new_x="LMARGIN", new_y="NEXT",
        align='C')

    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(84, 110, 122)
    pdf.cell(
        0, 6,
        f'Genere le '
        f'{datetime.now().strftime("%d/%m/%Y a %H:%M:%S")}',
        new_x="LMARGIN", new_y="NEXT",
        align='C')
    pdf.ln(8)

    # ==========================
    # SECTION 1 : RÉSULTAT ML
    # ==========================
    pdf.add_section_title(
        "Resultat de Detection",
        "1.")

    # KPI
    pred_class = prediction.get(
        "prediction", "N/A")
    confidence = prediction.get(
        "confidence", 0)
    alert_level = prediction.get(
        "alert_level", "N/A")

    alert_colors = {
        "CRITICAL": (198, 40, 40),
        "HIGH": (245, 124, 0),
        "MEDIUM": (255, 160, 0),
        "INFO": (25, 118, 210),
    }

    pdf.add_kpi_box(
        "Classe Detectee",
        pred_class)
    pdf.add_kpi_box(
        "Confiance",
        f"{confidence*100:.1f}%")
    pdf.add_kpi_box(
        "Niveau Alerte",
        alert_level,
        alert_colors.get(
            alert_level,
            (13, 71, 161)))
    pdf.ln(20)

    # Détails modèles
    details = prediction.get(
        "details", {})

    pdf.add_text(
        "Details des Modeles :",
        bold=True, size=11)
    pdf.ln(2)

    # LightGBM
    lgbm = details.get("lightgbm", {})
    pdf.add_text(
        f"LightGBM : "
        f"{lgbm.get('predicted_class', 'N/A')} "
        f"(Confiance: "
        f"{lgbm.get('confidence', 0)*100:.1f}%)",
        size=10)

    # Autoencoder
    ae = details.get("autoencoder", {})
    is_anom = ae.get("is_anomaly", False)
    ae_err = ae.get(
        "reconstruction_error", 0)
    ae_thr = ae.get("threshold", 0)

    status = ("ANOMALIE DETECTEE"
               if is_anom
               else "Normal")
    pdf.add_text(
        f"Autoencoder : {status} "
        f"(Erreur: {ae_err:.6f} / "
        f"Seuil: {ae_thr:.6f})",
        size=10,
        color=((198, 40, 40)
                if is_anom
                else (46, 125, 50)))

    pdf.ln(3)
    pdf.add_separator()

    # ==========================
    # SECTION 2 : ANALYSE LLM
    # ==========================
    if analysis and "error" not in analysis:
        pdf.add_section_title(
            "Analyse Intelligente (LLM)",
            "2.")

        # Résumé
        summary = analysis.get(
            "summary", "")
        if summary:
            pdf.add_text(
                "Resume :", bold=True)
            pdf.add_text(
                summary, size=10,
                color=(33, 33, 33))
            pdf.ln(3)

        # Sévérité et Risque
        severity = analysis.get(
            "severity", "N/A")
        risk = analysis.get(
            "risk_score", 0)
        try:
            risk = float(risk)
        except:
            risk = 0

        pdf.add_text(
            f"Severite : {severity}",
            bold=True, size=11,
            color=alert_colors.get(
                severity,
                (13, 71, 161)))

        sev_reason = analysis.get(
            "severity_reason", "")
        if sev_reason:
            pdf.add_text(
                sev_reason, size=9,
                color=(84, 110, 122))

        pdf.add_text(
            f"Score de Risque : "
            f"{risk}/10",
            bold=True, size=11)
        pdf.ln(3)

        # Détails attaque
        desc = analysis.get(
            "attack_description", "")
        vector = analysis.get(
            "attack_vector", "")
        fp = analysis.get(
            "false_positive_likelihood",
            "")
        fp_reason = analysis.get(
            "false_positive_reason", "")

        if desc or vector:
            pdf.add_text(
                "Analyse de l'Attaque :",
                bold=True)
            if desc:
                pdf.add_text(
                    f"Description : {desc}",
                    size=10)
            if vector:
                pdf.add_text(
                    f"Vecteur : {vector}",
                    size=10)
            if fp:
                pdf.add_text(
                    f"Risque Faux Positif : "
                    f"{fp}",
                    size=10)
            if fp_reason:
                pdf.add_text(
                    fp_reason, size=9,
                    color=(84, 110, 122))
            pdf.ln(3)

        pdf.add_separator()

        # ==========================
        # SECTION 3 : RECOMMANDATIONS
        # ==========================
        pdf.add_section_title(
            "Recommandations",
            "3.")

        # Actions immédiates
        actions = analysis.get(
            "immediate_actions", [])
        if actions and isinstance(
                actions, list):
            pdf.add_text(
                "Actions Immediates :",
                bold=True)
            for i, action in enumerate(
                    actions, 1):
                pdf.add_bullet(
                    action, number=i)
            pdf.ln(3)

        # Investigation
        steps = analysis.get(
            "investigation_steps", [])
        if steps and isinstance(
                steps, list):
            pdf.add_text(
                "Investigation "
                "Suggeree :",
                bold=True)
            for step in steps:
                pdf.add_bullet(step)
            pdf.ln(3)

        # Mesures préventives
        measures = analysis.get(
            "preventive_measures", [])
        if measures and isinstance(
                measures, list):
            pdf.add_text(
                "Mesures Preventives :",
                bold=True)
            for m in measures:
                pdf.add_bullet(m)
            pdf.ln(3)

        pdf.add_separator()

        # ==========================
        # SECTION 4 : CONTEXTE
        # ==========================
        context = analysis.get(
            "technical_context", "")
        if context:
            pdf.add_section_title(
                "Contexte Technique",
                "4.")
            pdf.add_text(
                context, size=10)
            pdf.ln(3)
            pdf.add_separator()

    # ==========================
    # SECTION 5 : MÉTADONNÉES
    # ==========================
    pdf.add_section_title(
        "Metadonnees",
        "5." if analysis else "3.")

    pdf.add_text(
        f"Date : "
        f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
        size=9, color=(84, 110, 122))
    pdf.add_text(
        f"Dataset : CIC-IDS2017",
        size=9, color=(84, 110, 122))
    pdf.add_text(
        f"Modele Principal : "
        f"LightGBM (F1=99.78%)",
        size=9, color=(84, 110, 122))
    pdf.add_text(
        f"Detection Anomalie : "
        f"Autoencoder",
        size=9, color=(84, 110, 122))

    if analysis and "llm_model" in analysis:
        pdf.add_text(
            f"LLM : "
            f"{analysis['llm_model']} "
            f"(Groq)",
            size=9,
            color=(84, 110, 122))

    pdf.add_text(
        f"Features : 24 (RFECV)",
        size=9, color=(84, 110, 122))
    pdf.add_text(
        f"Classes : 9 types d'attaques",
        size=9, color=(84, 110, 122))

    # Sauvegarder
    tmp_dir = tempfile.gettempdir()
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S")
    pdf_path = os.path.join(
        tmp_dir,
        f"SOC_IA_Rapport_{timestamp}.pdf")

    pdf.output(pdf_path)

    return pdf_path