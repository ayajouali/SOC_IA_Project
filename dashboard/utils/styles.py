"""
SOC-IA | Styles CSS pour Dashboard
"""

CUSTOM_CSS = """
<style>
    /* Police générale */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
    }

    /* Couleurs principales */
    :root {
        --primary: #1565C0;
        --primary-dark: #0A2472;
        --primary-light: #42A5F5;
        --teal: #00796B;
        --bg: #FAFCFF;
        --text: #1A237E;
    }

    /* Titre principal */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #0A2472;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(
            90deg, #0A2472, #1565C0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        font-size: 1.1rem;
        color: #546E7A;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Cards KPI */
    .kpi-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1565C0;
        box-shadow:
            0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }

    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0A2472;
        margin: 0;
    }

    .kpi-label {
        font-size: 0.9rem;
        color: #546E7A;
        margin: 0;
    }

    /* Alert badges */
    .alert-critical {
        background: #C62828;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
        display: inline-block;
    }

    .alert-high {
        background: #F57C00;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
        display: inline-block;
    }

    .alert-medium {
        background: #FFA000;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
        display: inline-block;
    }

    .alert-info {
        background: #1976D2;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
        display: inline-block;
    }

    .alert-review {
        background: #7B1FA2;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
        display: inline-block;
    }

    /* Sidebar */
    .sidebar-title {
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem 0;
    }

    /* Boutons */
    .stButton > button {
        background: linear-gradient(
            90deg, #1565C0, #1976D2);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.7rem 2rem;
        border-radius: 8px;
        transition: all 0.3s;
    }

    .stButton > button:hover {
        background: linear-gradient(
            90deg, #0D47A1, #1565C0);
        transform: translateY(-2px);
        box-shadow:
            0 4px 12px rgba(21,101,192,0.4);
    }

    /* Cacher footer Streamlit */
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
"""


def get_alert_badge(level: str) -> str:
    """Retourne un badge HTML coloré."""
    level_lower = level.lower()
    return (
        f'<span class="alert-{level_lower}">'
        f'{level}</span>')


def get_alert_color(level: str) -> str:
    """Retourne la couleur d'une alerte."""
    colors = {
        "CRITICAL": "#C62828",
        "HIGH": "#F57C00",
        "MEDIUM": "#FFA000",
        "LOW": "#1976D2",
        "INFO": "#1976D2",
        "REVIEW": "#7B1FA2",
    }
    return colors.get(level, "#546E7A")