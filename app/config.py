"""
SOC-IA | Configuration de l'API
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Charger .env
load_dotenv()


class Settings:
    """Configuration centralisée."""

    # API
    API_HOST: str = os.getenv(
        "API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv(
        "API_PORT", "8000"))
    API_DEBUG: bool = os.getenv(
        "API_DEBUG", "true").lower() == "true"
    API_VERSION: str = os.getenv(
        "API_VERSION", "1.0.0")
    API_TITLE: str = os.getenv(
        "API_TITLE", "SOC-IA API")

    # Chemins
    BASE_DIR: Path = Path(
        __file__).parent.parent
    MODELS_DIR: Path = BASE_DIR / "models"
    LOG_DIR: Path = (
        BASE_DIR / "data" / "logs")
    DATA_DIR: Path = BASE_DIR / "data"

    # Fichiers modèles
    LGBM_PATH: Path = (
        MODELS_DIR /
        "lightgbm_principal.joblib")
    RF_PATH: Path = (
        MODELS_DIR /
        "randomforest_backup.joblib")
    AE_PATH: Path = (
        MODELS_DIR /
        "autoencoder_zeroday.keras")
    SCALER_ROB_PATH: Path = (
        MODELS_DIR /
        "scaler_robust.joblib")
    SCALER_MM_PATH: Path = (
        MODELS_DIR /
        "scaler_minmax.joblib")
    METADATA_PATH: Path = (
        MODELS_DIR / "metadata.json")
    CONFIG_PATH: Path = (
        MODELS_DIR / "config.json")

    # Sécurité
    API_KEY: str = os.getenv(
        "API_KEY",
        "soc-ia-dev-key-2024")

    # ✅ Groq LLM
    GROQ_API_KEY: str = os.getenv(
        "GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv(
        "GROQ_MODEL",
        "llama-3.3-70b-versatile")

    def __init__(self):
        # Créer les dossiers nécessaires
        self.LOG_DIR.mkdir(
            parents=True, exist_ok=True)
        self.DATA_DIR.mkdir(
            parents=True, exist_ok=True)
        (self.DATA_DIR / "input").mkdir(
            exist_ok=True)
        (self.DATA_DIR / "output").mkdir(
            exist_ok=True)


# Instance globale
settings = Settings()