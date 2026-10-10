"""
SOC-IA | Point d'Entrée Principal de l'API
===========================================

Lancement :
    uvicorn app.main:app --reload

Documentation :
    http://localhost:8000/docs
"""

import time
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import (
    CORSMiddleware)

from app.config import settings
from app.predictor import SOC_IA_Predictor

# Logger
logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(name)s | "
        "%(levelname)s | %(message)s"))
logger = logging.getLogger("SOC-IA")

# ============================================================
# INITIALISATION
# ============================================================

app = FastAPI(
    title=settings.API_TITLE,
    description=(
        "API de détection automatique de "
        "cyberattaques basée sur des "
        "modèles IA complémentaires : "
        "LightGBM, RandomForest et "
        "Autoencoder, avec analyse "
        "intelligente par LLM (Groq).\n\n"
        "**Dataset** : CIC-IDS2017\n"
        "**Classes** : 9 types d'attaques\n"
        "**Features** : 24 caractéristiques "
        "réseau\n"
        "**Meilleur modèle** : LightGBM "
        "(F1=99.78%)"),
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# CHARGEMENT DES MODÈLES
# ============================================================

logger.info("="*50)
logger.info("  SOC-IA API - Demarrage")
logger.info("="*50)

predictor = SOC_IA_Predictor()

# ============================================================
# IMPORT DES ROUTES
# ============================================================

from app.routes.health import (
    router as health_router)
from app.routes.predict import (
    router as predict_router)
from app.routes.models_info import (
    router as models_router)
from app.routes.stats import (
    router as stats_router)
from app.routes.llm_analysis import (
    router as llm_router)

app.include_router(health_router)
app.include_router(predict_router)
app.include_router(models_router)
app.include_router(stats_router)
app.include_router(llm_router)


# ============================================================
# ENDPOINT RACINE
# ============================================================

@app.get(
    "/",
    tags=["Root"],
    summary="Page d'accueil")
async def root():
    """Page d'accueil de l'API."""
    return {
        "project": "SOC-IA",
        "description": (
            "API de detection de "
            "cyberattaques avec LLM"),
        "version": settings.API_VERSION,
        "documentation": "/docs",
        "health": "/health",
        "endpoints": {
            "predict": (
                "POST /api/v1/predict"),
            "batch": (
                "POST /api/v1/predict/batch"),
            "csv": (
                "POST /api/v1/predict/csv"),
            "llm_explain": (
                "POST /api/v1/llm/explain"),
            "llm_chat": (
                "POST /api/v1/llm/chat"),
            "llm_status": (
                "GET /api/v1/llm/status"),
            "models": (
                "GET /api/v1/models/info"),
            "features": (
                "GET /api/v1/models/features"),
            "classes": (
                "GET /api/v1/models/classes"),
            "stats": (
                "GET /api/v1/stats"),
            "health": "GET /health",
        },
        "models": {
            "principal": "LightGBM (99.78%)",
            "backup": "RandomForest (99.62%)",
            "zero_day": (
                "Autoencoder (70.84%)"),
            "ai_analyst": (
                "Llama 3.3 70B (Groq)"),
        }
    }


# ============================================================
# ÉVÉNEMENTS
# ============================================================

@app.on_event("startup")
async def startup():
    logger.info("API demarree avec succes")
    logger.info(
        f"Documentation : "
        f"http://localhost:"
        f"{settings.API_PORT}/docs")


@app.on_event("shutdown")
async def shutdown():
    logger.info("Arret de l'API")


# ============================================================
# POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_DEBUG,
    )