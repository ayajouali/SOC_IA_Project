"""
SOC-IA | Endpoint Health Check
"""

from fastapi import APIRouter
from datetime import datetime
import time

from app.schemas import HealthResponse

router = APIRouter(tags=["Health"])

# Temps de démarrage
START_TIME = time.time()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Vérifier le statut de l'API",
    description="Retourne le statut de "
                "l'API et des modèles")
async def health_check():
    """Health check endpoint."""
    from app.main import predictor

    return HealthResponse(
        status="ok" if predictor.loaded
               else "error",
        version="1.0.0",
        models_loaded=predictor.loaded,
        uptime_seconds=round(
            time.time() - START_TIME, 1),
        timestamp=datetime.now().isoformat(),
    )