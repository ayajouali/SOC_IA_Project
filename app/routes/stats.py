"""
SOC-IA | Endpoint Statistiques
"""

from fastapi import APIRouter

from app.schemas import StatsResponse

router = APIRouter(
    prefix="/api/v1",
    tags=["Statistics"])


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Statistiques d'utilisation",
    description="Compteurs de prédictions, "
                "alertes, anomalies, etc.")
async def get_stats():
    """Statistiques de l'API."""
    from app.main import predictor

    stats = predictor.get_stats()

    return StatsResponse(**stats)