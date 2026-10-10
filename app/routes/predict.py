"""
SOC-IA | Endpoints de Prédiction
"""

import time
import io
import csv
from fastapi import (
    APIRouter, HTTPException,
    UploadFile, File)
from typing import List

from app.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionDetails,
    ModelDetail,
    AutoencoderDetail,
    ErrorResponse)

router = APIRouter(
    prefix="/api/v1",
    tags=["Predictions"])


def get_predictor():
    """Récupère l'instance du predictor."""
    from app.main import predictor
    if not predictor.loaded:
        raise HTTPException(
            status_code=503,
            detail="Modeles non charges")
    return predictor


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Prédire UN flux réseau",
    description="Envoie 24 features et "
                "reçoit la prédiction "
                "avec niveau d'alerte",
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse}})
async def predict_single(
        request: PredictionRequest):
    """Prédiction pour un seul flux."""

    pred = get_predictor()

    try:
        result = pred.predict_single(
            request.features)

        # Construire la réponse
        details = None
        if "details" in result:
            d = result["details"]
            details = PredictionDetails(
                lightgbm=ModelDetail(
                    **d["lightgbm"]),
                randomforest=ModelDetail(
                    **d["randomforest"]),
                autoencoder=AutoencoderDetail(
                    **d["autoencoder"]),
                models_agree=d["models_agree"],
            )

        return PredictionResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
            alert_level=result["alert_level"],
            timestamp=result["timestamp"],
            details=details,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur prediction : {e}")


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Prédire PLUSIEURS flux",
    description="Envoie N flux et reçoit "
                "N prédictions")
async def predict_batch(
        request: BatchPredictionRequest):
    """Prédiction batch."""

    pred = get_predictor()
    t_start = time.time()

    try:
        results_raw = pred.predict_batch(
            request.samples)

        # Convertir en réponses Pydantic
        predictions = []
        summary = {}

        for r in results_raw:
            alert = r["alert_level"]
            summary[alert] = summary.get(
                alert, 0) + 1

            details = None
            if "details" in r:
                d = r["details"]
                details = PredictionDetails(
                    lightgbm=ModelDetail(
                        **d["lightgbm"]),
                    randomforest=ModelDetail(
                        **d["randomforest"]),
                    autoencoder=AutoencoderDetail(
                        **d["autoencoder"]),
                    models_agree=d[
                        "models_agree"],
                )

            predictions.append(
                PredictionResponse(
                    prediction=r["prediction"],
                    confidence=r["confidence"],
                    alert_level=r[
                        "alert_level"],
                    timestamp=r["timestamp"],
                    details=details,
                ))

        t_ms = (time.time() - t_start) * 1000

        return BatchPredictionResponse(
            total=len(predictions),
            predictions=predictions,
            summary=summary,
            processing_time_ms=round(t_ms, 2),
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e))


@router.post(
    "/predict/csv",
    response_model=BatchPredictionResponse,
    summary="Prédire depuis un fichier CSV",
    description="Upload un CSV avec "
                "24 colonnes de features")
async def predict_csv(
        file: UploadFile = File(...)):
    """Prédiction depuis CSV."""

    pred = get_predictor()
    t_start = time.time()

    # Vérifier extension
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Fichier .csv requis")

    try:
        # Lire le CSV
        content = await file.read()
        text = content.decode('utf-8')
        reader = csv.reader(
            io.StringIO(text))

        # Lire le header (optionnel)
        header = next(reader, None)

        samples = []
        for row_num, row in enumerate(
                reader, 1):
            try:
                features = [
                    float(x.strip())
                    for x in row
                    if x.strip()]
                if len(features) == 24:
                    samples.append(features)
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Ligne {row_num}: "
                            f"attendu 24 valeurs, "
                            f"reçu {len(features)}"))
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Ligne {row_num}: "
                        f"valeurs non numériques"))

        if not samples:
            raise HTTPException(
                status_code=400,
                detail="Fichier CSV vide")

        # Prédire
        results_raw = pred.predict_batch(
            samples)

        predictions = []
        summary = {}
        for r in results_raw:
            alert = r["alert_level"]
            summary[alert] = summary.get(
                alert, 0) + 1
            predictions.append(
                PredictionResponse(
                    prediction=r["prediction"],
                    confidence=r["confidence"],
                    alert_level=r[
                        "alert_level"],
                    timestamp=r["timestamp"],
                ))

        t_ms = (time.time() - t_start) * 1000

        return BatchPredictionResponse(
            total=len(predictions),
            predictions=predictions,
            summary=summary,
            processing_time_ms=round(t_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e))