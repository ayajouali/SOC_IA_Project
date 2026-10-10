"""
SOC-IA | Schémas Pydantic
Validation automatique des données entrantes/sortantes
"""

from pydantic import (
    BaseModel, Field, validator)
from typing import (
    List, Optional, Dict, Any)
from datetime import datetime


# ============================================================
# REQUÊTES (Input)
# ============================================================

class PredictionRequest(BaseModel):
    """Requête de prédiction pour 1 flux."""

    features: List[float] = Field(
        ...,
        description="24 features du flux réseau",
        min_length=24,
        max_length=24,
        examples=[[0.5, 1.2, 0.8, 0.3,
                    0.1, 0.9, 0.4, 0.7,
                    0.2, 0.6, 0.5, 0.8,
                    0.3, 0.1, 0.9, 0.4,
                    0.7, 0.2, 0.6, 0.5,
                    0.8, 0.3, 0.1, 0.9]]
    )

    @validator('features')
    def check_features_count(cls, v):
        if len(v) != 24:
            raise ValueError(
                f"Attendu 24 features, "
                f"reçu {len(v)}")
        return v


class BatchPredictionRequest(BaseModel):
    """Requête de prédiction pour N flux."""

    samples: List[List[float]] = Field(
        ...,
        description="Liste de flux "
                    "(chaque flux = 24 features)",
        min_length=1,
        max_length=10000,
    )

    @validator('samples')
    def check_all_samples(cls, v):
        for i, sample in enumerate(v):
            if len(sample) != 24:
                raise ValueError(
                    f"Flux #{i+1}: attendu 24 "
                    f"features, reçu {len(sample)}")
        return v


# ============================================================
# RÉPONSES (Output)
# ============================================================

class ModelDetail(BaseModel):
    """Détails d'un modèle dans la prédiction."""

    model_name: str
    predicted_class: str
    confidence: Optional[float] = None


class AutoencoderDetail(BaseModel):
    """Détails du score Autoencoder."""

    is_anomaly: bool
    reconstruction_error: float
    threshold: float


class PredictionDetails(BaseModel):
    """Détails complets d'une prédiction."""

    lightgbm: ModelDetail
    randomforest: ModelDetail
    autoencoder: AutoencoderDetail
    models_agree: bool


class PredictionResponse(BaseModel):
    """Réponse pour 1 flux."""

    prediction: str = Field(
        ...,
        description="Classe prédite")
    confidence: float = Field(
        ...,
        description="Confiance (0-1)")
    alert_level: str = Field(
        ...,
        description="Niveau d'alerte")
    timestamp: str = Field(
        default_factory=lambda:
            datetime.now().isoformat())
    details: Optional[PredictionDetails] = None


class BatchPredictionResponse(BaseModel):
    """Réponse pour N flux."""

    total: int
    predictions: List[PredictionResponse]
    summary: Dict[str, int] = Field(
        default_factory=dict,
        description="Compteur par alerte")
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Réponse du health check."""

    status: str
    version: str
    models_loaded: bool
    uptime_seconds: float
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Infos sur un modèle."""

    name: str
    role: str
    type: str
    performance: Dict[str, float]
    file_size_mb: float


class ModelsListResponse(BaseModel):
    """Liste de tous les modèles."""

    total_models: int
    models: List[ModelInfoResponse]
    best_model: str
    best_f1: float


class FeatureInfo(BaseModel):
    """Info sur une feature."""

    rank: int
    name: str
    importance: float
    importance_pct: float


class FeaturesResponse(BaseModel):
    """Liste des features."""

    total_features: int
    features: List[FeatureInfo]


class ClassInfo(BaseModel):
    """Info sur une classe."""

    id: int
    name: str
    alert_level: str


class ClassesResponse(BaseModel):
    """Liste des classes."""

    total_classes: int
    classes: List[ClassInfo]


class StatsResponse(BaseModel):
    """Statistiques d'utilisation."""

    total_predictions: int
    predictions_per_class: Dict[str, int]
    predictions_per_alert: Dict[str, int]
    avg_confidence: float
    anomalies_detected: int
    disagreements: int
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Réponse d'erreur."""

    error: str
    detail: str
    status_code: int