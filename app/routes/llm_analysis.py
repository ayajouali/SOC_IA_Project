"""
SOC-IA | Endpoints LLM Analysis
"""

from fastapi import (
    APIRouter, HTTPException)
from pydantic import (
    BaseModel, Field, validator)
from typing import (
    List, Optional, Dict, Any)

from app.llm_analyst import analyst

router = APIRouter(
    prefix="/api/v1/llm",
    tags=["LLM Analysis"])


# ============================================================
# SCHEMAS
# ============================================================

class ExplainRequest(BaseModel):
    """Requête pour analyse LLM."""

    features: List[float] = Field(
        ...,
        description="24 features",
        min_length=24,
        max_length=24)

    @validator('features')
    def check_features_count(cls, v):
        if len(v) != 24:
            raise ValueError(
                f"Attendu 24 features, "
                f"reçu {len(v)}")
        return v


class ChatRequest(BaseModel):
    """Requête pour chat libre."""

    question: str = Field(
        ...,
        description="Question",
        min_length=1,
        max_length=2000)
    context: Optional[str] = Field(
        None,
        description="Contexte optionnel")


class ChatResponse(BaseModel):
    """Réponse de chat."""

    response: str
    model: str


class LLMStatusResponse(BaseModel):
    """Statut du LLM."""

    available: bool
    model: str
    provider: str


# ============================================================
# ENDPOINTS
# ============================================================

@router.get(
    "/status",
    response_model=LLMStatusResponse,
    summary="Statut du LLM",
    description="Vérifie si le LLM "
                "est disponible")
async def llm_status():
    """Vérifie le statut du LLM."""
    return LLMStatusResponse(
        available=analyst.available,
        model=analyst.model,
        provider="Groq")


@router.post(
    "/explain",
    summary="Prédire + Expliquer avec LLM",
    description="Fait une prédiction et "
                "génère une analyse "
                "détaillée par LLM")
async def explain_prediction(
        request: ExplainRequest):
    """Prédiction + Analyse LLM."""
    from app.main import predictor

    if not predictor.loaded:
        raise HTTPException(
            status_code=503,
            detail="Modeles non charges")

    if not analyst.available:
        raise HTTPException(
            status_code=503,
            detail=(
                "LLM non disponible. "
                "Verifiez GROQ_API_KEY "
                "dans .env"))

    try:
        # 1. Faire la prédiction
        prediction = (
            predictor.predict_single(
                request.features))

        # 2. Analyser avec LLM
        analysis = analyst.analyze(
            prediction=prediction[
                "prediction"],
            confidence=prediction[
                "confidence"],
            alert_level=prediction[
                "alert_level"],
            details=prediction["details"])

        # 3. Combiner les résultats
        return {
            "prediction": prediction,
            "llm_analysis": analysis,
        }

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e))


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat libre avec LLM",
    description="Pose une question "
                "au LLM expert SOC")
async def chat_with_llm(
        request: ChatRequest):
    """Chat libre avec le LLM."""

    if not analyst.available:
        raise HTTPException(
            status_code=503,
            detail="LLM non disponible")

    try:
        response = analyst.chat(
            question=request.question,
            context=request.context)

        return ChatResponse(
            response=response,
            model=analyst.model)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e))