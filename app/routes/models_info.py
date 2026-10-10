"""
SOC-IA | Endpoints Information Modèles
"""

from fastapi import APIRouter

from app.schemas import (
    ModelsListResponse,
    ModelInfoResponse,
    FeaturesResponse,
    FeatureInfo,
    ClassesResponse,
    ClassInfo)

router = APIRouter(
    prefix="/api/v1/models",
    tags=["Models Info"])


def get_predictor():
    from app.main import predictor
    return predictor


@router.get(
    "/info",
    response_model=ModelsListResponse,
    summary="Informations sur les modèles",
    description="Retourne les performances "
                "et métadonnées de chaque modèle")
async def models_info():
    """Liste des modèles et performances."""
    pred = get_predictor()
    models = pred.get_models_info()

    return ModelsListResponse(
        total_models=len(models),
        models=[
            ModelInfoResponse(**m)
            for m in models],
        best_model="LightGBM",
        best_f1=pred.meta["ranking"][
            "best_f1"],
    )


@router.get(
    "/features",
    response_model=FeaturesResponse,
    summary="Liste des 24 features",
    description="Retourne les features "
                "utilisées avec leur importance")
async def features_list():
    """Features et importance."""
    pred = get_predictor()
    features = pred.get_feature_importance()

    return FeaturesResponse(
        total_features=pred.n_features,
        features=[
            FeatureInfo(**f)
            for f in features],
    )


@router.get(
    "/classes",
    response_model=ClassesResponse,
    summary="Classes d'attaques détectables",
    description="Retourne les 9 classes "
                "avec leur niveau d'alerte")
async def classes_list():
    """Classes d'attaques."""
    pred = get_predictor()
    classes = pred.get_classes_info()

    return ClassesResponse(
        total_classes=len(classes),
        classes=[
            ClassInfo(**c)
            for c in classes],
    )