"""
SOC-IA | Tests du Predictor
"""

import pytest
import numpy as np
from app.predictor import SOC_IA_Predictor


@pytest.fixture(scope="module")
def predictor():
    """Fixture : charge le predictor une fois."""
    return SOC_IA_Predictor()


def test_loaded(predictor):
    """Test : modèles chargés."""
    assert predictor.loaded == True


def test_classes_count(predictor):
    """Test : 9 classes."""
    assert len(predictor.classes) == 9


def test_features_count(predictor):
    """Test : 24 features."""
    assert predictor.n_features == 24


def test_predict_single(predictor):
    """Test : prédiction unique."""
    np.random.seed(42)
    features = np.random.randn(24).tolist()
    result = predictor.predict_single(
        features)

    assert "prediction" in result
    assert "confidence" in result
    assert "alert_level" in result
    assert 0 <= result["confidence"] <= 1


def test_predict_wrong_features(predictor):
    """Test : mauvais nb features."""
    with pytest.raises(ValueError):
        predictor.predict_single(
            [1.0, 2.0, 3.0])


def test_predict_batch(predictor):
    """Test : prédiction batch."""
    np.random.seed(42)
    samples = [
        np.random.randn(24).tolist()
        for _ in range(5)
    ]
    results = predictor.predict_batch(
        samples)

    assert len(results) == 5
    for r in results:
        assert "prediction" in r


def test_alert_levels(predictor):
    """Test : niveaux d'alerte valides."""
    valid_levels = [
        "CRITICAL", "HIGH", "MEDIUM",
        "LOW", "INFO", "REVIEW"]

    np.random.seed(42)
    for _ in range(20):
        features = np.random.randn(24)\
            .tolist()
        result = predictor.predict_single(
            features)
        assert result["alert_level"] \
               in valid_levels


def test_stats_update(predictor):
    """Test : stats mises à jour."""
    initial = predictor.stats[
        "total_predictions"]

    np.random.seed(42)
    features = np.random.randn(24).tolist()
    predictor.predict_single(features)

    assert predictor.stats[
        "total_predictions"] == initial + 1


def test_get_models_info(predictor):
    """Test : infos modèles."""
    info = predictor.get_models_info()
    assert len(info) >= 3
    for m in info:
        assert "name" in m
        assert "performance" in m


def test_get_feature_importance(predictor):
    """Test : feature importance."""
    feats = predictor.get_feature_importance(
        top_n=10)
    assert len(feats) == 10
    assert feats[0]["importance"] >= \
           feats[-1]["importance"]


def test_get_classes_info(predictor):
    """Test : infos classes."""
    classes = predictor.get_classes_info()
    assert len(classes) == 9
    for c in classes:
        assert "name" in c
        assert "alert_level" in c


def test_get_stats(predictor):
    """Test : statistiques."""
    stats = predictor.get_stats()
    assert "total_predictions" in stats
    assert "uptime_seconds" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])