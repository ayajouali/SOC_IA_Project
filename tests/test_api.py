"""
SOC-IA | Tests de l'API
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root():
    """Test page d'accueil."""
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["project"] == "SOC-IA"


def test_health():
    """Test health check."""
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["models_loaded"] == True


def test_predict_single():
    """Test prédiction unique."""
    import numpy as np
    np.random.seed(42)
    features = np.random.randn(24).tolist()

    r = client.post(
        "/api/v1/predict",
        json={"features": features})

    assert r.status_code == 200
    data = r.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "alert_level" in data
    assert data["confidence"] >= 0
    assert data["confidence"] <= 1


def test_predict_wrong_features():
    """Test avec mauvais nombre de features."""
    r = client.post(
        "/api/v1/predict",
        json={"features": [1.0, 2.0, 3.0]})

    assert r.status_code == 422


def test_predict_batch():
    """Test prédiction batch."""
    import numpy as np
    np.random.seed(42)
    samples = np.random.randn(5, 24).tolist()

    r = client.post(
        "/api/v1/predict/batch",
        json={"samples": samples})

    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 5
    assert len(data["predictions"]) == 5


def test_models_info():
    """Test infos modèles."""
    r = client.get("/api/v1/models/info")
    assert r.status_code == 200
    data = r.json()
    assert data["total_models"] >= 3


def test_features():
    """Test liste features."""
    r = client.get("/api/v1/models/features")
    assert r.status_code == 200
    data = r.json()
    assert data["total_features"] == 24


def test_classes():
    """Test liste classes."""
    r = client.get("/api/v1/models/classes")
    assert r.status_code == 200
    data = r.json()
    assert data["total_classes"] == 9


def test_stats():
    """Test statistiques."""
    r = client.get("/api/v1/stats")
    assert r.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])