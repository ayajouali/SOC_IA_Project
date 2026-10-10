"""
SOC-IA | Exemple de Client API
"""

import requests
import numpy as np
import json

API_URL = "http://localhost:8000"


def test_health():
    """Vérifier que l'API tourne."""
    r = requests.get(f"{API_URL}/health")
    print("Health :", r.json())


def test_predict():
    """Prédire un flux."""
    np.random.seed(42)
    features = np.random.randn(24).tolist()

    r = requests.post(
        f"{API_URL}/api/v1/predict",
        json={"features": features})

    data = r.json()
    print("\n--- Prediction ---")
    print(f"Classe    : {data['prediction']}")
    print(f"Confiance : "
          f"{data['confidence']*100:.1f}%")
    print(f"Alerte    : {data['alert_level']}")


def test_batch():
    """Prédire plusieurs flux."""
    np.random.seed(42)
    samples = np.random.randn(10, 24).tolist()

    r = requests.post(
        f"{API_URL}/api/v1/predict/batch",
        json={"samples": samples})

    data = r.json()
    print(f"\n--- Batch ({data['total']}) ---")
    print(f"Alertes : {data['summary']}")
    print(f"Temps   : "
          f"{data['processing_time_ms']:.1f}ms")


def test_models():
    """Infos modèles."""
    r = requests.get(
        f"{API_URL}/api/v1/models/info")
    data = r.json()
    print(f"\n--- Modeles ---")
    for m in data['models']:
        print(f"  {m['name']} : {m['role']}")


def test_features():
    """Top features."""
    r = requests.get(
        f"{API_URL}/api/v1/models/features")
    data = r.json()
    print(f"\n--- Features ---")
    for f in data['features'][:5]:
        print(f"  {f['rank']}. {f['name']} "
              f"({f['importance_pct']:.1f}%)")


if __name__ == "__main__":
    print("="*50)
    print("  SOC-IA Client Test")
    print("="*50)

    test_health()
    test_predict()
    test_batch()
    test_models()
    test_features()

    print("\n[OK] Tous les tests passes !")