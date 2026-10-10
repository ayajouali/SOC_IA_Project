"""
SOC-IA | Client API pour Dashboard
Communication avec l'API FastAPI
"""

import requests
import json
from typing import (
    List, Dict, Any, Optional)

API_URL = "http://localhost:8000"
TIMEOUT = 10
TIMEOUT_LLM = 60


class SOC_IA_Client:
    """Client pour l'API SOC-IA."""

    def __init__(
            self,
            base_url: str = API_URL):
        self.base_url = base_url

    def health(self) -> Dict[str, Any]:
        """Vérifie le statut de l'API."""
        try:
            r = requests.get(
                f"{self.base_url}/health",
                timeout=TIMEOUT)
            return r.json()
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "models_loaded": False
            }

    def predict_single(
            self,
            features: List[float]
            ) -> Dict[str, Any]:
        """Prédire un seul flux."""
        try:
            r = requests.post(
                f"{self.base_url}"
                f"/api/v1/predict",
                json={
                    "features": features},
                timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def predict_batch(
            self,
            samples: List[List[float]]
            ) -> Dict[str, Any]:
        """Prédire plusieurs flux."""
        try:
            r = requests.post(
                f"{self.base_url}"
                f"/api/v1/predict/batch",
                json={"samples": samples},
                timeout=TIMEOUT * 6)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def predict_csv(
            self, file_path
            ) -> Dict[str, Any]:
        """Prédire depuis un fichier CSV."""
        try:
            with open(file_path,
                       "rb") as f:
                files = {"file": f}
                r = requests.post(
                    f"{self.base_url}"
                    f"/api/v1/predict/csv",
                    files=files,
                    timeout=TIMEOUT * 6)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def get_models_info(
            self) -> Dict[str, Any]:
        """Récupère infos modèles."""
        try:
            r = requests.get(
                f"{self.base_url}"
                f"/api/v1/models/info",
                timeout=TIMEOUT)
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def get_features(
            self) -> Dict[str, Any]:
        """Récupère les features."""
        try:
            r = requests.get(
                f"{self.base_url}"
                f"/api/v1/models/features",
                timeout=TIMEOUT)
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def get_classes(
            self) -> Dict[str, Any]:
        """Récupère les classes."""
        try:
            r = requests.get(
                f"{self.base_url}"
                f"/api/v1/models/classes",
                timeout=TIMEOUT)
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def get_stats(
            self) -> Dict[str, Any]:
        """Récupère les statistiques."""
        try:
            r = requests.get(
                f"{self.base_url}"
                f"/api/v1/stats",
                timeout=TIMEOUT)
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    # ============================
    # NOUVELLES MÉTHODES LLM
    # ============================

    def explain_with_llm(
            self,
            features: List[float]
            ) -> Dict[str, Any]:
        """
        Prédire + Expliquer avec LLM.
        """
        try:
            r = requests.post(
                f"{self.base_url}"
                f"/api/v1/llm/explain",
                json={
                    "features": features},
                timeout=TIMEOUT_LLM)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def chat_llm(
            self,
            question: str,
            context: Optional[str] = None
            ) -> Dict[str, Any]:
        """Chat libre avec LLM."""
        try:
            r = requests.post(
                f"{self.base_url}"
                f"/api/v1/llm/chat",
                json={
                    "question": question,
                    "context": context},
                timeout=TIMEOUT_LLM)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def llm_status(
            self) -> Dict[str, Any]:
        """Statut LLM."""
        try:
            r = requests.get(
                f"{self.base_url}"
                f"/api/v1/llm/status",
                timeout=TIMEOUT)
            return r.json()
        except Exception as e:
            return {
                "available": False,
                "error": str(e)
            }


# Instance globale
client = SOC_IA_Client()