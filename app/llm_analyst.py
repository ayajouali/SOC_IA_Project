"""
SOC-IA | Module d'Analyse par LLM
Utilise Groq + Llama 3.3 pour expliquer
les prédictions et donner des recommandations
"""

import json
import logging
from typing import (
    Dict, Any, Optional)

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

from app.config import settings

logger = logging.getLogger("SOC-IA-LLM")


class SOC_AI_Analyst:
    """
    Analyste IA basé sur LLM (Llama 3.3).
    Génère des analyses détaillées
    et des recommandations.
    """

    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
        self.model = settings.GROQ_MODEL
        self.client = None
        self.available = False

        if not GROQ_AVAILABLE:
            logger.warning(
                "Module 'groq' non installe. "
                "pip install groq")
            return

        if not self.api_key:
            logger.warning(
                "GROQ_API_KEY manquant "
                "dans .env")
            return

        try:
            self.client = Groq(
                api_key=self.api_key)
            self.available = True
            logger.info(
                f"[OK] LLM Analyst pret "
                f"({self.model})")
        except Exception as e:
            logger.error(
                f"Erreur init LLM : {e}")
            self.available = False

    def _build_prompt(
            self,
            prediction: str,
            confidence: float,
            alert_level: str,
            details: Dict[str, Any]
            ) -> str:
        """
        Construit le prompt pour le LLM.
        """
        ae_details = details.get(
            "autoencoder", {})
        ae_anomaly = ae_details.get(
            "is_anomaly", False)
        ae_error = ae_details.get(
            "reconstruction_error", 0)

        # Description des classes
        class_descriptions = {
            "BENIGN":
                "trafic réseau normal",
            "DDoS":
                "Distributed Denial of "
                "Service - submerge "
                "le serveur",
            "DoS":
                "Denial of Service - "
                "saturation des ressources",
            "PortScan":
                "scan de ports pour "
                "identifier les services",
            "Botnet":
                "réseau de machines "
                "compromises",
            "BruteForce":
                "tentatives de connexion "
                "par force brute",
            "WebAttack":
                "attaque ciblant les "
                "applications web",
            "Heartbleed":
                "exploitation faille "
                "OpenSSL",
            "Infiltration":
                "intrusion réseau "
                "sophistiquée",
        }

        attack_desc = class_descriptions.get(
            prediction,
            "type d'attaque inconnu")

        anomaly_str = (
            "OUI" if ae_anomaly else "NON")

        prompt = f"""Tu es un analyste SOC \
(Security Operations Center) expert en \
cybersécurité avec 15 ans d'expérience.

Analyse cette détection d'attaque \
réseau et fournis un rapport \
professionnel structuré.

## DÉTECTION

- Type détecté : {prediction} \
({attack_desc})
- Confiance modèle : {confidence*100:.2f}%
- Niveau d'alerte : {alert_level}
- Anomalie Autoencoder : {anomaly_str}
- Score d'anomalie : {ae_error:.6f}

## INSTRUCTIONS

Fournis un rapport structuré au format \
JSON valide avec les sections suivantes :

{{
  "summary": "Résumé en 2-3 phrases \
de l'incident",
  "severity": "CRITICAL ou HIGH ou \
MEDIUM ou LOW",
  "severity_reason": "Pourquoi ce niveau",
  "attack_type": "{prediction}",
  "attack_description": "Description \
technique en 1-2 phrases",
  "attack_vector": "Vecteur d'attaque \
probable",
  "risk_score": 8.5,
  "immediate_actions": [
    "Action 1",
    "Action 2",
    "Action 3"
  ],
  "investigation_steps": [
    "Étape 1",
    "Étape 2",
    "Étape 3"
  ],
  "preventive_measures": [
    "Mesure 1",
    "Mesure 2"
  ],
  "technical_context": "Explication \
technique de l'attaque en 2-3 phrases",
  "false_positive_likelihood": "FAIBLE \
ou MOYENNE ou ÉLEVÉE",
  "false_positive_reason": "Pourquoi"
}}

IMPORTANT :
- Réponds UNIQUEMENT avec le JSON valide
- Pas de texte avant ou après le JSON
- Tout en français
- Sois précis et professionnel
- risk_score entre 0 et 10
"""
        return prompt

    def analyze(
            self,
            prediction: str,
            confidence: float,
            alert_level: str,
            details: Dict[str, Any]
            ) -> Dict[str, Any]:
        """
        Analyse une prédiction avec le LLM.
        """
        if not self.available:
            return {
                "error": "LLM non disponible",
                "summary": (
                    "Le LLM n'est pas "
                    "configuré. Vérifiez "
                    "GROQ_API_KEY."),
            }

        try:
            prompt = self._build_prompt(
                prediction, confidence,
                alert_level, details)

            logger.info(
                f"Analyse LLM pour : "
                f"{prediction}")

            # Appel à Groq
            response = (
                self.client.chat.completions
                .create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Tu es un "
                                "analyste SOC "
                                "expert. "
                                "Réponds "
                                "uniquement "
                                "en JSON "
                                "valide.")
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3,
                    max_tokens=2048,
                    response_format={
                        "type": "json_object"
                    }
                )
            )

            # Récupérer le contenu
            content = (
                response.choices[0]
                .message.content)

            # Parser le JSON
            analysis = json.loads(content)

            # Ajouter métadonnées
            analysis["llm_model"] = self.model
            analysis["llm_provider"] = "Groq"

            logger.info("[OK] Analyse LLM")
            return analysis

        except json.JSONDecodeError as e:
            logger.error(
                f"JSON invalide : {e}")
            return {
                "error": "Reponse LLM "
                          "invalide",
                "raw_content": (
                    content
                    if 'content' in locals()
                    else "N/A"),
            }
        except Exception as e:
            logger.error(
                f"Erreur LLM : {e}")
            return {
                "error": str(e),
                "summary": (
                    "Erreur lors de "
                    "l'analyse LLM"),
            }

    def chat(
            self,
            question: str,
            context: Optional[str] = None
            ) -> str:
        """
        Mode chat libre avec le LLM.
        """
        if not self.available:
            return (
                "LLM non disponible. "
                "Vérifiez GROQ_API_KEY.")

        try:
            system_msg = (
                "Tu es un expert en "
                "cybersécurité et SOC. "
                "Réponds en français de "
                "manière professionnelle, "
                "concise et structurée.")

            if context:
                system_msg += (
                    f"\n\nContexte : "
                    f"{context}")

            response = (
                self.client.chat.completions
                .create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content":
                                system_msg
                        },
                        {
                            "role": "user",
                            "content":
                                question
                        }
                    ],
                    temperature=0.7,
                    max_tokens=1024,
                )
            )

            return (
                response.choices[0]
                .message.content)

        except Exception as e:
            return (
                f"Erreur : {str(e)}")


# Instance globale
analyst = SOC_AI_Analyst()