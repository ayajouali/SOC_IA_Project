"""
SOC-IA | Classe Predictor
"""

import json
import time
import logging
import joblib
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SOC-IA")


class SOC_IA_Predictor:
    """Moteur de prédiction SOC-IA."""

    def __init__(self):
        self.loaded = False
        self.start_time = time.time()

        self.stats = {
            "total_predictions": 0,
            "per_class": {},
            "per_alert": {},
            "anomalies": 0,
            "disagreements": 0,
            "total_confidence": 0.0,
        }

        self._load_models()

    def _load_models(self):
        logger.info("Chargement des modeles...")

        try:
            with open(settings.METADATA_PATH,
                       "r", encoding="utf-8") as f:
                self.meta = json.load(f)

            with open(settings.CONFIG_PATH,
                       "r") as f:
                self.config = json.load(f)

            self.classes = self.meta[
                "classes"]["names"]
            self.features = self.meta[
                "features"]["names"]
            self.n_features = self.meta[
                "features"]["n_features"]
            self.threshold = self.meta[
                "models"]["autoencoder"][
                "threshold"]

            self.lgbm = joblib.load(
                settings.LGBM_PATH)
            logger.info("  [OK] LightGBM")

            self.rf = joblib.load(
                settings.RF_PATH)
            logger.info("  [OK] RandomForest")

            # ✅ Utiliser tf.keras
            self.ae = tf.keras.models.load_model(
                settings.AE_PATH)
            logger.info("  [OK] Autoencoder")

            self.scaler_rob = joblib.load(
                settings.SCALER_ROB_PATH)
            self.scaler_mm = joblib.load(
                settings.SCALER_MM_PATH)
            logger.info("  [OK] Scalers")

            for cls in self.classes:
                self.stats["per_class"][cls] = 0

            self.loaded = True
            logger.info(
                f"Modeles prets : "
                f"{len(self.classes)} classes, "
                f"{self.n_features} features")

        except Exception as e:
            logger.error(
                f"Erreur chargement : {e}")
            self.loaded = False
            raise e

    def _get_alert_level(self, pred_class,
                          confidence,
                          disagreement,
                          anomaly):
        cls = self.classes[pred_class]
        alerts = self.config[
            "production"]["alert_levels"]

        if anomaly and cls == "BENIGN":
            return "HIGH"
        if disagreement:
            return "REVIEW"
        if cls in alerts.get("CRITICAL", []):
            return "CRITICAL"
        if cls in alerts.get("HIGH", []):
            return "HIGH"
        if cls in alerts.get("MEDIUM", []):
            return "MEDIUM"
        if confidence < 0.85:
            return "REVIEW"
        return "INFO"

    def predict_single(self, features):
        if not self.loaded:
            raise RuntimeError(
                "Modeles non charges")

        X = np.array(
            [features], dtype=np.float32)

        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Attendu {self.n_features} "
                f"features, recu {X.shape[1]}")

        X_rob = self.scaler_rob.transform(
            X).astype(np.float32)
        X_mm = self.scaler_mm.transform(
            X).astype(np.float32)

        pred_lgbm = int(
            self.lgbm.predict(X_rob)[0])
        proba_lgbm = self.lgbm.predict_proba(
            X_rob)[0]
        conf = float(np.max(proba_lgbm))

        pred_rf = int(
            self.rf.predict(X_rob)[0])

        recon = self.ae.predict(
            X_mm, verbose=0)
        err = float(np.mean(
            (X_mm - recon)**2))
        anomaly = err > self.threshold

        disagreement = pred_lgbm != pred_rf

        alert = self._get_alert_level(
            pred_lgbm, conf,
            disagreement, anomaly)

        cls_name = self.classes[pred_lgbm]

        self.stats["total_predictions"] += 1
        self.stats["per_class"][cls_name] = \
            self.stats["per_class"].get(
                cls_name, 0) + 1
        self.stats["per_alert"][alert] = \
            self.stats["per_alert"].get(
                alert, 0) + 1
        self.stats["total_confidence"] += conf
        if anomaly:
            self.stats["anomalies"] += 1
        if disagreement:
            self.stats["disagreements"] += 1

        result = {
            "prediction": cls_name,
            "confidence": round(conf, 4),
            "alert_level": alert,
            "timestamp": datetime.now()\
                .isoformat(),
            "details": {
                "lightgbm": {
                    "model_name": "LightGBM",
                    "predicted_class": cls_name,
                    "confidence": round(conf, 4),
                },
                "randomforest": {
                    "model_name": "RandomForest",
                    "predicted_class":
                        self.classes[pred_rf],
                    "confidence": None,
                },
                "autoencoder": {
                    "is_anomaly": anomaly,
                    "reconstruction_error":
                        round(err, 6),
                    "threshold":
                        round(self.threshold, 6),
                },
                "models_agree":
                    not disagreement,
            }
        }

        return result

    def predict_batch(self, samples):
        results = []
        for sample in samples:
            result = self.predict_single(sample)
            results.append(result)
        return results

    def get_stats(self):
        total = self.stats["total_predictions"]
        avg_conf = 0.0
        if total > 0:
            avg_conf = (
                self.stats["total_confidence"]
                / total)

        return {
            "total_predictions": total,
            "predictions_per_class":
                dict(self.stats["per_class"]),
            "predictions_per_alert":
                dict(self.stats["per_alert"]),
            "avg_confidence":
                round(avg_conf, 4),
            "anomalies_detected":
                self.stats["anomalies"],
            "disagreements":
                self.stats["disagreements"],
            "uptime_seconds":
                round(time.time()
                      - self.start_time, 1),
        }

    def get_models_info(self):
        models_meta = self.meta["models"]
        models_list = []

        for name, info in models_meta.items():
            perf = info.get(
                "performance",
                info.get("performance_binary",
                          {}))
            file_path = settings.MODELS_DIR \
                        / info.get("file", "")
            file_size = 0
            if file_path.exists():
                file_size = round(
                    file_path.stat().st_size
                    / 1024**2, 2)

            models_list.append({
                "name": name,
                "role": info.get("role", ""),
                "type": info.get("type", ""),
                "performance": perf,
                "file_size_mb": file_size,
            })

        return models_list

    def get_feature_importance(self, top_n=24):
        imp = self.rf.feature_importances_
        idx = np.argsort(imp)[::-1][:top_n]

        features = []
        for rank, i in enumerate(idx, 1):
            features.append({
                "rank": rank,
                "name": self.features[i],
                "importance":
                    round(float(imp[i]), 6),
                "importance_pct":
                    round(float(imp[i])*100, 2),
            })
        return features

    def get_classes_info(self):
        alerts = self.config[
            "production"]["alert_levels"]

        classes = []
        for i, cls in enumerate(self.classes):
            alert_level = "INFO"
            for level, cls_list in \
                    alerts.items():
                if cls in cls_list:
                    alert_level = level
                    break

            classes.append({
                "id": i,
                "name": cls,
                "alert_level": alert_level,
            })
        return classes