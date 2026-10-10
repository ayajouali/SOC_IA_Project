"""
SOC-IA | Script de Prediction Production
"""

import json
import joblib
import numpy as np
import tensorflow as tf
from pathlib import Path


class SOC_IA_Predictor:
    """3 modeles complementaires."""

    def __init__(self, models_dir="./"):
        self.dir = Path(models_dir)

        with open(self.dir/"metadata.json",
                   "r", encoding='utf-8') as f:
            self.meta = json.load(f)

        with open(self.dir/"config.json",
                   "r") as f:
            self.config = json.load(f)

        self.classes = self.meta["classes"]["names"]
        self.features = self.meta["features"]["names"]
        self.n_features = self.meta["features"]["n_features"]
        self.threshold = self.meta["models"]["autoencoder"]["threshold"]

        print("Chargement des modeles...")
        self.lgbm = joblib.load(
            self.dir/"lightgbm_principal.joblib")
        print("  [OK] LightGBM")

        self.rf = joblib.load(
            self.dir/"randomforest_backup.joblib")
        print("  [OK] RandomForest")

        self.ae = tf.keras.models.load_model(
            self.dir/"autoencoder_zeroday.keras")
        print("  [OK] Autoencoder")

        self.scaler_rob = joblib.load(
            self.dir/"scaler_robust.joblib")
        self.scaler_mm = joblib.load(
            self.dir/"scaler_minmax.joblib")
        print("  [OK] Scalers")

        print("\nPredicteur pret !")

    def _get_alert_level(self, pred_class,
                          confidence,
                          disagreement,
                          anomaly):
        cls = self.classes[pred_class]
        alert_lvls = self.config["production"]["alert_levels"]

        if anomaly and cls == "BENIGN":
            return "HIGH"
        if disagreement:
            return "REVIEW"
        if cls in alert_lvls["CRITICAL"]:
            return "CRITICAL"
        if cls in alert_lvls["HIGH"]:
            return "HIGH"
        if cls in alert_lvls["MEDIUM"]:
            return "MEDIUM"
        if confidence < 0.85:
            return "REVIEW"
        return "INFO"

    def predict(self, X_raw, return_details=True):
        X_raw = np.asarray(X_raw, dtype=np.float32)
        if X_raw.ndim == 1:
            X_raw = X_raw.reshape(1, -1)

        if X_raw.shape[1] != self.n_features:
            raise ValueError(
                "Attendu " + str(self.n_features) +
                " features, recu " + str(X_raw.shape[1]))

        X_rob = self.scaler_rob.transform(X_raw).astype(np.float32)
        X_mm = self.scaler_mm.transform(X_raw).astype(np.float32)

        pred_lgbm = self.lgbm.predict(X_rob)
        proba_lgbm = self.lgbm.predict_proba(X_rob)
        conf_lgbm = np.max(proba_lgbm, axis=1)

        pred_rf = self.rf.predict(X_rob)
        proba_rf = self.rf.predict_proba(X_rob)

        recon = self.ae.predict(X_mm, verbose=0)
        err = np.mean((X_mm - recon)**2, axis=1)
        anomaly = err > self.threshold

        disagreement = pred_lgbm != pred_rf

        results = []
        for i in range(len(X_raw)):
            alert = self._get_alert_level(
                pred_lgbm[i], conf_lgbm[i],
                disagreement[i], anomaly[i])

            result = {
                "prediction": self.classes[int(pred_lgbm[i])],
                "confidence": float(conf_lgbm[i]),
                "alert_level": alert,
            }

            if return_details:
                result["details"] = {
                    "lightgbm": {
                        "class": self.classes[int(pred_lgbm[i])],
                        "confidence": float(conf_lgbm[i]),
                    },
                    "randomforest": {
                        "class": self.classes[int(pred_rf[i])],
                    },
                    "autoencoder": {
                        "anomaly": bool(anomaly[i]),
                        "reconstruction_error": float(err[i]),
                        "threshold": self.threshold,
                    },
                    "disagreement": bool(disagreement[i]),
                }

            results.append(result)

        return results

    def predict_one(self, sample):
        return self.predict(np.array([sample]))[0]

    def get_feature_importance(self, top_n=10):
        imp = self.rf.feature_importances_
        idx = np.argsort(imp)[::-1][:top_n]
        return [
            {
                "rank": i+1,
                "feature": self.features[idx[i]],
                "importance": float(imp[idx[i]])
            }
            for i in range(top_n)
        ]


if __name__ == "__main__":
    soc = SOC_IA_Predictor(".")

    np.random.seed(42)
    X_test = np.random.randn(5, 24).astype(np.float32)

    print("\n" + "="*60)
    print("  PREDICTIONS")
    print("="*60)

    results = soc.predict(X_test)

    for i, r in enumerate(results):
        print("\nFlux #" + str(i+1))
        print("  Prediction : " + r['prediction'])
        print("  Confiance  : " + str(round(r['confidence']*100,1)) + "%")
        print("  Alerte     : " + r['alert_level'])

        d = r['details']
        if d['disagreement']:
            print("  [!] Desaccord LGBM/RF")
            print("    LGBM : " + d['lightgbm']['class'])
            print("    RF   : " + d['randomforest']['class'])

        if d['autoencoder']['anomaly']:
            err = d['autoencoder']['reconstruction_error']
            print("  [!] Zero-Day detecte !")
            print("    Erreur recon : " + str(round(err,6)))

    print("\n" + "="*60)
    print("  TOP 5 FEATURES IMPORTANTES")
    print("="*60)
    for f in soc.get_feature_importance(5):
        print("  " + str(f['rank']) + ". " +
              f['feature'][:30] +
              " (" + str(round(f['importance']*100,2)) + "%)")
