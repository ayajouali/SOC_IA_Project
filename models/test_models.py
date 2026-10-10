"""
SOC-IA | Script de Test
"""

import sys
import numpy as np
from predict import SOC_IA_Predictor


def test_chargement():
    print("\n[TEST 1] Chargement...")
    try:
        soc = SOC_IA_Predictor(".")
        print("  [OK] Modeles charges")
        return soc
    except Exception as e:
        print("  [FAIL] " + str(e))
        sys.exit(1)


def test_prediction_unique(soc):
    print("\n[TEST 2] Prediction unique...")
    try:
        X = np.random.randn(1, soc.n_features).astype(np.float32)
        result = soc.predict_one(X[0])
        print("  [OK] Classe : " + result['prediction'])
        print("  [OK] Alerte : " + result['alert_level'])
    except Exception as e:
        print("  [FAIL] " + str(e))


def test_prediction_batch(soc):
    print("\n[TEST 3] Prediction batch...")
    try:
        X = np.random.randn(100, soc.n_features).astype(np.float32)
        results = soc.predict(X)
        print("  [OK] " + str(len(results)) + " predictions")

        alerts = {}
        for r in results:
            lvl = r['alert_level']
            alerts[lvl] = alerts.get(lvl, 0) + 1
        print("  Distribution alertes :")
        for lvl, cnt in alerts.items():
            print("    " + lvl + ": " + str(cnt))
    except Exception as e:
        print("  [FAIL] " + str(e))


def test_feature_importance(soc):
    print("\n[TEST 4] Feature importance...")
    try:
        feats = soc.get_feature_importance(5)
        print("  [OK] Top 5 features :")
        for f in feats:
            print("    " + str(f['rank']) + ". " +
                  f['feature'][:20] +
                  " (" + str(round(f['importance']*100,1)) + "%)")
    except Exception as e:
        print("  [FAIL] " + str(e))


if __name__ == "__main__":
    print("="*60)
    print("  SOC-IA | TESTS DES MODELES")
    print("="*60)

    soc = test_chargement()
    test_prediction_unique(soc)
    test_prediction_batch(soc)
    test_feature_importance(soc)

    print("\n" + "="*60)
    print("  TESTS TERMINES")
    print("="*60)
