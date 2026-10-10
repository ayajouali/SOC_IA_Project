# SOC-IA — Modeles de Production

> Detection automatique de cyberattaques
> basee sur 3 modeles IA complementaires

## Contenu du Package

### Modeles Entraines

| Fichier | Description | Taille | Performance |
|---------|-------------|--------|-------------|
| `lightgbm_principal.joblib` | LightGBM (Principal) | 2.2 MB | F1=99.78% |
| `randomforest_backup.joblib` | RandomForest (Backup) | 3.1 MB | F1=99.62% |
| `autoencoder_zeroday.keras` | Autoencoder (Zero-Day) | 0.4 MB | F1=70.84% |

### Scalers

| Fichier | Type | Utilise par |
|---------|------|-------------|
| `scaler_robust.joblib` | RobustScaler | LightGBM, RandomForest |
| `scaler_minmax.joblib` | MinMaxScaler | Autoencoder |

### Fichiers Support

| Fichier | Description |
|---------|-------------|
| `metadata.json` | Metadonnees JSON |
| `config.json` | Configuration production |
| `predict.py` | Script de prediction |
| `test_models.py` | Script de test |
| `requirements.txt` | Dependances |
| `feature_importance.csv` | Importance features |

## Performances Detaillees

### LightGBM (Modele Principal)
- Accuracy    : 99.78%
- F1-Weighted : 99.78%
- F1-Macro    : 99.78%
- Precision   : 99.78%
- Recall      : 99.78%
- Temps       : 43.4s

### RandomForest (Backup)
- Accuracy    : 99.62%
- F1-Weighted : 99.62%
- F1-Macro    : 98.79%
- Temps       : 7.7s
- Interpretable (feature importance)

### Autoencoder (Zero-Day)
- Accuracy    : 61.93%
- F1-Score    : 70.84%
- Precision   : 96.94%
- Recall      : 55.81%
- Seuil       : 0.010398
- Temps       : 43.9s

## Installation

```bash
pip install -r requirements.txt
```

## Test Rapide

```bash
python test_models.py
```

## Utilisation Python

```python
from predict import SOC_IA_Predictor
import numpy as np

soc = SOC_IA_Predictor('.')
X = np.random.randn(10, 24)
results = soc.predict(X)

for r in results:
    print(r['prediction'])
    print(r['alert_level'])
```

## Classes d'Attaques Detectees

- **BENIGN**
- **Botnet**
- **BruteForce**
- **DDoS**
- **DoS**
- **Heartbleed**
- **Infiltration**
- **PortScan**
- **WebAttack**

## Features (24)

1. `Packet Length Mean`
2. `Init_Win_bytes_backward`
3. `Total Length of Fwd Packets`
4. `Fwd Packet Length Max`
5. `Packet Length Variance`
6. `Fwd Packet Length Mean`
7. `Bwd Packets/s`
8. `Flow IAT Max`
9. `Flow Bytes/s`
10. `Max Packet Length`
11. `Init_Win_bytes_forward`
12. `Flow Duration`
13. `Destination Port`
14. `Flow IAT Mean`
15. `Bwd Packet Length Mean`
16. `Bwd Packet Length Max`
17. `Fwd Packet Length Std`
18. `Total Backward Packets`
19. `Total Fwd Packets`
20. `Fwd IAT Std`
21. `min_seg_size_forward`
22. `act_data_pkt_fwd`
23. `Fwd IAT Min`
24. `Flow IAT Min`

## Architecture

```
Flux Reseau --> Scalers
                  |
        +---------+---------+
        |         |         |
     LightGBM  RandomF  Autoencoder
        |         |         |
        +---------+---------+
                  |
            Moteur SOC
                  |
              Alertes
```

## Niveaux d'Alerte

| Niveau | Description | Classes |
|--------|-------------|---------|
| CRITICAL | Action immediate | Heartbleed, Infiltration |
| HIGH | Attaque serieuse | DDoS, DoS, Botnet |
| MEDIUM | A surveiller | BruteForce, WebAttack, PortScan |
| LOW | Information | BENIGN |
| REVIEW | Desaccord modeles | Analyse manuelle |
| INFO | Aucune action | Trafic normal |

## Dataset

- Source : CIC-IDS2017
- Train  : 20,399 echantillons
- Val    : 2,915 echantillons
- Test   : 5,829 echantillons

## Prochaines Etapes

- [ ] API REST FastAPI
- [ ] Dashboard Grafana
- [ ] Integration LLM
- [ ] Container Docker

---

**SOC-IA v1.0.0** | Cree le 2026-06-14