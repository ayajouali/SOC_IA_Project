# SOC_IA_Project
Plateforme de détection d'intrusions réseau (ML + LLM + Streamlit + FastAPI)
# 🛡️ SOC-IA — Détection d'Intrusions Réseau par IA

Plateforme de cybersécurité combinant **Machine Learning** et **LLM** pour détecter, expliquer et recommander des actions face aux attaques réseau.

## 🚀 Fonctionnalités

- 🎯 Détection multi-classes (LightGBM) — 9 types d'attaques (CIC-IDS2017)
- 🔍 Détection Zero-Day (Autoencoder)
- 🤖 Analyse intelligente par LLM (Groq / Llama)
- 📊 Dashboard Streamlit interactif
- 📄 Génération de rapports PDF
- ⚡ API FastAPI

## 🛠️ Stack technique

| Couche | Technologie |
|--------|-------------|
| Frontend | Streamlit |
| Backend API | FastAPI |
| ML | LightGBM, Autoencoder (Keras), Scikit-learn |
| LLM | Groq (Llama 3.1) |
| Dataset | CIC-IDS2017 |

## 📦 Installation

```bash
git clone https://github.com/VOTRE_USERNAME/SOC-IA.git
cd SOC-IA
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
