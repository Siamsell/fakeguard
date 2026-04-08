---
title: FakeGuard - Fake News Detector
emoji: 🛡️
colorFrom: gold
colorTo: red
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🛡️ FakeGuard – Détecteur de Fake News par SVM

**FakeGuard** est une application web de détection de fausses informations (fake news) utilisant un **modèle SVM (Support Vector Machine)** combiné à un pipeline NLP avancé.  
Avec une **précision de 96.84%**, l’outil analyse le contenu textuel d’un article en anglais et prédit s’il est **REAL** (vrai) ou **FAKE** (faux).

![Démonstration](https://img.shields.io/badge/demo-live-brightgreen) ![Accuracy](https://img.shields.io/badge/accuracy-96.8%25-gold) ![Python](https://img.shields.io/badge/python-3.10-blue) ![Gradio](https://img.shields.io/badge/gradio-4.44.0-orange)

---

## ✨ Fonctionnalités

- 🔍 **Analyse instantanée** d’un texte en anglais (article, tweet, blog, etc.)
- 🧠 **Modèle SVM** avec kernel linéaire (LinearSVC) + calibration probabiliste
- 📊 **Score de confiance** (entre 55% et 99%)
- 📝 **Visualisation des signaux linguistiques** détectés (ex: sensationalism, formalité, appel à l'émotion)
- 📈 **Métriques détaillées** : précision, rappel, F1-score, AUC-ROC, matrice de confusion
- 🧪 **Exemples intégrés** pour tester rapidement l’outil
- 📱 Interface responsive moderne (or et rubis)

---

## 🧠 Pipeline technique

| Étape | Description |
|-------|-------------|
| **1. Prétraitement NLP** | Nettoyage, suppression des stopwords, stemming (Porter) et lemmatisation (WordNet) |
| **2. Vectorisation** | TF-IDF (Term Frequency – Inverse Document Frequency) avec 5000 features |
| **3. Classification** | SVM linéaire (LinearSVC) avec `CalibratedClassifierCV` pour les probabilités |
| **4. Prédiction** | Sortie : `REAL` / `FAKE` + score de confiance + signaux d’alerte |

---

## 📊 Performance du modèle

| Métrique | Valeur |
|----------|--------|
| Précision globale | **96.84%** |
| F1-Score | **97%** |
| AUC-ROC | **0.99** |
| Recall (Fake) | **98%** |
| Recall (Real) | **95%** |

### Matrice de confusion (test set)

|                | Prédit FAKE | Prédit REAL |
|----------------|-------------|--------------|
| **Réel FAKE**  | 2 847 (TP)  | 148 (FN)     |
| **Réel REAL**  | 93 (FP)     | 2 712 (TN)   |

---

## 🚀 Utilisation

### En ligne (Hugging Face Spaces)
> *Lien vers l’espace une fois déployé*  
> `https://huggingface.co/spaces/[votre-nom]/fakeguard`

### Localement

```bash
# 1. Cloner le dépôt
git clone https://github.com/votre-nom/fakeguard.git
cd fakeguard

# 2. Créer un environnement virtuel (optionnel)
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sous Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l’application Gradio
python app.py
