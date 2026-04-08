# 🛡️ FakeGuard – Détecteur de Fake News par SVM

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.9%2B-green)
![Flask](https://img.shields.io/badge/Flask-2.0-red)
![SVM](https://img.shields.io/badge/Model-SVM-gold)

**FakeGuard** est une application web de détection de fake news basée sur un **modèle SVM (Support Vector Machine)** associé à un pipeline NLP complet.  
Elle permet d'analyser un texte en anglais et de prédire s'il s'agit d'une information **réelle** ou **fausse**, avec un score de confiance associé.

> 🧠 Projet réalisé dans le cadre d'une démonstration de modèles de classification supervisée appliqués au traitement automatique du langage naturel.

---

## 🚀 Fonctionnalités principales

| Fonctionnalité | Description |
|----------------|-------------|
| ✅ **Analyse de texte** | Anglais uniquement, minimum 20 caractères |
| 🔍 **Pipeline NLP** | Stop-words, Stemming Porter, Lemmatisation WordNet, TF-IDF |
| 🤖 **Modèle SVM** | Entraîné sur 11 000+ articles |
| 📊 **Métriques performance** | Précision 96.8%, F1-score 97%, AUC-ROC 0.99 |
| 📈 **Visualisations** | Matrice de confusion, top features, distribution |
| 🧪 **Exemples intégrés** | Tests rapides avec textes prédéfinis |
| ⚡ **Interface responsive** | Design moderne, slideshow, ticker breaking news |

---

## 🧠 Pourquoi le modèle SVM ?

Trois modèles ont été évalués avant la sélection finale :

| Modèle | Précision | F1-score | Temps d'inférence | Décision |
|--------|-----------|----------|-------------------|----------|
| **SVM (linéaire)** | **96.8%** | **97%** | < 0.2 s | ✅ **Retenu** |
| Random Forest | 94.2% | 94% | 0.3 s | ❌ Non retenu |
| Logistic Regression | 93.5% | 93% | < 0.1 s | ❌ Non retenu |

**Le modèle SVM a été retenu** car il offre le meilleur compromis entre **précision**, **généralisation** et **rapidité** sur des textes courts et longs.  
Son hyperplan sépare efficacement les classes `REAL` et `FAKE` dans un espace vectoriel de haute dimension généré par TF-IDF.

---

## 🏗️ Architecture du projet
