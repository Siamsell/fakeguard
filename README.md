---
title: Fake News Detector
emoji: 🔍
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🔍 Fake News Detector

Un modèle NLP de détection de fake news basé sur un **SVM Linéaire** avec une accuracy de **96.84%**.

## Description

Ce projet utilise le machine learning pour classifier automatiquement les articles de presse en deux catégories :
- ✅ **REAL NEWS** — Information vérifiée
- 🚨 **FAKE NEWS** — Fausse information

## Pipeline technique

1. **Preprocessing** : Nettoyage du texte, suppression des stopwords, stemming + lemmatization
2. **Feature Engineering** : TF-IDF Vectorization (5000 features)
3. **Modèle** : SVM Linéaire (LinearSVC + CalibratedClassifierCV)
4. **Accuracy** : 96.84% sur le jeu de test

## Utilisation

Colle simplement le texte d'un article en anglais dans la zone de texte et clique sur **Analyser**.

## Remarques

- Le modèle est entraîné sur des **news en anglais** uniquement
- Dataset : [Fake News Classification](https://www.kaggle.com/datasets/oussamaberredjem/fake-news-classification)
