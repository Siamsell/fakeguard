# 🛡️ FakeGuard — Fake News Detection with SVM

> A complete Machine Learning pipeline for automated fake news detection, powered by a Support Vector Machine (SVM) and exposed through an interactive web interface.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [ML Pipeline](#ml-pipeline)
- [Models Compared](#models-compared)
- [Results](#results)
- [Web Interface — FakeGuard](#web-interface--fakeguard)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Dependencies](#dependencies)

---

## Overview

**FakeGuard** is an end-to-end machine learning project that trains, evaluates, and deploys a fake news classifier. The notebook (`projet-8-ml.ipynb`) covers the full ML workflow — from raw text preprocessing to hyperparameter optimization and statistical model comparison — and the best model (Linear SVM, ~96.84% accuracy) is served through a polished web application (`index.html`).

---

## Project Structure

```
├── projet-8-ml.ipynb       # Full ML pipeline (training, evaluation, export)
├── index.html              # FakeGuard web interface (SVM inference)
├── app.py                  # Flask/FastAPI backend (serves /analyze & /charts)
├── model/
│   ├── svm_pipeline.pkl    # Serialized best SVM pipeline (TF-IDF + SVM)
│   └── ...
└── README.md
```

---

## ML Pipeline

The notebook follows a structured 9-step methodology:

### Step 1 — Data Collection
- Dataset: `fake-news-classification` (Kaggle)
- Format: CSV with `text` and `label` columns (0 = Fake, 1 = Real)
- Columns `Unnamed: 0` and `title` dropped

### Step 2 — Text Preprocessing
A custom `preprocess()` function applies the following sequentially:

| Step | Description |
|---|---|
| **Normalization** | Lowercase, currency expansion (`$5B → 5 dollar`), symbol removal |
| **Stop word removal** | NLTK English stop word list; words of length ≤ 1 excluded |
| **Stemming** | Porter Stemmer |
| **Lemmatization** | WordNet Lemmatizer |

### Step 3 — Feature Engineering
- **TF-IDF Vectorization** with `max_features=5000`
- N-gram range tuned via Grid Search: `(1, 2)` (unigrams + bigrams)

### Step 4 — Hyperparameter Optimization (GridSearchCV)

| Model | Parameters Searched |
|---|---|
| **SVM** | `C ∈ {0.1, 1, 10}`, `kernel ∈ {linear, rbf}`, `max_df`, `min_df` |
| **Naive Bayes** | `alpha ∈ {0.1, 0.3, 1}`, `ngram_range`, `max_df`, `min_df` |
| **Logistic Regression** | `C ∈ {0.1, 1, 10}`, `solver=liblinear`, `ngram_range`, `max_df` |

All searches used `cv=3` (SVM) or `cv=5` (NB, LR) with `n_jobs=-1`.

### Step 5 — Bias-Variance Analysis
Training vs. validation scores compared for each model. A gap > 0.10 flags overfitting (High Variance); scores < 0.75 flag underfitting (High Bias).

### Step 6 — Statistical Comparison
- **10-Fold Stratified Cross-Validation** on training set
- **ANOVA (F-test)** across all three models
- **Paired t-tests** for pairwise comparisons (LR vs NB, LR vs SVM, NB vs SVM)

### Step 7 — Training Time Benchmark
Wall-clock training time measured and compared visually via bar chart.

### Step 8 — Variable Interpretation
- Linear SVM coefficients extracted to identify the **top 20 words** driving FAKE vs REAL predictions
- **Word Clouds** generated for each class from the cleaned corpus

---

## Models Compared

| Model | Cross-Val Accuracy | Notes |
|---|---|---|
| Support Vector Machine | **~96.84%** ✅ | Best overall — linear kernel, interpretable coefficients |
| Logistic Regression | High | Fast, competitive |
| Naive Bayes | Moderate | Fastest training time |

> **Winner: Linear SVM** — selected for deployment based on best bias-variance balance, highest cross-validation accuracy, and directly interpretable coefficients (unlike RBF kernel which acts as a black box).

---

## Results

| Metric | SVM (Best Model) |
|---|---|
| Test Accuracy | **~96.84%** |
| Kernel | `linear` |
| Regularization C | Tuned via GridSearchCV |
| TF-IDF N-gram range | `(1, 2)` |
| Features | 5 000 TF-IDF features |

Key linguistic signals found by the model:
- **Fake News** words: emotionally charged, politically extreme, sensationalist vocabulary
- **Real News** words: factual, source-referenced, neutral terminology

---

## Web Interface — FakeGuard

The `index.html` file is the frontend of the **FakeGuard** web application. It communicates with a backend server that exposes the trained SVM model.

### Features

| Feature | Description |
|---|---|
| 📝 Text Input | Paste any news article or headline (min. 20 characters) |
| ⚡ SVM Analysis | Real-time classification via `POST /analyze` |
| 📊 Confidence Score | Percentage confidence displayed with an animated bar |
| 🏷️ Linguistic Signals | Color-coded badges for detected fake/real/neutral keywords |
| 📈 Charts Panel | Distribution, feature importance, and evolution charts via `GET /charts` |
| 📋 Copy Verdict | One-click copy of the classification result |
| 📱 Responsive Design | Mobile-friendly with hamburger navigation |
| ⌨️ Keyboard Shortcut | `Ctrl + Enter` to trigger analysis |

### API Endpoints (Backend Required)

```
POST /analyze
  Body:    { "text": "..." }
  Returns: { "label": 0|1, "is_fake": bool, "confidence": float,
             "verdict": str, "explication": str,
             "word_count": int, "char_count": int,
             "signals": { "fake": [], "ok": [], "neutral": [] } }

GET /charts
  Returns: { "distribution": "<base64_img>",
             "features": "<base64_img>",
             "evolution": "<base64_img>" }
```

### NLP Processing Flow (Frontend Display)

```
Input Text → NLP Preprocessing → TF-IDF Vectorization → SVM Prediction → Result
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/fakeguard.git
cd fakeguard

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the notebook to train and export the model
jupyter notebook projet-8-ml.ipynb

# 5. Start the backend server
python app.py

# 6. Open the web interface
# Navigate to http://localhost:5000 in your browser
```

---

## Usage

### Run the full ML pipeline
```bash
jupyter notebook projet-8-ml.ipynb
```
Execute all cells in order. The best SVM pipeline will be serialized to `model/svm_pipeline.pkl`.

### Launch the web app
```bash
python app.py
# → http://localhost:5000
```

### Analyze text via CLI
```python
import pickle

with open("model/svm_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

text = "Breaking: Scientists confirm major discovery..."
prediction = model.predict([text])
print("FAKE" if prediction[0] == 0 else "REAL")
```

---

## Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — Fake News Classification](https://www.kaggle.com/datasets/oussamaberredjem/fake-news-classification) |
| Format | CSV (semicolon-separated) |
| Features used | `text` (article body) |
| Target | `label` (0 = Fake, 1 = Real) |
| Split | 80% train / 20% test (stratified) |

---

## Dependencies

```
scikit-learn
pandas
numpy
nltk
matplotlib
scipy
wordcloud
pickle
flask          # or fastapi / uvicorn for the backend
```

Install all at once:
```bash
pip install scikit-learn pandas numpy nltk matplotlib scipy wordcloud flask
```

Download required NLTK data:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## Author

Project developed as part of a Machine Learning course (Projet 8).  
Frontend: **FakeGuard** — Fake News Detection Interface.  
Model: **Linear SVM** with TF-IDF (n-gram) features, ~96.84% accuracy.

---

> *"In a world flooded with misinformation, every classification matters."*
