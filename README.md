<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>FakeGuard — README</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&family=Instrument+Sans:wght@400;500&display=swap" rel="stylesheet" />
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg: #0a0a0b;
  --bg-2: #111113;
  --bg-3: #18181b;
  --bg-4: #1e1e22;
  --border: rgba(255,255,255,0.07);
  --border-2: rgba(255,255,255,0.12);
  --text: #e4e4e7;
  --text-2: #a1a1aa;
  --text-3: #71717a;
  --accent: #d4af37;
  --accent-dim: rgba(212,175,55,0.12);
  --accent-border: rgba(212,175,55,0.25);
  --red: #e05252;
  --red-dim: rgba(224,82,82,0.1);
  --green: #4ade80;
  --green-dim: rgba(74,222,128,0.1);
  --teal: #2dd4bf;
  --teal-dim: rgba(45,212,191,0.1);
  --mono: 'JetBrains Mono', monospace;
  --sans: 'Instrument Sans', sans-serif;
  --display: 'Syne', sans-serif;
}

html { scroll-behavior: smooth; }
body {
  font-family: var(--sans);
  background: var(--bg);
  color: var(--text);
  line-height: 1.7;
  font-size: 15px;
}

/* Layout */
.wrapper {
  max-width: 900px;
  margin: 0 auto;
  padding: 0 2rem 6rem;
}

/* Hero */
.hero {
  padding: 5rem 0 4rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 4rem;
}
.hero-label {
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 1.25rem;
  display: flex;
  align-items: center;
  gap: 10px;
}
.hero-label::after {
  content: '';
  flex: 1;
  max-width: 60px;
  height: 1px;
  background: var(--accent);
  opacity: 0.4;
}
.hero-title {
  font-family: var(--display);
  font-size: clamp(2.4rem, 6vw, 4rem);
  font-weight: 800;
  line-height: 1.05;
  letter-spacing: -0.02em;
  color: #fff;
  margin-bottom: 1.25rem;
}
.hero-title span {
  color: var(--accent);
}
.hero-desc {
  font-size: 16px;
  color: var(--text-2);
  max-width: 560px;
  line-height: 1.8;
  margin-bottom: 2rem;
}
.hero-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.badge {
  font-family: var(--mono);
  font-size: 11px;
  padding: 5px 12px;
  border-radius: 4px;
  border: 1px solid;
}
.badge-gold { color: var(--accent); border-color: var(--accent-border); background: var(--accent-dim); }
.badge-red  { color: var(--red);    border-color: rgba(224,82,82,0.25);  background: var(--red-dim); }
.badge-teal { color: var(--teal);   border-color: rgba(45,212,191,0.25); background: var(--teal-dim); }
.badge-green{ color: var(--green);  border-color: rgba(74,222,128,0.25); background: var(--green-dim); }

/* TOC */
.toc {
  background: var(--bg-2);
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent);
  border-radius: 0 8px 8px 0;
  padding: 1.5rem 2rem;
  margin-bottom: 4rem;
}
.toc-title {
  font-family: var(--display);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.15em;
  color: var(--text-3);
  margin-bottom: 1rem;
}
.toc ol {
  list-style: none;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 4px 2rem;
}
.toc a {
  font-family: var(--mono);
  font-size: 12px;
  color: var(--text-2);
  text-decoration: none;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 0;
  transition: color 0.2s;
}
.toc a:hover { color: var(--accent); }
.toc-num {
  color: var(--text-3);
  min-width: 18px;
}

/* Sections */
section { margin-bottom: 4rem; }

.section-label {
  font-family: var(--mono);
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.18em;
  color: var(--text-3);
  margin-bottom: 0.5rem;
}
.section-title {
  font-family: var(--display);
  font-size: 1.6rem;
  font-weight: 700;
  color: #fff;
  margin-bottom: 1.25rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid var(--border);
}
p { color: var(--text-2); margin-bottom: 1rem; }

/* Subsections */
.sub-title {
  font-family: var(--display);
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--text);
  margin: 2rem 0 0.75rem;
  display: flex;
  align-items: center;
  gap: 10px;
}
.sub-title::before {
  content: '';
  width: 4px;
  height: 14px;
  background: var(--accent);
  border-radius: 2px;
  flex-shrink: 0;
}

/* File tree */
.file-tree {
  font-family: var(--mono);
  font-size: 12.5px;
  background: var(--bg-3);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1.25rem 1.5rem;
  line-height: 2;
  color: var(--text-2);
  overflow-x: auto;
}
.file-tree .dir  { color: var(--accent); }
.file-tree .file { color: var(--text-2); }
.file-tree .comment { color: var(--text-3); }

/* Tables */
.table-wrap {
  overflow-x: auto;
  margin: 1rem 0 1.5rem;
  border: 1px solid var(--border);
  border-radius: 8px;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
thead tr {
  background: var(--bg-3);
  border-bottom: 1px solid var(--border-2);
}
th {
  font-family: var(--mono);
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--text-3);
  padding: 10px 16px;
  text-align: left;
  font-weight: 500;
}
td {
  padding: 10px 16px;
  border-bottom: 1px solid var(--border);
  color: var(--text-2);
  vertical-align: top;
}
tr:last-child td { border-bottom: none; }
tr:hover td { background: var(--bg-2); }
td code, th code {
  font-family: var(--mono);
  font-size: 11.5px;
  background: var(--bg-4);
  border: 1px solid var(--border);
  border-radius: 3px;
  padding: 1px 5px;
  color: var(--accent);
}
td strong { color: var(--text); font-weight: 500; }
.winner-row td { background: var(--accent-dim); }
.winner-row td:first-child { border-left: 2px solid var(--accent); }

/* Code blocks */
pre {
  background: var(--bg-3);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow-x: auto;
  margin: 1rem 0 1.5rem;
  position: relative;
}
.pre-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 14px;
  border-bottom: 1px solid var(--border);
  background: var(--bg-4);
  border-radius: 8px 8px 0 0;
}
.pre-lang {
  font-family: var(--mono);
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--text-3);
}
.pre-dots { display: flex; gap: 5px; }
.pre-dots span {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--border-2);
}
pre code {
  display: block;
  padding: 1.25rem 1.5rem;
  font-family: var(--mono);
  font-size: 12.5px;
  line-height: 1.9;
  color: var(--text-2);
  white-space: pre;
}
pre code .kw  { color: #c792ea; }
pre code .str { color: #c3e88d; }
pre code .cmt { color: var(--text-3); }
pre code .fn  { color: #82aaff; }
pre code .num { color: var(--accent); }

/* Inline code */
p code, li code {
  font-family: var(--mono);
  font-size: 12px;
  background: var(--bg-4);
  border: 1px solid var(--border-2);
  border-radius: 3px;
  padding: 1px 6px;
  color: var(--accent);
}

/* Pipeline steps */
.pipeline {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1px;
  background: var(--border);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
  margin: 1.25rem 0 1.5rem;
}
.pipe-step {
  background: var(--bg-2);
  padding: 1.25rem;
}
.pipe-n {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--accent);
  letter-spacing: 0.1em;
  margin-bottom: 0.4rem;
}
.pipe-name {
  font-family: var(--display);
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 0.35rem;
}
.pipe-desc {
  font-size: 12px;
  color: var(--text-3);
  line-height: 1.6;
}

/* Metrics row */
.metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px;
  margin: 1.25rem 0 1.5rem;
}
.metric-card {
  background: var(--bg-2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1.1rem 1.25rem;
}
.metric-label {
  font-family: var(--mono);
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--text-3);
  margin-bottom: 0.4rem;
}
.metric-value {
  font-family: var(--display);
  font-size: 1.6rem;
  font-weight: 700;
  color: #fff;
}
.metric-value.gold { color: var(--accent); }
.metric-value.teal { color: var(--teal); }
.metric-sub {
  font-size: 11px;
  color: var(--text-3);
  margin-top: 2px;
}

/* Feature grid */
.feature-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1px;
  background: var(--border);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
  margin: 1.25rem 0;
}
.feat-card {
  background: var(--bg-2);
  padding: 1.1rem 1.25rem;
  transition: background 0.2s;
}
.feat-card:hover { background: var(--bg-3); }
.feat-name {
  font-family: var(--display);
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 0.3rem;
}
.feat-desc {
  font-size: 12px;
  color: var(--text-3);
  line-height: 1.6;
}

/* API block */
.api-block {
  background: var(--bg-3);
  border: 1px solid var(--border);
  border-left: 3px solid var(--teal);
  border-radius: 0 8px 8px 0;
  padding: 1.1rem 1.5rem;
  margin-bottom: 1rem;
  font-family: var(--mono);
  font-size: 12px;
  line-height: 1.9;
}
.api-method {
  display: inline-block;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.08em;
  padding: 2px 7px;
  border-radius: 3px;
  margin-right: 8px;
}
.api-post { background: var(--accent-dim); color: var(--accent); }
.api-get  { background: var(--teal-dim);   color: var(--teal); }
.api-path { color: var(--text); }
.api-field { color: var(--text-2); padding-left: 1.5rem; }
.api-type  { color: var(--text-3); }

/* Callout */
.callout {
  border: 1px solid var(--accent-border);
  background: var(--accent-dim);
  border-radius: 8px;
  padding: 1rem 1.25rem;
  font-size: 13.5px;
  color: var(--text-2);
  margin: 1rem 0 1.5rem;
  display: flex;
  gap: 10px;
}
.callout-bar {
  width: 3px;
  background: var(--accent);
  border-radius: 2px;
  flex-shrink: 0;
}

/* Flow */
.flow {
  display: flex;
  align-items: center;
  gap: 0;
  flex-wrap: wrap;
  margin: 1.25rem 0 1.5rem;
  font-family: var(--mono);
  font-size: 12px;
}
.flow-step {
  background: var(--bg-3);
  border: 1px solid var(--border-2);
  border-radius: 6px;
  padding: 8px 14px;
  color: var(--text-2);
  white-space: nowrap;
}
.flow-arrow {
  color: var(--text-3);
  padding: 0 6px;
  font-size: 14px;
}

/* Steps list */
.steps { list-style: none; margin: 1rem 0 1.5rem; }
.steps li {
  display: flex;
  gap: 14px;
  margin-bottom: 12px;
  align-items: flex-start;
}
.step-num {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--accent);
  background: var(--accent-dim);
  border: 1px solid var(--accent-border);
  border-radius: 4px;
  padding: 3px 7px;
  flex-shrink: 0;
  margin-top: 1px;
}
.step-text { font-size: 13.5px; color: var(--text-2); line-height: 1.6; }
.step-text code {
  font-family: var(--mono);
  font-size: 11.5px;
  background: var(--bg-4);
  border: 1px solid var(--border-2);
  border-radius: 3px;
  padding: 1px 5px;
  color: var(--text);
}

/* Footer */
.footer {
  border-top: 1px solid var(--border);
  padding-top: 3rem;
  margin-top: 4rem;
}
.footer-inner {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  flex-wrap: wrap;
  gap: 1rem;
}
.footer-brand {
  font-family: var(--display);
  font-size: 1.2rem;
  font-weight: 800;
  color: #fff;
}
.footer-brand span { color: var(--accent); }
.footer-quote {
  font-family: var(--display);
  font-style: italic;
  font-size: 12px;
  color: var(--text-3);
  max-width: 360px;
  text-align: right;
}

/* Responsive */
@media (max-width: 600px) {
  .toc ol { grid-template-columns: 1fr; }
  .footer-inner { flex-direction: column; align-items: flex-start; }
  .footer-quote { text-align: left; }
  .flow { flex-direction: column; align-items: flex-start; }
  .flow-arrow { transform: rotate(90deg); }
}
</style>
</head>
<body>
<div class="wrapper">

  <!-- Hero -->
  <header class="hero">
    <div class="hero-label">Machine Learning — Project 8</div>
    <h1 class="hero-title">Fake<span>Guard</span></h1>
    <p class="hero-desc">
      An end-to-end fake news detection system built with a Linear SVM classifier,
      TF-IDF feature engineering, and a full-stack web interface for real-time inference.
    </p>
    <div class="hero-badges">
      <span class="badge badge-gold">SVM · 96.84% accuracy</span>
      <span class="badge badge-teal">TF-IDF · n-grams (1,2)</span>
      <span class="badge badge-red">Fake News Detection</span>
      <span class="badge badge-green">Python · scikit-learn · Flask</span>
    </div>
  </header>

  <!-- TOC -->
  <nav class="toc" aria-label="Table of contents">
    <div class="toc-title">Contents</div>
    <ol>
      <li><a href="#overview"><span class="toc-num">01</span> Overview</a></li>
      <li><a href="#structure"><span class="toc-num">02</span> Project Structure</a></li>
      <li><a href="#pipeline"><span class="toc-num">03</span> ML Pipeline</a></li>
      <li><a href="#models"><span class="toc-num">04</span> Model Comparison</a></li>
      <li><a href="#results"><span class="toc-num">05</span> Results</a></li>
      <li><a href="#interface"><span class="toc-num">06</span> Web Interface</a></li>
      <li><a href="#installation"><span class="toc-num">07</span> Installation</a></li>
      <li><a href="#usage"><span class="toc-num">08</span> Usage</a></li>
      <li><a href="#dataset"><span class="toc-num">09</span> Dataset</a></li>
      <li><a href="#dependencies"><span class="toc-num">10</span> Dependencies</a></li>
    </ol>
  </nav>

  <!-- Overview -->
  <section id="overview">
    <div class="section-label">01 / Overview</div>
    <h2 class="section-title">Overview</h2>
    <p>
      <strong style="color:var(--text)">FakeGuard</strong> is a complete machine learning project that trains, evaluates, and deploys a binary text classifier.
      The notebook <code>projet-8-ml.ipynb</code> covers the full workflow — from raw text preprocessing to hyperparameter optimization
      and statistical model comparison. The best model (Linear SVM, ~96.84% accuracy) is served through a polished web application built in <code>index.html</code>.
    </p>
    <p>
      Three classifiers were benchmarked — Support Vector Machine, Naive Bayes, and Logistic Regression — using GridSearchCV,
      10-fold cross-validation, ANOVA F-tests, and pairwise t-tests. The SVM with a linear kernel won on every metric that matters.
    </p>
  </section>

  <!-- Structure -->
  <section id="structure">
    <div class="section-label">02 / Structure</div>
    <h2 class="section-title">Project Structure</h2>
    <div class="file-tree">
<span class="dir">fakeguard/</span>
├── <span class="file">projet-8-ml.ipynb</span>       <span class="comment"># Full ML pipeline — training, evaluation, export</span>
├── <span class="file">index.html</span>              <span class="comment"># FakeGuard web interface — SVM inference</span>
├── <span class="file">app.py</span>                  <span class="comment"># Flask backend — /analyze and /charts endpoints</span>
├── <span class="file">requirements.txt</span>
├── <span class="dir">model/</span>
│   └── <span class="file">svm_pipeline.pkl</span>    <span class="comment"># Serialized pipeline (TF-IDF + Linear SVM)</span>
└── <span class="file">README.html</span>             <span class="comment"># This document</span>
    </div>
  </section>

  <!-- Pipeline -->
  <section id="pipeline">
    <div class="section-label">03 / Pipeline</div>
    <h2 class="section-title">ML Pipeline</h2>
    <p>The notebook follows a structured 9-step methodology from raw CSV to deployed model.</p>

    <div class="pipeline">
      <div class="pipe-step">
        <div class="pipe-n">STEP 01</div>
        <div class="pipe-name">Data Collection</div>
        <div class="pipe-desc">CSV import from Kaggle. Drop <code>title</code> and index columns.</div>
      </div>
      <div class="pipe-step">
        <div class="pipe-n">STEP 02</div>
        <div class="pipe-name">Text Preprocessing</div>
        <div class="pipe-desc">Normalization, stop word removal, stemming, lemmatization.</div>
      </div>
      <div class="pipe-step">
        <div class="pipe-n">STEP 03</div>
        <div class="pipe-name">Feature Engineering</div>
        <div class="pipe-desc">TF-IDF vectorization with 5,000 features and bigrams.</div>
      </div>
      <div class="pipe-step">
        <div class="pipe-n">STEP 04</div>
        <div class="pipe-name">Hyperparameter Search</div>
        <div class="pipe-desc">GridSearchCV across kernels, C, n-gram ranges, and df thresholds.</div>
      </div>
      <div class="pipe-step">
        <div class="pipe-n">STEP 05</div>
        <div class="pipe-name">Bias-Variance</div>
        <div class="pipe-desc">Train vs. validation gap analysis. Overfitting / underfitting flags.</div>
      </div>
      <div class="pipe-step">
        <div class="pipe-n">STEP 06</div>
        <div class="pipe-name">Statistical Comparison</div>
        <div class="pipe-desc">10-fold CV, ANOVA F-test, paired t-tests across all models.</div>
      </div>
      <div class="pipe-step">
        <div class="pipe-n">STEP 07</div>
        <div class="pipe-name">Training Benchmark</div>
        <div class="pipe-desc">Wall-clock training time measured and visualized per model.</div>
      </div>
      <div class="pipe-step">
        <div class="pipe-n">STEP 08</div>
        <div class="pipe-name">Evaluation</div>
        <div class="pipe-desc">Final accuracy on held-out test set (80/20 stratified split).</div>
      </div>
      <div class="pipe-step">
        <div class="pipe-n">STEP 09</div>
        <div class="pipe-name">Interpretation</div>
        <div class="pipe-desc">SVM coefficients, top-20 words, Word Clouds per class.</div>
      </div>
    </div>

    <h3 class="sub-title">Preprocessing details</h3>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Step</th><th>Description</th></tr></thead>
        <tbody>
          <tr><td><strong>Normalization</strong></td><td>Lowercase, currency expansion (<code>$5B &rarr; 5 dollar</code>), remove symbols and digits</td></tr>
          <tr><td><strong>Stop word removal</strong></td><td>NLTK English stop word list — words of length &le; 1 excluded</td></tr>
          <tr><td><strong>Stemming</strong></td><td>Porter Stemmer (reduces words to root form)</td></tr>
          <tr><td><strong>Lemmatization</strong></td><td>WordNet Lemmatizer (context-aware canonical forms)</td></tr>
        </tbody>
      </table>
    </div>

    <h3 class="sub-title">Hyperparameter search space</h3>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Model</th><th>Parameters</th><th>CV folds</th></tr></thead>
        <tbody>
          <tr><td><strong>SVM</strong></td><td><code>C &isin; {0.1, 1, 10}</code>, <code>kernel &isin; {linear, rbf}</code>, <code>max_df</code>, <code>min_df</code></td><td>3</td></tr>
          <tr><td><strong>Naive Bayes</strong></td><td><code>alpha &isin; {0.1, 0.3, 1}</code>, <code>ngram_range</code>, <code>max_df</code>, <code>min_df</code></td><td>5</td></tr>
          <tr><td><strong>Logistic Regression</strong></td><td><code>C &isin; {0.1, 1, 10}</code>, <code>solver=liblinear</code>, <code>ngram_range</code>, <code>max_df</code></td><td>5</td></tr>
        </tbody>
      </table>
    </div>
  </section>

  <!-- Models -->
  <section id="models">
    <div class="section-label">04 / Models</div>
    <h2 class="section-title">Model Comparison</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Model</th><th>Cross-Val Accuracy</th><th>Training Time</th><th>Notes</th></tr></thead>
        <tbody>
          <tr class="winner-row">
            <td><strong>SVM (Linear)</strong></td>
            <td><strong>~96.84%</strong></td>
            <td>Moderate</td>
            <td>Best overall — interpretable coefficients</td>
          </tr>
          <tr>
            <td>Logistic Regression</td>
            <td>Competitive</td>
            <td>Fast</td>
            <td>Strong baseline, linear decision boundary</td>
          </tr>
          <tr>
            <td>Naive Bayes</td>
            <td>Moderate</td>
            <td>Fastest</td>
            <td>Lowest accuracy; fastest inference</td>
          </tr>
        </tbody>
      </table>
    </div>
    <div class="callout">
      <div class="callout-bar"></div>
      <div>
        <strong style="color:var(--accent)">Winner: Linear SVM.</strong>
        Selected for deployment based on best bias-variance balance, highest cross-validation accuracy,
        and directly interpretable coefficients — unlike the RBF kernel variant which acts as a black box.
      </div>
    </div>
  </section>

  <!-- Results -->
  <section id="results">
    <div class="section-label">05 / Results</div>
    <h2 class="section-title">Results</h2>

    <div class="metrics">
      <div class="metric-card">
        <div class="metric-label">Test Accuracy</div>
        <div class="metric-value gold">96.84%</div>
        <div class="metric-sub">Linear SVM</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">TF-IDF Features</div>
        <div class="metric-value teal">5,000</div>
        <div class="metric-sub">max_features</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">N-gram Range</div>
        <div class="metric-value" style="font-size:1.2rem; color:#fff;">(1, 2)</div>
        <div class="metric-sub">Unigrams + Bigrams</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">CV Folds</div>
        <div class="metric-value" style="color:#fff;">10</div>
        <div class="metric-sub">Stratified K-Fold</div>
      </div>
    </div>

    <h3 class="sub-title">Linguistic signals</h3>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Class</th><th>Vocabulary Profile</th></tr></thead>
        <tbody>
          <tr><td><span class="badge badge-red" style="font-size:10px;">Fake</span></td><td>Emotionally charged, politically extreme, sensationalist, hyperbolic vocabulary</td></tr>
          <tr><td><span class="badge badge-green" style="font-size:10px;">Real</span></td><td>Factual, source-referenced, neutral, domain-specific terminology</td></tr>
        </tbody>
      </table>
    </div>
  </section>

  <!-- Interface -->
  <section id="interface">
    <div class="section-label">06 / Interface</div>
    <h2 class="section-title">Web Interface — FakeGuard</h2>
    <p>
      <code>index.html</code> is the frontend of the FakeGuard web application.
      It communicates with a Flask backend that exposes the trained SVM pipeline via REST API.
    </p>

    <h3 class="sub-title">Features</h3>
    <div class="feature-grid">
      <div class="feat-card">
        <div class="feat-name">Text Input</div>
        <div class="feat-desc">Paste any news article or headline. Minimum 20 characters. Live character counter with progress bar.</div>
      </div>
      <div class="feat-card">
        <div class="feat-name">SVM Analysis</div>
        <div class="feat-desc">Real-time classification via <code>POST /analyze</code>. Animated loading with NLP step indicators.</div>
      </div>
      <div class="feat-card">
        <div class="feat-name">Confidence Score</div>
        <div class="feat-desc">Percentage confidence displayed with an animated progress bar and verdict label.</div>
      </div>
      <div class="feat-card">
        <div class="feat-name">Linguistic Signals</div>
        <div class="feat-desc">Color-coded badges for detected fake / real / neutral keywords in the input text.</div>
      </div>
      <div class="feat-card">
        <div class="feat-name">Charts Panel</div>
        <div class="feat-desc">Distribution, feature importance, and evolution charts loaded via <code>GET /charts</code>.</div>
      </div>
      <div class="feat-card">
        <div class="feat-name">Responsive Design</div>
        <div class="feat-desc">Mobile-first layout with hamburger navigation. <code>Ctrl + Enter</code> shortcut to trigger analysis.</div>
      </div>
    </div>

    <h3 class="sub-title">NLP Processing Flow</h3>
    <div class="flow">
      <div class="flow-step">Input Text</div>
      <div class="flow-arrow">&#8594;</div>
      <div class="flow-step">NLP Preprocessing</div>
      <div class="flow-arrow">&#8594;</div>
      <div class="flow-step">TF-IDF Vectorization</div>
      <div class="flow-arrow">&#8594;</div>
      <div class="flow-step">SVM Prediction</div>
      <div class="flow-arrow">&#8594;</div>
      <div class="flow-step">Result + Confidence</div>
    </div>

    <h3 class="sub-title">API Endpoints</h3>

    <div class="api-block">
      <div><span class="api-method api-post">POST</span> <span class="api-path">/analyze</span></div>
      <div class="api-field">Body: <span class="api-type">{ "text": "..." }</span></div>
      <div class="api-field">Returns:</div>
      <div class="api-field" style="padding-left:3rem;">label: <span class="api-type">0 | 1</span></div>
      <div class="api-field" style="padding-left:3rem;">is_fake: <span class="api-type">bool</span></div>
      <div class="api-field" style="padding-left:3rem;">confidence: <span class="api-type">float</span></div>
      <div class="api-field" style="padding-left:3rem;">verdict, explication: <span class="api-type">str</span></div>
      <div class="api-field" style="padding-left:3rem;">word_count, char_count: <span class="api-type">int</span></div>
      <div class="api-field" style="padding-left:3rem;">signals: <span class="api-type">{ fake: [], ok: [], neutral: [] }</span></div>
    </div>

    <div class="api-block">
      <div><span class="api-method api-get">GET</span> <span class="api-path">/charts</span></div>
      <div class="api-field">Returns:</div>
      <div class="api-field" style="padding-left:3rem;">distribution: <span class="api-type">base64 PNG</span></div>
      <div class="api-field" style="padding-left:3rem;">features: <span class="api-type">base64 PNG</span></div>
      <div class="api-field" style="padding-left:3rem;">evolution: <span class="api-type">base64 PNG</span></div>
    </div>
  </section>

  <!-- Installation -->
  <section id="installation">
    <div class="section-label">07 / Installation</div>
    <h2 class="section-title">Installation</h2>

    <ol class="steps">
      <li>
        <span class="step-num">01</span>
        <div class="step-text">Clone the repository<br>
          <code>git clone https://github.com/your-username/fakeguard.git && cd fakeguard</code>
        </div>
      </li>
      <li>
        <span class="step-num">02</span>
        <div class="step-text">Create and activate a virtual environment<br>
          <code>python -m venv venv && source venv/bin/activate</code>
        </div>
      </li>
      <li>
        <span class="step-num">03</span>
        <div class="step-text">Install all dependencies<br>
          <code>pip install -r requirements.txt</code>
        </div>
      </li>
      <li>
        <span class="step-num">04</span>
        <div class="step-text">Run the notebook to train and export the model<br>
          <code>jupyter notebook projet-8-ml.ipynb</code>
        </div>
      </li>
      <li>
        <span class="step-num">05</span>
        <div class="step-text">Start the Flask backend<br>
          <code>python app.py</code> &mdash; then open <code>http://localhost:5000</code>
        </div>
      </li>
    </ol>
  </section>

  <!-- Usage -->
  <section id="usage">
    <div class="section-label">08 / Usage</div>
    <h2 class="section-title">Usage</h2>

    <h3 class="sub-title">Run the full ML pipeline</h3>
    <pre>
      <div class="pre-header">
        <div class="pre-dots"><span></span><span></span><span></span></div>
        <div class="pre-lang">bash</div>
      </div>
      <code>jupyter notebook projet-8-ml.ipynb
<span class="cmt"># Execute all cells in order.</span>
<span class="cmt"># Exports model/svm_pipeline.pkl on completion.</span></code>
    </pre>

    <h3 class="sub-title">Launch the web app</h3>
    <pre>
      <div class="pre-header">
        <div class="pre-dots"><span></span><span></span><span></span></div>
        <div class="pre-lang">bash</div>
      </div>
      <code>python app.py
<span class="cmt"># Listening on http://localhost:5000</span></code>
    </pre>

    <h3 class="sub-title">Programmatic inference</h3>
    <pre>
      <div class="pre-header">
        <div class="pre-dots"><span></span><span></span><span></span></div>
        <div class="pre-lang">python</div>
      </div>
      <code><span class="kw">import</span> pickle

<span class="kw">with</span> <span class="fn">open</span>(<span class="str">"model/svm_pipeline.pkl"</span>, <span class="str">"rb"</span>) <span class="kw">as</span> f:
    model = pickle.<span class="fn">load</span>(f)

text = <span class="str">"Scientists confirm groundbreaking climate discovery..."</span>
prediction = model.<span class="fn">predict</span>([text])
<span class="fn">print</span>(<span class="str">"FAKE"</span> <span class="kw">if</span> prediction[<span class="num">0</span>] == <span class="num">0</span> <span class="kw">else</span> <span class="str">"REAL"</span>)</code>
    </pre>
  </section>

  <!-- Dataset -->
  <section id="dataset">
    <div class="section-label">09 / Dataset</div>
    <h2 class="section-title">Dataset</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Property</th><th>Value</th></tr></thead>
        <tbody>
          <tr><td>Source</td><td>Kaggle — fake-news-classification (oussamaberredjem)</td></tr>
          <tr><td>Format</td><td>CSV, semicolon-separated</td></tr>
          <tr><td>Feature used</td><td><code>text</code> (article body)</td></tr>
          <tr><td>Target</td><td><code>label</code> — <code>0</code> = Fake, <code>1</code> = Real</td></tr>
          <tr><td>Split</td><td>80% train / 20% test, stratified by label</td></tr>
        </tbody>
      </table>
    </div>
  </section>

  <!-- Dependencies -->
  <section id="dependencies">
    <div class="section-label">10 / Dependencies</div>
    <h2 class="section-title">Dependencies</h2>
    <pre>
      <div class="pre-header">
        <div class="pre-dots"><span></span><span></span><span></span></div>
        <div class="pre-lang">requirements.txt</div>
      </div>
      <code>scikit-learn
pandas
numpy
nltk
matplotlib
scipy
wordcloud
flask</code>
    </pre>

    <pre>
      <div class="pre-header">
        <div class="pre-dots"><span></span><span></span><span></span></div>
        <div class="pre-lang">bash</div>
      </div>
      <code>pip install scikit-learn pandas numpy nltk matplotlib scipy wordcloud flask

<span class="cmt"># Required NLTK corpora</span>
python -c <span class="str">"import nltk; nltk.download('stopwords'); nltk.download('wordnet')"</span></code>
    </pre>
  </section>

  <!-- Footer -->
  <footer class="footer">
    <div class="footer-inner">
      <div>
        <div class="footer-brand">Fake<span>Guard</span></div>
        <div style="font-size:12px; color:var(--text-3); margin-top:4px; font-family:var(--mono);">
          Machine Learning — Project 8 &mdash; Linear SVM &mdash; 96.84% accuracy
        </div>
      </div>
      <div class="footer-quote">
        In a world flooded with misinformation, every classification matters.
      </div>
    </div>
  </footer>

</div>
</body>
</html>
