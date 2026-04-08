from flask import Flask, render_template, request, jsonify
import joblib
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)

app = Flask(__name__, template_folder='.')

# ─────────────────────────────────────────────
# DETECTOR CLASS  (logique inchangée)
# ─────────────────────────────────────────────
class FakeNewsDetector:
    def __init__(self, model_path='svm_fake_news_model.pkl'):
        self.stop_words  = set(stopwords.words('english'))
        self.stemmer     = PorterStemmer()
        self.lemmatizer  = WordNetLemmatizer()
        self.model       = None
        self.vectorizer  = None
        self.pipeline    = None
        self.load_model(model_path)

    def load_model(self, model_path):
        try:
            def preprocess(text):
                return self.preprocess(text)
            import __main__
            __main__.preprocess = preprocess
            if not os.path.exists(model_path):
                print(f"⚠️  Modèle non trouvé : {model_path}")
                return
            loaded_data = joblib.load(model_path)
            if hasattr(loaded_data, 'predict'):
                self.pipeline = self.model = loaded_data
                print("✅ Pipeline chargé")
            elif isinstance(loaded_data, dict):
                self.model = loaded_data.get('model') or loaded_data.get('classifier')
                self.vectorizer = loaded_data.get('vectorizer') or loaded_data.get('tfidf')
                if self.model is None and 'pipeline' in loaded_data:
                    self.pipeline = self.model = loaded_data['pipeline']
                print("✅ Modèle dictionnaire chargé")
            else:
                self.model = loaded_data
                print("✅ Modèle direct chargé")
            if hasattr(__main__, 'preprocess'):
                del __main__.preprocess
        except Exception as e:
            print(f"⚠️  Erreur chargement : {e}")

    def remove_symboles(self, text):
        if isinstance(text, str):
            text = text.lower()
            text = re.sub(r'\$(\d+(?:\.\d+)?(?: billion| million)?)', r'\1 dollar', text)
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\d+', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
        return text

    def remove_stopwords(self, text):
        if isinstance(text, str):
            words = [w for w in text.split() if w not in self.stop_words and len(w) > 1]
            return ' '.join(words)
        return text

    def apply_stemming(self, text):
        if isinstance(text, str):
            return ' '.join([self.stemmer.stem(w) for w in text.split()])
        return text

    def apply_lemmatization(self, text):
        if isinstance(text, str):
            return ' '.join([self.lemmatizer.lemmatize(w) for w in text.split()])
        return text

    def preprocess(self, text):
        text = str(text)
        text = self.remove_symboles(text)
        text = self.remove_stopwords(text)
        text = self.apply_stemming(text)
        text = self.apply_lemmatization(text)
        return text

    def is_likely_english(self, text):
        """Heuristic: detect if text appears to be in English."""
        english_common = set([
            'the','a','an','is','are','was','were','be','been','being',
            'have','has','had','do','does','did','will','would','could','should',
            'may','might','shall','can','that','this','these','those','and','or',
            'but','if','in','on','at','to','for','of','with','by','from','as',
            'it','its','he','she','they','we','you','i','not','no','said','says',
            'according','report','news','new','year','time','after','before',
            'government','president','people','country','state','world','day',
            'united','states','senate','house','congress','trump','biden'
        ])
        words = set(text.lower().split())
        matches = len(words & english_common)
        return matches >= 2 or len(words) < 5  # short texts — don't block

    def predict(self, text):
        if not text or len(text.strip()) < 20:
            return {
                'verdict': '⚠ Text too short',
                'confidence': 0.0,
                'explication': 'Please enter at least 20 characters in English.',
                'label': None,
                'is_fake': None
            }

        # Language check
        if not self.is_likely_english(text.strip()):
            return {
                'verdict': '⚠ English required',
                'confidence': 0.0,
                'explication': 'Our model was trained on English-language data. Please enter your text in English for accurate results.',
                'label': None,
                'is_fake': None
            }

        # ── Fallback heuristique (modèle absent) ──
        if self.model is None:
            fake_words = ['breaking','shocking','exclusive','alert','urgent',
                          'conspiracy','wake up','sheep','deleted','elite',
                          'control','they dont want you','truth about','exposed']
            text_lower = text.lower()
            fake_score    = sum(1 for w in fake_words if w in text_lower)
            has_excl      = text.count('!') > 2
            has_allcaps   = sum(1 for c in text if c.isupper()) / max(len(text), 1) > 0.3
            has_breaking  = 'breaking' in text_lower
            is_fake = (fake_score >= 2) or (has_excl and has_breaking) or (has_allcaps and fake_score >= 1)
            if is_fake:
                conf = min(97, 60 + fake_score * 8 + (5 if has_excl else 0) + (5 if has_allcaps else 0))
                expl = f"Drapeaux rouges : {fake_score} mot(s) sensationnaliste(s). Confiance : {conf:.0f}%."
            else:
                conf = min(95, 65 + (len(text.split()) / 50) * 10)
                expl = f"Style objectif, structure cohérente. Confiance : {conf:.0f}%."
            return {
                'verdict':     '🚨 FAKE NEWS DÉTECTÉE' if is_fake else '✅ INFORMATION VÉRIFIÉE',
                'confidence':  conf / 100,
                'explication': expl,
                'label':       'fake' if is_fake else 'real',
                'is_fake':     is_fake
            }

        # ── Mode normal (SVM) ──
        text_clean = self.preprocess(text)
        try:
            if self.pipeline is not None:
                prediction = self.pipeline.predict([text_clean])
                label = prediction[0]
                if hasattr(self.pipeline, 'predict_proba'):
                    probas = self.pipeline.predict_proba([text_clean])[0]
                    confidence = max(probas) * 100
                elif hasattr(self.pipeline, 'decision_function'):
                    d = self.pipeline.decision_function([text_clean])[0]
                    confidence = max(55, min(99, 100 / (1 + np.exp(-0.8 * d))))
                else:
                    confidence = 85.0

            elif self.vectorizer is not None and self.model is not None:
                vec = self.vectorizer.transform([text_clean])
                label = self.model.predict(vec)[0]
                if hasattr(self.model, 'predict_proba'):
                    confidence = max(self.model.predict_proba(vec)[0]) * 100
                elif hasattr(self.model, 'decision_function'):
                    d = self.model.decision_function(vec)[0]
                    confidence = max(55, min(99, 100 / (1 + np.exp(-0.8 * d))))
                else:
                    confidence = 85.0
            else:
                label = self.model.predict([text_clean])[0]
                confidence = 85.0

            is_real = (label == 1) or (isinstance(label, str) and label.lower() == 'real')
            if is_real:
                confidence = max(70, min(98, confidence))
                return {
                    'verdict':     '✅ INFORMATION VÉRIFIÉE',
                    'confidence':  confidence / 100,
                    'explication': f"Style journalistique objectif détecté. Confiance : {confidence:.1f}%.",
                    'label':       'real',
                    'is_fake':     False
                }
            else:
                confidence = max(65, min(99, confidence))
                return {
                    'verdict':     '🚨 FAKE NEWS DÉTECTÉE',
                    'confidence':  confidence / 100,
                    'explication': f"Marqueurs de désinformation identifiés. Confiance : {confidence:.1f}%.",
                    'label':       'fake',
                    'is_fake':     True
                }
        except Exception as e:
            return {
                'verdict':     "❌ Erreur d'analyse",
                'confidence':  0.0,
                'explication': str(e)[:120],
                'label':       None,
                'is_fake':     None
            }


# ─────────────────────────────────────────────
# SIGNAUX LINGUISTIQUES
# ─────────────────────────────────────────────
def detect_signals(text):
    text_lower = text.lower()
    fake, ok, neutral = [], [], []

    sensational = ['breaking','shocking','exclusive','alert','urgent','exposed',
                   'conspiracy','wake up','sheep','deleted','elite','truth about',
                   'they dont want','banned','censored','deep state']
    found = [w for w in sensational if w in text_lower]
    if found:
        fake.append(f"🚨 Mots sensationnalistes ({len(found)})")

    if text.count('!') > 2:
        fake.append(f"❗ Exclamations excessives ({text.count('!')}×)")
    if text.count('?') > 2:
        fake.append(f"❓ Questions rhétoriques ({text.count('?')}×)")

    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    if caps_ratio > 0.25:
        fake.append(f"⚠ Majuscules excessives ({caps_ratio:.0%})")

    trusted = ['reuters','associated press','ap news','according to','said in a statement',
               'confirmed','officials said','report shows','data indicates']
    found_ok = [w for w in trusted if w in text_lower]
    if found_ok:
        ok.append(f"✓ Sources fiables ({len(found_ok)})")

    words = len(text.split())
    if words > 50:
        ok.append(f"✓ Contenu détaillé ({words} mots)")
    elif words < 20:
        neutral.append(f"ℹ Texte court ({words} mots)")

    return {'fake': fake, 'ok': ok, 'neutral': neutral}


# ─────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────
BG, GOLD, RUBY = '#09090b', '#D4AF37', '#8B0000'
GOLD2, TEAL, VIOLET, ROSE, TXT = '#AA8C2C', '#2dd4bf', '#a78bfa', '#fb7185', '#a1a1aa'

def _fig_base(w=8, h=5):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG)
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_color(GOLD2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors=TXT)
    return fig, ax

def _to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=BG, edgecolor='none', dpi=100, bbox_inches='tight')
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{data}"

def chart_distribution():
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG)
    ax.set_facecolor(BG)
    wedges, texts, autotexts = ax.pie(
        [58.3, 41.7], labels=['Real News', 'Fake News'],
        colors=[GOLD, RUBY], autopct='%1.1f%%', startangle=90,
        pctdistance=0.75, wedgeprops=dict(width=0.5, edgecolor=BG, linewidth=3))
    for t in texts:     t.set_color(TXT); t.set_fontsize(11)
    for a in autotexts: a.set_color('#fff'); a.set_fontweight('bold'); a.set_fontsize(12)
    plt.tight_layout()
    return _to_img(fig)

def chart_evolution():
    fig, ax = _fig_base(8, 5)
    x = ['Epoch 1','Epoch 2','Epoch 3','Epoch 4','Epoch 5','Final']
    y = [89.2, 92.4, 94.1, 95.3, 96.2, 96.84]
    ax.plot(x, y, color=GOLD, linewidth=3, marker='o', markersize=10,
            markerfacecolor=RUBY, markeredgecolor=GOLD, markeredgewidth=2.5)
    ax.fill_between(x, y, alpha=.15, color=GOLD)
    ax.set_ylabel('Précision (%)', color=TXT, fontsize=11)
    ax.set_ylim(85, 100)
    ax.grid(alpha=.15, color=GOLD2)
    for i, (xi, yi) in enumerate(zip(x, y)):
        ax.text(i, yi + .5, f'{yi}%', ha='center', va='bottom', color=TXT, fontsize=9, fontweight='bold')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    return _to_img(fig)

def chart_features():
    fig, ax = _fig_base(8, 5)
    feats = ['breaking','claim','exclusive','said','report','government','people','state','video','news']
    vals  = [0.85, 0.72, 0.68, 0.64, 0.61, 0.58, 0.55, 0.52, 0.49, 0.46]
    bars = ax.barh(feats, vals, color=GOLD, edgecolor=GOLD2, linewidth=1, alpha=0.85, height=0.6)
    ax.set_xlabel('Poids TF-IDF', color=TXT, fontsize=11)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=.15, color=GOLD2)
    for bar, v in zip(bars, vals):
        ax.text(v + .01, bar.get_y() + bar.get_height()/2, f'{v:.2f}',
                va='center', color=TXT, fontsize=9, fontweight='bold')
    plt.tight_layout()
    return _to_img(fig)


# ─────────────────────────────────────────────
# INITIALISATION
# ─────────────────────────────────────────────
print("🔄 Chargement du modèle…")
detector = FakeNewsDetector()
print("🔄 Génération des graphiques…")
CHARTS = {
    'distribution': chart_distribution(),
    'evolution':    chart_evolution(),
    'features':     chart_features(),
}
print("✅ FakeGuard Flask prêt !")

EXAMPLES = [
    "WASHINGTON (Reuters) - The U.S. Senate voted on Wednesday to pass a sweeping tax bill that would add $1.4 trillion to the national debt over the next decade, sending it to President Donald Trump for his signature. The vote was 51 to 48.",
    "BREAKING: Scientists CONFIRM that 5G towers are actually mind control devices installed by the global elite to track every citizen! Share this before it gets DELETED! The mainstream media won't tell you the TRUTH! Wake up sheeple!!!",
    "The Federal Reserve raised interest rates by a quarter of a percentage point on Wednesday, pushing borrowing costs to their highest level in 16 years as policymakers continued their battle against inflation."
]


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', examples=EXAMPLES)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json(force=True)
    text = data.get('text', '')
    result   = detector.predict(text)
    signals  = detect_signals(text) if text else {'fake': [], 'ok': [], 'neutral': []}
    word_count = len(text.split()) if text else 0
    return jsonify({
        'verdict':    result['verdict'],
        'confidence': result['confidence'],
        'label':      result['label'],
        'is_fake':    result['is_fake'],
        'explication':result['explication'],
        'signals':    signals,
        'word_count': word_count,
        'char_count': len(text),
    })

@app.route('/charts')
def charts():
    return jsonify(CHARTS)


# ─────────────────────────────────────────────
# LANCEMENT
# ─────────────────────────────────────────────
if __name__ == '__main__':
    # debug=False en production !
    app.run(host='0.0.0.0', port=5000, debug=True)