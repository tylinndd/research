import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow import keras
from tensorflow.keras import layers
import urllib.request
import zipfile
import os
import warnings
warnings.filterwarnings('ignore')

# ─── 1. Download & Load SMS Spam Dataset ─────────────────────────────────────
DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
ZIP_PATH = 'smsspamcollection.zip'
TSV_NAME = 'SMSSpamCollection'

if not os.path.exists(TSV_NAME):
    print("Downloading SMS Spam dataset …")
    urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall('.')
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)

df = pd.read_csv(TSV_NAME, sep='\t', header=None, names=['label', 'text'],
                 encoding='latin-1')
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

print(f"Dataset: SMS Spam Collection")
print(f"Samples: {len(df)}  |  ham: {(df['label']==0).sum()}  spam: {(df['label']==1).sum()}")

# ─── 2. Split 70 / 15 / 15 (identical to D6) ────────────────────────────────
X_text = df['text'].values
y = df['label'].values

X_train_text, X_temp_text, y_train, y_temp = train_test_split(
    X_text, y, test_size=0.30, random_state=42, stratify=y)
X_val_text, X_test_text, y_val, y_test = train_test_split(
    X_temp_text, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"Split sizes — train: {len(X_train_text)}  val: {len(X_val_text)}  test: {len(X_test_text)}")

# ─── 3. Metrics Helper ──────────────────────────────────────────────────────
def get_metrics(name, y_true, y_pred):
    return {
        'Model': name,
        'Accuracy':  round(accuracy_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred), 4),
        'Recall':    round(recall_score(y_true, y_pred), 4),
        'F1 Score':  round(f1_score(y_true, y_pred), 4),
    }

# ─── 4. TF-IDF Baseline (reproduced from D6 for comparison) ─────────────────
print("\n--- Reproducing TF-IDF baseline (D6) for comparison ---")
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_val_tfidf   = vectorizer.transform(X_val_text)
X_test_tfidf  = vectorizer.transform(X_test_text)

lr_tfidf = LogisticRegression(max_iter=1000, random_state=42)
lr_tfidf.fit(X_train_tfidf, y_train)
metrics_tfidf_lr = get_metrics('LR  (TF-IDF)', y_test, lr_tfidf.predict(X_test_tfidf))

X_train_tfidf_dense = X_train_tfidf.toarray()
X_val_tfidf_dense   = X_val_tfidf.toarray()
X_test_tfidf_dense  = X_test_tfidf.toarray()

mlp_tfidf = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_tfidf_dense.shape[1],)),
    layers.Dense(1,  activation='sigmoid'),
])
mlp_tfidf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
mlp_tfidf.fit(X_train_tfidf_dense, y_train,
              validation_data=(X_val_tfidf_dense, y_val),
              epochs=20, batch_size=32, verbose=0)
mlp_tfidf_preds = (mlp_tfidf.predict(X_test_tfidf_dense, verbose=0).ravel() >= 0.5).astype(int)
metrics_tfidf_mlp = get_metrics('MLP (TF-IDF)', y_test, mlp_tfidf_preds)

print("TF-IDF baseline ready ✓")

# ─── 5. spaCy Document Embeddings ───────────────────────────────────────────
print("\nLoading spaCy model (en_core_web_md) …")
nlp = spacy.load('en_core_web_md')

def texts_to_embeddings(texts, nlp_model):
    vdim = nlp_model.vocab.vectors_length
    embeddings = np.zeros((len(texts), vdim), dtype=np.float32)
    for i, text in enumerate(texts):
        doc = nlp_model(text)
        if doc.vector.any():
            embeddings[i] = doc.vector
    return embeddings

print("Computing document embeddings …")
X_train_emb = texts_to_embeddings(X_train_text, nlp)
X_val_emb   = texts_to_embeddings(X_val_text,   nlp)
X_test_emb  = texts_to_embeddings(X_test_text,  nlp)

print(f"Embedding dimension: {X_train_emb.shape[1]}")

scaler = StandardScaler()
X_train_emb = scaler.fit_transform(X_train_emb)
X_val_emb   = scaler.transform(X_val_emb)
X_test_emb  = scaler.transform(X_test_emb)

# ─── 6. Model 1 — LR on Embeddings ──────────────────────────────────────────
lr_emb = LogisticRegression(max_iter=1000, random_state=42)
lr_emb.fit(X_train_emb, y_train)
metrics_emb_lr = get_metrics('LR  (Embedding)', y_test, lr_emb.predict(X_test_emb))

# ─── 7. Model 2 — Keras MLP on Embeddings ───────────────────────────────────
mlp_emb = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_emb.shape[1],)),
    layers.Dense(1,  activation='sigmoid'),
])
mlp_emb.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\nTraining Keras MLP on embeddings …")
mlp_emb.fit(X_train_emb, y_train,
            validation_data=(X_val_emb, y_val),
            epochs=20, batch_size=32, verbose=1)
mlp_emb_preds = (mlp_emb.predict(X_test_emb, verbose=0).ravel() >= 0.5).astype(int)
metrics_emb_mlp = get_metrics('MLP (Embedding)', y_test, mlp_emb_preds)

# ─── 8. Embedding-Only Metrics Table ────────────────────────────────────────
emb_df = pd.DataFrame([metrics_emb_lr, metrics_emb_mlp])
print("\n=== DELIVERABLE 7 — EMBEDDING-BASED MODEL METRICS ===")
print(emb_df.to_string(index=False))

# ─── 9. Comparison Table: TF-IDF vs Embeddings ──────────────────────────────
all_metrics = pd.DataFrame([
    metrics_tfidf_lr, metrics_tfidf_mlp,
    metrics_emb_lr,   metrics_emb_mlp,
])
print("\n=== TF-IDF vs EMBEDDING — FULL COMPARISON ===")
print(all_metrics.to_string(index=False))

# ─── 10. Comparison Plot ────────────────────────────────────────────────────
BG     = '#1a1d27'
TEXT   = 'white'
GRID   = '#2a2d3a'
ACCENT = ['#4C9BE8', '#E8724C', '#4CE89B', '#E84C9B']

fig, ax = plt.subplots(figsize=(12, 7), facecolor='#0f1117')
ax.set_facecolor(BG)

metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
models  = [metrics_tfidf_lr, metrics_tfidf_mlp, metrics_emb_lr, metrics_emb_mlp]
labels  = [m['Model'] for m in models]
x       = np.arange(len(metric_names))
total_w = 0.72
w       = total_w / len(models)

for i, (m_dict, label, color) in enumerate(zip(models, labels, ACCENT)):
    offsets = x - total_w / 2 + w * (i + 0.5)
    vals = [m_dict[m] for m in metric_names]
    bars = ax.bar(offsets, vals, w, label=label, color=color, alpha=0.9)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom',
                color=TEXT, fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(metric_names, color=TEXT, fontsize=12)
ax.set_ylim(0, 1.15)
ax.set_ylabel('Score', color=TEXT)
ax.set_title('Deliverable 7 — TF-IDF vs spaCy Embedding: Full Comparison',
             color=TEXT, fontsize=14, fontweight='bold', pad=12)
ax.legend(facecolor=BG, labelcolor=TEXT, fontsize=10, loc='lower right')
ax.tick_params(colors=TEXT)
ax.yaxis.set_tick_params(labelcolor=TEXT)
ax.xaxis.set_tick_params(labelcolor=TEXT)
for spine in ax.spines.values():
    spine.set_color(GRID)
ax.grid(True, color=GRID, alpha=0.4, axis='y')

plt.tight_layout()
plt.savefig('deliverable7_comparison.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("\nComparison plot saved → deliverable7_comparison.png")

# ─── 11. Hypothesis ─────────────────────────────────────────────────────────
hypothesis = """
=== HYPOTHESIS: WHY TF-IDF AND EMBEDDINGS DIFFER ===

TF-IDF and averaged word embeddings capture fundamentally different textual
signals, and their relative effectiveness depends on the nature of the task.

TF-IDF represents each document as a sparse, high-dimensional vector weighted
by term frequency and inverse document frequency. This makes it extremely
sensitive to the presence of specific discriminative keywords. SMS spam
detection is largely a keyword-driven problem: spam messages cluster around
distinctive tokens such as "free", "win", "prize", "claim", and "txt" that
are rare in legitimate messages. TF-IDF directly encodes the importance of
these tokens, giving classifiers sharp decision boundaries.

spaCy's pre-trained word embeddings (en_core_web_md, 300-dimensional) capture
semantic similarity — words with related meanings receive nearby vectors.
However, creating a document-level representation by averaging all word vectors
has two key limitations: (1) it dilutes the signal from rare but highly
informative spam keywords among common everyday words, and (2) it discards
word-order and compositional information entirely. Additionally, the
pre-trained vectors were learned from general web text and may not optimally
represent the idiosyncratic vocabulary of SMS spam.

Therefore, TF-IDF is expected to perform comparably or better than averaged
embeddings on this task because spam detection rewards exact lexical matching
over semantic generalization. If embeddings still achieve reasonable accuracy,
it indicates that the distributional properties of spam vocabulary are
distinctive enough to survive the averaging process — but the inherent
compression from thousands of TF-IDF features to 300 embedding dimensions
inevitably discards task-relevant information.
""".strip()

print(f"\n{hypothesis}")
