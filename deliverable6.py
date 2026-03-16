import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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

# ─── 2. Split 70 / 15 / 15 ──────────────────────────────────────────────────
X_text = df['text'].values
y = df['label'].values

X_train_text, X_temp_text, y_train, y_temp = train_test_split(
    X_text, y, test_size=0.30, random_state=42, stratify=y)
X_val_text, X_test_text, y_val, y_test = train_test_split(
    X_temp_text, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"Split sizes — train: {len(X_train_text)}  val: {len(X_val_text)}  test: {len(X_test_text)}")

# ─── 3. TF-IDF Vectorization ────────────────────────────────────────────────
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_val_tfidf   = vectorizer.transform(X_val_text)
X_test_tfidf  = vectorizer.transform(X_test_text)

print(f"TF-IDF vocabulary size: {len(vectorizer.get_feature_names_out())}")

# ─── 4. Metrics Helper ──────────────────────────────────────────────────────
def get_metrics(name, y_true, y_pred):
    return {
        'Model': name,
        'Accuracy':  round(accuracy_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred), 4),
        'Recall':    round(recall_score(y_true, y_pred), 4),
        'F1 Score':  round(f1_score(y_true, y_pred), 4),
    }

# ─── 5. Model 1 — Logistic Regression ───────────────────────────────────────
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_tfidf, y_train)
lr_preds = lr.predict(X_test_tfidf)
metrics_lr = get_metrics('Logistic Regression', y_test, lr_preds)

print("\nLogistic Regression trained ✓")

# ─── 6. Model 2 — Keras MLP ─────────────────────────────────────────────────
X_train_dense = X_train_tfidf.toarray()
X_val_dense   = X_val_tfidf.toarray()
X_test_dense  = X_test_tfidf.toarray()

mlp = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_dense.shape[1],)),
    layers.Dense(1,  activation='sigmoid'),
])
mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\nTraining Keras MLP …")
mlp.fit(X_train_dense, y_train,
        validation_data=(X_val_dense, y_val),
        epochs=20, batch_size=32, verbose=1)

mlp_prob  = mlp.predict(X_test_dense, verbose=0).ravel()
mlp_preds = (mlp_prob >= 0.5).astype(int)
metrics_mlp = get_metrics('Keras MLP', y_test, mlp_preds)

# ─── 7. Side-by-Side Metrics Comparison ─────────────────────────────────────
metrics_df = pd.DataFrame([metrics_lr, metrics_mlp])

print("\n=== DELIVERABLE 6 — TF-IDF MODEL COMPARISON ===")
print(metrics_df.to_string(index=False))

# ─── 8. Top 10 Most Informative Words ───────────────────────────────────────
feature_names = vectorizer.get_feature_names_out()
coefs = lr.coef_[0]
top_indices = np.argsort(np.abs(coefs))[::-1][:10]

print("\n=== TOP 10 MOST INFORMATIVE WORDS (Logistic Regression) ===")
print(f"{'Rank':<6}{'Word':<20}{'Coefficient':>12}")
print("─" * 38)
for rank, idx in enumerate(top_indices, 1):
    print(f"{rank:<6}{feature_names[idx]:<20}{coefs[idx]:>12.4f}")

# ─── 9. Metrics Plot ────────────────────────────────────────────────────────
BG     = '#1a1d27'
TEXT   = 'white'
GRID   = '#2a2d3a'
ACCENT = ['#4C9BE8', '#E8724C']

fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0f1117')
ax.set_facecolor(BG)

metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
lr_vals  = [metrics_lr[m] for m in metric_names]
mlp_vals = [metrics_mlp[m] for m in metric_names]
x = np.arange(len(metric_names))
w = 0.3

bars1 = ax.bar(x - w/2, lr_vals,  w, label='Logistic Regression', color=ACCENT[0], alpha=0.9)
bars2 = ax.bar(x + w/2, mlp_vals, w, label='Keras MLP',           color=ACCENT[1], alpha=0.9)

for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', va='bottom', color=TEXT, fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels(metric_names, color=TEXT, fontsize=12)
ax.set_ylim(0, 1.12)
ax.set_ylabel('Score', color=TEXT)
ax.set_title('Deliverable 6 — TF-IDF: Logistic Regression vs Keras MLP',
             color=TEXT, fontsize=14, fontweight='bold', pad=12)
ax.legend(facecolor=BG, labelcolor=TEXT, fontsize=11)
ax.tick_params(colors=TEXT)
ax.yaxis.set_tick_params(labelcolor=TEXT)
ax.xaxis.set_tick_params(labelcolor=TEXT)
for spine in ax.spines.values():
    spine.set_color(GRID)
ax.grid(True, color=GRID, alpha=0.4, axis='y')

plt.tight_layout()
plt.savefig('deliverable6_metrics.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("\nMetrics plot saved → deliverable6_metrics.png")
