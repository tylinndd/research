import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

# ─── 1. Load Data ────────────────────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target

print(f"Dataset: {data.DESCR.splitlines()[0]}")
print(f"Features: {X.shape[1]}  |  Samples: {X.shape[0]}")
print(f"Class distribution: benign={np.sum(y==1)}, malignant={np.sum(y==0)}")

# ─── 2. Split 70 / 15 / 15 ──────────────────────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"\nSplit sizes — train: {len(X_train)}  val: {len(X_val)}  test: {len(X_test)}")

# ─── 3. Scale (fit on train only) ───────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# ─── 4. Build Keras Model ───────────────────────────────────────────────────
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1,  activation='sigmoid'),
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# ─── 5. Train ───────────────────────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    verbose=1,
)

# ─── 6. Plot Training vs Validation Loss ────────────────────────────────────
BG     = '#1a1d27'
TEXT   = 'white'
GRID   = '#2a2d3a'
ACCENT = ['#4C9BE8', '#E8724C']

fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0f1117')
ax.set_facecolor(BG)

epochs_range = np.arange(1, len(history.history['loss']) + 1)
ax.plot(epochs_range, history.history['loss'],
        color=ACCENT[0], linewidth=2, label='Training Loss')
ax.plot(epochs_range, history.history['val_loss'],
        color=ACCENT[1], linewidth=2, label='Validation Loss')

ax.set_xlabel('Epoch', color=TEXT, fontsize=12)
ax.set_ylabel('Loss (Binary Cross-Entropy)', color=TEXT, fontsize=12)
ax.set_title('Deliverable 5 — Training vs Validation Loss',
             color=TEXT, fontsize=14, fontweight='bold', pad=12)
ax.legend(facecolor=BG, labelcolor=TEXT, fontsize=11)
ax.tick_params(colors=TEXT)
ax.yaxis.set_tick_params(labelcolor=TEXT)
ax.xaxis.set_tick_params(labelcolor=TEXT)
for spine in ax.spines.values():
    spine.set_color(GRID)
ax.grid(True, color=GRID, alpha=0.4)

plt.tight_layout()
plt.savefig('deliverable5_loss.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("\nLoss plot saved → deliverable5_loss.png")

# ─── 7. Over/Underfitting Assessment ────────────────────────────────────────
train_loss_final = history.history['loss'][-1]
val_loss_final   = history.history['val_loss'][-1]
val_loss_min     = min(history.history['val_loss'])
val_loss_min_ep  = int(np.argmin(history.history['val_loss'])) + 1

print("\n=== FIT ASSESSMENT ===")
print(f"  Final training loss : {train_loss_final:.4f}")
print(f"  Final validation loss: {val_loss_final:.4f}")
print(f"  Best validation loss : {val_loss_min:.4f} (epoch {val_loss_min_ep})")

gap = val_loss_final - train_loss_final
late_rise = val_loss_final - val_loss_min

if gap > 0.10 or late_rise > 0.05:
    assessment = "OVERFITTING"
    detail = (
        "The validation loss is notably higher than training loss "
        f"(gap = {gap:.4f}) and/or rose after its minimum "
        f"(rise = {late_rise:.4f}). The model has memorised training "
        "patterns that do not generalise. Consider adding dropout, "
        "reducing model capacity, or using early stopping."
    )
elif train_loss_final > 0.30:
    assessment = "UNDERFITTING"
    detail = (
        f"Training loss remains high ({train_loss_final:.4f}), "
        "suggesting the model has not learned the training data well. "
        "Consider increasing model capacity or training longer."
    )
else:
    assessment = "GOOD FIT"
    detail = (
        f"Training and validation losses converge closely (gap = {gap:.4f}) "
        f"with minimal late-epoch rise ({late_rise:.4f}). "
        "The model generalises well to unseen data."
    )

print(f"  Verdict: **{assessment}**")
print(f"  {detail}")

# ─── 8. Test Set Metrics ────────────────────────────────────────────────────
y_pred_prob = model.predict(X_test, verbose=0).ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)

def get_metrics(name, y_true, y_pred):
    return {
        'Model': name,
        'Accuracy':  round(accuracy_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred), 4),
        'Recall':    round(recall_score(y_true, y_pred), 4),
        'F1 Score':  round(f1_score(y_true, y_pred), 4),
    }

metrics = get_metrics('Keras FF Network', y_test, y_pred)
metrics_df = pd.DataFrame([metrics])

print("\n=== TEST SET METRICS ===")
print(metrics_df.to_string(index=False))
