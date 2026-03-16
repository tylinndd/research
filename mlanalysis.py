import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ─── 1. Generate realistic Titanic-like dataset ───────────────────────────────
np.random.seed(42)
n = 891

pclass  = np.random.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55])
sex     = np.random.choice([0, 1],    n, p=[0.65, 0.35])   # 0=male, 1=female
age     = np.clip(np.random.normal(30, 14, n), 1, 80)
sibsp   = np.random.choice([0,1,2,3], n, p=[0.68,0.23,0.06,0.03])
parch   = np.random.choice([0,1,2,3], n, p=[0.76,0.13,0.09,0.02])
fare    = np.clip(np.random.exponential(32, n), 0, 512)

# Survival probability based on known Titanic patterns
logit = (-1.2
         + 2.5 * sex
         - 0.6 * (pclass == 2).astype(float)
         - 1.4 * (pclass == 3).astype(float)
         - 0.01 * age
         + 0.003 * fare)
prob = 1 / (1 + np.exp(-logit))
survived = (np.random.rand(n) < prob).astype(int)

df = pd.DataFrame({'Survived': survived, 'Pclass': pclass, 'Sex': sex,
                   'Age': age, 'SibSp': sibsp, 'Parch': parch, 'Fare': fare})

X = df.drop('Survived', axis=1)
y = df['Survived'].astype(int)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─── 2. Train/Test Split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# ─── 3. Train Models ─────────────────────────────────────────────────────────
lr = LogisticRegression(random_state=42, max_iter=1000)
dt = DecisionTreeClassifier(random_state=42, max_depth=5)

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)

lr_preds = lr.predict(X_test)
dt_preds = dt.predict(X_test)

# ─── 4. Metrics Function ─────────────────────────────────────────────────────
def get_metrics(name, y_true, y_pred):
    return {
        'Model': name,
        'Accuracy':  round(accuracy_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred), 4),
        'Recall':    round(recall_score(y_true, y_pred), 4),
        'F1 Score':  round(f1_score(y_true, y_pred), 4),
    }

metrics_lr = get_metrics('Logistic Regression', y_test, lr_preds)
metrics_dt = get_metrics('Decision Tree', y_test, dt_preds)
metrics_df = pd.DataFrame([metrics_lr, metrics_dt])

print("\n=== SINGLE SPLIT METRICS ===")
print(metrics_df.to_string(index=False))

# ─── 5. Confusion Matrices ───────────────────────────────────────────────────
cm_lr = confusion_matrix(y_test, lr_preds)
cm_dt = confusion_matrix(y_test, dt_preds)

# ─── 6. Stratified K-Fold Cross Validation ───────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'precision', 'recall', 'f1']

cv_lr = cross_validate(LogisticRegression(random_state=42, max_iter=1000),
                       X_scaled, y, cv=skf, scoring=scoring)
cv_dt = cross_validate(DecisionTreeClassifier(random_state=42, max_depth=5),
                       X_scaled, y, cv=skf, scoring=scoring)

def cv_summary(name, cv_results):
    return {
        'Model': name,
        'CV Accuracy':  round(cv_results['test_accuracy'].mean(), 4),
        'CV Precision': round(cv_results['test_precision'].mean(), 4),
        'CV Recall':    round(cv_results['test_recall'].mean(), 4),
        'CV F1 Score':  round(cv_results['test_f1'].mean(), 4),
    }

cv_df = pd.DataFrame([cv_summary('Logistic Regression', cv_lr),
                      cv_summary('Decision Tree', cv_dt)])

print("\n=== STRATIFIED 5-FOLD CV METRICS ===")
print(cv_df.to_string(index=False))

# ─── 7. Plotting ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 22), facecolor='#0f1117')
fig.suptitle('Titanic — Model Comparison Report', fontsize=22, fontweight='bold',
             color='white', y=0.98)

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.4)

ACCENT  = ['#4C9BE8', '#E8724C']
BG      = '#1a1d27'
TEXT    = 'white'
GRID    = '#2a2d3a'

# ── Plot 1: Metrics bar chart (single split) ──────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor(BG)
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
lr_vals = [metrics_lr[m] for m in metric_names]
dt_vals = [metrics_dt[m] for m in metric_names]
x = np.arange(len(metric_names))
w = 0.3
bars1 = ax1.bar(x - w/2, lr_vals, w, label='Logistic Regression', color=ACCENT[0], alpha=0.9)
bars2 = ax1.bar(x + w/2, dt_vals, w, label='Decision Tree',        color=ACCENT[1], alpha=0.9)
for bar in list(bars1) + list(bars2):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{bar.get_height():.3f}', ha='center', va='bottom', color=TEXT, fontsize=10)
ax1.set_xticks(x); ax1.set_xticklabels(metric_names, color=TEXT, fontsize=12)
ax1.set_ylim(0, 1.1); ax1.set_ylabel('Score', color=TEXT)
ax1.set_title('Q5a — Single Split Metrics: Logistic Regression vs Decision Tree',
              color=TEXT, fontsize=13, pad=12)
ax1.tick_params(colors=TEXT); ax1.yaxis.label.set_color(TEXT)
ax1.spines[['top','right','left','bottom']].set_color(GRID)
ax1.set_facecolor(BG); ax1.legend(facecolor=BG, labelcolor=TEXT, fontsize=11)
ax1.yaxis.set_tick_params(labelcolor=TEXT)
ax1.set_facecolor(BG)
for spine in ax1.spines.values(): spine.set_color(GRID)

# ── Plot 2 & 3: Confusion Matrices ───────────────────────────────────────────
for idx, (cm, name, preds) in enumerate(
        [(cm_lr, 'Logistic Regression', lr_preds),
         (cm_dt, 'Decision Tree', dt_preds)]):
    ax = fig.add_subplot(gs[1, idx])
    ax.set_facecolor(BG)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted\nDied (0)', 'Predicted\nSurvived (1)'],
                yticklabels=['Actual\nDied (0)', 'Actual\nSurvived (1)'],
                ax=ax, linewidths=1, linecolor=GRID,
                annot_kws={"size": 14, "weight": "bold", "color": "white"})
    ax.set_title(f'Q6 — Confusion Matrix: {name}', color=TEXT, fontsize=12, pad=10)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
    for spine in ax.spines.values(): spine.set_color(GRID)
    # Annotate FP and FN
    fp = cm[0][1]; fn = cm[1][0]
    ax.set_xlabel(f'FP={fp} (Predicted survived, actually died)  |  FN={fn} (Predicted died, actually survived)',
                  color='#aaaaaa', fontsize=8)

# ── Plot 4: CV vs Single Split comparison ────────────────────────────────────
ax4 = fig.add_subplot(gs[2, :])
ax4.set_facecolor(BG)
cv_metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
single_lr = [metrics_lr[m] for m in cv_metric_names]
single_dt = [metrics_dt[m] for m in cv_metric_names]
cv_lr_vals = [cv_df.loc[0, f'CV {m}'] for m in cv_metric_names]
cv_dt_vals = [cv_df.loc[1, f'CV {m}'] for m in cv_metric_names]

x = np.arange(len(cv_metric_names))
w = 0.2
b1 = ax4.bar(x - 1.5*w, single_lr,  w, label='LR — Single Split', color='#4C9BE8', alpha=0.7)
b2 = ax4.bar(x - 0.5*w, cv_lr_vals, w, label='LR — CV Average',    color='#4C9BE8', alpha=1.0)
b3 = ax4.bar(x + 0.5*w, single_dt,  w, label='DT — Single Split', color='#E8724C', alpha=0.7)
b4 = ax4.bar(x + 1.5*w, cv_dt_vals, w, label='DT — CV Average',    color='#E8724C', alpha=1.0)
for bar in list(b1)+list(b2)+list(b3)+list(b4):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
             f'{bar.get_height():.3f}', ha='center', va='bottom', color=TEXT, fontsize=8)
ax4.set_xticks(x); ax4.set_xticklabels(cv_metric_names, color=TEXT, fontsize=12)
ax4.set_ylim(0, 1.15); ax4.set_ylabel('Score', color=TEXT)
ax4.set_title('Q7 — Single Split vs Stratified 5-Fold CV Performance',
              color=TEXT, fontsize=13, pad=12)
ax4.tick_params(colors=TEXT); ax4.yaxis.label.set_color(TEXT)
ax4.legend(facecolor=BG, labelcolor=TEXT, fontsize=10)
for spine in ax4.spines.values(): spine.set_color(GRID)
ax4.yaxis.set_tick_params(labelcolor=TEXT)

plt.savefig('/mnt/user-data/outputs/ml_analysis.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("\nPlot saved successfully.")

# ─── Print detailed explanation ──────────────────────────────────────────────
print("\n=== WHICH MODEL IS BETTER? ===")
better = 'Logistic Regression' if metrics_lr['F1 Score'] >= metrics_dt['F1 Score'] else 'Decision Tree'
print(f"Based on F1 Score: {better} performs better on this split.")

print("\n=== CV vs SINGLE SPLIT CHANGE ===")
for m, cm_key in zip(['Accuracy','F1 Score'], ['CV Accuracy','CV F1 Score']):
    lr_diff = round(cv_df.loc[0, cm_key] - metrics_lr[m], 4)
    dt_diff = round(cv_df.loc[1, cm_key] - metrics_dt[m], 4)
    print(f"  LR  {m}: single={metrics_lr[m]} → CV avg={cv_df.loc[0,cm_key]} (Δ{lr_diff:+.4f})")
    print(f"  DT  {m}: single={metrics_dt[m]} → CV avg={cv_df.loc[1,cm_key]} (Δ{dt_diff:+.4f})")