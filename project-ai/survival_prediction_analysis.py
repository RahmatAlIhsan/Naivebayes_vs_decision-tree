#!/usr/bin/env python3
"""
SURVIVAL_PREDICTION ANALYSIS - Memory Optimized
================================================
Target: Survival_Prediction | SMOTE K=5 | 9 fitur terbaik
4 Algoritma: SVM, XGBoost, RF, KNN
Output: visualisasi_survival_prediction.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import time
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_auc_score, roc_curve)
from imblearn.over_sampling import SMOTE

# Force XGBoost to use limited resources
import os as _os
_os.environ['OMP_NUM_THREADS'] = '2'
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
gc.collect()

print("=" * 70)
print("SURVIVAL_PREDICTION ANALYSIS - 9 Fitur + SMOTE K=5")
print("=" * 70)

# ================================================================
# 1. LOAD DATA (10K sample for memory)
# ================================================================
print("\n1. LOAD DATA (10K sample)")
df = pd.read_csv('preprocessed/dataset_no_outlier.csv')
df['Survival_Prediction'] = df['Survival_Prediction'].map({'Yes': 1, 'No': 0})
print(f"Total: {df.shape[0]:,} baris")

df_sample = df.sample(n=10000, random_state=42)
y_full = df_sample['Survival_Prediction'].values
print(f"Sample: 10,000 baris")
print(f"  Yes=1: {y_full.sum():,}")
print(f"  No=0:  {(y_full==0).sum():,}")

# ================================================================
# 2. FEATURES - 9 terbaik
# ================================================================
num_best = ['Age', 'Tumor_Size_mm', 'Healthcare_Costs', 'Incidence_Rate_per_100K', 'Mortality_Rate_per_100K']
cat_best = ['Alcohol_Consumption', 'Diet_Risk', 'Economic_Classification', 'Healthcare_Access']
all_cols = num_best + cat_best + ['Survival_Prediction']

df_sel = df_sample[all_cols].copy()
df_enc = pd.get_dummies(df_sel, columns=cat_best, drop_first=True, dtype=int)

feature_names = [c for c in df_enc.columns if c != 'Survival_Prediction']
print(f"\nFitur ({len(feature_names)}):")
for i, f in enumerate(feature_names):
    print(f"  {i+1}. {f}")

X = df_enc[feature_names].values
y = df_enc['Survival_Prediction'].values

# Clean up
del df, df_sample, df_sel, df_enc
gc.collect()

# ================================================================
# 3. SPLIT + SMOTE
# ================================================================
print("\n2. SPLIT + SMOTE K=5")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {X_train.shape[0]:,} (Yes={y_train.sum():,}, No={(y_train==0).sum():,})")
print(f"Test:  {X_test.shape[0]:,} (Yes={y_test.sum():,}, No={(y_test==0).sum():,})")

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {X_train_res.shape[0]:,} (Yes={y_train_res.sum():,}, No={(y_train_res==0).sum():,})")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)
gc.collect()

# ================================================================
# 4. TRAIN MODELS
# ================================================================
print("\n3. TRAINING 4 ALGORITMA")

models_def = {
    'SVM': CalibratedClassifierCV(LinearSVC(C=1.0, class_weight='balanced', max_iter=2000, random_state=42, dual='auto'), cv=3, n_jobs=1),
    'XGBoost': XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=1, eval_metric='logloss'),
    'RF': RF(n_estimators=30, max_depth=6, random_state=42, n_jobs=1),
    'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=1)
}

results = {}
predictions = {}
probabilities = {}
model_list = list(models_def.keys())

for name, model in models_def.items():
    print(f"  [{name}]...", end=' ')
    X_tr = X_train_scaled if name in ['SVM', 'KNN'] else X_train_res
    X_te = X_test_scaled if name in ['SVM', 'KNN'] else X_test

    start = time.time()
    model.fit(X_tr, y_train_res)
    elapsed = time.time() - start

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    predictions[name] = y_pred
    probabilities[name] = y_prob

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1, 'ROC-AUC': auc, 'Time': elapsed}

    cm = confusion_matrix(y_test, y_pred)
    print(f"Acc={acc:.4f} F1={f1:.4f} AUC={auc:.4f} ({elapsed:.1f}s)")
    print(f"    CM: TN={cm[0,0]:,} FP={cm[0,1]:,} FN={cm[1,0]:,} TP={cm[1,1]:,}")

# Free memory before plotting
del model, X_tr, X_te, X_train_res, X_train_scaled
gc.collect()

# ================================================================
# 5. COMPACT VISUALIZATION (1 PNG)
# ================================================================
print("\n4. GENERATING VISUALISASI...")

fig, axes = plt.subplots(3, 4, figsize=(14, 11))
plt.subplots_adjust(hspace=0.4, wspace=0.35)
# Flatten axes for easier indexing
axs = axes.flatten()

# Row 1: 4 Confusion Matrices (cols 0-3)
cmap_names = ['Blues', 'Greens', 'Oranges', 'Purples']
for i, name in enumerate(model_list):
    ax = axs[i]  # indices 0,1,2,3
    cm = confusion_matrix(y_test, predictions[name])
    cm_pct = cm / cm.sum(axis=1)[:, np.newaxis] * 100

    im = ax.imshow(cm, cmap=cmap_names[i], interpolation='nearest')
    for j in range(2):
        for k in range(2):
            color = 'white' if cm[j, k] > cm.max()/2 else 'black'
            ax.text(k, j, f'{cm[j, k]:,}\n({cm_pct[j, k]:.1f}%)',
                    ha='center', va='center', fontsize=10, fontweight='bold', color=color)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred No', 'Pred Yes'], fontsize=7)
    ax.set_yticklabels(['Act No', 'Act Yes'], fontsize=7)
    ax.set_title(name, fontsize=11, fontweight='bold')

# Row 2: ROC Curves (span cols 0-1)
ax_roc = axs[4]
colors = {'SVM': 'blue', 'XGBoost': 'green', 'RF': 'orange', 'KNN': 'purple'}
styles = {'SVM': '-', 'XGBoost': '--', 'RF': '-.', 'KNN': ':'}

for name in model_list:
    fpr, tpr, _ = roc_curve(y_test, probabilities[name])
    ax_roc.plot(fpr, tpr, styles[name], color=colors[name], lw=2,
                label=f'{name} (AUC={results[name]["ROC-AUC"]:.4f})')
ax_roc.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.3, label='Random (0.5)')
ax_roc.set_xlabel('FPR', fontsize=9)
ax_roc.set_ylabel('TPR', fontsize=9)
ax_roc.set_title('ROC Curves', fontsize=11, fontweight='bold')
ax_roc.legend(fontsize=7, loc='lower right')
ax_roc.grid(True, alpha=0.3)

# Row 2: Target Distribution (col 2)
ax_dist = axs[6]
y_counts = [int((y==0).sum()), int(y.sum())]
ax_dist.pie(y_counts, labels=['No', 'Yes'], autopct='%1.1f%%',
            colors=['#e74c3c', '#2ecc71'], explode=[0.05, 0.05],
            textprops={'fontsize': 10, 'fontweight': 'bold'})
ax_dist.set_title('Target Distribution', fontsize=10, fontweight='bold')

# Row 2: Empty (col 3) - hide
axs[7].set_visible(False)

# Row 3: Metrics Bar Chart (span cols 0-1)
ax_bar = axs[8]
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(model_list))
width = 0.15
for i, metric in enumerate(metrics_names):
    vals = [results[m][metric] for m in model_list]
    ax_bar.bar(x + i*width, vals, width, label=metric)
ax_bar.set_xticks(x + width*2)
ax_bar.set_xticklabels(model_list, fontsize=8)
ax_bar.set_ylabel('Score', fontsize=9)
ax_bar.set_title('Metrics Comparison', fontsize=10, fontweight='bold')
ax_bar.set_ylim(0, 1.05)
ax_bar.legend(fontsize=6, loc='lower right')
ax_bar.grid(True, alpha=0.3, axis='y')

# Row 3: Feature Importance (col 2)
ax_fi = axs[10]
rf_temp = RF(n_estimators=30, max_depth=6, random_state=42, n_jobs=1)
rf_temp.fit(X_train, y_train)
importances = rf_temp.feature_importances_
fi_idx = np.argsort(importances)[-9:]

ax_fi.barh(range(9), importances[fi_idx], 
           color=plt.cm.Greens(np.linspace(0.3, 0.85, 9))[::-1])
ax_fi.set_yticks(range(9))
ax_fi.set_yticklabels([feature_names[i] for i in fi_idx], fontsize=6)
ax_fi.set_xlabel('Importance', fontsize=9)
ax_fi.set_title('Feature Importance', fontsize=10, fontweight='bold')
ax_fi.invert_yaxis()

# Row 3: Empty (col 3) - hide
axs[11].set_visible(False)

plt.suptitle(f'SURVIVAL_PREDICTION ANALYSIS - SMOTE K=5 (10K samples)',
             fontsize=13, fontweight='bold', y=0.98)
plt.savefig('visualisasi_survival_prediction.png', dpi=100, bbox_inches='tight',
            facecolor='white')
plt.close()
gc.collect()
print(f"  => Saved: visualisasi_survival_prediction.png")

# ================================================================
# 6. SUMMARY TABLE
# ================================================================
print("\n" + "=" * 70)
print("HASIL PERBANDINGAN")
print("=" * 70)
print(f"\n{'Model':<12} {'Accuracy':>9} {'Precision':>9} {'Recall':>9} {'F1':>9} {'AUC':>9} {'Time':>8}")
print("-" * 65)
for name in model_list:
    r = results[name]
    print(f"{name:<12} {r['Accuracy']:>9.4f} {r['Precision']:>9.4f} {r['Recall']:>9.4f} {r['F1-Score']:>9.4f} {r['ROC-AUC']:>9.4f} {r['Time']:>7.2f}s")
print("-" * 65)

print(f"\n{'Model':<12} {'TN':>6} {'FP':>6} {'FN':>6} {'TP':>6} {'Total':>6}")
print("-" * 42)
for name in model_list:
    cm = confusion_matrix(y_test, predictions[name])
    print(f"{name:<12} {cm[0,0]:>6,} {cm[0,1]:>6,} {cm[1,0]:>6,} {cm[1,1]:>6,} {cm.sum():>6,}")

print(f"\n{'='*70}")
print(f"Output: visualisasi_survival_prediction.png")
print(f"{'='*70}")
