#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
======================================================================
MODEL COMPARISON - Perbandingan 8 Algoritma Klasifikasi
Prediksi Stunting pada Balita
======================================================================
Algoritma:
1. Logistic Regression
2. Gaussian Naive Bayes
3. Decision Tree
4. Random Forest
5. Support Vector Machine (SVM)
6. K-Nearest Neighbors (KNN)
7. XGBoost
8. Gradient Boosting

Skenario:
A. Tanpa SMOTE (data imbalance asli)
B. Dengan SMOTE K=3 (data seimbang)
======================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Peringatan: XGBoost tidak tersedia. Akan dilewati.")

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams.update({'figure.figsize': (10, 6), 'font.size': 12})

OUTPUT_DIR = 'data/processed/model_comparison'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("MODEL COMPARISON - 8 Algoritma Klasifikasi")
print("Prediksi Stunting pada Balita")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════
# 1. LOAD DATA & FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BAGIAN 1: LOAD DATA & FEATURE ENGINEERING")
print("=" * 70)

df = pd.read_csv('data/processed/stunting_clean.csv')
print(f"Dataset: {df.shape[0]} baris, {df.shape[1]} kolom")

# Encode features
df['jk_encoded'] = df['jk'].map({'Laki-Laki': 1, 'Perempuan': 0})
target_mapping = {'Normal': 0, 'Stunted': 1, 'Severely Stunted': 2}
df['status_encoded'] = df['status'].map(target_mapping)

feature_cols = ['umur_bulan', 'tinggi', 'jk_encoded']
X = df[feature_cols].values
y = df['status_encoded'].values

print(f"Features: {feature_cols}")
print(f"X: {X.shape}, y: {y.shape}")
print(f"Target classes: {np.unique(y)}")
print(f"Distribusi:\n{df['status'].value_counts()}")

# ═══════════════════════════════════════════════════════════════
# 2. SPLIT DATA
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BAGIAN 2: SPLIT DATA 80:20 (STRATIFIED)")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]} sampel")
print(f"Test : {X_test.shape[0]} sampel")
print(f"Train distribution:\n{pd.Series(y_train).value_counts().sort_index()}")
print(f"Test distribution:\n{pd.Series(y_test).value_counts().sort_index()}")

# Standard Scaling (untuk model yang sensitif terhadap skala)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ═══════════════════════════════════════════════════════════════
# 3. DEFINISI 8 ALGORITMA
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BAGIAN 3: DEFINISI 8 ALGORITMA KLASIFIKASI")
print("=" * 70)

algorithms = {
    'Logistic Regression': {
        'model': LogisticRegression(random_state=42, max_iter=1000, multi_class='ovr'),
        'use_scaling': True
    },
    'Naive Bayes': {
        'model': GaussianNB(),
        'use_scaling': False
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=42),
        'use_scaling': False
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42, n_estimators=100),
        'use_scaling': False
    },
    'SVM': {
        'model': SVC(random_state=42, kernel='rbf', probability=True),
        'use_scaling': True
    },
    'KNN': {
        'model': KNeighborsClassifier(n_neighbors=5),
        'use_scaling': True
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(random_state=42, n_estimators=100),
        'use_scaling': False
    }
}

if XGB_AVAILABLE:
    algorithms['XGBoost'] = {
        'model': XGBClassifier(random_state=42, n_estimators=100, eval_metric='mlogloss'),
        'use_scaling': False
    }

print(f"Total algoritma: {len(algorithms)}")
for name in algorithms:
    print(f"  - {name}")


def evaluate_model(model, X_tr, X_te, y_tr, y_te, model_name):
    """Train and evaluate a model, return metrics."""
    start = time()
    model.fit(X_tr, y_tr)
    train_time = time() - start

    start = time()
    y_pred = model.predict(X_te)
    pred_time = time() - start

    # Handle predict_proba for ROC AUC (not all models support it)
    try:
        y_proba = model.predict_proba(X_te)
        if y_proba.shape[1] == 3:  # multiclass
            roc_auc = roc_auc_score(y_te, y_proba, multi_class='ovr')
        else:
            roc_auc = roc_auc_score(y_te, y_proba[:, 1])
    except Exception:
        roc_auc = None

    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_te, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_te, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_te, y_pred)

    # Per-class metrics
    cr = classification_report(y_te, y_pred,
                                target_names=['Normal', 'Stunted', 'Severely Stunted'],
                                zero_division=0, output_dict=True)

    return {
        'Model': model_name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4),
        'ROC AUC': round(roc_auc, 4) if roc_auc is not None else None,
        'Train Time (s)': round(train_time, 4),
        'Pred Time (s)': round(pred_time, 4),
        'Confusion Matrix': cm,
        'Classification Report': cr,
        'y_pred': y_pred
    }


# ═══════════════════════════════════════════════════════════════
# 4. SKENARIO A: TANPA SMOTE
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BAGIAN 4: SKENARIO A - TANPA SMOTE")
print("=" * 70)

results_no_smote = []

print(f"\n{'Model':20} {'Accuracy':10} {'Precision':10} {'Recall':10} {'F1-Score':10} {'Time(s)':10}")
print(f"{'-'*70}")

for name, config in algorithms.items():
    model = config['model']
    use_scaling = config['use_scaling']

    X_tr = X_train_scaled if use_scaling else X_train
    X_te = X_test_scaled if use_scaling else X_test

    result = evaluate_model(model, X_tr, X_te, y_train, y_test, name)
    result['SMOTE'] = 'Tidak'
    results_no_smote.append(result)

    print(f"{name:20} {result['Accuracy']:<10.4f} {result['Precision']:<10.4f} "
          f"{result['Recall']:<10.4f} {result['F1-Score']:<10.4f} {result['Train Time (s)']:<10.4f}")

# ═══════════════════════════════════════════════════════════════
# 5. SKENARIO B: DENGAN SMOTE K=3
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BAGIAN 5: SKENARIO B - DENGAN SMOTE K=3")
print("=" * 70)

smote = SMOTE(random_state=42, k_neighbors=3)

# Apply SMOTE on training data (after scaling if applicable)
X_train_smote = X_train.copy()
y_train_smote = y_train.copy()

# For models needing scaling, apply SMOTE after scaling
X_train_resampled = {}
y_train_resampled = {}

for use_scaling_name in [True, False]:
    X_tr = X_train_scaled if use_scaling_name else X_train
    X_res, y_res = smote.fit_resample(X_tr, y_train)
    X_train_resampled[use_scaling_name] = X_res
    y_train_resampled[use_scaling_name] = y_res

print(f"Train distribution SEBELUM SMOTE:\n{pd.Series(y_train).value_counts().sort_index()}")
print(f"\nTrain distribution SETELAH SMOTE K=3:\n{pd.Series(y_train_resampled[False]).value_counts().sort_index()}")
print(f"Total samples after SMOTE: {len(y_train_resampled[False])}")

results_with_smote = []

print(f"\n{'Model':20} {'Accuracy':10} {'Precision':10} {'Recall':10} {'F1-Score':10} {'Time(s)':10}")
print(f"{'-'*70}")

for name, config in algorithms.items():
    model = config['model']
    use_scaling = config['use_scaling']

    # Use resampled data with appropriate scaling
    X_tr = X_train_resampled[use_scaling]
    y_tr = y_train_resampled[use_scaling]
    X_te = X_test_scaled if use_scaling else X_test

    # Clone model for fresh training
    if hasattr(model, 'get_params'):
        new_model = model.__class__(**model.get_params())
        # Set random_state if the original had it
        if hasattr(model, 'random_state'):
            new_model.set_params(**{'random_state': model.random_state})
    else:
        new_model = model

    # Re-evaluate
    result = evaluate_model(new_model, X_tr, X_te, y_tr, y_test, name)
    result['SMOTE'] = 'Ya (K=3)'
    results_with_smote.append(result)

    print(f"{name:20} {result['Accuracy']:<10.4f} {result['Precision']:<10.4f} "
          f"{result['Recall']:<10.4f} {result['F1-Score']:<10.4f} {result['Train Time (s)']:<10.4f}")

# ═══════════════════════════════════════════════════════════════
# 6. RANGKUMAN HASIL
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BAGIAN 6: RANGKUMAN HASIL")
print("=" * 70)

# Build results dataframe (SMOTE column already in results dict)
df_results_no = pd.DataFrame([{k: v for k, v in r.items() if k not in ['Confusion Matrix', 'Classification Report', 'y_pred']}
                               for r in results_no_smote])

df_results_yes = pd.DataFrame([{k: v for k, v in r.items() if k not in ['Confusion Matrix', 'Classification Report', 'y_pred']}
                                for r in results_with_smote])

df_all = pd.concat([df_results_no, df_results_yes], ignore_index=True)
df_all = df_all.sort_values('F1-Score', ascending=False).reset_index(drop=True)

# Save CSV
csv_path = f'{OUTPUT_DIR}/hasil_perbandingan_8_algoritma.csv'
df_all.to_csv(csv_path, index=False)
print(f"\nHasil tersimpan: {csv_path}")
print("\n" + df_all.to_string(index=False))

# ═══════════════════════════════════════════════════════════════
# 7. VISUALISASI
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BAGIAN 7: VISUALISASI")
print("=" * 70)

# 7a. Perbandingan F1-Score Semua Model (Bar Chart)
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(algorithms))
width = 0.35

f1_no = [r['F1-Score'] for r in results_no_smote]
f1_yes = [r['F1-Score'] for r in results_with_smote]
model_names = list(algorithms.keys())

bars1 = ax.bar(x - width/2, f1_no, width, label='Tanpa SMOTE',
               color='#E74C3C', alpha=0.85, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, f1_yes, width, label='Dengan SMOTE K=3',
               color='#2ECC71', alpha=0.85, edgecolor='black', linewidth=0.5)

for bar, val in zip(bars1, f1_no):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', va='bottom', fontsize=8, rotation=45)
for bar, val in zip(bars2, f1_yes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', va='bottom', fontsize=8, rotation=45)

ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=10, rotation=30, ha='right')
ax.set_ylabel('F1-Score (Weighted)', fontsize=12)
ax.set_title('Perbandingan F1-Score: 8 Algoritma - Tanpa vs Dengan SMOTE K=3',
             fontsize=14, fontweight='bold')
ax.set_ylim(0, 1.15)
ax.legend(loc='lower right', fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/perbandingan_f1_8_algoritma.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: perbandingan_f1_8_algoritma.png")

# 7b. Heatmap Perbandingan Metrik
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Perbandingan Metrik: Tanpa SMOTE vs Dengan SMOTE K=3',
             fontsize=14, fontweight='bold')

metrics_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

for ax, (results, title) in zip(axes, [
    (results_no_smote, 'Tanpa SMOTE'),
    (results_with_smote, 'Dengan SMOTE K=3')
]):
    data_matrix = np.array([[r[m] for m in metrics_plot] for r in results])
    sns.heatmap(data_matrix, annot=True, fmt='.4f', cmap='YlOrRd' if 'Tanpa' in title else 'YlGn',
                xticklabels=metrics_plot, yticklabels=model_names,
                ax=ax, cbar_kws={'label': 'Score'}, vmin=0, vmax=1)
    ax.set_title(title, fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/heatmap_metrik_8_algoritma.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: heatmap_metrik_8_algoritma.png")

# 7c. Confusion Matrices (Top 4 Models with SMOTE)
# Sort by F1-Score descending
sorted_indices = np.argsort(f1_yes)[::-1]
top4_idx = sorted_indices[:4]
top4_names = [model_names[i] for i in top4_idx]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Confusion Matrix - 4 Model Terbaik (Dengan SMOTE K=3)',
             fontsize=14, fontweight='bold')

for i, (idx, ax) in enumerate(zip(top4_idx, axes.ravel())):
    name = model_names[idx]
    cm = results_with_smote[idx]['Confusion Matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Stunted', 'SS'],
                yticklabels=['Normal', 'Stunted', 'SS'])
    ax.set_title(f'{name} (F1: {results_with_smote[idx]["F1-Score"]:.4f})',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/confusion_matrix_top4.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: confusion_matrix_top4.png")

# 7d. Perbandingan Kinerja per Model (Scatter Plot)
fig, ax = plt.subplots(figsize=(12, 7))
for i, name in enumerate(model_names):
    ax.scatter(f1_no[i], f1_yes[i], s=150, alpha=0.8,
               edgecolors='black', linewidth=1, label=name)
    ax.annotate(name, (f1_no[i], f1_yes[i]),
                textcoords="offset points", xytext=(5, 5), fontsize=9)

# Garis diagonal y=x
min_val = min(min(f1_no), min(f1_yes))
max_val = max(max(f1_no), max(f1_yes))
ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5,
        label='y = x (no change)')
# Area improved
ax.fill_between([min_val, max_val], [min_val, max_val], [max_val, max_val],
                alpha=0.1, color='green', label='SMOTE Improved')
ax.fill_between([min_val, max_val], [min_val, min_val], [min_val, max_val],
                alpha=0.1, color='red', label='SMOTE Worsened')

ax.set_xlabel('F1-Score Tanpa SMOTE', fontsize=12)
ax.set_ylabel('F1-Score Dengan SMOTE K=3', fontsize=12)
ax.set_title('Efek SMOTE K=3 pada Performa Model', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=9, ncol=2)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/smote_effect_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: smote_effect_scatter.png")

# 7e. Ranking Table
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('off')

table_data = []
for _, row in df_all.iterrows():
    table_data.append([
        row['Model'], row['SMOTE'],
        f"{row['Accuracy']:.4f}", f"{row['Precision']:.4f}",
        f"{row['Recall']:.4f}", f"{row['F1-Score']:.4f}",
        f"{row.get('ROC AUC', '-')}" if row.get('ROC AUC') else '-',
        f"{row['Train Time (s)']:.4f}"
    ])

table = ax.table(cellText=table_data,
                 colLabels=['Model', 'SMOTE', 'Accuracy', 'Precision',
                           'Recall', 'F1-Score', 'ROC AUC', 'Train Time'],
                 loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.6)

for (i, j), cell in table.get_celld().items():
    if i == 0:
        cell.set_facecolor('#2C3E50')
        cell.set_text_props(color='white', fontweight='bold')
    elif i == 1:  # Top rank
        cell.set_facecolor('#d4edda')
        cell.set_text_props(fontweight='bold')
    elif i % 2 == 0:
        cell.set_facecolor('#f8f9fa')
    else:
        cell.set_facecolor('white')

ax.set_title('Ranking Perbandingan: 8 Algoritma × 2 Skenario',
             fontsize=14, fontweight='bold', pad=20)

plt.savefig(f'{OUTPUT_DIR}/ranking_table_8_algoritma.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: ranking_table_8_algoritma.png")

# ═══════════════════════════════════════════════════════════════
# 8. KESIMPULAN
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BAGIAN 8: KESIMPULAN")
print("=" * 70)

# Find best model overall
best_no = max(results_no_smote, key=lambda x: x['F1-Score'])
best_yes = max(results_with_smote, key=lambda x: x['F1-Score'])

print(f"\n--- Model Terbaik Tanpa SMOTE ---")
print(f"  Model: {best_no['Model']}")
print(f"  F1-Score: {best_no['F1-Score']:.4f}")
print(f"  Accuracy: {best_no['Accuracy']:.4f}")

print(f"\n--- Model Terbaik Dengan SMOTE K=3 ---")
print(f"  Model: {best_yes['Model']}")
print(f"  F1-Score: {best_yes['F1-Score']:.4f}")
print(f"  Accuracy: {best_yes['Accuracy']:.4f}")

# Effect of SMOTE
improved = sum(1 for r_no, r_yes in zip(results_no_smote, results_with_smote)
               if r_yes['F1-Score'] > r_no['F1-Score'])
worsened = len(algorithms) - improved

print(f"\n--- Efek SMOTE K=3 pada {len(algorithms)} Model ---")
print(f"  Meningkat: {improved} model")
print(f"  Menurun  : {worsened} model")

models_improved = [(name, r_yes['F1-Score'] - r_no['F1-Score'])
                   for name, r_no, r_yes in zip(model_names, results_no_smote, results_with_smote)
                   if r_yes['F1-Score'] > r_no['F1-Score']]
models_improved.sort(key=lambda x: x[1], reverse=True)
print(f"\n  Model dengan peningkatan terbesar:")
for name, gain in models_improved[:3]:
    print(f"    +{name}: +{gain:.4f}")

print(f"\n  {'='*50}")
print(f"  MODEL TERBAIK KESELURUHAN:")
print(f"  {best_yes['Model']} dengan SMOTE K=3")
print(f"  F1-Score: {best_yes['F1-Score']:.4f}")
print(f"  {'='*50}")

print("\n" + "=" * 70)
print("MODEL COMPARISON SELESAI!")
print(f"Semua output tersimpan di: {OUTPUT_DIR}/")
print("=" * 70)
