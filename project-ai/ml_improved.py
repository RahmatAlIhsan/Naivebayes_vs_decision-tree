#!/usr/bin/env python3
"""
IMPROVED ML COMPARISON - 9 FITUR TERBAIK + SMOTE K=5
======================================================
Membandingkan performa SVM, XGBoost, Random Forest, KNN
dengan hanya fitur terpenting + SMOTE oversampling
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

output_dir = 'ml_improved'
os.makedirs(output_dir, exist_ok=True)

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100

print("=" * 80)
print("IMPROVED ML: 9 FITUR TERBAIK + SMOTE K=5")
print("=" * 80)

# ================================================================
# 1. LOAD DATA
# ================================================================
print("\n1. LOAD DATA")
df = pd.read_csv('preprocessed/dataset_no_outlier.csv')
print(f"Dataset: {df.shape[0]:,} baris")

df['Survival_5_years'] = df['Survival_5_years'].map({'Yes': 1, 'No': 0})
print(f"Target: Yes={df['Survival_5_years'].sum():,}, No={(df['Survival_5_years']==0).sum():,}")

# Sample 30K stratified
df_sample, _ = train_test_split(df, train_size=30000, random_state=42, stratify=df['Survival_5_years'])
print(f"Sample: {df_sample.shape[0]:,} baris")

# ================================================================
# 2. PILIH 9 FITUR TERBAIK
# ================================================================
print("\n2. 9 FITUR TERBAIK (berdasarkan Feature Importance)")

# 5 Numerik + 4 Kategorikal terbaik
num_best = ['Age', 'Tumor_Size_mm', 'Healthcare_Costs', 'Incidence_Rate_per_100K', 'Mortality_Rate_per_100K']
cat_best = ['Alcohol_Consumption', 'Diet_Risk', 'Economic_Classification', 'Healthcare_Access']

print(f"  Numerik (5): {num_best}")
print(f"  Kategorikal (4): {cat_best}")

# Ambil 9 kolom + target
selected_cols = num_best + cat_best + ['Survival_5_years']
df_9 = df_sample[selected_cols].copy()

print(f"  Shape: {df_9.shape}")
print(f"  Kolom: {list(df_9.columns)}")

# One-Hot Encoding untuk 4 kategorikal
df_encoded = pd.get_dummies(df_9, columns=cat_best, drop_first=True, dtype=int)
print(f"  Setelah OHE: {df_encoded.shape[1]} fitur ({len(num_best)} numerik + sisanya encoded)")

X = df_encoded.drop(columns=['Survival_5_years'])
y = df_encoded['Survival_5_years']
feature_names = X.columns.tolist()
print(f"\n  Fitur final ({len(feature_names)}):")
for i, f in enumerate(feature_names):
    print(f"    {i+1}. {f}")

# ================================================================
# 3. SPLIT + SMOTE
# ================================================================
print("\n3. SPLIT DATA + SMOTE OVERSAMPLING (K=5)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  SEBELUM SMOTE:")
print(f"    Train: Yes={y_train.sum():,}, No={(y_train==0).sum():,}")
print(f"    Ratio: {y_train.mean()*100:.1f}% / {(1-y_train.mean())*100:.1f}%")

# SMOTE
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"\n  SETELAH SMOTE (K=5):")
print(f"    Train: Yes={y_train_res.sum():,}, No={(y_train_res==0).sum():,}")
print(f"    Ratio: {y_train_res.mean()*100:.1f}% / {(1-y_train_res.mean())*100:.1f}%")
print(f"    Baris train: {X_train.shape[0]:,} -> {X_train_res.shape[0]:,} (+{X_train_res.shape[0]-X_train.shape[0]:,})")

# StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# ================================================================
# 4. TRAIN MODELS
# ================================================================
print("\n4. TRAINING 4 MODEL + BASELINE")

neg_pos_ratio = len(y_train_res[y_train_res==0]) / len(y_train_res[y_train_res==1])

svm_base = LinearSVC(C=1.0, class_weight='balanced', max_iter=2000, random_state=42, dual='auto')
models = {
    'SVM': CalibratedClassifierCV(svm_base, cv=3, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                              random_state=42, n_jobs=-1, eval_metric='logloss',
                              scale_pos_weight=1.0),  # balance sudah dari SMOTE
    'Random Forest': RF(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1),
    'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1),
    'Baseline': DummyClassifier(strategy='most_frequent')
}

results = {}
predictions = {}
probabilities = {}
model_names = list(models.keys())

for name, model in models.items():
    print(f"\n  [{name}] Training...")
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
    roc_auc = roc_auc_score(y_test, y_prob)

    results[name] = {
        'Accuracy': acc, 'Precision': prec, 'Recall': rec,
        'F1-Score': f1, 'ROC-AUC': roc_auc, 'Training_Time': elapsed
    }

    cm = confusion_matrix(y_test, y_pred)
    print(f"    Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")
    print(f"    F1: {f1:.4f}  | AUC: {roc_auc:.4f} | Time: {elapsed:.2f}s")
    print(f"    CM: TN={cm[0,0]:,} FP={cm[0,1]:,} FN={cm[1,0]:,} TP={cm[1,1]:,}")

# ================================================================
# 5. CONFUSION MATRICES
# ================================================================
print("\n5. GENERATING VISUALIZATIONS...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
cmaps = ['Blues', 'Greens', 'Oranges', 'Purples', 'Reds']

for i, name in enumerate(model_names):
    ax = axes[i]
    cm = confusion_matrix(y_test, predictions[name])
    cm_pct = cm / cm.sum(axis=1)[:, np.newaxis] * 100

    sns.heatmap(cm, annot=True, fmt=',', cmap=cmaps[i],
                ax=ax, cbar=True, square=True,
                xticklabels=['Predicted No', 'Predicted Yes'],
                yticklabels=['Actual No', 'Actual Yes'],
                linewidths=1, linecolor='white',
                annot_kws={'fontsize': 14, 'fontweight': 'bold'})

    for j in range(2):
        for k in range(2):
            color = 'white' if cm_pct[j, k] > 50 else 'black'
            ax.text(k + 0.5, j + 0.65, f'({cm_pct[j, k]:.1f}%)',
                    ha='center', va='center', fontsize=9, fontweight='bold', color=color)

    ax.set_title(f'{name} (9 Fitur + SMOTE)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    metrics_text = (f"Acc: {results[name]['Accuracy']:.4f}\n"
                    f"F1:  {results[name]['F1-Score']:.4f}\n"
                    f"AUC: {results[name]['ROC-AUC']:.4f}")
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

for j in range(len(model_names), 6):
    axes[j].set_visible(False)

plt.suptitle('CONFUSION MATRIX - 9 Fitur Terbaik + SMOTE K=5', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/01_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  => Saved: {output_dir}/01_confusion_matrices.png")

# ================================================================
# 6. ROC CURVES
# ================================================================
roc_models = [n for n in model_names if n != 'Baseline']

fig, ax = plt.subplots(figsize=(10, 8))
colors_roc = {'SVM': 'blue', 'XGBoost': 'green', 'Random Forest': 'orange', 'KNN': 'purple'}
linestyles = {'SVM': '-', 'XGBoost': '--', 'Random Forest': '-.', 'KNN': ':'}

for name in roc_models:
    y_score = probabilities[name]
    if y_score is not None and len(np.unique(y_score)) > 1:
        fpr, tpr, _ = roc_curve(y_test, y_score)
        auc_val = results[name]['ROC-AUC']
        ax.plot(fpr, tpr, linestyles[name], color=colors_roc[name],
                linewidth=2.5, label=f'{name} (AUC = {auc_val:.4f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.3, label='Random (AUC = 0.5)')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC CURVES - 9 Fitur Terbaik + SMOTE K=5', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/02_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  => Saved: {output_dir}/02_roc_curves.png")

# ================================================================
# 7. HASIL PERBANDINGAN
# ================================================================
print("\n" + "=" * 80)
print("HASIL PERBANDINGAN - 9 FITUR + SMOTE K=5")
print("=" * 80)

results_df = pd.DataFrame(results).T.round(4)
results_df['Rank_Acc'] = results_df['Accuracy'].rank(ascending=False).astype(int)
results_df['Rank_F1'] = results_df['F1-Score'].rank(ascending=False).astype(int)
results_df['Rank_AUC'] = results_df['ROC-AUC'].rank(ascending=False).astype(int)
results_df['Avg_Rank'] = results_df[['Rank_Acc', 'Rank_F1', 'Rank_AUC']].mean(axis=1).round(2)
results_df = results_df.sort_values('Avg_Rank')

print(f"\n{'Model':<16} {'Accuracy':>9} {'Precision':>9} {'Recall':>9} {'F1':>9} {'AUC':>9} {'Time':>8} {'Rank':>5}")
print("-" * 75)
for idx, row in results_df.iterrows():
    print(f"{idx:<16} {row['Accuracy']:>9.4f} {row['Precision']:>9.4f} {row['Recall']:>9.4f} {row['F1-Score']:>9.4f} {row['ROC-AUC']:>9.4f} {row['Training_Time']:>7.2f}s {row['Avg_Rank']:>4.1f}")
print("-" * 75)

# ================================================================
# 8. PERBANDINGAN DENGAN SEBELUMNYA
# ================================================================
print("\n" + "=" * 80)
print("PERBANDINGAN SEBELUM (47 fitur, tanpa SMOTE) vs SESUDAH (9 fitur + SMOTE)")
print("=" * 80)

before_data = {
    'SVM':       {'Accuracy': 0.5993, 'F1-Score': 0.7495, 'ROC-AUC': 0.5062},
    'XGBoost':   {'Accuracy': 0.5203, 'F1-Score': 0.5776, 'ROC-AUC': 0.5092},
    'Random Forest': {'Accuracy': 0.5178, 'F1-Score': 0.5776, 'ROC-AUC': 0.5132},
    'KNN':       {'Accuracy': 0.5348, 'F1-Score': 0.6343, 'ROC-AUC': 0.5006}
}

print(f"\n{'Model':<16} {'Metrik':<12} {'Sebelum':>10} {'Sesudah':>10} {'Perubahan':>10}")
print("-" * 58)
for name in ['SVM', 'XGBoost', 'Random Forest', 'KNN']:
    if name in results:
        for metric in ['ROC-AUC', 'F1-Score', 'Accuracy']:
            before = before_data[name][metric]
            after = results[name][metric]
            change = after - before
            arrow = '↑' if change > 0 else ('↓' if change < 0 else '→')
            print(f"{name:<16} {metric:<12} {before:>10.4f} {after:>10.4f} {arrow}{abs(change):>8.4f}")
print("-" * 58)

# Visualisasi perbandingan
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

metrics_comp = ['ROC-AUC', 'F1-Score', 'Accuracy']
titles = ['ROC-AUC Comparison', 'F1-Score Comparison', 'Accuracy Comparison']

for i, (metric, title) in enumerate(zip(metrics_comp, titles)):
    ax = axes[i]
    models_list = ['SVM', 'XGBoost', 'Random Forest', 'KNN']
    before_vals = [before_data[m][metric] for m in models_list]
    after_vals = [results[m][metric] for m in models_list]

    x = np.arange(len(models_list))
    width = 0.35
    bars1 = ax.bar(x - width/2, before_vals, width, label='Before (47 fitur)', color='#e74c3c', alpha=0.7)
    bars2 = ax.bar(x + width/2, after_vals, width, label='After (9 fitur + SMOTE)', color='#2ecc71', alpha=0.7)

    for bar, val in zip(bars1, before_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{val:.3f}', ha='center', fontsize=7, rotation=45)
    for bar, val in zip(bars2, after_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{val:.3f}', ha='center', fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(models_list, fontsize=10)
    ax.set_ylabel(metric)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    if metric == 'ROC-AUC':
        ax.set_ylim(0.4, 0.6)
    else:
        ax.set_ylim(0.4, 0.85)

plt.suptitle('PERBANDINGAN SEBELUM vs SESUDAH - 9 Fitur + SMOTE K=5', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/03_sebelum_vs_sesudah.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  => Saved: {output_dir}/03_sebelum_vs_sesudah.png")

# Summary table image
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis('off')

table_data = []
for idx, row in results_df.iterrows():
    table_data.append([
        idx,
        f"{row['Accuracy']:.4f}", f"{row['Precision']:.4f}",
        f"{row['Recall']:.4f}", f"{row['F1-Score']:.4f}",
        f"{row['ROC-AUC']:.4f}", f"{row['Training_Time']:.2f}s"
    ])

col_labels = ['Algoritma', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Time']
table = ax.table(cellText=table_data, colLabels=col_labels,
                 cellLoc='center', loc='center',
                 colWidths=[0.15, 0.14, 0.14, 0.14, 0.14, 0.14, 0.10])
table.auto_set_font_size(False)
table.set_fontsize(11)

for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor('#1a1a2e')
    cell.set_text_props(color='white', fontweight='bold')

for metric_idx in range(1, 6):
    vals = [float(table_data[i][metric_idx]) for i in range(len(table_data))]
    best_i = np.argmax(vals)
    for i in range(len(table_data)):
        cell = table[i+1, metric_idx]
        if i == best_i:
            cell.set_facecolor('#d4edda')
            cell.set_text_props(fontweight='bold')
        else:
            cell.set_facecolor('#f8f9fa')

for i in range(len(table_data)):
    table[i+1, 0].set_facecolor('#e3f2fd')
    table[i+1, 0].set_text_props(fontweight='bold')

ax.set_title('HASIL - 9 Fitur Terbaik + SMOTE K=5', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/04_summary_table.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  => Saved: {output_dir}/04_summary_table.png")

# ================================================================
# 9. SAVE RESULTS
# ================================================================
results_df.to_csv(f'{output_dir}/results_9fitur_smote.csv')
print(f"\n  => Saved: {output_dir}/results_9fitur_smote.csv")

# Feature list
feat_df = pd.DataFrame({'Feature': feature_names})
feat_df.to_csv(f'{output_dir}/selected_features.csv', index=False)

# ================================================================
# 10. FINAL SUMMARY
# ================================================================
print("\n" + "=" * 80)
print("KESIMPULAN")
print("=" * 80)

print(f"""
KONFIGURASI:
  - Fitur: 9 terbaik ({len(num_best)} numerik + {len(cat_best)} kategorikal -> {len(feature_names)} encoded)
  - SMOTE: K=5 (balance classes)
  - Data: 30.000 sampel stratified, 80/20 split
  - Data train setelah SMOTE: {X_train_res.shape[0]:,} ({y_train_res.sum():,} Yes, {(y_train_res==0).sum():,} No)

PERINGKAT:
""")
for i, idx in enumerate(results_df.index):
    print(f"  {i+1}st: {idx} (Avg Rank: {results_df['Avg_Rank'].loc[idx]})")

print(f"""
PERBANDINGAN DENGAN SEBELUMNYA:
""")
for name in ['SVM', 'XGBoost', 'Random Forest', 'KNN']:
    if name in results:
        auc_before = before_data[name]['ROC-AUC']
        auc_after = results[name]['ROC-AUC']
        delta = auc_after - auc_before
        arrow = '↑' if delta > 0 else ('↓' if delta < 0 else '→')
        print(f"  {name:<16} AUC: {auc_before:.4f} -> {auc_after:.4f} {arrow}{abs(delta):.4f}")

print(f"""
OUTPUT: folder '{output_dir}/'
  01_confusion_matrices.png    - Confusion matrix semua model
  02_roc_curves.png            - ROC Curves
  03_sebelum_vs_sesudah.png    - Perbandingan Before vs After
  04_summary_table.png         - Tabel hasil
  results_9fitur_smote.csv     - Hasil metrics
  selected_features.csv        - Daftar 9 fitur terpilih
""")
print("=" * 80)
