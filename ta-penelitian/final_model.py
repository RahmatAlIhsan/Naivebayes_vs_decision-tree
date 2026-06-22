#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
======================================================================
FINAL MODEL - Prediksi Stunting pada Balita
======================================================================
Naive Bayes (Pembanding) vs Decision Tree (Model Terbaik)

- Naive Bayes  : Tanpa SMOTE, Tanpa Tuning (sebagai baseline pembanding)
- Decision Tree: SMOTE K=3 + Hyperparameter Tuning (model terbaik)

Berdasarkan hasil Model Comparison dari 8 algoritma, Decision Tree
dengan SMOTE K=3 + Tuning terpilih sebagai model terbaik.
======================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Handle Windows Unicode encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams.update({
    'figure.figsize': (10, 6), 'font.size': 12,
    'axes.titlesize': 14, 'axes.titleweight': 'bold'
})

OUTPUT_DIR = 'data/processed/final_model'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("FINAL MODEL - Prediksi Stunting pada Balita")
print("Naive Bayes (Pembanding) vs Decision Tree (Terbaik)")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════
# 1. LOAD DATA & FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BAGIAN 1: LOAD DATA & FEATURE ENGINEERING")
print("=" * 70)

df = pd.read_csv('data/processed/stunting_clean.csv')
print(f"Dataset: {df.shape[0]} baris, {df.shape[1]} kolom")

# Encode
df['jk_encoded'] = df['jk'].map({'Laki-Laki': 1, 'Perempuan': 0})
target_mapping = {'Normal': 0, 'Stunted': 1, 'Severely Stunted': 2}
df['status_encoded'] = df['status'].map(target_mapping)

feature_cols = ['umur_bulan', 'tinggi', 'jk_encoded']
X = df[feature_cols].values
y = df['status_encoded'].values

# Distribusi
dist = df['status'].value_counts()
print("\nDistribusi Target:")
for s in ['Normal', 'Stunted', 'Severely Stunted']:
    cnt = dist.get(s, 0)
    print(f"  {s:20} {cnt:5} ({cnt/len(df)*100:.2f}%)")

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
print(f"\nTrain distribution:\n{pd.Series(y_train).value_counts().sort_index()}")
print(f"\nTest distribution:\n{pd.Series(y_test).value_counts().sort_index()}")

# ═══════════════════════════════════════════════════════════════
# 3. NAIVE BAYES - PEMBANDING (Tanpa SMOTE, Tanpa Tuning)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BAGIAN 3: NAIVE BAYES (PEMBANDING)")
print("Tanpa SMOTE - Tanpa Hyperparameter Tuning")
print("=" * 70)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)

nb_acc  = accuracy_score(y_test, nb_pred)
nb_prec = precision_score(y_test, nb_pred, average='weighted', zero_division=0)
nb_rec  = recall_score(y_test, nb_pred, average='weighted', zero_division=0)
nb_f1   = f1_score(y_test, nb_pred, average='weighted', zero_division=0)
nb_cm   = confusion_matrix(y_test, nb_pred)
nb_cr   = classification_report(y_test, nb_pred,
            target_names=['Normal', 'Stunted', 'Severely Stunted'],
            zero_division=0, output_dict=True)

print(f"\n  Accuracy : {nb_acc:.4f}")
print(f"  Precision: {nb_prec:.4f}")
print(f"  Recall   : {nb_rec:.4f}")
print(f"  F1-Score : {nb_f1:.4f}")
print(f"\n  Confusion Matrix:\n{nb_cm}")

# Detail per kelas
print(f"\n  Detail per Kelas:")
for kelas in ['Normal', 'Stunted', 'Severely Stunted']:
    if kelas in nb_cr:
        print(f"  {kelas:20} | Precision: {nb_cr[kelas]['precision']:.4f} | "
              f"Recall: {nb_cr[kelas]['recall']:.4f} | "
              f"F1: {nb_cr[kelas]['f1-score']:.4f} | "
              f"Support: {int(nb_cr[kelas]['support'])}")

# ═══════════════════════════════════════════════════════════════
# 4. DECISION TREE - MODEL TERBAIK (SMOTE K=3 + Tuning)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BAGIAN 4: DECISION TREE (MODEL TERBAIK)")
print("SMOTE K=3 + Hyperparameter Tuning")
print("=" * 70)

# 4a. SMOTE K=3
print("\n[4a] SMOTE K=3")
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"  Sebelum SMOTE: {X_train.shape[0]} sampel")
print(f"  Sesudah SMOTE: {X_train_res.shape[0]} sampel")
res_dist = pd.Series(y_train_res).value_counts().sort_index()
print(f"  Distribusi: Normal={res_dist.get(0,0)}, Stunted={res_dist.get(1,0)}, "
      f"Severely Stunted={res_dist.get(2,0)}")

# 4b. Hyperparameter Tuning
print("\n[4b] Hyperparameter Tuning (GridSearchCV)")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

dt_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42, k_neighbors=3)),
    ('dt', DecisionTreeClassifier(random_state=42))
])

dt_param_grid = {
    'dt__criterion': ['gini', 'entropy'],
    'dt__max_depth': [3, 5, 7, 10, 15, None],
    'dt__min_samples_split': [2, 5, 10, 20],
    'dt__min_samples_leaf': [1, 2, 5, 10],
    'dt__max_features': [None, 'sqrt', 'log2']
}

n_combinations = 1
for v in dt_param_grid.values():
    n_combinations *= len(v)
print(f"  Total kombinasi: {n_combinations}")

dt_grid = GridSearchCV(dt_pipeline, dt_param_grid, cv=cv,
                       scoring='f1_weighted', n_jobs=-1, verbose=1)
dt_grid.fit(X_train, y_train)

best_params = dt_grid.best_params_
print(f"\n  Best params: {best_params}")
print(f"  Best CV F1  : {dt_grid.best_score_:.4f}")

# Evaluasi pada test set
dt_pred = dt_grid.predict(X_test)

dt_acc  = accuracy_score(y_test, dt_pred)
dt_prec = precision_score(y_test, dt_pred, average='weighted', zero_division=0)
dt_rec  = recall_score(y_test, dt_pred, average='weighted', zero_division=0)
dt_f1   = f1_score(y_test, dt_pred, average='weighted', zero_division=0)
dt_cm   = confusion_matrix(y_test, dt_pred)
dt_cr   = classification_report(y_test, dt_pred,
            target_names=['Normal', 'Stunted', 'Severely Stunted'],
            zero_division=0, output_dict=True)

print(f"\n  Accuracy : {dt_acc:.4f}")
print(f"  Precision: {dt_prec:.4f}")
print(f"  Recall   : {dt_rec:.4f}")
print(f"  F1-Score : {dt_f1:.4f}")
print(f"\n  Confusion Matrix:\n{dt_cm}")

print(f"\n  Detail per Kelas:")
for kelas in ['Normal', 'Stunted', 'Severely Stunted']:
    if kelas in dt_cr:
        print(f"  {kelas:20} | Precision: {dt_cr[kelas]['precision']:.4f} | "
              f"Recall: {dt_cr[kelas]['recall']:.4f} | "
              f"F1: {dt_cr[kelas]['f1-score']:.4f} | "
              f"Support: {int(dt_cr[kelas]['support'])}")

# ═══════════════════════════════════════════════════════════════
# 5. SIMPAN HASIL
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BAGIAN 5: SIMPAN HASIL")
print("=" * 70)

# 5a. Tabel Perbandingan Final
final_results = pd.DataFrame([
    {
        'Model': 'Naive Bayes (Pembanding)',
        'SMOTE': 'Tidak',
        'Tuning': 'Tidak',
        'Accuracy': round(nb_acc, 4),
        'Precision': round(nb_prec, 4),
        'Recall': round(nb_rec, 4),
        'F1-Score': round(nb_f1, 4),
        'Keterangan': 'Baseline - Gagal deteksi minoritas (Accuracy Paradox)'
    },
    {
        'Model': 'Decision Tree (Terbaik)',
        'SMOTE': 'Ya (K=3)',
        'Tuning': 'Ya',
        'Accuracy': round(dt_acc, 4),
        'Precision': round(dt_prec, 4),
        'Recall': round(dt_rec, 4),
        'F1-Score': round(dt_f1, 4),
        'Keterangan': 'Model Terbaik - Mampu deteksi Stunted & Severely Stunted'
    }
])

final_results.to_csv(f'{OUTPUT_DIR}/hasil_final.csv', index=False)
print(f"  CSV: {OUTPUT_DIR}/hasil_final.csv")
print(final_results.to_string(index=False))

# 5b. Detail per Kelas per Model
for name, cr in [('Naive Bayes', nb_cr), ('Decision Tree', dt_cr)]:
    detail = []
    for kelas in ['Normal', 'Stunted', 'Severely Stunted']:
        if kelas in cr:
            detail.append({
                'Model': name, 'Kelas': kelas,
                'Precision': round(cr[kelas]['precision'], 4),
                'Recall': round(cr[kelas]['recall'], 4),
                'F1-Score': round(cr[kelas]['f1-score'], 4),
                'Support': int(cr[kelas]['support'])
            })
    fname = name.lower().replace(' ', '_')
    pd.DataFrame(detail).to_csv(f'{OUTPUT_DIR}/detail_{fname}.csv', index=False)
    print(f"  Detail {name}: {OUTPUT_DIR}/detail_{fname}.csv")

# 5c. Simpan tuning detail
dt_results = pd.DataFrame(dt_grid.cv_results_)
dt_top = dt_results.sort_values('rank_test_score').head(10)
dt_top.to_csv(f'{OUTPUT_DIR}/tuning_decision_tree.csv', index=False)
print(f"  Tuning DT: {OUTPUT_DIR}/tuning_decision_tree.csv")

# ═══════════════════════════════════════════════════════════════
# 6. VISUALISASI
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BAGIAN 6: VISUALISASI")
print("=" * 70)

# 6a. Confusion Matrix - Dua Model Berdampingan
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle('Perbandingan Confusion Matrix', fontsize=16, fontweight='bold', y=1.02)

for ax, (name, cm, cmap) in zip(axes, [
    ('Naive Bayes (Pembanding)\nTanpa SMOTE, Tanpa Tuning', nb_cm, 'Blues'),
    ('Decision Tree (Terbaik)\nSMOTE K=3 + Tuning', dt_cm, 'Greens')
]):
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=['Normal', 'Stunted', 'SS'],
                yticklabels=['Normal', 'Stunted', 'SS'],
                cbar=False, linewidths=1, linecolor='white',
                annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_title(name, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    total = cm.sum()
    benar = cm[0, 0] + cm[1, 1] + cm[2, 2]
    acc_local = benar / total
    ax.text(0.5, -0.18, f'Correct: {benar}/{total} ({acc_local:.1%})',
            ha='center', va='top', transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/confusion_matrix_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: confusion_matrix_comparison.png")

# 6b. Bar Chart Perbandingan Metrik
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(final_results))
width = 0.2
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

for i, metric in enumerate(metrics):
    bars = ax.bar(x + i * width, final_results[metric].values, width,
                  label=metric, color=colors[i], alpha=0.85,
                  edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, final_results[metric].values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(['Naive Bayes\n(Pembanding)', 'Decision Tree\n(Terbaik)'], fontsize=11)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Perbandingan Metrik: Naive Bayes vs Decision Tree', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1.15)
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Annotation: Accuracy Paradox
ax.annotate('Accuracy Paradox!\nNB akurasi {:.1%} tapi\ntidak bisa deteksi stunting'.format(nb_acc),
            xy=(0, nb_acc), xytext=(0.35, 0.5),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=9, color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#FFE4E4', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/metric_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: metric_comparison.png")

# 6c. Distribusi Kelas - Efek SMOTE
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle('Efek SMOTE K=3 pada Distribusi Kelas Training', fontsize=15, fontweight='bold')

labels_bar = ['Normal', 'Stunted', 'Severely\nStunted']
colors_status = ['#2ECC71', '#F39C12', '#E74C3C']

for i, (data, title, ax) in enumerate([
    (pd.Series(y_train), 'Sebelum SMOTE (Imbalance)', axes[0]),
    (pd.Series(y_train_res), 'Sesudah SMOTE K=3 (Balanced)', axes[1])
]):
    counts = data.value_counts().sort_index()
    bar_colors = [colors_status[j] for j in range(len(counts))]
    bars = ax.bar(labels_bar[:len(counts)], counts.values, color=bar_colors,
                  edgecolor='black', linewidth=1.5, width=0.6)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{val}', ha='center', va='bottom', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel('Jumlah Sampel', fontsize=11)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/smote_effect.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: smote_effect.png")

# 6d. Classification Report Tables
for model_name, cr, color in [
    ('Naive Bayes (Pembanding)', nb_cr, '#E74C3C'),
    ('Decision Tree (Terbaik)', dt_cr, '#2ECC71')
]:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')

    table_data = []
    for kelas in ['Normal', 'Stunted', 'Severely Stunted']:
        if kelas in cr:
            table_data.append([
                kelas,
                f"{cr[kelas]['precision']:.4f}",
                f"{cr[kelas]['recall']:.4f}",
                f"{cr[kelas]['f1-score']:.4f}",
                f"{int(cr[kelas]['support'])}"
            ])

    table_data.append(['─' * 5, '─' * 8, '─' * 8, '─' * 8, '─' * 7])
    for avg_type in ['weighted avg', 'macro avg']:
        if avg_type in cr:
            table_data.append([
                avg_type,
                f"{cr[avg_type]['precision']:.4f}",
                f"{cr[avg_type]['recall']:.4f}",
                f"{cr[avg_type]['f1-score']:.4f}",
                f"{int(cr[avg_type]['support'])}"
            ])

    acc_val = nb_acc if 'Naive' in model_name else dt_acc
    table_data.append(['Accuracy', f"{acc_val:.4f}", f"{acc_val:.4f}", f"{acc_val:.4f}", ''])

    table = ax.table(cellText=table_data,
                     colLabels=['Kelas', 'Precision', 'Recall', 'F1-Score', 'Support'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor(color)
            cell.set_text_props(color='white', fontweight='bold')
        elif i == len(table_data) - 1:
            cell.set_facecolor('#d4edda')
            cell.set_text_props(fontweight='bold')
        elif i % 2 == 0:
            cell.set_facecolor('#f8f9fa')
        else:
            cell.set_facecolor('white')

    ax.set_title(f'Classification Report - {model_name}', fontsize=13, fontweight='bold', pad=20)

    fname = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(f'{OUTPUT_DIR}/classification_report_{fname}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: classification_report_{fname}.png")

# 6e. Learning Curve: Top 15 DT Parameters
fig, ax = plt.subplots(figsize=(14, 6))
dt_top_15 = dt_results.sort_values('rank_test_score').head(15)

param_labels = []
for _, row in dt_top_15.iterrows():
    label = (f"depth={row['param_dt__max_depth']}, "
             f"split={row['param_dt__min_samples_split']}, "
             f"leaf={row['param_dt__min_samples_leaf']}")
    param_labels.append(label)

x = np.arange(len(param_labels))
bars = ax.bar(x, dt_top_15['mean_test_score'].values,
              yerr=dt_top_15['std_test_score'].values,
              color='#2ECC71', alpha=0.8, edgecolor='black',
              capsize=3, width=0.7)

bars[0].set_color('#27AE60')
bars[0].set_edgecolor('black')
bars[0].set_linewidth(2)

ax.set_xticks(x)
ax.set_xticklabels(param_labels, fontsize=8, rotation=45, ha='right')
ax.set_ylabel('F1-Weighted Score (CV)', fontsize=12)
ax.set_title('Top 15 Kombinasi Parameter - Decision Tree (SMOTE K=3)',
             fontsize=14, fontweight='bold')
best_val = dt_top_15['mean_test_score'].values[0]
ax.axhline(y=best_val, color='red', linestyle='--', linewidth=1.5,
           label=f"Best: {best_val:.4f}")
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/dt_tuning_top_params.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: dt_tuning_top_params.png")

# 6f. Radar Chart
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

nb_vals = [nb_acc, nb_prec, nb_rec, nb_f1] + [nb_acc]
dt_vals = [dt_acc, dt_prec, dt_rec, dt_f1] + [dt_acc]

ax.plot(angles, nb_vals, 'o-', linewidth=2, label='Naive Bayes (Pembanding)', color='#E74C3C')
ax.fill(angles, nb_vals, alpha=0.1, color='#E74C3C')
ax.plot(angles, dt_vals, 'o-', linewidth=2, label='Decision Tree (Terbaik)', color='#2ECC71')
ax.fill(angles, dt_vals, alpha=0.1, color='#2ECC71')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.1)
ax.set_title('Radar Chart Perbandingan Model', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
ax.grid(True)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/radar_chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: radar_chart.png")

# 6g. Tabel Perbandingan Lengkap
fig, ax = plt.subplots(figsize=(12, 3.5))
ax.axis('off')

comparison_data = []
for _, row in final_results.iterrows():
    comparison_data.append([
        row['Model'], row['SMOTE'], row['Tuning'],
        f"{row['Accuracy']:.4f}", f"{row['Precision']:.4f}",
        f"{row['Recall']:.4f}", f"{row['F1-Score']:.4f}",
        row['Keterangan']
    ])

table = ax.table(cellText=comparison_data,
                 colLabels=['Model', 'SMOTE', 'Tuning', 'Accuracy',
                           'Precision', 'Recall', 'F1-Score', 'Keterangan'],
                 loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.2)

for (i, j), cell in table.get_celld().items():
    if i == 0:
        cell.set_facecolor('#2C3E50')
        cell.set_text_props(color='white', fontweight='bold')
    elif i == 2:
        cell.set_facecolor('#d4edda')
        cell.set_text_props(fontweight='bold')
    elif i % 2 == 0:
        cell.set_facecolor('#f8f9fa')
    else:
        cell.set_facecolor('white')

ax.set_title('Tabel Perbandingan Final: Naive Bayes vs Decision Tree',
             fontsize=14, fontweight='bold', pad=20)

plt.savefig(f'{OUTPUT_DIR}/final_comparison_table.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: final_comparison_table.png")

# ═══════════════════════════════════════════════════════════════
# 7. RINGKASAN FINAL
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RINGKASAN FINAL")
print("=" * 70)

# Hitung metrik tambahan untuk ringkasan
nb_stunted_recall = nb_cr.get('Stunted', {}).get('recall', 0)
nb_ss_recall = nb_cr.get('Severely Stunted', {}).get('recall', 0)
dt_stunted_recall = dt_cr.get('Stunted', {}).get('recall', 0)
dt_ss_recall = dt_cr.get('Severely Stunted', {}).get('recall', 0)

print(f"""
Dataset         : stunting_clean.csv ({df.shape[0]} sampel, {df.shape[1]} kolom)
Split           : 80:20 (Stratified)
Features        : {feature_cols}
Target          : Normal(0), Stunted(1), Severely Stunted(2)

┌─────────────────────────────────────────────────────────────┐
│ MODEL A: NAIVE BAYES (PEMBANDING)                            │
├─────────────────────────────────────────────────────────────┤
│ Preprocessing  : Tanpa SMOTE, Tanpa Tuning                   │
│ F1-Score       : {nb_f1:.4f}                                              │
│ Recall Stunted : {nb_stunted_recall:.1%} ({'GAGAL total!' if nb_stunted_recall == 0 else 'Terbatas'})                    │
│ Recall SS      : {nb_ss_recall:.1%} ({'GAGAL total!' if nb_ss_recall == 0 else 'Terbatas'})                    │
│ Catatan        : Accuracy Paradox - akurasi {nb_acc:.1%} tapi                 │
│                   tidak bisa mendeteksi stunting sama sekali!  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ MODEL B: DECISION TREE (TERBAIK)                             │
├─────────────────────────────────────────────────────────────┤
│ Preprocessing  : SMOTE K=3 + Hyperparameter Tuning           │
│ Best Params    : {best_params}           │
│ F1-Score       : {dt_f1:.4f}                                              │
│ Recall Stunted : {dt_stunted_recall:.1%} ({'Baik' if dt_stunted_recall > 0.5 else 'Perlu ditingkatkan'})                      │
│ Recall SS      : {dt_ss_recall:.1%} ({'Baik' if dt_ss_recall > 0.5 else 'Perlu ditingkatkan'})                      │
│ Catatan        : Mampu deteksi Stunted & Severely Stunted     │
│                   dengan keseimbangan precision-recall yang baik │
└─────────────────────────────────────────────────────────────┘

KESIMPULAN:
  Decision Tree dengan SMOTE K=3 + Hyperparameter Tuning
  terpilih sebagai MODEL TERBAIK untuk prediksi stunting.

  Alasan:
  1. F1-Score tertinggi: {dt_f1:.4f}
  2. Mampu mendeteksi kelas minoritas (Stunted & Severely Stunted)
  3. Tidak terpengaruh korelasi fitur (beda dengan Naive Bayes)
  4. SMOTE efektif meningkatkan recall kelas minoritas

  HASIL FINAL:
  Accuracy : {dt_acc:.4f} | Precision : {dt_prec:.4f}
  Recall   : {dt_rec:.4f} | F1-Score  : {dt_f1:.4f}
""")

print("=" * 70)
print("FINAL MODEL SELESAI!")
print(f"Semua output tersimpan di: {OUTPUT_DIR}/")
print("=" * 70)
