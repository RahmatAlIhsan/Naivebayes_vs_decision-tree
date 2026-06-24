#!/usr/bin/env python
"""
============================================================
EKSPERIMEN 1: TANPA SMOTE - TANPA HYPERPARAMETER TUNING
Model: Naive Bayes vs Decision Tree
Dataset: stunting_clean.csv
============================================================
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams.update({'figure.figsize': (10, 6), 'font.size': 12})

OUTPUT_DIR = 'data/processed/eksperimen_1'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("EKSPERIMEN 1: TANPA SMOTE - TANPA HYPERPARAMETER TUNING")
print("=" * 70)

# ── 1. LOAD DATA ──
print("\n[1] LOAD DATA")
df = pd.read_csv('data/processed/stunting_clean.csv')
print("Shape: {}".format(df.shape))
print("Status counts:\n{}".format(df['status'].value_counts()))

# ── 2. FEATURE ENGINEERING ──
print("\n[2] FEATURE ENGINEERING")
df['jk_encoded'] = df['jk'].map({'Laki-Laki': 1, 'Perempuan': 0})
target_mapping = {'Normal': 0, 'Stunted': 1, 'Severely Stunted': 2}
df['status_encoded'] = df['status'].map(target_mapping)

feature_cols = ['umur_bulan', 'tinggi', 'jk_encoded']
X = df[feature_cols].values
y = df['status_encoded'].values

print("Features: {}".format(feature_cols))
print("X: {}, y: {}".format(X.shape, y.shape))

# ── 3. SPLIT DATA ──
print("\n[3] SPLIT DATA 80:20 (STRATIFIED)")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train: {}, Test: {}".format(X_train.shape[0], X_test.shape[0]))
train_dist = pd.Series(y_train).value_counts().sort_index()
print("Train distribution:\n{}".format(train_dist))

# ── 4. MODEL ──
print("\n[4] DEFINISI MODEL (DEFAULT)")
models = {
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}
print("Models: {}".format(list(models.keys())))

# ── 5. TRAINING ──
print("\n[5] TRAINING & EVALUASI")
header = "{:20} {:10} {:10} {:10} {:10}".format(
    'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score')
print("=" * 80)
print(header)
print("=" * 80)

results = []
confusion_matrices = {}
classification_reports = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    results.append({
        'Model': name,
        'SMOTE': 'Tidak',
        'Hyperparameter Tuning': 'Tidak',
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4)
    })

    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices[name] = cm

    cr = classification_report(y_test, y_pred,
                               target_names=['Normal', 'Stunted', 'Severely Stunted'],
                               zero_division=0, output_dict=True)
    classification_reports[name] = cr

    print("{:20} {:.4f}     {:.4f}     {:.4f}     {:.4f}".format(
        name, acc, prec, rec, f1))
    print("  Confusion Matrix:\n{}".format(cm))
    print()

print("=" * 80)

# ── 6. SAVE RESULTS ──
print("\n[6] SIMPAN HASIL")
df_results = pd.DataFrame(results)
csv_path = os.path.join(OUTPUT_DIR, 'hasil_perbandingan.csv')
df_results.to_csv(csv_path, index=False)
print("CSV saved: {}".format(csv_path))
print(df_results.to_string(index=False))

# ── 7. VISUALISASI ──
print("\n[7] VISUALISASI")

# 7a. Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Eksperimen 1: Tanpa SMOTE - Tanpa Tuning', fontsize=15, fontweight='bold')

for i, (name, cm) in enumerate(confusion_matrices.items()):
    ax = axes[i]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Stunted', 'Severely Stunted'],
                yticklabels=['Normal', 'Stunted', 'Severely Stunted'])
    ax.set_title(name, fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: confusion_matrices.png")

# 7b. Metrik Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))
x_pos = np.arange(len(df_results))
width = 0.2
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

for i, (metric, color) in enumerate(zip(metrics, colors)):
    bars = ax.bar(x_pos + i * width, df_results[metric].values, width,
                  label=metric, color=color, alpha=0.85, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, df_results[metric].values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                '{:.4f}'.format(val), ha='center', va='bottom', fontsize=9)

ax.set_xticks(x_pos + width * 1.5)
ax.set_xticklabels(df_results['Model'].values, fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Perbandingan Metrik: Tanpa SMOTE - Tanpa Tuning', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1.1)
ax.legend(loc='lower right', fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'perbandingan_metrik.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: perbandingan_metrik.png")

# 7c. Distribusi Kelas
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Distribusi Kelas: Data Asli (Imbalance)', fontsize=14, fontweight='bold')

train_series = pd.Series(y_train).map({0: 'Normal', 1: 'Stunted', 2: 'Severely Stunted'})
test_series = pd.Series(y_test).map({0: 'Normal', 1: 'Stunted', 2: 'Severely Stunted'})

colors_status = {'Normal': '#2ECC71', 'Stunted': '#F39C12', 'Severely Stunted': '#E74C3C'}

for i, (data, title, ax) in enumerate([
    (train_series, 'Train Set (80%)', axes[0]),
    (test_series, 'Test Set (20%)', axes[1])
]):
    counts = data.value_counts()
    bar_colors = [colors_status.get(s, '#95A5A6') for s in counts.index]
    bars = ax.bar(counts.index, counts.values, color=bar_colors,
                  edgecolor='black', linewidth=1)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13)
    ax.set_ylabel('Jumlah Sampel')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'distribusi_kelas.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: distribusi_kelas.png")

# 7d. Classification Report per Model
for name, cr in classification_reports.items():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    table_data = []
    for kelas in ['Normal', 'Stunted', 'Severely Stunted']:
        if kelas in cr:
            table_data.append([
                kelas,
                '{:.4f}'.format(cr[kelas]['precision']),
                '{:.4f}'.format(cr[kelas]['recall']),
                '{:.4f}'.format(cr[kelas]['f1-score']),
                str(int(cr[kelas]['support']))
            ])

    table_data.append(['---', '---', '---', '---', '---'])
    for avg_type in ['weighted avg', 'macro avg']:
        if avg_type in cr:
            table_data.append([
                avg_type,
                '{:.4f}'.format(cr[avg_type]['precision']),
                '{:.4f}'.format(cr[avg_type]['recall']),
                '{:.4f}'.format(cr[avg_type]['f1-score']),
                str(int(cr[avg_type]['support']))
            ])

    table = ax.table(cellText=table_data,
                     colLabels=['Kelas', 'Precision', 'Recall', 'F1-Score', 'Support'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#2E86AB')
            cell.set_text_props(color='white', fontweight='bold')
        elif i == len(table_data) - 1:
            cell.set_facecolor('#d4edda')
        elif i % 2 == 0:
            cell.set_facecolor('#f8f9fa')
        else:
            cell.set_facecolor('white')

    safe_name = name.lower().replace(' ', '_')
    ax.set_title('Classification Report: {} (Tanpa SMOTE - Tanpa Tuning)'.format(name),
                 fontsize=13, fontweight='bold', pad=20)

    img_path = os.path.join(OUTPUT_DIR, 'classification_report_{}.png'.format(safe_name))
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: classification_report_{}.png'.format(safe_name))

print("\n" + "=" * 70)
print("EKSPERIMEN 1 SELESAI!")
print("Semua output tersimpan di: {}/".format(OUTPUT_DIR))
print("=" * 70)
