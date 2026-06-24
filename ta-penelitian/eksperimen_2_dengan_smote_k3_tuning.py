#!/usr/bin/env python3
"""
============================================================
EKSPERIMEN 2: SMOTE K=3 - DENGAN HYPERPARAMETER TUNING
Model: Naive Bayes vs Decision Tree
Dataset: stunting_clean.csv
============================================================
"""

import os
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

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams.update({'figure.figsize': (10, 6), 'font.size': 12})

OUTPUT_DIR = 'data/processed/eksperimen_2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("EKSPERIMEN 2: SMOTE K=3 - DENGAN HYPERPARAMETER TUNING")
print("=" * 70)

# ── 1. LOAD DATA ──
print("\n[1] LOAD DATA")
df = pd.read_csv('data/processed/stunting_clean.csv')
print(f"Shape: {df.shape}")
print(f"Status counts:\n{df['status'].value_counts()}")

# ── 2. FEATURE ENGINEERING ──
print("\n[2] FEATURE ENGINEERING")
df['jk_encoded'] = df['jk'].map({'Laki-Laki': 1, 'Perempuan': 0})
target_mapping = {'Normal': 0, 'Stunted': 1, 'Severely Stunted': 2}
df['status_encoded'] = df['status'].map(target_mapping)

feature_cols = ['umur_bulan', 'tinggi', 'jk_encoded']
X = df[feature_cols].values
y = df['status_encoded'].values

print(f"Features: {feature_cols}")
print(f"X: {X.shape}, y: {y.shape}")
print(f"Target classes: {np.unique(y)}")

# ── 3. SPLIT DATA (80:20 STRATIFIED) ──
print("\n[3] SPLIT DATA 80:20 (STRATIFIED)")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

train_dist = pd.Series(y_train).value_counts().sort_index()
test_dist = pd.Series(y_test).value_counts().sort_index()
print(f"\nTrain distribution (SEBELUM SMOTE):\n{train_dist}")

# ── 4. SMOTE K=3 ──
print("\n[4] SMOTE DENGAN K=3")
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

train_res_dist = pd.Series(y_train_res).value_counts().sort_index()
print(f"Train distribution (SETELAH SMOTE K=3):")
print(f"Total samples: {X_train_res.shape[0]}")
print(train_res_dist)

# ── 5. VISUALISASI PERBANDINGAN SEBELUM VS SESUDAH SMOTE ──
print("\n[5] VISUALISASI SMOTE")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Perbandingan Distribusi Kelas: Sebelum vs Sesudah SMOTE K=3',
             fontsize=14, fontweight='bold')

colors_status = ['#2ECC71', '#F39C12', '#E74C3C']
labels = ['Normal', 'Stunted', 'Severely Stunted']

for i, (data, title, ax) in enumerate([
    (pd.Series(y_train), 'Sebelum SMOTE (Imbalance)', axes[0]),
    (pd.Series(y_train_res), 'Sesudah SMOTE K=3 (Balanced)', axes[1])
]):
    counts = data.value_counts().sort_index()
    bars = ax.bar(labels, counts.values, color=colors_status,
                  edgecolor='black', linewidth=1)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13)
    ax.set_ylabel('Jumlah Sampel')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/perbandingan_smote_k3.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/perbandingan_smote_k3.png")

# ── 6. MODEL DENGAN SMOTE + HYPERPARAMETER TUNING ──
print("\n[6] HYPERPARAMETER TUNING DENGAN GRIDSEARCHCV")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 6a. Naive Bayes - Hyperparameter Tuning
print("\n--- Naive Bayes Tuning ---")
nb_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42, k_neighbors=3)),
    ('classifier', GaussianNB())
])

nb_param_grid = {
    'classifier__var_smoothing': np.logspace(-12, -5, 8)
}

nb_grid = GridSearchCV(
    nb_pipeline, nb_param_grid, cv=cv,
    scoring='f1_weighted', n_jobs=-1, verbose=1
)
nb_grid.fit(X_train, y_train)

print(f"Best params (Naive Bayes): {nb_grid.best_params_}")
print(f"Best CV score: {nb_grid.best_score_:.4f}")

# 6b. Decision Tree - Hyperparameter Tuning
print("\n--- Decision Tree Tuning ---")
dt_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42, k_neighbors=3)),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

dt_param_grid = {
    'classifier__max_depth': [3, 5, 7, 10, 15, None],
    'classifier__min_samples_split': [2, 5, 10, 20],
    'classifier__min_samples_leaf': [1, 2, 5, 10],
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__max_features': [None, 'sqrt', 'log2']
}

dt_grid = GridSearchCV(
    dt_pipeline, dt_param_grid, cv=cv,
    scoring='f1_weighted', n_jobs=-1, verbose=1
)
dt_grid.fit(X_train, y_train)

print(f"Best params (Decision Tree): {dt_grid.best_params_}")
print(f"Best CV score: {dt_grid.best_score_:.4f}")

# ── 7. EVALUASI PADA TEST SET ──
print("\n[7] EVALUASI PADA TEST SET")
print("=" * 80)
print(f"{'Model':20} {'Accuracy':10} {'Precision':10} {'Recall':10} {'F1-Score':10}")
print("=" * 80)

models_tuned = {
    'Naive Bayes (Tuned)': nb_grid.best_estimator_,
    'Decision Tree (Tuned)': dt_grid.best_estimator_
}

results = []
confusion_matrices = {}
classification_reports = {}

for name, pipeline in models_tuned.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    results.append({
        'Model': name,
        'SMOTE': 'Ya (K=3)',
        'Hyperparameter Tuning': 'Ya',
        'Best Params': str(pipeline.named_steps['classifier'].get_params()),
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

    print(f"{name:20} {acc:.4f}{'':6} {prec:.4f}{'':6} {rec:.4f}{'':6} {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print()

print("=" * 80)

# ── 8. DETAIL HASIL TUNING ──
print("\n[8] DETAIL HASIL TUNING")

print("\n--- Naive Bayes Tuning Details ---")
nb_results = pd.DataFrame(nb_grid.cv_results_)
nb_display = nb_results[['param_classifier__var_smoothing', 'mean_test_score', 'std_test_score', 'rank_test_score']]
nb_display = nb_display.sort_values('rank_test_score')
print(nb_display.to_string(index=False))

print("\n--- Decision Tree Tuning Details ---")
dt_results = pd.DataFrame(dt_grid.cv_results_)
dt_cols = [c for c in dt_results.columns if c.startswith('param_') or c in ['mean_test_score', 'std_test_score', 'rank_test_score']]
dt_display = dt_results[dt_cols].sort_values('rank_test_score').head(10)
print(f"Top 10 kombinasi parameter (dari {len(dt_results)} total):")
print(dt_display.to_string(index=False))

# ── 9. SAVE RESULTS ──
print("\n[9] SIMPAN HASIL")
df_results = pd.DataFrame(results)
df_results.to_csv(f'{OUTPUT_DIR}/hasil_perbandingan.csv', index=False)
print(f"CSV saved: {OUTPUT_DIR}/hasil_perbandingan.csv")
print(df_results.to_string(index=False))

nb_display.to_csv(f'{OUTPUT_DIR}/tuning_naive_bayes.csv', index=False)
dt_display.to_csv(f'{OUTPUT_DIR}/tuning_decision_tree.csv', index=False)
print("Tuning details saved")

# ── 10. VISUALISASI ──
print("\n[10] VISUALISASI")

# 10a. Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Eksperimen 2: SMOTE K=3 + Hyperparameter Tuning', fontsize=15, fontweight='bold')

for i, (name, cm) in enumerate(confusion_matrices.items()):
    ax = axes[i]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax,
                xticklabels=['Normal', 'Stunted', 'Severely Stunted'],
                yticklabels=['Normal', 'Stunted', 'Severely Stunted'])
    ax.set_title(name, fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/confusion_matrices.png")

# 10b. Perbandingan Metrik
fig, ax = plt.subplots(figsize=(10, 6))
x_pos = np.arange(len(df_results))
width = 0.2
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

for i, (metric, color) in enumerate(zip(metrics, colors)):
    bars = ax.bar(x_pos + i*width, df_results[metric].values, width,
                  label=metric, color=color, alpha=0.85, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, df_results[metric].values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

ax.set_xticks(x_pos + width*1.5)
ax.set_xticklabels(df_results['Model'].values, fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Perbandingan Metrik: SMOTE K=3 + Hyperparameter Tuning', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1.1)
ax.legend(loc='lower right', fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/perbandingan_metrik.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/perbandingan_metrik.png")

# 10c. Learning Curve: Decision Tree Tuning
fig, ax = plt.subplots(figsize=(12, 6))
dt_display_top = dt_results.sort_values('rank_test_score').head(20)

param_labels = []
for _, row in dt_display_top.iterrows():
    label = "depth={}, split={}, leaf={}".format(
        row['param_classifier__max_depth'],
        row['param_classifier__min_samples_split'],
        row['param_classifier__min_samples_leaf']
    )
    param_labels.append(label)

x = np.arange(len(param_labels))
ax.bar(x, dt_display_top['mean_test_score'].values, yerr=dt_display_top['std_test_score'].values,
       color='#2ECC71', alpha=0.8, edgecolor='black', capsize=3)
ax.set_xticks(x)
ax.set_xticklabels(param_labels, fontsize=7, rotation=45, ha='right')
ax.set_ylabel('F1-Weighted Score (CV)', fontsize=12)
ax.set_title('Top 20 Kombinasi Parameter Decision Tree (SMOTE K=3)', fontsize=14, fontweight='bold')
best_val = dt_display_top['mean_test_score'].values[0]
ax.axhline(y=best_val, color='red', linestyle='--',
           label=f"Best: {best_val:.4f}")
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/tuning_decision_tree_chart.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/tuning_decision_tree_chart.png")

# 10d. Naive Bayes var_smoothing tuning curve
fig, ax = plt.subplots(figsize=(10, 6))
nb_tuning = nb_results.sort_values('param_classifier__var_smoothing')
smoothing_vals = nb_tuning['param_classifier__var_smoothing'].values
ax.plot(range(len(smoothing_vals)), nb_tuning['mean_test_score'].values,
        'o-', color='#A23B72', linewidth=2, markersize=8)
ax.set_xticks(range(len(smoothing_vals)))
smoothing_labels = ['{:.1e}'.format(v) for v in smoothing_vals]
ax.set_xticklabels(smoothing_labels, rotation=45, ha='right')
ax.set_xlabel('var_smoothing', fontsize=12)
ax.set_ylabel('F1-Weighted Score (CV)', fontsize=12)
ax.set_title('GridSearch: Naive Bayes - var_smoothing (SMOTE K=3)', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)

best_idx = nb_tuning['rank_test_score'].idxmin()
best_nb_val = nb_tuning.loc[best_idx, 'mean_test_score']
ax.plot(best_idx, best_nb_val, 'r*', markersize=15,
        label="Best: {:.4f}".format(best_nb_val))
ax.legend()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/tuning_naive_bayes_chart.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/tuning_naive_bayes_chart.png")

# 10e. Classification Report per Model
for name, cr in classification_reports.items():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    table_data = []
    for kelas in ['Normal', 'Stunted', 'Severely Stunted']:
        if kelas in cr:
            table_data.append([
                kelas,
                "{:.4f}".format(cr[kelas]['precision']),
                "{:.4f}".format(cr[kelas]['recall']),
                "{:.4f}".format(cr[kelas]['f1-score']),
                "{}".format(int(cr[kelas]['support']))
            ])

    table_data.append(['---', '---', '---', '---', '---'])
    for avg_type in ['weighted avg', 'macro avg']:
        if avg_type in cr:
            table_data.append([
                avg_type,
                "{:.4f}".format(cr[avg_type]['precision']),
                "{:.4f}".format(cr[avg_type]['recall']),
                "{:.4f}".format(cr[avg_type]['f1-score']),
                "{}".format(int(cr[avg_type]['support']))
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

    safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    ax.set_title('Classification Report: {} (SMOTE K=3 + Tuning)'.format(name),
                 fontsize=13, fontweight='bold', pad=20)

    plt.savefig('{}/classification_report_{}.png'.format(OUTPUT_DIR, safe_name),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved classification report: {}".format(name))

print("\n{}".format('=' * 70))
print("EKSPERIMEN 2 SELESAI!")
print("Semua output tersimpan di: {}/".format(OUTPUT_DIR))
print("{}".format('=' * 70))
