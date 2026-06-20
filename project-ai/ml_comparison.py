#!/usr/bin/env python3
"""
PERBANDINGAN ALGORITMA MACHINE LEARNING
=========================================
SVM, XGBoost, Random Forest, dan KNN
Prediksi Penyakit Kanker Kolorektal (Target: Survival_5_years)
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

warnings.filterwarnings('ignore')

output_dir = 'ml_comparison'
os.makedirs(output_dir, exist_ok=True)

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 11

print("=" * 80)
print("PERBANDINGAN ALGORITMA MACHINE LEARNING")
print("SVM vs XGBoost vs Random Forest vs KNN")
print("Prediksi Survival_5_years - Kanker Kolorektal")
print("=" * 80)

# ================================================================
# 1. LOAD & PREPARE DATA
# ================================================================
print("\n1. LOAD & PREPARE DATA")
df = pd.read_csv('preprocessed/dataset_no_outlier.csv')
print(f"Dataset: {df.shape[0]:,} baris, {df.shape[1]} kolom")

df['Survival_5_years'] = df['Survival_5_years'].map({'Yes': 1, 'No': 0})
print(f"  Survival (Yes=1): {df['Survival_5_years'].sum():,}")
print(f"  Non-Survival (No=0): {(df['Survival_5_years'] == 0).sum():,}")

# Sample 50K rows for faster training (stratified)
from sklearn.model_selection import train_test_split

df_sample, _ = train_test_split(df, train_size=30000, random_state=42, stratify=df['Survival_5_years'])
print(f"\nSampled: {df_sample.shape[0]:,} baris (stratified)")
print(f"  Survival: {df_sample['Survival_5_years'].sum():,} ({df_sample['Survival_5_years'].mean()*100:.1f}%)")
print(f"  Non-Survival: {(df_sample['Survival_5_years']==0).sum():,} ({(1-df_sample['Survival_5_years'].mean())*100:.1f}%)")

drop_cols = ['Patient_ID', 'Survival_Prediction', 'Mortality']
df_ml = df_sample.drop(columns=drop_cols)

num_features = ['Age', 'Tumor_Size_mm', 'Healthcare_Costs', 'Incidence_Rate_per_100K', 'Mortality_Rate_per_100K']
cat_features = [c for c in df_ml.columns if c not in num_features and c != 'Survival_5_years']

df_encoded = pd.get_dummies(df_ml, columns=cat_features, drop_first=True, dtype=int)
print(f"\nSetelah One-Hot Encoding: {df_encoded.shape[1]} fitur")

# ================================================================
# 2. SPLIT DATA
# ================================================================
X = df_encoded.drop(columns=['Survival_5_years'])
y = df_encoded['Survival_5_years']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n2. SPLIT: Train={X_train.shape[0]:,}, Test={X_test.shape[0]:,}")

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

feature_names = X.columns.tolist()

# ================================================================
# 3. FEATURE SELECTION
# ================================================================
print("\n3. FEATURE SELECTION")
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2

print("\n  3a. Random Forest Importance...")
rf_fi = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
rf_fi.fit(X_train, y_train)
fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': rf_fi.feature_importances_})
fi_df = fi_df.sort_values('Importance', ascending=False).reset_index(drop=True)

print("  Top 10:")
for i in range(10):
    print(f"    {i+1}. {fi_df.loc[i, 'Feature'][:55]:55s} {fi_df.loc[i, 'Importance']:.4f}")

print("\n  3b. Mutual Information...")
mi_selector = SelectKBest(mutual_info_classif, k='all')
mi_scores = mi_selector.fit(X_train, y_train)
mi_df = pd.DataFrame({'Feature': feature_names, 'MI_Score': mi_scores.scores_})
mi_df = mi_df.sort_values('MI_Score', ascending=False).reset_index(drop=True)

print("  Top 10:")
for i in range(10):
    print(f"    {i+1}. {mi_df.loc[i, 'Feature'][:55]:55s} {mi_df.loc[i, 'MI_Score']:.4f}")

print("\n  3c. Chi-Square Test...")
X_train_nn = X_train.copy()
for col in X_train_nn.columns:
    if X_train_nn[col].min() < 0:
        X_train_nn[col] = X_train_nn[col] - X_train_nn[col].min()

chi2_selector = SelectKBest(chi2, k='all')
chi2_scores = chi2_selector.fit(X_train_nn, y_train)
chi2_df = pd.DataFrame({'Feature': feature_names, 'Chi2_Score': chi2_scores.scores_})
chi2_df = chi2_df.sort_values('Chi2_Score', ascending=False).reset_index(drop=True)

print("  Top 10:")
for i in range(10):
    print(f"    {i+1}. {chi2_df.loc[i, 'Feature'][:55]:55s} {chi2_df.loc[i, 'Chi2_Score']:.2f}")

# Consensus features
consensus = (set(fi_df.head(20)['Feature']) & 
             set(mi_df.head(20)['Feature']) & 
             set(chi2_df.head(20)['Feature']))
print(f"\n  Fitur konsensus ({len(consensus)}):")
for f in sorted(consensus):
    print(f"    - {f}")

# --- Visualisasi Feature Importance ---
fig, axes = plt.subplots(1, 3, figsize=(20, 8))
top_n = 15

for ax, df_i, title, color, fmt in [
    (axes[0], fi_df.head(top_n), 'Random Forest', plt.cm.Greens, '{:.4f}'),
    (axes[1], mi_df.head(top_n), 'Mutual Information', plt.cm.Blues, '{:.4f}'),
    (axes[2], chi2_df.head(top_n), 'Chi-Square', plt.cm.Oranges, '{:.0f}')
]:
    vals = df_i.iloc[:, 1].values
    colors = color(np.linspace(0.4, 0.9, top_n))
    ax.barh(range(top_n), vals, color=colors[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(df_i['Feature'].values, fontsize=9)
    ax.set_xlabel('Score')
    ax.set_title(f'Top 15 Features - {title}', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + (max(vals)*0.02), i, fmt.format(v), va='center', fontsize=8)

plt.suptitle('FEATURE IMPORTANCE ANALYSIS', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/01_feature_importance_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  => Saved: {output_dir}/01_feature_importance_comparison.png")

# ================================================================
# 4. TRAIN MODELS (with optimized parameters for speed)
# ================================================================
print("\n4. TRAINING 4 ALGORITMA MACHINE LEARNING")

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

# Use LinearSVC instead of SVC for speed, with calibration for probabilities
# Class imbalance ratio
neg_pos_ratio = len(y_train[y_train==0]) / len(y_train[y_train==1])

svm_base = LinearSVC(C=1.0, class_weight='balanced', max_iter=2000, random_state=42, dual='auto')
models = {
    'SVM': CalibratedClassifierCV(svm_base, cv=3, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                              random_state=42, n_jobs=-1, eval_metric='logloss',
                              scale_pos_weight=neg_pos_ratio),
    'Random Forest': RF(n_estimators=50, max_depth=8, class_weight='balanced',
                         random_state=42, n_jobs=-1),
    'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1),
    'Baseline': DummyClassifier(strategy='most_frequent')
}

results = {}
predictions = {}
probabilities = {}
training_times = {}

for name, model in models.items():
    print(f"\n  [{name}] Training...")
    X_tr = X_train_scaled if name in ['SVM', 'KNN'] else X_train
    X_te = X_test_scaled if name in ['SVM', 'KNN'] else X_test

    start = time.time()
    model.fit(X_tr, y_train)
    elapsed = time.time() - start
    training_times[name] = elapsed

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
    print(f"    Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
    print(f"    F1-Score: {f1:.4f} | ROC-AUC: {roc_auc:.4f} | Time: {elapsed:.2f}s")
    print(f"    CM: TN={cm[0,0]:,} FP={cm[0,1]:,} FN={cm[1,0]:,} TP={cm[1,1]:,}")

# ================================================================
# 5. CONFUSION MATRICES
# ================================================================
print("\n5. GENERATING CONFUSION MATRICES...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
cmap_list = ['Blues', 'Greens', 'Oranges', 'Purples', 'Reds']
model_names = list(models.keys())

for i, name in enumerate(model_names):
    ax = axes[i]
    cm = confusion_matrix(y_test, predictions[name])
    cm_pct = cm / cm.sum(axis=1)[:, np.newaxis] * 100

    sns.heatmap(cm, annot=True, fmt=',', cmap=cmap_list[i],
                ax=ax, cbar=True, square=True,
                xticklabels=['Predicted No', 'Predicted Yes'],
                yticklabels=['Actual No', 'Actual Yes'],
                linewidths=1, linecolor='white',
                annot_kws={'fontsize': 14, 'fontweight': 'bold'})

    for j in range(2):
        for k in range(2):
            color = 'white' if cm_pct[j, k] > 50 else 'black'
            ax.text(k + 0.5, j + 0.65, f'({cm_pct[j, k]:.1f}%)',
                    ha='center', va='center', fontsize=10, fontweight='bold', color=color)

    ax.set_title(f'{name} - Confusion Matrix', fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('Actual Label', fontsize=11)
    metrics_text = (f"Accuracy:  {results[name]['Accuracy']:.4f}\n"
                    f"Precision: {results[name]['Precision']:.4f}\n"
                    f"Recall:    {results[name]['Recall']:.4f}\n"
                    f"F1-Score:  {results[name]['F1-Score']:.4f}")
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Hide unused subplot
for j in range(len(model_names), 6):
    axes[j].set_visible(False)
plt.suptitle('CONFUSION MATRIX - Perbandingan Algoritma', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/02_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  => Saved: {output_dir}/02_confusion_matrices.png")

# Individual confusion matrices
for name in model_names:
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y_test, predictions[name])
    cm_pct = cm / cm.sum(axis=1)[:, np.newaxis] * 100
    ci = model_names.index(name)

    sns.heatmap(cm, annot=True, fmt=',', cmap=cmap_list[ci],
                ax=ax, cbar=True, square=True,
                xticklabels=['Predicted No', 'Predicted Yes'],
                yticklabels=['Actual No', 'Actual Yes'],
                linewidths=1.5, linecolor='white',
                annot_kws={'fontsize': 16, 'fontweight': 'bold'})

    for j in range(2):
        for k in range(2):
            color = 'white' if cm_pct[j, k] > 50 else 'black'
            ax.text(k + 0.5, j + 0.65, f'({cm_pct[j, k]:.1f}%)',
                    ha='center', va='center', fontsize=12, fontweight='bold', color=color)

    ax.set_title(f'{name} - Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('Actual Label', fontsize=12)
    plt.tight_layout()
    safe_name = name.replace(' ', '_')
    plt.savefig(f'{output_dir}/02_cm_{safe_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  => Saved: {output_dir}/02_cm_{safe_name}.png")

# ================================================================
# 6. ROC CURVES
# ================================================================
print("\n6. GENERATING ROC CURVES...")

# Exclude Baseline from ROC plot since it predicts constant probabilities
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

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.3, label='Random Classifier (AUC = 0.5)')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC CURVES - Perbandingan Algoritma', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/03_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  => Saved: {output_dir}/03_roc_curves.png")

# ================================================================
# 7. PERFORMANCE TABLE
# ================================================================
print("\n7. PERFORMANCE COMPARISON")

results_df = pd.DataFrame(results).T.round(4)
results_df['Rank_Acc'] = results_df['Accuracy'].rank(ascending=False).astype(int)
results_df['Rank_F1'] = results_df['F1-Score'].rank(ascending=False).astype(int)
results_df['Rank_AUC'] = results_df['ROC-AUC'].rank(ascending=False).astype(int)
results_df['Avg_Rank'] = results_df[['Rank_Acc', 'Rank_F1', 'Rank_AUC']].mean(axis=1).round(2)
results_df = results_df.sort_values('Avg_Rank')

print(f"\n{'Algoritma':<18} {'Accuracy':>9} {'Precision':>9} {'Recall':>9} {'F1':>9} {'AUC':>9} {'Time':>8} {'Rank':>5}")
print("-" * 77)
for idx, row in results_df.iterrows():
    stars = '*' * (6 - int(row['Avg_Rank']))
    print(f"{idx:<18} {row['Accuracy']:>9.4f} {row['Precision']:>9.4f} {row['Recall']:>9.4f} {row['F1-Score']:>9.4f} {row['ROC-AUC']:>9.4f} {row['Training_Time']:>7.2f}s {row['Avg_Rank']:>4.1f} {stars}")
print("-" * 77)

# Visualisasi
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

# Bar chart metrics
ax = axes[0]
x = np.arange(len(model_names))
width = 0.2
metrics_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
for i, metric in enumerate(metrics_plot):
    vals = [results[m][metric] for m in model_names]
    bars = ax.bar(x + i * width, vals, width, label=metric, color=plt.cm.Set2(i))
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.005,
                f'{val:.3f}', ha='center', fontsize=7, rotation=45)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylabel('Score')
ax.set_title('Perbandingan Metrics', fontsize=13, fontweight='bold')
ax.set_ylim(0, 1.15)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# ROC-AUC
ax = axes[1]
roc_vals = [results[m]['ROC-AUC'] for m in model_names]
colors_alg = ['#4285F4', '#34A853', '#FBBC05', '#EA4335', '#888888']
bars = ax.bar(range(len(model_names)), roc_vals, color=colors_alg, width=0.6, edgecolor='black')
for bar, val in zip(bars, roc_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f'{val:.4f}', ha='center', fontsize=12, fontweight='bold')
ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names)
ax.set_ylabel('ROC-AUC')
ax.set_title('ROC-AUC Comparison', fontsize=13, fontweight='bold')
ax.set_ylim(0.5, 1.02)
ax.grid(True, alpha=0.3, axis='y')

# Training Time
ax = axes[2]
time_vals = [results[m]['Training_Time'] for m in model_names]
bars = ax.bar(range(len(model_names)), time_vals, color=colors_alg, width=0.6, edgecolor='black')
for bar, val in zip(bars, time_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'{val:.1f}s', ha='center', fontsize=11, fontweight='bold')
ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names)
ax.set_ylabel('Training Time (s)')
ax.set_title('Training Time Comparison', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Radar Chart
ax = axes[3]
categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
N = len(categories)
angles = [n/float(N)*2*np.pi for n in range(N)] + [0]
for name in model_names:
    if name not in colors_roc:
        continue
    vals = [results[name][m] for m in categories] + [results[name][categories[0]]]
    ax.plot(angles, vals, 'o-', linewidth=2, label=name, color=colors_roc[name])
    ax.fill(angles, vals, alpha=0.1, color=colors_roc[name])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylim(0.5, 1.0)
ax.set_title('Radar Chart', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.5)

plt.suptitle('PERBANDINGAN KINERJA ALGORITMA ML', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/04_performance_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  => Saved: {output_dir}/04_performance_comparison.png")

# ================================================================
# 8. SUMMARY TABLE (IMAGE)
# ================================================================
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')

table_data = []
for idx, row in results_df.iterrows():
    table_data.append([
        idx,
        f"{row['Accuracy']:.4f}", f"{row['Precision']:.4f}",
        f"{row['Recall']:.4f}", f"{row['F1-Score']:.4f}",
        f"{row['ROC-AUC']:.4f}", f"{row['Training_Time']:.2f}s",
        f"{row['Avg_Rank']:.1f}"
    ])

col_labels = ['Algoritma', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Time', 'Rank']
table = ax.table(cellText=table_data, colLabels=col_labels,
                 cellLoc='center', loc='center',
                 colWidths=[0.15, 0.12, 0.12, 0.12, 0.12, 0.12, 0.10, 0.08])
table.auto_set_font_size(False)
table.set_fontsize(11)

for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor('#1a1a2e')
    cell.set_text_props(color='white', fontweight='bold', fontsize=12)

for metric_idx in range(1, 6):
    vals = [float(table_data[i][metric_idx]) for i in range(len(table_data))]
    best_i = np.argmax(vals)
    svals = sorted(set(vals), reverse=True)
    second = svals[1] if len(svals) > 1 else vals[0]
    for i in range(len(table_data)):
        cell = table[i+1, metric_idx]
        if i == best_i:
            cell.set_facecolor('#d4edda')
            cell.set_text_props(fontweight='bold')
        elif vals[i] == second:
            cell.set_facecolor('#e8f5e9')
        else:
            cell.set_facecolor('#f8f9fa')

for i in range(len(table_data)):
    cell = table[i+1, 0]
    cell.set_facecolor('#e3f2fd')
    cell.set_text_props(fontweight='bold')

ax.set_title('TABEL PERBANDINGAN ALGORITMA - Prediksi Kanker Kolorektal', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/05_summary_table.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  => Saved: {output_dir}/05_summary_table.png")

# ================================================================
# 9. SAVE RESULTS
# ================================================================
results_df.to_csv(f'{output_dir}/model_comparison_results.csv')
fi_df.to_csv(f'{output_dir}/feature_importance_rf.csv', index=False)
mi_df.to_csv(f'{output_dir}/feature_importance_mutual_info.csv', index=False)
chi2_df.to_csv(f'{output_dir}/feature_importance_chisquare.csv', index=False)
print(f"\n  => Saved CSVs to {output_dir}/")

# ================================================================
# 10. FINAL SUMMARY
# ================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"""
Dataset: 30,000 samples (stratified), target: Survival_5_years
Split: 80/20 Train/Test

PERINGKAT ALGORITMA:
  1st: {results_df.index[0]} (Avg Rank: {results_df['Avg_Rank'].iloc[0]})
  2nd: {results_df.index[1]} (Avg Rank: {results_df['Avg_Rank'].iloc[1]})
  3rd: {results_df.index[2]} (Avg Rank: {results_df['Avg_Rank'].iloc[2]})
  4th: {results_df.index[3]} (Avg Rank: {results_df['Avg_Rank'].iloc[3]})

BEST METRICS:
  Accuracy:  {results_df['Accuracy'].max():.4f} ({results_df['Accuracy'].idxmax()})
  F1-Score:  {results_df['F1-Score'].max():.4f} ({results_df['F1-Score'].idxmax()})
  ROC-AUC:   {results_df['ROC-AUC'].max():.4f} ({results_df['ROC-AUC'].idxmax()})
  Fastest:   {results_df['Training_Time'].min():.2f}s ({results_df['Training_Time'].idxmin()})

OUTPUT FILES (folder '{output_dir}/'):
  Feature Importance  -> 01_feature_importance_comparison.png
  Confusion Matrices  -> 02_confusion_matrices.png + 02_cm_*.png
  ROC Curves          -> 03_roc_curves.png
  Performance Charts  -> 04_performance_comparison.png
  Summary Table       -> 05_summary_table.png
  Results CSV         -> model_comparison_results.csv
  Feature CSVs        -> feature_importance_*.csv
""")
print("=" * 80)
