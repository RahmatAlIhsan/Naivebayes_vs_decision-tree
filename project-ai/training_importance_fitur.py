#!/usr/bin/env python3
"""
TRAINING IMPORTANCE FITUR - Survival_Prediction
================================================
- Feature Importance (RF) untuk Survival_Prediction
- Seleksi fitur terpenting
- 4 Model: SVM, XGBoost, Random Forest, KNN
- SMOTE K=5 untuk imbalance data
- Output: Confusion Matrix, Prediction Table, Performance Table
Semua output ke folder: Training_Importance_Fitur/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import warnings
import os
import time
import gc

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from imblearn.over_sampling import SMOTE

# Suppress XGBoost warnings
import os as _os
_os.environ['OMP_NUM_THREADS'] = '2'
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
gc.collect()

# ================================================================
# SETUP OUTPUT FOLDER
# ================================================================
output_dir = 'Training_Importance_Fitur'
os.makedirs(output_dir, exist_ok=True)

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 11

print("=" * 80)
print("TRAINING IMPORTANCE FITUR - SURVIVAL_PREDICTION")
print("Target: Survival_Prediction | SMOTE K=5 | 4 Algoritma")
print("=" * 80)

# ================================================================
# 1. LOAD DATA (stratified sample 30K)
# ================================================================
print("\n" + "=" * 80)
print("1. LOAD & PREPARE DATA")
print("=" * 80)

df = pd.read_csv('preprocessed/dataset_no_outlier.csv')
print(f"Total dataset: {df.shape[0]:,} baris, {df.shape[1]} kolom")

# Target encoding
df['Survival_Prediction'] = df['Survival_Prediction'].map({'Yes': 1, 'No': 0})

# Sampling stratified 30K
df_sample, _ = train_test_split(df, train_size=30000, random_state=42,
                                 stratify=df['Survival_Prediction'])
print(f"Sample: {df_sample.shape[0]:,} baris (stratified)")
print(f"  Yes=1: {df_sample['Survival_Prediction'].sum():,} ({df_sample['Survival_Prediction'].mean()*100:.1f}%)")
print(f"  No=0:  {(df_sample['Survival_Prediction']==0).sum():,} ({(1-df_sample['Survival_Prediction'].mean())*100:.1f}%)")

# Drop unnecessary columns
drop_cols = ['Patient_ID', 'Survival_5_years', 'Mortality']
df_ml = df_sample.drop(columns=[c for c in drop_cols if c in df_sample.columns])

# Identify numeric and categorical columns
num_cols = ['Age', 'Tumor_Size_mm', 'Healthcare_Costs', 'Incidence_Rate_per_100K', 'Mortality_Rate_per_100K']
cat_cols = [c for c in df_ml.columns if c not in num_cols and c != 'Survival_Prediction']
print(f"\nFitur numerik ({len(num_cols)}): {num_cols}")
print(f"Fitur kategorikal ({len(cat_cols)}): {cat_cols}")

# One-Hot Encoding
df_encoded = pd.get_dummies(df_ml, columns=cat_cols, drop_first=True, dtype=int)
feature_names = [c for c in df_encoded.columns if c != 'Survival_Prediction']
print(f"Total fitur setelah encoding: {len(feature_names)}")

X = df_encoded[feature_names].values
y = df_encoded['Survival_Prediction'].values

# Clean up
del df, df_sample, df_ml, df_encoded
gc.collect()

# ================================================================
# 2. FEATURE IMPORTANCE ANALYSIS (untuk Survival_Prediction)
# ================================================================
print("\n" + "=" * 80)
print("2. FEATURE IMPORTANCE ANALYSIS (untuk Survival_Prediction)")
print("=" * 80)

print("\n  2a. Random Forest Feature Importance...")
rf_importance = RF(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_importance.fit(X, y)

fi_scores = rf_importance.feature_importances_
fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': fi_scores})
fi_df = fi_df.sort_values('Importance', ascending=False).reset_index(drop=True)

print("\n  Top 15 fitur terpenting untuk Survival_Prediction:")
print(f"  {'No':>3} {'Feature':<55} {'Importance':>10}")
print(f"  {'-'*70}")
for i in range(min(15, len(fi_df))):
    print(f"  {i+1:>3}. {fi_df.loc[i, 'Feature']:<55} {fi_df.loc[i, 'Importance']:.6f}")

print("\n  2b. Mutual Information Score...")
mi_selector = SelectKBest(mutual_info_classif, k='all')
mi_scores = mi_selector.fit(X, y)
mi_df = pd.DataFrame({'Feature': feature_names, 'MI_Score': mi_scores.scores_})
mi_df = mi_df.sort_values('MI_Score', ascending=False).reset_index(drop=True)

print("\n  Top 15 fitur (Mutual Information):")
for i in range(min(15, len(mi_df))):
    print(f"  {i+1:>3}. {mi_df.loc[i, 'Feature']:<55} {mi_df.loc[i, 'MI_Score']:.6f}")

# Save feature importance to CSV
fi_df.to_csv(f'{output_dir}/feature_importance_rf.csv', index=False)
mi_df.to_csv(f'{output_dir}/feature_importance_mutual_info.csv', index=False)
print(f"\n  => Saved: feature_importance_rf.csv")
print(f"  => Saved: feature_importance_mutual_info.csv")

# Plot feature importance
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
top_n = 20

# RF Importance
ax = axes[0]
fi_top = fi_df.head(top_n)
colors = plt.cm.Greens(np.linspace(0.3, 0.9, top_n))[::-1]
bars = ax.barh(range(top_n), fi_top['Importance'].values, color=colors)
ax.set_yticks(range(top_n))
ax.set_yticklabels(fi_top['Feature'].values, fontsize=8)
ax.set_xlabel('Importance Score', fontsize=11)
ax.set_title(f'Top {top_n} Features - Random Forest Importance', fontsize=13, fontweight='bold')
ax.invert_yaxis()
for i, v in enumerate(fi_top['Importance'].values):
    ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=7)

# Mutual Information
ax = axes[1]
mi_top = mi_df.head(top_n)
colors2 = plt.cm.Blues(np.linspace(0.3, 0.9, top_n))[::-1]
bars = ax.barh(range(top_n), mi_top['MI_Score'].values, color=colors2)
ax.set_yticks(range(top_n))
ax.set_yticklabels(mi_top['Feature'].values, fontsize=8)
ax.set_xlabel('Mutual Information Score', fontsize=11)
ax.set_title(f'Top {top_n} Features - Mutual Information', fontsize=13, fontweight='bold')
ax.invert_yaxis()
for i, v in enumerate(mi_top['MI_Score'].values):
    ax.text(v + 0.0001, i, f'{v:.4f}', va='center', fontsize=7)

plt.suptitle('FEATURE IMPORTANCE untuk TARGET Survival_Prediction', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/01_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  => Saved: {output_dir}/01_feature_importance.png")

# ================================================================
# 3. PILIH FITUR TERBAIK (Top N features based on consensus)
# ================================================================
print("\n" + "=" * 80)
print("3. SELEKSI FITUR TERBAIK")
print("=" * 80)

# Ambil fitur yang masuk top 20 di RF Importance ATAU MI Score
top_rf = set(fi_df.head(20)['Feature'])
top_mi = set(mi_df.head(20)['Feature'])
# Gunakan INTERSECTION (fitur yang penting di kedua metode) + top 10 RF
# untuk memastikan fitur yang benar-benar penting saja
selected_features = list(top_rf & top_mi)  # Intersection - fitur penting di kedua metode

# Jika intersection terlalu sedikit, tambahkan dari top RF
if len(selected_features) < 10:
    # Tambahkan dari top RF yang belum ada
    extra = [f for f in fi_df.head(15)['Feature'] if f not in selected_features]
    selected_features.extend(extra[:min(10-len(selected_features), len(extra))])

selected_features.sort()

print(f"  Top 20 RF:        {len(top_rf)} fitur")
print(f"  Top 20 MI:        {len(top_mi)} fitur")
print(f"  Intersection:     {len(top_rf & top_mi)} fitur")
print(f"  Selected:         {len(selected_features)} fitur")

# Ini adalah fitur yang akan digunakan
# Ambil index dari fitur terpilih
selected_indices = [feature_names.index(f) for f in selected_features]
X_selected = X[:, selected_indices]

print(f"\n  Fitur terpilih ({len(selected_features)}):")
for i, f in enumerate(selected_features):
    print(f"    {i+1:>3}. {f}")

# Save selected features to CSV
pd.DataFrame({'Feature': selected_features}).to_csv(f'{output_dir}/selected_features.csv', index=False)
print(f"  => Saved: selected_features.csv")

# ================================================================
# 4. SPLIT + SMOTE OVERSAMPLING (K=5)
# ================================================================
print("\n" + "=" * 80)
print("4. SPLIT DATA + SMOTE OVERSAMPLING K=5")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n  SEBELUM SMOTE:")
print(f"    Train: {X_train.shape[0]:,} baris")
print(f"    Yes={y_train.sum():,}, No={(y_train==0).sum():,}")
print(f"    Ratio: {y_train.mean()*100:.1f}% / {(1-y_train.mean())*100:.1f}%")
print(f"    Test:  {X_test.shape[0]:,} baris")

# SMOTE dengan K=5
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"\n  SETELAH SMOTE (K=5):")
print(f"    Train: {X_train_res.shape[0]:,} baris (+{X_train_res.shape[0]-X_train.shape[0]:,})")
print(f"    Yes={y_train_res.sum():,}, No={(y_train_res==0).sum():,}")
print(f"    Ratio: {y_train_res.mean()*100:.1f}% / {(1-y_train_res.mean())*100:.1f}%")

# StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# ================================================================
# 5. TRAIN 4 MODELS + BASELINE
# ================================================================
print("\n" + "=" * 80)
print("5. TRAINING 4 ALGORITMA + BASELINE")
print("=" * 80)

models = {
    'SVM': CalibratedClassifierCV(
        LinearSVC(C=1.0, class_weight='balanced', max_iter=2000, random_state=42, dual='auto'),
        cv=3, n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=42, n_jobs=-1, eval_metric='logloss',
        scale_pos_weight=1.0
    ),
    'Random Forest': RF(
        n_estimators=50, max_depth=8, random_state=42, n_jobs=-1
    ),
    'KNN': KNeighborsClassifier(
        n_neighbors=5, weights='distance', n_jobs=-1
    ),
    'Baseline': DummyClassifier(strategy='most_frequent')
}

results = {}
predictions = {}
probabilities = {}
model_names = list(models.keys())
cmap_list = ['Blues', 'Greens', 'Oranges', 'Purples', 'Reds']

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
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    results[name] = {
        'Accuracy': acc, 'Precision': prec, 'Recall': rec,
        'F1-Score': f1, 'ROC-AUC': roc_auc, 'Training_Time': elapsed
    }

    cm = confusion_matrix(y_test, y_pred)
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1-Score:  {f1:.4f}")
    print(f"    ROC-AUC:   {roc_auc:.4f}")
    print(f"    Time:      {elapsed:.2f}s")
    print(f"    Confusion Matrix:")
    print(f"      TN={cm[0,0]:>6,}  FP={cm[0,1]:>5,}")
    print(f"      FN={cm[1,0]:>6,}  TP={cm[1,1]:>5,}")

gc.collect()

# ================================================================
# 6. PREDICTION TABLE (Actual vs Predicted - 50 samples)
# ================================================================
print("\n" + "=" * 80)
print("6. GENERATING PREDICTION TABLE")
print("=" * 80)

# Ambil 50 sampel test untuk ditampilkan
n_show = 50
pred_table_data = []
for i in range(min(n_show, len(y_test))):
    row_data = {'Actual': 'Yes' if y_test[i] == 1 else 'No'}
    for name in model_names:
        row_data[f'{name}'] = 'Yes' if predictions[name][i] == 1 else 'No'
    pred_table_data.append(row_data)

pred_df = pd.DataFrame(pred_table_data)
print(f"\n  50 sampel prediksi (ditampilkan 10 pertama):")
print(pred_df.head(10).to_string(index=False))

# Save prediction table as CSV
pred_df.to_csv(f'{output_dir}/prediction_table.csv', index=False)
print(f"\n  => Saved: {output_dir}/prediction_table.csv")

# Buat gambar prediction table
fig, ax = plt.subplots(figsize=(14, max(6, len(pred_table_data) * 0.35)))
ax.axis('off')

table_data = []
for i in range(len(pred_table_data)):
    row = [str(i+1), pred_table_data[i]['Actual']]
    for name in model_names:
        row.append(pred_table_data[i][name])
    table_data.append(row)

col_labels = ['No', 'Actual'] + model_names
table = ax.table(cellText=table_data, colLabels=col_labels,
                 cellLoc='center', loc='center',
                 colWidths=[0.05, 0.08] + [0.12] * len(model_names))

table.auto_set_font_size(False)
table.set_fontsize(8)

# Style header
for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor('#1a1a2e')
    cell.set_text_props(color='white', fontweight='bold', fontsize=9)

# Color cells - green if correct, red if wrong
for i in range(len(pred_table_data)):
    actual = pred_table_data[i]['Actual']
    # Row number
    table[i+1, 0].set_facecolor('#f0f0f0')
    # Actual value
    table[i+1, 1].set_facecolor('#e3f2fd')
    table[i+1, 1].set_text_props(fontweight='bold')
    # Predictions
    for j, name in enumerate(model_names):
        cell = table[i+1, j+2]
        pred_val = pred_table_data[i][name]
        if pred_val == actual:
            cell.set_facecolor('#d4edda')  # Green = correct
            cell.set_text_props(fontweight='bold')
        else:
            cell.set_facecolor('#f8d7da')  # Red = wrong
            cell.set_text_props(fontweight='bold')

ax.set_title('PREDICTION TABLE - Actual vs Predicted (50 Samples)',
             fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{output_dir}/02_prediction_table.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  => Saved: {output_dir}/02_prediction_table.png")

# ================================================================
# 7. INDIVIDUAL CONFUSION MATRICES (each model as separate PNG)
# ================================================================
print("\n" + "=" * 80)
print("7. GENERATING CONFUSION MATRICES")
print("=" * 80)

model_short_names = {
    'SVM': 'SVM',
    'XGBoost': 'XGBoost',
    'Random Forest': 'Random_Forest',
    'KNN': 'KNN',
    'Baseline': 'Baseline'
}

for name in model_names:
    fig, ax = plt.subplots(figsize=(8, 7))
    cm = confusion_matrix(y_test, predictions[name])
    cm_pct = cm / cm.sum(axis=1)[:, np.newaxis] * 100

    sns.heatmap(cm, annot=True, fmt=',', cmap=cmap_list[model_names.index(name)],
                ax=ax, cbar=True, square=True,
                xticklabels=['Predicted No', 'Predicted Yes'],
                yticklabels=['Actual No', 'Actual Yes'],
                linewidths=2, linecolor='white',
                annot_kws={'fontsize': 18, 'fontweight': 'bold'})

    # Add percentage text
    for j in range(2):
        for k in range(2):
            color = 'white' if cm_pct[j, k] > 50 else 'black'
            ax.text(k + 0.5, j + 0.65, f'({cm_pct[j, k]:.1f}%)',
                    ha='center', va='center', fontsize=13, fontweight='bold', color=color)

    # Metrics display
    r = results[name]
    metrics_str = (f"Accuracy:  {r['Accuracy']:.4f}\n"
                   f"Precision: {r['Precision']:.4f}\n"
                   f"Recall:    {r['Recall']:.4f}\n"
                   f"F1-Score:  {r['F1-Score']:.4f}\n"
                   f"ROC-AUC:   {r['ROC-AUC']:.4f}")
    ax.text(0.02, 0.98, metrics_str, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    ax.set_title(f'{name} - Confusion Matrix\nTarget: Survival_Prediction + SMOTE K=5',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('Actual Label', fontsize=12)

    plt.tight_layout()
    safe_name = model_short_names[name]
    plt.savefig(f'{output_dir}/03_cm_{safe_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  => Saved: {output_dir}/03_cm_{safe_name}.png")

# ================================================================
# 8. ALL CONFUSION MATRICES IN ONE FIGURE
# ================================================================
print("\n  8a. All Confusion Matrices (1 figure)...")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

for i, name in enumerate(model_names):
    ax = axes[i]
    cm = confusion_matrix(y_test, predictions[name])
    cm_pct = cm / cm.sum(axis=1)[:, np.newaxis] * 100

    sns.heatmap(cm, annot=True, fmt=',', cmap=cmap_list[i],
                ax=ax, cbar=True, square=True,
                xticklabels=['Pred No', 'Pred Yes'],
                yticklabels=['Act No', 'Act Yes'],
                linewidths=1, linecolor='white',
                annot_kws={'fontsize': 14, 'fontweight': 'bold'})

    for j in range(2):
        for k in range(2):
            color = 'white' if cm_pct[j, k] > 50 else 'black'
            ax.text(k + 0.5, j + 0.65, f'({cm_pct[j, k]:.1f}%)',
                    ha='center', va='center', fontsize=10, fontweight='bold', color=color)

    ax.set_title(f'{name}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('Actual', fontsize=10)

    metrics_text = (f"Acc: {results[name]['Accuracy']:.4f}\n"
                    f"F1:  {results[name]['F1-Score']:.4f}\n"
                    f"AUC: {results[name]['ROC-AUC']:.4f}")
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Hide unused
for j in range(len(model_names), 6):
    axes[j].set_visible(False)

plt.suptitle('CONFUSION MATRICES - Survival_Prediction (SMOTE K=5)',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/04_all_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  => Saved: {output_dir}/04_all_confusion_matrices.png")

# ================================================================
# 9. ROC CURVES
# ================================================================
print("\n  9. ROC Curves...")

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
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC CURVES - Survival_Prediction (SMOTE K=5)',
             fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/05_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  => Saved: {output_dir}/05_roc_curves.png")

# ================================================================
# 10. PERFORMANCE COMPARISON TABLE (PNG)
# ================================================================
print("\n" + "=" * 80)
print("10. PERFORMANCE COMPARISON TABLE")
print("=" * 80)

results_df = pd.DataFrame(results).T.round(4)
results_df['Rank_Acc'] = results_df['Accuracy'].rank(ascending=False).astype(int)
results_df['Rank_F1'] = results_df['F1-Score'].rank(ascending=False).astype(int)
results_df['Rank_AUC'] = results_df['ROC-AUC'].rank(ascending=False).astype(int)
results_df['Avg_Rank'] = results_df[['Rank_Acc', 'Rank_F1', 'Rank_AUC']].mean(axis=1).round(2)
results_df = results_df.sort_values('Avg_Rank')

print(f"\n{'Algoritma':<16} {'Accuracy':>9} {'Precision':>9} {'Recall':>9} {'F1':>9} {'AUC':>9} {'Time':>8} {'Rank':>5}")
print("-" * 75)
for idx, row in results_df.iterrows():
    print(f"{idx:<16} {row['Accuracy']:>9.4f} {row['Precision']:>9.4f} {row['Recall']:>9.4f} {row['F1-Score']:>9.4f} {row['ROC-AUC']:>9.4f} {row['Training_Time']:>7.2f}s {row['Avg_Rank']:>4.1f}")
print("-" * 75)

# Save results
results_df.to_csv(f'{output_dir}/model_comparison_results.csv')
print(f"\n  => Saved: {output_dir}/model_comparison_results.csv")

# --- Performance Comparison as Table Image ---
fig, ax = plt.subplots(figsize=(14, 5))
ax.axis('off')

table_data = []
for idx, row in results_df.iterrows():
    table_data.append([
        idx,
        f"{row['Accuracy']:.4f}",
        f"{row['Precision']:.4f}",
        f"{row['Recall']:.4f}",
        f"{row['F1-Score']:.4f}",
        f"{row['ROC-AUC']:.4f}",
        f"{row['Training_Time']:.2f}s",
        f"{row['Avg_Rank']:.1f}"
    ])

col_labels = ['Algoritma', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Time', 'Rank']
table = ax.table(cellText=table_data, colLabels=col_labels,
                 cellLoc='center', loc='center',
                 colWidths=[0.15, 0.12, 0.12, 0.12, 0.12, 0.12, 0.10, 0.08])
table.auto_set_font_size(False)
table.set_fontsize(11)

# Header style
for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor('#1a1a2e')
    cell.set_text_props(color='white', fontweight='bold', fontsize=12)

# Highlight best values
for metric_idx in range(1, 6):  # Accuracy to ROC-AUC
    vals = [float(table_data[i][metric_idx]) for i in range(len(table_data))]
    best_i = np.argmax(vals)
    for i in range(len(table_data)):
        cell = table[i+1, metric_idx]
        if i == best_i:
            cell.set_facecolor('#d4edda')
            cell.set_text_props(fontweight='bold')
        else:
            cell.set_facecolor('#f8f9fa')

# Style algorithm names
for i in range(len(table_data)):
    cell = table[i+1, 0]
    cell.set_facecolor('#e3f2fd')
    cell.set_text_props(fontweight='bold')

ax.set_title('PERBANDINGAN KINERJA MODEL - Survival_Prediction (SMOTE K=5)',
             fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{output_dir}/06_performance_table.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  => Saved: {output_dir}/06_performance_table.png")

# --- Bar Chart Performance ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

# 1. Metrics Bar Chart
ax = axes[0]
x = np.arange(len(model_names))
width = 0.2
metrics_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
for i, metric in enumerate(metrics_plot):
    vals = [results[m][metric] for m in model_names]
    bars = ax.bar(x + i * width, vals, width, label=metric, color=plt.cm.Set2(i))
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', fontsize=7, rotation=45)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(model_names, fontsize=10)
ax.set_ylabel('Score')
ax.set_title('Perbandingan Metrics', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.15)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# 2. ROC-AUC Comparison
ax = axes[1]
colors_alg = ['#4285F4', '#34A853', '#FBBC05', '#EA4335', '#888888']
roc_vals = [results[m]['ROC-AUC'] for m in model_names]
bars = ax.bar(range(len(model_names)), roc_vals, color=colors_alg, width=0.6, edgecolor='black')
for bar, val in zip(bars, roc_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')
ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names)
ax.set_ylabel('ROC-AUC')
ax.set_title('ROC-AUC Comparison', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.02)
ax.grid(True, alpha=0.3, axis='y')

# 3. Training Time
ax = axes[2]
time_vals = [results[m]['Training_Time'] for m in model_names]
bars = ax.bar(range(len(model_names)), time_vals, color=colors_alg, width=0.6, edgecolor='black')
for bar, val in zip(bars, time_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'{val:.1f}s', ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names)
ax.set_ylabel('Training Time (s)')
ax.set_title('Training Time Comparison', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 4. Feature Importance Pie (top 10)
ax = axes[3]
fi_top10 = fi_df.head(10)
colors_pie = plt.cm.Set3(np.linspace(0, 1, 10))
wedges, texts, autotexts = ax.pie(
    fi_top10['Importance'].values,
    labels=fi_top10['Feature'].values,
    autopct='%1.1f%%',
    colors=colors_pie,
    textprops={'fontsize': 7},
    pctdistance=0.75
)
for t in autotexts:
    t.set_fontsize(7)
    t.set_fontweight('bold')
ax.set_title('Top 10 Feature Importance', fontsize=12, fontweight='bold')

plt.suptitle('PERBANDINGAN KINERJA & ANALISIS FITUR', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/07_performance_charts.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  => Saved: {output_dir}/07_performance_charts.png")

# ================================================================
# 11. CONFUSION MATRIX SUMMARY TABLE (as image)
# ================================================================
print("\n  11. Confusion Matrix Summary Table...")

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.axis('off')

cm_data = []
for name in model_names:
    cm = confusion_matrix(y_test, predictions[name])
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    total = cm.sum()
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    cm_data.append([
        name,
        f"{tn:,}", f"{fp:,}", f"{fn:,}", f"{tp:,}",
        f"{accuracy:.4f}",
        f"{precision:.4f}",
        f"{recall:.4f}",
        f"{f1:.4f}"
    ])

col_labels = ['Model', 'TN', 'FP', 'FN', 'TP', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
table = ax.table(cellText=cm_data, colLabels=col_labels,
                 cellLoc='center', loc='center',
                 colWidths=[0.12, 0.08, 0.08, 0.08, 0.08, 0.10, 0.10, 0.10, 0.10])
table.auto_set_font_size(False)
table.set_fontsize(10)

for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor('#1a1a2e')
    cell.set_text_props(color='white', fontweight='bold')

for i in range(len(cm_data)):
    for j in range(len(col_labels)):
        cell = table[i+1, j]
        if j == 0:
            cell.set_facecolor('#e3f2fd')
            cell.set_text_props(fontweight='bold')
        elif j in [5, 6, 7, 8]:  # Metrics columns
            vals = [float(cm_data[k][j]) for k in range(len(cm_data))]
            if float(cm_data[i][j]) == max(vals):
                cell.set_facecolor('#d4edda')
                cell.set_text_props(fontweight='bold')
            else:
                cell.set_facecolor('#f8f9fa')
        else:
            cell.set_facecolor('#f8f9fa')

ax.set_title('CONFUSION MATRIX SUMMARY - Survival_Prediction (SMOTE K=5)',
             fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{output_dir}/08_cm_summary_table.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  => Saved: {output_dir}/08_cm_summary_table.png")

# ================================================================
# 12. FINAL SUMMARY
# ================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(f"""
KONFIGURASI:
  - Target:       Survival_Prediction
  - Fitur:        {len(selected_features)} fitur terpilih (dari {len(feature_names)} total)
  - SMOTE:        K=5 (balancing classes)
  - Data:         30,000 sampel stratified, 80/20 split
  - Setelah SMOTE: {X_train_res.shape[0]:,} baris train
  - Algoritma:    {', '.join(model_names)}

PERINGKAT (berdasarkan Avg Rank):
""")
for i, idx in enumerate(results_df.index):
    print(f"  {i+1}. {idx} (Rank: {results_df['Avg_Rank'].loc[idx]})")

print(f"""
BEST METRICS:
  Accuracy:  {results_df['Accuracy'].max():.4f} ({results_df['Accuracy'].idxmax()})
  F1-Score:  {results_df['F1-Score'].max():.4f} ({results_df['F1-Score'].idxmax()})
  ROC-AUC:   {results_df['ROC-AUC'].max():.4f} ({results_df['ROC-AUC'].idxmax()})
  Fastest:   {results_df['Training_Time'].min():.2f}s ({results_df['Training_Time'].idxmin()})

OUTPUT (folder '{output_dir}/'):
  01_feature_importance.png          - Feature Importance charts
  02_prediction_table.png            - Prediction table (50 samples)
  03_cm_*.png                        - Individual confusion matrices (5 files)
  04_all_confusion_matrices.png      - All confusion matrices in 1 figure
  05_roc_curves.png                  - ROC Curves
  06_performance_table.png           - Performance comparison table
  07_performance_charts.png          - Performance charts
  08_cm_summary_table.png            - Confusion matrix summary table
  feature_importance_rf.csv          - RF Importance scores
  feature_importance_mutual_info.csv - Mutual Information scores
  model_comparison_results.csv       - All model metrics
  prediction_table.csv               - Actual vs Predicted (50 samples)
""")
print("=" * 80)
