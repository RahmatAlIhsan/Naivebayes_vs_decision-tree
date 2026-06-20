#!/usr/bin/env python3
"""
ANALISIS SEMUA TARGET - Cari target paling prediktif
=====================================================
Membandingkan prediktabilitas: Survival_5_years, Cancer_Stage, 
Early_Detection, Treatment_Type, Mortality, Healthcare_Costs
"""

import pandas as pd
import numpy as np
import warnings
import os
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score, mean_squared_error
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings('ignore')

output_dir = 'target_analysis'
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("ANALISIS SEMUA TARGET - MANA YANG PALING PREDIKTIF?")
print("=" * 80)

# ================================================================
# 1. LOAD DATA
# ================================================================
print("\n1. LOAD DATA")
df = pd.read_csv('preprocessed/dataset_no_outlier.csv')
print(f"Dataset: {df.shape[0]:,} baris, {df.shape[1]} kolom")

# Sample 50K stratified for speed
df_sample = df.sample(n=50000, random_state=42)
print(f"Sample: {df_sample.shape[0]:,} baris")

# Drop identifiers
drop_cols = ['Patient_ID', 'Survival_Prediction']
df_ml = df_sample.drop(columns=[c for c in drop_cols if c in df_sample.columns])

# ================================================================
# 2. DEFINE TARGET CANDIDATES
# ================================================================
print("\n2. TARGET CANDIDATES")

target_candidates = {
    'Survival_5_years': {
        'type': 'classification',
        'desc': 'Survival 5 tahun (Yes/No)',
        'is_binary': True
    },
    'Cancer_Stage': {
        'type': 'classification',
        'desc': 'Stage kanker (Localized/Regional/Distant)',
        'is_binary': False
    },
    'Early_Detection': {
        'type': 'classification',
        'desc': 'Deteksi dini (Yes/No)',
        'is_binary': True
    },
    'Treatment_Type': {
        'type': 'classification',
        'desc': 'Jenis treatment (multi-class)',
        'is_binary': False
    },
    'Mortality': {
        'type': 'classification',
        'desc': 'Mortalitas (Yes/No)',
        'is_binary': True
    },
    'Healthcare_Costs': {
        'type': 'regression',
        'desc': 'Biaya healthcare (numeric)',
        'is_binary': False
    },
    'Tumor_Size_mm': {
        'type': 'regression',
        'desc': 'Ukuran tumor (numeric)',
        'is_binary': False
    }
}

results_list = []

for target_name, target_info in target_candidates.items():
    if target_name not in df_ml.columns:
        print(f"\n  [!] {target_name} tidak ditemukan di dataset, skip")
        continue

    print(f"\n  Target: {target_name} ({target_info['desc']})")
    print(f"  Tipe: {target_info['type']}")

    # Features: semua kolom kecuali target dan kolom target lain
    other_targets = [t for t in target_candidates.keys() if t != target_name and t in df_ml.columns]
    feature_cols = [c for c in df_ml.columns if c not in other_targets and c != target_name]

    # Kategorikal vs numerik
    num_cols = ['Age', 'Tumor_Size_mm', 'Healthcare_Costs', 'Incidence_Rate_per_100K', 'Mortality_Rate_per_100K']
    num_cols = [c for c in num_cols if c in feature_cols]
    cat_cols = [c for c in feature_cols if c not in num_cols]

    # One-Hot Encoding
    df_encoded = pd.get_dummies(df_ml[feature_cols], columns=cat_cols, drop_first=True, dtype=int)

    X = df_encoded.values
    y_raw = df_ml[target_name].values

    # Encode target jika kategorikal
    if target_info['type'] == 'classification':
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        n_classes = len(le.classes_)
        target_info['n_classes'] = n_classes
        target_info['classes'] = list(le.classes_)
        print(f"  Kelas ({n_classes}): {list(le.classes_)}")

        class_dist = pd.Series(y).value_counts(normalize=True)
        for i, cls in enumerate(le.classes_):
            print(f"    {cls}: {class_dist[i]*100:.1f}%")

        # Hitung MI Score untuk target ini
        mi_scores = mutual_info_classif(X, y, random_state=42)
        target_info['mi_mean'] = np.mean(mi_scores)
        target_info['mi_max'] = np.max(mi_scores)
        print(f"  MI Score: mean={target_info['mi_mean']:.4f}, max={target_info['mi_max']:.4f}")
    else:
        # Regression
        y = y_raw.astype(float)
        print(f"  Range: [{y.min():.2f}, {y.max():.2f}], Mean: {y.mean():.2f}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ================================================================
    # TRAIN & EVALUATE
    # ================================================================
    print(f"  Train: {X_train.shape[0]:,}, Test: {X_test.shape[0]:,}")

    if target_info['type'] == 'classification':
        # Random Forest
        start = time.time()
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train_s, y_train)
        elapsed = time.time() - start

        y_pred = rf.predict(X_test_s)
        acc = accuracy_score(y_test, y_pred)

        # Baseline Dummy
        dummy = DummyClassifier(strategy='most_frequent')
        dummy.fit(X_train_s, y_train)
        dummy_acc = accuracy_score(y_test, dummy.predict(X_test_s))

        # Improvement over baseline
        improvement = ((acc - dummy_acc) / dummy_acc) * 100

        # Untuk binary classification, hitung ROC-AUC
        if target_info['is_binary'] and n_classes == 2:
            y_prob = rf.predict_proba(X_test_s)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        else:
            # Untuk multi-class, gunakan OvR AUC
            try:
                y_prob = rf.predict_proba(X_test_s)
                auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
            except:
                auc = 0.0

        # Cross-validation score (5-fold)
        try:
            cv_scores = cross_val_score(rf, X_train_s, y_train, cv=3, scoring='accuracy')
            cv_mean = cv_scores.mean()
        except:
            cv_mean = 0.0

        result = {
            'Target': target_name,
            'Type': target_info['type'],
            'Description': target_info['desc'],
            'Classes': n_classes,
            'Accuracy': round(acc, 4),
            'Baseline (Dummy)': round(dummy_acc, 4),
            'Improvement (%)': round(improvement, 2),
            'ROC-AUC': round(auc, 4),
            'CV Score (3-fold)': round(cv_mean, 4),
            'MI Mean': round(target_info.get('mi_mean', 0), 4),
            'MI Max': round(target_info.get('mi_max', 0), 4),
            'Time (s)': round(elapsed, 2)
        }

        print(f"  Accuracy: {acc:.4f} (Baseline: {dummy_acc:.4f}, +{improvement:.1f}%)")
        print(f"  ROC-AUC: {auc:.4f}")
        print(f"  CV Score: {cv_mean:.4f}")

    else:
        # Regression
        start = time.time()
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train_s, y_train)
        elapsed = time.time() - start

        y_pred = rf.predict(X_test_s)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Baseline Dummy
        dummy = DummyRegressor(strategy='mean')
        dummy.fit(X_train_s, y_train)
        dummy_r2 = r2_score(y_test, dummy.predict(X_test_s))

        result = {
            'Target': target_name,
            'Type': target_info['type'],
            'Description': target_info['desc'],
            'Classes': '-',
            'R2 Score': round(r2, 4),
            'Baseline R2': round(dummy_r2, 4),
            'RMSE': round(rmse, 2),
            'CV Score (3-fold)': '-',
            'ROC-AUC': '-',
            'MI Mean': '-',
            'MI Max': '-',
            'Time (s)': round(elapsed, 2)
        }

        print(f"  R2 Score: {r2:.4f} (Baseline R2: {dummy_r2:.4f})")
        print(f"  RMSE: {rmse:.2f}")

    results_list.append(result)

# ================================================================
# 3. COMPARISON TABLE
# ================================================================
print("\n" + "=" * 80)
print("3. PERBANDINGAN SEMUA TARGET")
print("=" * 80)

results_df = pd.DataFrame(results_list)
classification_results = results_df[results_df['Type'] == 'classification'].copy()
regression_results = results_df[results_df['Type'] == 'regression'].copy()

if len(classification_results) > 0:
    classification_results = classification_results.sort_values('Improvement (%)', ascending=False)
    print("\n--- CLASSIFICATION TARGETS (Ranked by Improvement over Baseline) ---")
    print("-" * 100)
    print(f"{'Target':<25} {'Accuracy':>10} {'Baseline':>10} {'Improve':>10} {'AUC':>10} {'MI Mean':>10} {'Time':>8}")
    print("-" * 100)
    for _, row in classification_results.iterrows():
        print(f"{row['Target']:<25} {row['Accuracy']:>10.4f} {row['Baseline (Dummy)']:>10.4f} {row['Improvement (%)']:>8.1f}% {row['ROC-AUC']:>10.4f} {row['MI Mean']:>10.4f} {row['Time (s)']:>7.1f}s")
    print("-" * 100)

if len(regression_results) > 0:
    regression_results = regression_results.sort_values('R2 Score', ascending=False)
    print("\n--- REGRESSION TARGETS (Ranked by R2 Score) ---")
    print("-" * 80)
    print(f"{'Target':<25} {'R2 Score':>10} {'Baseline R2':>10} {'RMSE':>12} {'Time':>8}")
    print("-" * 80)
    for _, row in regression_results.iterrows():
        print(f"{row['Target']:<25} {row['R2 Score']:>10.4f} {row['Baseline R2']:>10.4f} {row['RMSE']:>12.2f} {row['Time (s)']:>7.1f}s")
    print("-" * 80)

# ================================================================
# 4. DETAIL TARGET DISTRIBUTIONS
# ================================================================
print("\n" + "=" * 80)
print("4. DISTRIBUSI MASING-MASING TARGET")
print("=" * 80)

for target_name in target_candidates.keys():
    if target_name not in df_ml.columns:
        continue
    print(f"\n--- {target_name} ---")
    if target_candidates[target_name]['type'] == 'classification':
        vc = df_ml[target_name].value_counts()
        vc_pct = df_ml[target_name].value_counts(normalize=True) * 100
        for idx in vc.index:
            print(f"  {idx:<20} {vc[idx]:>8,} ({vc_pct[idx]:.1f}%)")
    else:
        print(f"  Min: {df_ml[target_name].min():.2f}")
        print(f"  Max: {df_ml[target_name].max():.2f}")
        print(f"  Mean: {df_ml[target_name].mean():.2f}")
        print(f"  Std: {df_ml[target_name].std():.2f}")

# ================================================================
# 5. RECOMMENDATION
# ================================================================
print("\n" + "=" * 80)
print("5. REKOMENDASI TARGET TERBAIK")
print("=" * 80)

print("""
KRITERIA PENILAIAN:
  1. Improvement over Baseline (%) - semakin besar, semakin baik
  2. ROC-AUC - >0.7 = bagus, >0.8 = sangat bagus
  3. MI Mean - semakin besar, semakin banyak sinyal dari fitur
  4. Jumlah kelas - lebih sedikit = lebih mudah diprediksi
""")

if len(classification_results) > 0:
    best_class = classification_results.iloc[0]
    print(f"🏆 TARGET KLASIFIKASI TERBAIK: {best_class['Target']}")
    print(f"   Accuracy: {best_class['Accuracy']:.4f} (+{best_class['Improvement (%)']:.1f}% over baseline)")
    print(f"   ROC-AUC:  {best_class['ROC-AUC']:.4f}")
    print(f"   Kelas:    {best_class['Classes']}")

    second_class = classification_results.iloc[1] if len(classification_results) > 1 else None
    if second_class is not None:
        print(f"\n🥈 Runner-up: {second_class['Target']}")
        print(f"   Accuracy: {second_class['Accuracy']:.4f} (+{second_class['Improvement (%)']:.1f}% over baseline)")
        print(f"   ROC-AUC:  {second_class['ROC-AUC']:.4f}")

if len(regression_results) > 0:
    best_reg = regression_results.iloc[0]
    print(f"\n🏆 TARGET REGRESI TERBAIK: {best_reg['Target']}")
    print(f"   R2 Score: {best_reg['R2 Score']:.4f}")

# Save results
results_df.to_csv(f'{output_dir}/all_targets_comparison.csv', index=False)
print(f"\n  => Saved: {output_dir}/all_targets_comparison.csv")

# ================================================================
# 6. VISUALIZATION
# ================================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Chart 1: Classification targets - Improvement over Baseline
if len(classification_results) > 0:
    ax = axes[0]
    names = classification_results['Target'].values
    improvements = classification_results['Improvement (%)'].values
    aucs = classification_results['ROC-AUC'].values

    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(names)))
    bars = ax.barh(range(len(names)), improvements, color=colors[::-1])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel('Improvement over Baseline (%)', fontsize=12)
    ax.set_title('Classification Targets - Improvement over Baseline', fontsize=13, fontweight='bold')
    ax.invert_yaxis()

    for i, (imp, auc_val) in enumerate(zip(improvements, aucs)):
        auc_str = f'AUC={auc_val:.3f}'
        ax.text(imp + 0.5, i, f'{imp:.1f}% ({auc_str})', va='center', fontsize=10)

# Chart 2: MI Score comparison
if len(classification_results) > 0:
    ax = axes[1]
    mi_vals = classification_results['MI Mean'].values
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(names)))
    bars = ax.barh(range(len(names)), mi_vals, color=colors[::-1])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel('Mean Mutual Information Score', fontsize=12)
    ax.set_title('Feature Signal Strength (MI Score) by Target', fontsize=13, fontweight='bold')
    ax.invert_yaxis()

    for i, mi in enumerate(mi_vals):
        ax.text(mi + 0.0001, i, f'{mi:.4f}', va='center', fontsize=10)

plt.suptitle('PERBANDINGAN PREDIKTABILITAS TARGET', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/target_comparison.png', dpi=100, bbox_inches='tight')
plt.close()
print(f"  => Saved: {output_dir}/target_comparison.png")

print("\n" + "=" * 80)
print("ANALISIS SELESAI!")
print("=" * 80)
