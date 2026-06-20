#!/usr/bin/env python3
"""
SOLUSI LANJUTAN - Survival_Prediction Target
=============================================
Mencoba 6 pendekatan: NN, Feature Engineering, Polynomial, PCA, Stacking, GridSearch
"""

import pandas as pd
import numpy as np
import warnings
import time
import gc
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (RandomForestClassifier as RF, StackingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
from imblearn.over_sampling import SMOTE

import os
os.environ['OMP_NUM_THREADS'] = '2'
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
gc.collect()

print("=" * 80)
print("SOLUSI LANJUTAN - PREDIKSI SURVIVAL_PREDICTION")
print("=" * 80)

# ================================================================
# 1. LOAD DATA (5K sample for memory)
# ================================================================
print("\n1. LOAD DATA")
df = pd.read_csv('preprocessed/dataset_no_outlier.csv')
df['Survival_Prediction'] = df['Survival_Prediction'].map({'Yes': 1, 'No': 0})
print(f"Total dataset: {df.shape[0]:,} baris")

# Sample 5K stratified
df_sample, _ = train_test_split(df, train_size=5000, random_state=42,
                                 stratify=df['Survival_Prediction'])
y_sample = df_sample['Survival_Prediction'].values
print(f"Sample: 5,000 baris (Yes={y_sample.sum():,}, No={(y_sample==0).sum():,})")

# ================================================================
# 2. FEATURE ENGINEERING
# ================================================================
print("\n2. FEATURE ENGINEERING")

num_cols = ['Age', 'Tumor_Size_mm', 'Healthcare_Costs', 'Incidence_Rate_per_100K', 'Mortality_Rate_per_100K']
cat_cols_all = ['Country', 'Gender', 'Cancer_Stage', 'Family_History', 'Smoking_History',
                'Alcohol_Consumption', 'Obesity_BMI', 'Diet_Risk', 'Physical_Activity',
                'Diabetes', 'Inflammatory_Bowel_Disease', 'Genetic_Mutation',
                'Screening_History', 'Early_Detection', 'Treatment_Type',
                'Urban_or_Rural', 'Economic_Classification', 'Healthcare_Access', 'Insurance_Status']

# Buat fitur interaksi baru
df_fe = df_sample.copy()

# Risk Score (jumlah faktor risiko)
risk_factors = ['Smoking_History', 'Alcohol_Consumption', 'Obesity_BMI', 
                'Diabetes', 'Genetic_Mutation', 'Family_History', 'Inflammatory_Bowel_Disease']
for col in risk_factors:
    df_fe[col + '_bin'] = (df_fe[col] == 'Yes').astype(int)
df_fe['Risk_Score'] = df_fe[[c+'_bin' for c in risk_factors]].sum(axis=1)
print(f"  Created: Risk_Score (sum of {len(risk_factors)} risk factors)")

# Healthy Score (jumlah faktor positif)
healthy_factors = ['Physical_Activity', 'Diet_Risk', 'Screening_History']
df_fe['PA_bin'] = (df_fe['Physical_Activity'] == 'High').astype(int)
df_fe['Diet_bin'] = (df_fe['Diet_Risk'] == 'Low').astype(int)
df_fe['Screen_bin'] = (df_fe['Screening_History'] == 'Regular').astype(int)
df_fe['Healthy_Score'] = df_fe[['PA_bin', 'Diet_bin', 'Screen_bin']].sum(axis=1)
print(f"  Created: Healthy_Score")

# Rasio
df_fe['Tumor_Age_Ratio'] = df_fe['Tumor_Size_mm'] / (df_fe['Age'] + 1)
df_fe['Cost_Age_Ratio'] = df_fe['Healthcare_Costs'] / (df_fe['Age'] + 1)
df_fe['Cost_Tumor_Ratio'] = df_fe['Healthcare_Costs'] / (df_fe['Tumor_Size_mm'] + 1)
df_fe['Mortality_Incidence_Ratio'] = df_fe['Mortality_Rate_per_100K'] / (df_fe['Incidence_Rate_per_100K'] + 1)
print(f"  Created: Tumor_Age_Ratio, Cost_Age_Ratio, Cost_Tumor_Ratio, Mortality_Incidence_Ratio")

# Fitur kategorikal untuk encoding
eng_features = ['Risk_Score', 'Healthy_Score', 'Tumor_Age_Ratio', 'Cost_Age_Ratio',
                'Cost_Tumor_Ratio', 'Mortality_Incidence_Ratio']
all_feats = num_cols + eng_features + cat_cols_all

# One-Hot Encoding
df_enc = pd.get_dummies(df_fe[all_feats + ['Survival_Prediction']], 
                        columns=cat_cols_all, drop_first=True, dtype=int)

feature_names = [c for c in df_enc.columns if c != 'Survival_Prediction']
X = df_enc[feature_names].values
y = df_enc['Survival_Prediction'].values
print(f"Total fitur setelah engineering + encoding: {len(feature_names)}")

# Clean up
del df, df_sample, df_fe, df_enc
gc.collect()

# ================================================================
# 3. SPLIT + SMOTE
# ================================================================
print("\n3. SPLIT + SMOTE")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"Train: {X_train.shape[0]:,} -> {X_train_res.shape[0]:,} (after SMOTE)")
print(f"Test:  {X_test.shape[0]:,}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)
gc.collect()

results_all = {}

# ================================================================
# 4. BASELINE
# ================================================================
print("\n4. BASELINE")
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train_scaled, y_train_res)
y_pred = dummy.predict(X_test_scaled)
y_prob = dummy.predict_proba(X_test_scaled)[:, 1]
results_all['Baseline'] = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred, zero_division=0),
    'Recall': recall_score(y_test, y_pred, zero_division=0),
    'F1-Score': f1_score(y_test, y_pred, zero_division=0),
    'ROC-AUC': roc_auc_score(y_test, y_prob)
}
print(f"  Baseline: Acc={results_all['Baseline']['Accuracy']:.4f}, AUC={results_all['Baseline']['ROC-AUC']:.4f}")

# ================================================================
# 5. NEURAL NETWORK (MLPClassifier)
# ================================================================
print("\n5. NEURAL NETWORK (MLPClassifier)")
mlp = MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation='relu',
                     max_iter=300, random_state=42, early_stopping=True)
start = time.time()
mlp.fit(X_train_scaled, y_train_res)
elapsed = time.time() - start
y_pred = mlp.predict(X_test_scaled)
y_prob = mlp.predict_proba(X_test_scaled)[:, 1]
results_all['Neural Network'] = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred, zero_division=0),
    'Recall': recall_score(y_test, y_pred, zero_division=0),
    'F1-Score': f1_score(y_test, y_pred, zero_division=0),
    'ROC-AUC': roc_auc_score(y_test, y_prob),
    'Time': round(elapsed, 2)
}
print(f"  Acc={results_all['Neural Network']['Accuracy']:.4f}, AUC={results_all['Neural Network']['ROC-AUC']:.4f} ({elapsed:.1f}s)")

# ================================================================
# 6. POLYNOMIAL FEATURES + RF
# ================================================================
print("\n6. POLYNOMIAL FEATURES (degree=2, interaction_only)")
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
print(f"  Fitur: {X_train_scaled.shape[1]} -> {X_train_poly.shape[1]}")

rf_poly = RF(n_estimators=50, max_depth=6, random_state=42, n_jobs=1)
start = time.time()
rf_poly.fit(X_train_poly, y_train_res)
elapsed = time.time() - start
y_pred = rf_poly.predict(X_test_poly)
y_prob = rf_poly.predict_proba(X_test_poly)[:, 1]
results_all['RF + Polynomial'] = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred, zero_division=0),
    'Recall': recall_score(y_test, y_pred, zero_division=0),
    'F1-Score': f1_score(y_test, y_pred, zero_division=0),
    'ROC-AUC': roc_auc_score(y_test, y_prob),
    'Time': round(elapsed, 2)
}
print(f"  Acc={results_all['RF + Polynomial']['Accuracy']:.4f}, AUC={results_all['RF + Polynomial']['ROC-AUC']:.4f} ({elapsed:.1f}s)")

del X_train_poly, X_test_poly, rf_poly
gc.collect()

# ================================================================
# 7. PCA + XGBoost
# ================================================================
print("\n7. PCA + XGBoost")
pca = PCA(n_components=0.90)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f"  Fitur: {X_train_scaled.shape[1]} -> {X_train_pca.shape[1]} (90% variance)")

xgb_pca = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                         random_state=42, n_jobs=1, eval_metric='logloss')
start = time.time()
xgb_pca.fit(X_train_pca, y_train_res)
elapsed = time.time() - start
y_pred = xgb_pca.predict(X_test_pca)
y_prob = xgb_pca.predict_proba(X_test_pca)[:, 1]
results_all['XGBoost + PCA'] = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred, zero_division=0),
    'Recall': recall_score(y_test, y_pred, zero_division=0),
    'F1-Score': f1_score(y_test, y_pred, zero_division=0),
    'ROC-AUC': roc_auc_score(y_test, y_prob),
    'Time': round(elapsed, 2)
}
print(f"  Acc={results_all['XGBoost + PCA']['Accuracy']:.4f}, AUC={results_all['XGBoost + PCA']['ROC-AUC']:.4f} ({elapsed:.1f}s)")

del X_train_pca, X_test_pca, xgb_pca
gc.collect()

# ================================================================
# 8. STACKING ENSEMBLE
# ================================================================
print("\n8. STACKING ENSEMBLE")
estimators = [
    ('svm', CalibratedClassifierCV(LinearSVC(C=1.0, class_weight='balanced', max_iter=2000, random_state=42, dual='auto'), cv=3, n_jobs=1)),
    ('xgb', XGBClassifier(n_estimators=50, max_depth=4, random_state=42, n_jobs=1, eval_metric='logloss')),
    ('rf', RF(n_estimators=30, max_depth=6, random_state=42, n_jobs=1))
]
stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(),
                           cv=3, n_jobs=1)
start = time.time()
stack.fit(X_train_scaled, y_train_res)
elapsed = time.time() - start
y_pred = stack.predict(X_test_scaled)
y_prob = stack.predict_proba(X_test_scaled)[:, 1]
results_all['Stacking Ensemble'] = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred, zero_division=0),
    'Recall': recall_score(y_test, y_pred, zero_division=0),
    'F1-Score': f1_score(y_test, y_pred, zero_division=0),
    'ROC-AUC': roc_auc_score(y_test, y_prob),
    'Time': round(elapsed, 2)
}
print(f"  Acc={results_all['Stacking Ensemble']['Accuracy']:.4f}, AUC={results_all['Stacking Ensemble']['ROC-AUC']:.4f} ({elapsed:.1f}s)")

del stack
gc.collect()

# ================================================================
# 9. GRIDSEARCH XGBoost
# ================================================================
print("\n9. GRIDSEARCH XGBoost")
param_grid = {
    'n_estimators': [30, 50],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid = GridSearchCV(XGBClassifier(random_state=42, n_jobs=1, eval_metric='logloss'),
                    param_grid, cv=cv, scoring='roc_auc', n_jobs=1, verbose=0)
start = time.time()
grid.fit(X_train_scaled, y_train_res)
elapsed = time.time() - start
y_pred = grid.predict(X_test_scaled)
y_prob = grid.predict_proba(X_test_scaled)[:, 1]
results_all['XGBoost GridSearch'] = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred, zero_division=0),
    'Recall': recall_score(y_test, y_pred, zero_division=0),
    'F1-Score': f1_score(y_test, y_pred, zero_division=0),
    'ROC-AUC': roc_auc_score(y_test, y_prob),
    'Best Params': str(grid.best_params_),
    'Time': round(elapsed, 2)
}
print(f"  Best params: {grid.best_params_}")
print(f"  Acc={results_all['XGBoost GridSearch']['Accuracy']:.4f}, AUC={results_all['XGBoost GridSearch']['ROC-AUC']:.4f} ({elapsed:.1f}s)")

del grid
gc.collect()

# ================================================================
# 10. COMPARISON TABLE
# ================================================================
print("\n" + "=" * 80)
print("PERBANDINGAN SEMUA SOLUSI")
print("=" * 80)

results_df = pd.DataFrame(results_all).T.round(4)
results_df = results_df.sort_values('ROC-AUC', ascending=False)

print(f"\n{'Metode':<25} {'Accuracy':>9} {'F1':>9} {'AUC':>9} {'Time':>8}")
print("-" * 60)
for idx, row in results_df.iterrows():
    t = row.get('Time', 0)
    print(f"{idx:<25} {row['Accuracy']:>9.4f} {row['F1-Score']:>9.4f} {row['ROC-AUC']:>9.4f} {t:>7.1f}s")
print("-" * 60)

best_auc = results_df['ROC-AUC'].max()
best_method = results_df['ROC-AUC'].idxmax()
print(f"\n🏆 METODE TERBAIK: {best_method} (AUC = {best_auc:.4f})")

# Save results
results_df.to_csv('advanced_solutions_results.csv')
print(f"\n=> Saved: advanced_solutions_results.csv")

# ================================================================
# 11. CONCLUSION
# ================================================================
print("\n" + "=" * 80)
print("KESIMPULAN")
print("=" * 80)

if best_auc > 0.55:
    print(f"""
🎉 Ada peningkatan! {best_method} mencapai AUC {best_auc:.4f}
Ini menunjukkan bahwa dengan pendekatan yang tepat, Survival_Prediction 
bisa diprediksi lebih baik. Rekomendasi:
1. Gunakan {best_method} sebagai model utama
2. Tambah fitur interaksi lainnya
3. Gunakan full dataset (167K) dengan komputasi yang lebih besar
""")
else:
    print(f"""
⚠️  Semua metode masih AUC ~0.5 (terbaik: {best_method} = {best_auc:.4f})
Ini konfirmasi bahwa fitur di dataset memang TIDAK memiliki sinyal 
prediktif untuk Survival_Prediction.

REKOMENDASI:
1️⃣  Cek sumber dataset - Apakah dataset ini hasil simulasi/sintetik?
    Karena 0 missing values + semua fitur tidak berkorelasi dengan 
    target manapun adalah ciri khas data sintetik.
    
2️⃣  Cari dataset tambahan dengan fitur klinis riil:
    - TNM Staging detail
    - Biomarkers (CEA, CA19-9)
    - Hasil biopsi / histopatologi
    - Data genetik (mutasi KRAS, BRAF, dll)
    
3️⃣  Jika dataset ini memang dari penelitian Anda:
    - Periksa kembali proses labeling Survival_Prediction
    - Mungkin ada kesalahan dalam pembuatan target variable
    - Bandingkan dengan sumber data asli

4️⃣  Gunakan untuk analisis statistik deskriptif saja:
    - Dataset ini sangat baik untuk demografi dan distribusi
    - Tapi tidak cocok untuk predictive modeling
""")
print("=" * 80)
