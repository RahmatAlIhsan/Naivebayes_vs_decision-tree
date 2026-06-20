#!/usr/bin/env python3
"""
PREPROCESSING PIPELINE - Colorectal Cancer Dataset
===================================================
Tahapan preprocessing yang menghasilkan CSV terpisah:
  1. cleaning_dataset.csv        - Pembersihan data awal
  2. dataset_handling_missing_value.csv - Penanganan missing values
  3. dataset_no_outlier.csv      - Penghapusan outlier
  4. normalisasi_dan_standar_fitur.csv  - Normalisasi & Standarisasi fitur
"""

import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')

# Buat folder output
output_dir = 'preprocessed'
os.makedirs(output_dir, exist_ok=True)

# ================================================================
# LOAD DATA AWAL
# ================================================================
print("=" * 70)
print("PREPROCESSING PIPELINE - COLORECTAL CANCER DATASET")
print("=" * 70)

df = pd.read_csv('colorectal_cancer_dataset_asli.csv', sep=';')
print(f"\nDataset awal: {df.shape[0]:,} baris, {df.shape[1]} kolom")

# Identifikasi kolom
num_cols = ['Age', 'Tumor_Size_mm', 'Healthcare_Costs', 'Incidence_Rate_per_100K', 'Mortality_Rate_per_100K']
cat_cols = [c for c in df.columns if c not in num_cols and c != 'Patient_ID']

# ================================================================
# TAHAP 1: CLEANING DATASET
# ================================================================
print("\n" + "=" * 70)
print("TAHAP 1: CLEANING DATASET")
print("=" * 70)

df1 = df.copy()
cleaning_log = []

# 1a. Cek & hapus duplikat
n_duplicates = df1.duplicated().sum()
if n_duplicates > 0:
    df1 = df1.drop_duplicates()
    cleaning_log.append(f"Hapus {n_duplicates} baris duplikat")
print(f"  Duplikat: {n_duplicates} baris >> {'dihapus' if n_duplicates > 0 else 'tidak ada'}")

# 1b. Cek whitespace pada kolom string
for col in cat_cols:
    if col in df1.columns:
        has_ws = df1[col].astype(str).str.contains(r'^\s+|\s+$', na=False).sum()
        if has_ws > 0:
            df1[col] = df1[col].astype(str).str.strip()
            cleaning_log.append(f"Strip whitespace kolom {col} ({has_ws} nilai)")

# 1c. Validasi range numerik
invalid_age = ((df1['Age'] <= 0) | (df1['Age'] > 120)).sum()
if invalid_age > 0:
    df1 = df1[(df1['Age'] > 0) & (df1['Age'] <= 120)]
    cleaning_log.append(f"Hapus {invalid_age} baris dengan Age tidak valid")

invalid_tumor = (df1['Tumor_Size_mm'] <= 0).sum()
if invalid_tumor > 0:
    df1 = df1[df1['Tumor_Size_mm'] > 0]
    cleaning_log.append(f"Hapus {invalid_tumor} baris dengan Tumor_Size_mm <= 0")

invalid_cost = (df1['Healthcare_Costs'] <= 0).sum()
if invalid_cost > 0:
    df1 = df1[df1['Healthcare_Costs'] > 0]
    cleaning_log.append(f"Hapus {invalid_cost} baris dengan Healthcare_Costs <= 0")

print(f"  Hasil cleaning: {df1.shape[0]:,} baris, {df1.shape[1]} kolom")
for log in cleaning_log:
    print(f"    - {log}")

df1.to_csv(f'{output_dir}/cleaning_dataset.csv', index=False)
print(f"  => Saved: {output_dir}/cleaning_dataset.csv")

# ================================================================
# TAHAP 2: HANDLING MISSING VALUES
# ================================================================
print("\n" + "=" * 70)
print("TAHAP 2: HANDLING MISSING VALUES")
print("=" * 70)

df2 = df1.copy()
mv_log = []

missing_before = df2.isnull().sum()
missing_cols = missing_before[missing_before > 0]

if len(missing_cols) == 0:
    print("  Tidak ada missing values pada dataset")
    mv_log.append("Tidak ada missing values ditemukan")
else:
    print(f"  Missing values sebelum penanganan:")
    for col, val in missing_cols.items():
        print(f"    - {col}: {val} ({val/len(df2)*100:.2f}%)")
    for col in missing_cols.index:
        if col in num_cols:
            median_val = df2[col].median()
            df2[col] = df2[col].fillna(median_val)
            mv_log.append(f"Kolom {col}: isi {missing_cols[col]} missing dengan median ({median_val:.2f})")
        else:
            mode_val = df2[col].mode()[0]
            df2[col] = df2[col].fillna(mode_val)
            mv_log.append(f"Kolom {col}: isi {missing_cols[col]} missing dengan modus ({mode_val})")

missing_after = df2.isnull().sum().sum()
print(f"  Missing values setelah penanganan: {missing_after}")
for log in mv_log:
    print(f"    - {log}")

df2.to_csv(f'{output_dir}/dataset_handling_missing_value.csv', index=False)
print(f"  => Saved: {output_dir}/dataset_handling_missing_value.csv")

# ================================================================
# TAHAP 3: REMOVE OUTLIERS
# ================================================================
print("\n" + "=" * 70)
print("TAHAP 3: REMOVE OUTLIERS")
print("=" * 70)

df3 = df2.copy()
outlier_log = {}
outlier_rows_before = df3.shape[0]

for col in num_cols:
    Q1 = df3[col].quantile(0.25)
    Q3 = df3[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    n_outliers = ((df3[col] < lower_bound) | (df3[col] > upper_bound)).sum()
    outlier_log[col] = {
        'n_outliers': n_outliers,
        'lower_bound': round(lower_bound, 2),
        'upper_bound': round(upper_bound, 2),
        'pct': round(n_outliers / len(df3) * 100, 2)
    }
    df3 = df3[(df3[col] >= lower_bound) & (df3[col] <= upper_bound)]

n_total_outliers = outlier_rows_before - df3.shape[0]
print(f"  Total baris sebelum: {outlier_rows_before:,}")
print(f"  Total baris setelah:  {df3.shape[0]:,}")
print(f"  Baris terhapus:       {n_total_outliers:,} ({n_total_outliers/outlier_rows_before*100:.2f}%)")
print(f"\n  Detil outlier per kolom:")
for col, info in outlier_log.items():
    print(f"    {col}: {info['n_outliers']:,} outliers ({info['pct']}%) | bounds: [{info['lower_bound']}, {info['upper_bound']}]")

df3.to_csv(f'{output_dir}/dataset_no_outlier.csv', index=False)
print(f"  => Saved: {output_dir}/dataset_no_outlier.csv")

# ================================================================
# TAHAP 4: NORMALISASI & STANDARISASI FITUR
# ================================================================
print("\n" + "=" * 70)
print("TAHAP 4: NORMALISASI & STANDARISASI FITUR")
print("=" * 70)

df4 = df3.copy()

# --- 4a. ONE-HOT ENCODING untuk fitur kategorikal ---
encode_cols = cat_cols  # Patient_ID sudah tidak termasuk

before_encode_shape = df4.shape
df4 = pd.get_dummies(df4, columns=encode_cols, drop_first=False, dtype=int)
print(f"  One-Hot Encoding:")
print(f"    Sebelum: {before_encode_shape[1]} kolom")
print(f"    Sesudah:  {df4.shape[1]} kolom (+{df4.shape[1]-before_encode_shape[1]} fitur baru)")

# --- 4b. NORMALISASI (Min-Max Scaling) ---
for col in num_cols:
    col_min = df4[col].min()
    col_max = df4[col].max()
    df4[f"{col}_Normalized"] = (df4[col] - col_min) / (col_max - col_min)

print(f"\n  Normalisasi (Min-Max Scaling) [0,1]:")
for col in num_cols:
    print(f"    {col}_Normalized: min={df4[f'{col}_Normalized'].min():.4f}, max={df4[f'{col}_Normalized'].max():.4f}")

# --- 4c. STANDARISASI (Z-Score) ---
for col in num_cols:
    col_mean = df4[col].mean()
    col_std = df4[col].std()
    df4[f"{col}_Standardized"] = (df4[col] - col_mean) / col_std

print(f"\n  Standarisasi (Z-Score) [mean~0, std~1]:")
for col in num_cols:
    print(f"    {col}_Standardized: mean={df4[f'{col}_Standardized'].mean():.4f}, std={df4[f'{col}_Standardized'].std():.4f}")

# --- 4d. SIMPAN ---
# Normalisasi: kolom asli + normalized
orig_norm_cols = [c for c in df4.columns if not c.endswith('_Standardized')]
df4_norm = df4[orig_norm_cols]

# Standarisasi: kolom asli + standardized
orig_std_cols = [c for c in df4.columns if not c.endswith('_Normalized')]
df4_std = df4[orig_std_cols]

df4_norm.to_csv(f'{output_dir}/normalisasi_dan_standar_fitur.csv', index=False)
print(f"\n  => Saved: {output_dir}/normalisasi_dan_standar_fitur.csv ({df4_norm.shape[1]} kolom)")

df4_std.to_csv(f'{output_dir}/standarisasi_fitur.csv', index=False)
print(f"  => Saved: {output_dir}/standarisasi_fitur.csv ({df4_std.shape[1]} kolom)")

df4.to_csv(f'{output_dir}/dataset_preprocessing_lengkap.csv', index=False)
print(f"  => Saved: {output_dir}/dataset_preprocessing_lengkap.csv ({df4.shape[1]} kolom)")

# ================================================================
# RINGKASAN PREPROCESSING
# ================================================================
print("\n" + "=" * 70)
print("RINGKASAN PREPROCESSING")
print("=" * 70)

print(f"""
+-----------+--------------------------------------------+-------------+----------+
| Tahap     | File                                       | Baris       | Kolom    |
+-----------+--------------------------------------------+-------------+----------+
| 0. Raw    | colorectal_cancer_dataset_asli.csv         | {df.shape[0]:>9,} | {df.shape[1]:>6} |
| 1. Clean  | {output_dir}/cleaning_dataset.csv               | {df1.shape[0]:>9,} | {df1.shape[1]:>6} |
| 2. MV     | {output_dir}/dataset_handling_missing_value.csv  | {df2.shape[0]:>9,} | {df2.shape[1]:>6} |
| 3. No Out | {output_dir}/dataset_no_outlier.csv              | {df3.shape[0]:>9,} | {df3.shape[1]:>6} |
| 4. Norm   | {output_dir}/normalisasi_dan_standar_fitur.csv   | {df4_norm.shape[0]:>9,} | {df4_norm.shape[1]:>6} |
| 4. Std    | {output_dir}/standarisasi_fitur.csv              | {df4_std.shape[0]:>9,} | {df4_std.shape[1]:>6} |
+-----------+--------------------------------------------+-------------+----------+
""")

print(f"File Output (di folder '{output_dir}/'):")
print(f"  1. cleaning_dataset.csv               - Pembersihan data (duplikat, whitespace, validasi)")
print(f"  2. dataset_handling_missing_value.csv  - Penanganan missing values")
print(f"  3. dataset_no_outlier.csv              - Outlier removal (IQR method)")
print(f"  4. normalisasi_dan_standar_fitur.csv   - One-Hot Encoding + Min-Max Normalisasi")
print(f"  5. standarisasi_fitur.csv              - One-Hot Encoding + Z-Score Standarisasi")
print(f"  6. dataset_preprocessing_lengkap.csv   - Semua fitur (original + transformed)")
print("=" * 70)
