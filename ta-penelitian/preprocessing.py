#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
======================================================================
PREPROCESSING - Prediksi Stunting pada Balita
======================================================================
Tahapan:
1. Setup & Data Loading
2. Data Preprocessing (standardisasi, fix inkonsistensi, hapus duplikat)
3. Handling Missing Values
4. Outlier Detection & Handling (IQR, Z-Score, Capping)
5. Export Clean Dataset & Summary
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
from scipy import stats

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams.update({'figure.figsize': (10, 6), 'font.size': 12})

OUTPUT_DIR = 'data/processed'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("PREPROCESSING - Prediksi Stunting pada Balita")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════
# 1. SETUP & DATA LOADING
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BAGIAN 1: SETUP & DATA LOADING")
print("=" * 70)

# Load raw data
print("\n[1.1] Load Dataset")
df_raw = pd.read_csv('stunting.csv', sep=';')
print(f"  Dataset berhasil dimuat!")
print(f"  Shape: {df_raw.shape[0]} baris, {df_raw.shape[1]} kolom")
print(f"  Kolom: {df_raw.columns.tolist()}")

# Eksplorasi awal
print("\n[1.2] Eksplorasi Awal")
print("  5 data pertama:")
print(df_raw.head().to_string(index=False))
print(f"\n  Informasi dataset:")
print(f"  Tipe data:\n{df_raw.dtypes}")
print(f"\n  Statistik deskriptif:")
print(df_raw.describe().round(2).to_string())

# Distribusi target
print("\n[1.3] Distribusi Target (Status)")
status_counts = df_raw['Status'].value_counts()
print(f"\n{status_counts}")
print(f"\n  Persentase:")
print(df_raw['Status'].value_counts(normalize=True).mul(100).round(2))

# Visualisasi distribusi awal
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Distribusi Status Stunting (Data Awal)', fontsize=14, fontweight='bold')

colors_status = ['#2ECC71', '#F39C12', '#E74C3C']
c = df_raw['Status'].value_counts()
c.plot(kind='bar', ax=axes[0], color=colors_status[:len(c)])
axes[0].set_title('Bar Chart')
axes[0].set_xlabel('Status')
axes[0].set_ylabel('Jumlah')
axes[0].tick_params(axis='x', rotation=0)

c.plot(kind='pie', ax=axes[1], autopct='%1.1f%%',
       colors=colors_status[:len(c)], startangle=90)
axes[1].set_title('Proporsi')
axes[1].set_ylabel('')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/distribusi_status_awal.png', dpi=100, bbox_inches='tight')
plt.close()
print("  Visualisasi tersimpan: data/processed/distribusi_status_awal.png")

# Simpan data mentah & statistik
df_raw.to_csv(f'{OUTPUT_DIR}/stunting_00_raw.csv', index=False)
summary_stats = df_raw.describe(include='all').round(2)
summary_stats.to_csv(f'{OUTPUT_DIR}/stunting_00_summary_stats.csv')

dist_status = df_raw['Status'].value_counts().reset_index()
dist_status.columns = ['Status', 'Jumlah']
dist_status['Persentase'] = (dist_status['Jumlah'] / dist_status['Jumlah'].sum() * 100).round(2)
dist_status.to_csv(f'{OUTPUT_DIR}/stunting_00_distribusi_status.csv', index=False)
print(f"  Raw data exported: {OUTPUT_DIR}/stunting_00_raw.csv")

# ═══════════════════════════════════════════════════════════════
# 2. DATA PREPROCESSING
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BAGIAN 2: DATA PREPROCESSING")
print("=" * 70)

df = df_raw.copy()

# 2.1 Standardisasi Nama Kolom
print("\n[2.1] Standardisasi Nama Kolom")
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
print(f"  Kolom setelah standardisasi: {df.columns.tolist()}")

# 2.2 Fix Inkonsistensi String pada Status
print("\n[2.2] Fix Inkonsistensi Status")
print(f"  Nilai unik SEBELUM: {df['status'].unique()}")
print(f"  Value counts SEBELUM:\n{df['status'].value_counts()}")
df['status'] = df['status'].str.strip().str.title()
print(f"  Nilai unik SETELAH: {df['status'].unique()}")
print(f"  Value counts SETELAH:\n{df['status'].value_counts()}")

# 2.3 Standardisasi Kolom JK
print("\n[2.3] Standardisasi Kolom JK")
print(f"  Nilai unik SEBELUM: {df['jk'].unique()}")
jk_mapping = {'L': 'Laki-Laki', 'P': 'Perempuan'}
df['jk'] = df['jk'].str.strip().map(jk_mapping)
print(f"  Nilai unik SETELAH: {df['jk'].unique()}")
print(f"  Value counts SETELAH:\n{df['jk'].value_counts()}")

# 2.4 Deteksi & Hapus Duplikat
print("\n[2.4] Deteksi & Hapus Duplikat")
duplicates = df.duplicated().sum()
print(f"  Jumlah data duplikat: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"  {duplicates} duplikat telah dihapus.")
print(f"  Shape setelah hapus duplikat: {df.shape}")

# 2.5 Konversi Tipe Data
print("\n[2.5] Konversi Tipe Data")
print(f"  Tipe data SEBELUM:\n{df.dtypes}")
df['umur_bulan'] = pd.to_numeric(df['umur_bulan'], errors='coerce')
df['tinggi'] = pd.to_numeric(df['tinggi'], errors='coerce')
print(f"  Tipe data SETELAH:\n{df.dtypes}")

# Export intermediate
df.to_csv(f'{OUTPUT_DIR}/stunting_01_preprocessed.csv', index=False)
print(f"  Shape: {df.shape}")
print(f"  Intermediate CSV: {OUTPUT_DIR}/stunting_01_preprocessed.csv")

# ═══════════════════════════════════════════════════════════════
# 3. HANDLING MISSING VALUES
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BAGIAN 3: HANDLING MISSING VALUES")
print("=" * 70)

print("\n[3.1] Deteksi Missing Values")
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({'Jumlah Missing': missing, 'Persentase (%)': missing_pct.round(2)})

if missing.sum() > 0:
    print(f"\n{missing_df[missing_df['Jumlah Missing'] > 0]}")
    print(f"  Total missing values: {df.isnull().sum().sum()}")

    # Visualisasi
    plt.figure(figsize=(8, 4))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title('Peta Missing Values')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/missing_values_heatmap.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Heatmap saved: missing_values_heatmap.png")

    # Penanganan
    print("\n[3.2] Penanganan Missing Values")
    for col in df.columns:
        mc = df[col].isnull().sum()
        if mc > 0:
            if df[col].dtype in ['float64', 'int64']:
                median_val = df[col].median()
                df[col + '_was_missing'] = df[col].isnull().astype(int)
                df[col] = df[col].fillna(median_val)
                print(f"  Kolom '{col}': {mc} missing diisi dengan median ({median_val:.2f})")
            else:
                modus_val = df[col].mode()[0]
                df[col] = df[col].fillna(modus_val)
                print(f"  Kolom '{col}': {mc} missing diisi dengan modus ({modus_val})")
    print(f"  Total {df.isnull().sum().sum()} missing values tersisa (harusnya 0).")
else:
    print("  Tidak ada missing values! Dataset sudah lengkap.")

# Export intermediate
df.to_csv(f'{OUTPUT_DIR}/stunting_02_no_missing.csv', index=False)
print(f"\n  Intermediate CSV: {OUTPUT_DIR}/stunting_02_no_missing.csv")
print(f"  Shape: {df.shape}")

# ═══════════════════════════════════════════════════════════════
# 4. OUTLIER DETECTION & HANDLING
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BAGIAN 4: OUTLIER DETECTION & HANDLING")
print("=" * 70)

numeric_cols = ['umur_bulan', 'tinggi']

# 4.1 Statistik Deskriptif
print("\n[4.1] Statistik Deskriptif Kolom Numerik")
print(df[numeric_cols].describe().round(2).to_string())

# 4.2 Visualisasi Boxplot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Boxplot Sebelum Penanganan Outlier', fontsize=14, fontweight='bold')

sns.boxplot(y=df['umur_bulan'], ax=axes[0], color='skyblue')
axes[0].set_title('Umur (Bulan)')

sns.boxplot(y=df['tinggi'], ax=axes[1], color='lightcoral')
axes[1].set_title('Tinggi Badan (cm)')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/boxplot_sebelum_outlier.png', dpi=100, bbox_inches='tight')
plt.close()
print("  Boxplot saved: boxplot_sebelum_outlier.png")

# 4.3 Deteksi Outlier dengan IQR
print("\n[4.2] Deteksi Outlier dengan Metode IQR")
print(f"  {'Kolom':15} {'Q1':8} {'Q3':8} {'IQR':8} {'Lower':8} {'Upper':8} {'Outlier':8} {'%':8}")
print(f"  {'-'*63}")

outlier_info = {}
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    iqr = Q3 - Q1
    lower = Q1 - 1.5 * iqr
    upper = Q3 + 1.5 * iqr
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    pct_outliers = n_outliers / len(df) * 100
    print(f"  {col:15} {Q1:<8.2f} {Q3:<8.2f} {iqr:<8.2f} {lower:<8.2f} {upper:<8.2f} {n_outliers:<8} {pct_outliers:<8.2f}")
    outlier_info[col] = {'Q1': Q1, 'Q3': Q3, 'IQR': iqr, 'lower': lower, 'upper': upper,
                          'n_iqr': n_outliers, 'pct_iqr': pct_outliers}

# 4.4 Deteksi Outlier dengan Z-Score
print(f"\n[4.3] Deteksi Outlier dengan Metode Z-Score (Threshold: |Z| > 3)")
for col in numeric_cols:
    z_scores = np.abs(stats.zscore(df[col]))
    n_outliers = (z_scores > 3).sum()
    pct_outliers = n_outliers / len(df) * 100
    print(f"  {col:15} | Outlier: {n_outliers:<4} ({pct_outliers:.2f}%)")
    outlier_info[col]['n_zscore'] = n_outliers
    outlier_info[col]['pct_zscore'] = pct_outliers

# 4.5 Scatter Plot Umur vs Tinggi
fig, ax = plt.subplots(figsize=(10, 6))
color_map = {'Normal': '#2ECC71', 'Stunted': '#F39C12', 'Severely Stunted': '#E74C3C'}
colors = df['status'].map(color_map)

scatter = ax.scatter(df['umur_bulan'], df['tinggi'], c=colors, alpha=0.6,
                     edgecolors='black', linewidth=0.5)
ax.set_xlabel('Umur (Bulan)')
ax.set_ylabel('Tinggi Badan (cm)')
ax.set_title('Hubungan Umur vs Tinggi Badan (Berdasarkan Status Stunting)', fontsize=13)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ECC71', label='Normal'),
    Patch(facecolor='#F39C12', label='Stunted'),
    Patch(facecolor='#E74C3C', label='Severely Stunted')
]
ax.legend(handles=legend_elements, title='Status')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/scatter_umur_tinggi.png', dpi=100, bbox_inches='tight')
plt.close()
print("  Scatter plot saved: scatter_umur_tinggi.png")

# 4.6 Penanganan Outlier (Capping / Winsorization)
print(f"\n[4.4] Penanganan Outlier dengan Capping (Winsorization)")
df_clean = df.copy()
total_capped = 0

for col in numeric_cols:
    info = outlier_info[col]
    before = info['n_iqr']
    df_clean[col] = df_clean[col].clip(lower=info['lower'], upper=info['upper'])
    after = ((df_clean[col] < info['lower']) | (df_clean[col] > info['upper'])).sum()
    total_capped += before
    print(f"  {col:15} | Before: {before} outlier | After capping: {after} outlier")

print(f"\n  Total outlier ditangani (capping): {total_capped}")

# Visualisasi perbandingan sebelum vs sesudah
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Perbandingan Sebelum vs Sesudah Penanganan Outlier', fontsize=14, fontweight='bold')

for i, col in enumerate(numeric_cols):
    sns.boxplot(y=df[col], ax=axes[i, 0], color='lightcoral')
    axes[i, 0].set_title(f'{col.upper()} - SEBELUM')
    sns.boxplot(y=df_clean[col], ax=axes[i, 1], color='lightgreen')
    axes[i, 1].set_title(f'{col.upper()} - SETELAH')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/perbandingan_outlier.png', dpi=100, bbox_inches='tight')
plt.close()
print("  Perbandingan visual: perbandingan_outlier.png")

# Export intermediate
df_clean.to_csv(f'{OUTPUT_DIR}/stunting_03_after_outlier.csv', index=False)
print(f"  Intermediate CSV: {OUTPUT_DIR}/stunting_03_after_outlier.csv")
print(f"  Shape: {df_clean.shape}")

# ═══════════════════════════════════════════════════════════════
# 5. EXPORT CLEAN DATASET & SUMMARY
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BAGIAN 5: EXPORT CLEAN DATASET & SUMMARY")
print("=" * 70)

# 5.1 Simpan Dataset Bersih
df_clean.to_csv(f'{OUTPUT_DIR}/stunting_clean.csv', index=False)
print(f"\n[5.1] Dataset bersih tersimpan: {OUTPUT_DIR}/stunting_clean.csv")

# 5.2 Perbandingan Sebelum vs Sesudah
print(f"\n[5.2] Perbandingan Sebelum vs Sesudah Cleaning")
print(f"  {'Aspek':25} {'Sebelum':12} {'Sesudah':12}")
print(f"  {'-'*49}")
print(f"  {'Jumlah Baris':25} {df_raw.shape[0]:<12} {df_clean.shape[0]:<12}")
print(f"  {'Jumlah Kolom':25} {df_raw.shape[1]:<12} {df_clean.shape[1]:<12}")
print(f"  {'Missing Values':25} {df_raw.isnull().sum().sum():<12} {df_clean.isnull().sum().sum():<12}")
print(f"  {'Duplikat Dihapus':25} {'-':<12} {duplicates:<12}")

# 5.3 Ringkasan Final
print(f"\n[5.3] Ringkasan Final Cleaning Dataset")

# Pre-compute strings to avoid nested f-string issues
missing_status = "Tidak ditemukan missing values dalam dataset" if df.isnull().sum().sum() == 0 else "Missing values telah ditangani"
iqr_str = ', '.join([f'{c}: {outlier_info[c]["n_iqr"]} outlier' for c in numeric_cols])
zscore_str = ', '.join([f'{c}: {outlier_info[c]["n_zscore"]} outlier' for c in numeric_cols])

ringkasan = f"""
Dataset: stunting.csv
Total Sampel Awal: {df_raw.shape[0]}
Total Sampel Akhir: {df_clean.shape[0]}
Jumlah Kolom: {df_clean.shape[1]}

TAHAPAN YANG TELAH DILAKUKAN:
1. Data Loading & Eksplorasi Awal
2. Data Preprocessing:
   - Standardisasi nama kolom
   - Fix inkonsistensi nilai Status (spasi, kapitalisasi)
   - Standardisasi kolom JK (L/P -> Laki-Laki/Perempuan)
   - Deteksi dan hapus duplikasi data ({duplicates} duplikat dihapus)
   - Konversi tipe data numerik

3. Handling Missing Values:
   - {missing_status}
   - Handler tetap disiapkan untuk antisipasi data baru

4. Outlier Detection & Handling:
   - Metode IQR: {iqr_str}
   - Metode Z-Score: {zscore_str}
   - Penanganan: Capping/Winsorization (mengganti outlier dengan batas IQR)

5. Export Dataset Bersih ke CSV
"""

print(ringkasan)

# Simpan ringkasan ke file
with open(f'{OUTPUT_DIR}/ringkasan_cleaning.txt', 'w', encoding='utf-8') as f:
    f.write(ringkasan.strip())

print(f"  Ringkasan tersimpan: {OUTPUT_DIR}/ringkasan_cleaning.txt")

# 5.4 Distribusi Status Akhir
print(f"\n[5.4] Distribusi Status Akhir")
final_dist = df_clean['status'].value_counts()
print(final_dist)
print(f"\n  Persentase:")
print(df_clean['status'].value_counts(normalize=True).mul(100).round(2))

# 5.5 5 Data Pertama
print(f"\n[5.5] 5 Data Pertama (Dataset Bersih)")
print(df_clean.head().to_string(index=False))

print("\n" + "=" * 70)
print("PREPROCESSING SELESAI!")
print(f"Dataset bersih siap digunakan: {OUTPUT_DIR}/stunting_clean.csv")
print(f"Semua output preprocessing tersimpan di: {OUTPUT_DIR}/")
print("=" * 70)
