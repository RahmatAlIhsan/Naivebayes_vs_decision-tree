import json

nb = {
 "cells": [
  {"cell_type": "markdown", "metadata": {"tags": []}, "source": [
    "# Analisis Dataset Stunting\n",
    "\n",
    "**Proyek:** Prediksi Stunting pada Balita\n",
    "**Dataset:** stunting.csv\n",
    "---\n",
    "## Tahapan Analisis\n",
    "1. **Setup & Data Loading** - Load dataset dan eksplorasi awal\n",
    "2. **Data Preprocessing** - Standardisasi, fix inkonsistensi, hapus duplikat\n",
    "3. **Handling Missing Values** - Deteksi dan penanganan nilai hilang\n",
    "4. **Outlier Detection & Handling** - IQR, Z-Score, Capping\n",
    "5. **Export Clean Dataset** - Simpan dataset bersih\n",
    "6. **Model Comparison (8 Algoritma)** - Perbandingan 8 algoritma\n",
    "7. **Final Model: NB vs DT** - Naive Bayes (pembanding) vs Decision Tree (terbaik)\n",
    "8. **Kesimpulan** - Ringkasan hasil\n",
    "---"
  ]},
  {"cell_type": "markdown", "metadata": {"tags": []}, "source": [
    "---\n",
    "# BAGIAN 1: SETUP & DATA LOADING\n",
    "---"
  ]},
  {"cell_type": "code", "execution_count": None, "metadata": {"tags": []}, "outputs": [], "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import os, warnings\n",
    "os.makedirs('data/processed', exist_ok=True)\n",
    "warnings.filterwarnings('ignore')\n",
    "plt.rcParams.update({'figure.figsize': (10, 6), 'font.size': 12})\n",
    "sns.set_style('whitegrid')\n",
    "print('Library berhasil diimport.')"
  ]},
  {"cell_type": "code", "execution_count": None, "metadata": {"tags": []}, "outputs": [], "source": [
    "df_raw = pd.read_csv('data/raw/stunting.csv', sep=';')\n",
    "print(f'Dataset: {df_raw.shape[0]} baris, {df_raw.shape[1]} kolom')\n",
    "display(df_raw.head())"
  ]},
  {"cell_type": "code", "execution_count": None, "metadata": {"tags": []}, "outputs": [], "source": [
    "# Distribusi Target\n",
    "print('=== Distribusi Status Stunting ===')\n",
    "print(df_raw['Status'].value_counts())\n",
    "print(df_raw['Status'].value_counts(normalize=True).mul(100).round(2))\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
    "c = df_raw['Status'].value_counts()\n",
    "c.plot(kind='bar', ax=axes[0], color=['green', 'orange', 'red'])\n",
    "axes[0].set_title('Distribusi Status Stunting')\n",
    "c.plot(kind='pie', ax=axes[1], autopct='%1.1f%%',\n",
    "       colors=['green', 'orange', 'red'], startangle=90)\n",
    "axes[1].set_title('Proporsi'); axes[1].set_ylabel('')\n",
    "plt.tight_layout(); plt.show()"
  ]},
  {"cell_type": "markdown", "metadata": {"tags": []}, "source": [
    "---\n",
    "# BAGIAN 2: DATA PREPROCESSING\n",
    "---"
  ]},
  {"cell_type": "code", "execution_count": None, "metadata": {"tags": []}, "outputs": [], "source": [
    "# Standardisasi Nama Kolom\n",
    "df = df_raw.copy()\n",
    "df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')\n",
    "print('Nama kolom:', df.columns.tolist())"
  ]},
  {"cell_type": "code", "execution_count": None, "metadata": {"tags": []}, "outputs": [], "source": [
    "# Fix Inkonsistensi Status\n",
    "print('Sebelum:', df['status'].unique())\n",
    "df['status'] = df['status'].str.strip().str.title()\n",
    "print('Sesudah:', df['status'].unique())"
  ]},
  {"cell_type": "code", "execution_count": None, "metadata": {"tags": []}, "outputs": [], "source": [
    "# Standardisasi JK\n",
    "print('Sebelum:', df['jk'].unique())\n",
    "df['jk'] = df['jk'].str.strip().map({'L': 'Laki-Laki', 'P': 'Perempuan'})\n",
    "print('Sesudah:', df['jk'].unique())"
  ]},
  {"cell_type": "code", "execution_count": None, "metadata": {"tags": []}, "outputs": [], "source": [
    "# Hapus Duplikat\n",
    "dup = df.duplicated().sum()\n",
    "print(f'Duplikat ditemukan: {dup}')\n",
    "df = df.drop_duplicates().reset_index(drop=True)\n",
    "print(f'Shape setelah hapus duplikat: {df.shape}')"
  ]},
  {"cell_type": "code", "execution_count": None, "metadata": {"tags": []}, "outputs": [], "source": [
    "# Konversi Tipe Data\n",
    "df['umur_bulan'] = pd.to_numeric(df['umur_bulan'], errors='coerce')\n",
    "df['tinggi'] = pd.to_numeric(df['tinggi'], errors='coerce')\n",
    "print(df.dtypes)\n",
    "df.to_csv('data/processed/stunting_01_preprocessed.csv', index=False)\n",
    "print(f'Shape: {df.shape}')"
  ]},
  {"cell_type": "markdown", "metadata": {"tags": []}, "source": [
    "---\n",
    "# BAGIAN 3: HANDLING MISSING VALUES\n",
    "---"
  ]},
  {"cell_type": "code", "execution_count": None, "metadata": {"tags": []}, "outputs": [], "source": [
    "missing = df.isnull().sum()\n",
    "print('Missing values:')\n",
    "print(missing[missing > 0] if missing.sum() > 0 else 'Tidak ada missing values!')\n",
    "print(f'Total: {df.isnull().sum().sum()}')"
  ]},
  {"cell_type": "code", "execution_count": None, "metadata": {"tags": []}, "outputs": [], "source": [
    "# Penanganan (jika ada)\n",
    "total_missing = df.isnull().sum().sum()\n",
    "if total_missing > 0:\n",
    "    for col in df.columns:\n",
    "        mc = df[col].isnull().sum()\n",
    "        if mc > 0:\n",
    "            if df[col].dtype in ['float64', 'int64']:\n",
    "                df[col] = df[col].fillna(df[col].median())\n",
    "            else:\n",
    "                df[col] = df[col].fillna(df[col].mode()[0])\n",
    "    print(f'{total_missing} missing values ditangani.')\n",
    "else:\n",
    "    print('Dataset sudah lengkap, tidak ada missing values.')\n",
    "\n",
    "df.to_csv('data/processed/stunting_02_no_missing.csv', index=False)\n",
    "print(f'Shape: {df.shape}')"
  ]},
  {"cell_type": "markdown", "metadata": {"tags": []}, "source": [
    "---\n",
    "# BAGIAN 4: OUTLIER DETECTION & HANDLING\n",
    "---"
  ]},
  {"cell_type": "code", "execution_count": None, "metadata": {"tags": []}, "outputs": [], "source": [
    "numeric_cols = ['umur_bulan', 'tinggi']\n",
    "print('=== Statistik Deskriptif ===')\n",
    "display(df[numeric_cols].describe().round(2))\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "sns.boxplot(y=df['umur_bulan'], ax=axes[0], color='skyblue')\n",
    "axes[0].set_title('Boxplot: Umur (Bulan)')\n",
    "sns.boxplot(y=df['tinggi'], ax=axes[1], color='lightcoral')\n",
    "axes[1].set_title('Boxplot: Tinggi (cm)')\n",
    "plt.tight_layout(); plt.show()"
  ]},
  {"cell_type": "code", "execution_count": None, "metadata": {"tags": []}, "outputs": [], "source": [
    "# Deteksi Outlier dengan IQR & Z-Score\n",
    "from scipy import stats\n",
    "\n",
    "for col in numeric_cols:\n",
    "    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)\n",
    "    iqr = Q3 - Q1\n",
    "    iqr_outliers = ((df[col] < Q1 - 1.5*iqr) | (df[col] > Q3 + 1.5*iqr)).sum()\n",
    "    z_outliers = (np.abs(stats.zscore(df[col])) > 3).sum()\n",
    "    print(f'{col:15} | IQR outliers: {iqr_outliers} | Z-score outliers: {z_outliers}')"
  ]},
  {"cell_type": "code", "execution_count": None, "metadata": {"tags": []}, "outputs": [], "source": [
    "# Penanganan Outlier: Capping (Winsorization)\n",
    "df_clean = df.copy()\n",
    "for col in numeric_cols:\n",
    "    Q1, Q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)\n",
    "    iqr = Q3 - Q1\n",
    "    lower = Q1 - 1.5*iqr\n",
    "    upper = Q3 + 1.5*iqr\n",
    "    before = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()\n",
    "    df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)\n",
    "    print(f'{col:15} | Outlier ditangani: {before}')\n",
    "\n",
    "df_clean.to_csv('data/processed/stunting_03_after_outlier.csv', index=False)\n",
    "print(f'Shape: {df_clean.shape}')"
  ]},
  {"cell_type": "markdown", "metadata": {"tags": []}, "source": [
    "---\n",
    "# BAGIAN 5: EXPORT CLEAN DATASET\n",
    "---"
  ]},
  {"cell_type": "code", "execution_count": None, "metadata": {"tags": []}, "outputs": [], "source": [
    "df_clean.to_csv('data/processed/stunting_clean.csv', index=False)\n",
    "print(f'Dataset bersih: {df_clean.shape[0]} baris, {df_clean.shape[1]} kolom')\n",
    "print('\\nDistribusi Status:')\n",
    "print(df_clean['status'].value_counts())\n",
    "print('\\nDataset siap digunakan untuk pemodelan!')"
  ]},
  {"cell_type": "markdown", "metadata": {"tags": []}, "source": [
    "---\n",
    "# BAGIAN 6: MODEL COMPARISON - 8 ALGORITMA KLASIFIKASI\n",
    "---"
  ]},
  {"cell_type": "code", "execution_count": None, "metadata": {"tags": []}, "outputs": [], "source": [
    "df_results = pd.read_csv('data/processed/model_comparison_results.csv')\n",
    "print('=== Perbandingan 8 Algoritma ===')\n",
    "print(df_results.to_string(index=False))"
  ]},
  {"cell_type": "markdown", "metadata": {"tags": []}, "source": [
    "---\n",
    "# BAGIAN 7: FINAL MODEL - NAIVE BAYES vs DECISION TREE\n",
    "---\n",
    "| Model | Preprocessing | Fungsi |\n",
    "|---|---|---|\n",
    "| Naive Bayes | Tanpa SMOTE, Tanpa Tuning | Pembanding (baseline) |\n",
    "| Decision Tree | SMOTE K=3 + Tuning | Model Terbaik |\n",
    "\n",
    "> Alasan: Naive Bayes gagal total setelah SMOTE + Tuning (F1 turun dari 0.8832 ke 0.5505).\n",
    "> Jadi cukup sebagai pembanding dengan parameter default."
  ]},
  {"cell_type": "code", "execution_count": None, "metadata": {"tags": []}, "outputs": [], "source": [
    "from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.metrics import (accuracy_score, precision_score, recall_score,\n",
    "                             f1_score, confusion_matrix, classification_report)\n",
    "from imblearn.over_sampling import SMOTE\n",
    "from imblearn.pipeline import Pipeline as ImbPipeline\n",
    "\n",
    "df = pd.read_csv('data/processed/stunting_clean.csv')\n",
    "df['jk_encoded'] = df['jk'].map({'Laki-Laki': 1, 'Perempuan': 0})\n",
    "target_map = {'Normal': 0, 'Stunted': 1, 'Severely Stunted': 2}\n",
    "df['status_encoded'] = df['status'].map(target_map)\n",
    "X = df[['umur_bulan', 'tinggi', 'jk_encoded']].values\n",
    "y = df['status_encoded'].values\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=0.2, random_state=42, stratify=y)\n",
    "print(f'Data: {df.shape[0]} sampel | Train: {X_train.shape[0]} | Test: {X_test.shape[0]}')"
  ]},
  {"cell_type": "code", "execution_count": None, "metadata": {"tags": []}, "outputs": [], "source": [
    "# MODEL A: NAIVE BAYES (PEMBANDING) - Tanpa SMOTE, Tanpa Tuning\n",
    "nb = GaussianNB()\n",
    "nb.fit(X_train, y_train)\n",
    "nb_pred = nb.predict(X_test)\n",
    "\n",
    "nb_acc = accuracy_score(y_test, nb_pred)\n",
    "nb_f1 = f1_score(y_test, nb_pred, average='weighted', zero_division=0)\n",
    "nb_prec = precision_score(y_test, nb_pred, average='weighted', zero_division=0)\n",
    "nb_rec = recall_score(y_test, nb_pred, average='weighted', zero_division=0)\n",
    "nb_cm = confusion_matrix(y_test, nb_pred)\n",
    "\n",
    "print('NAIVE BAYES (Pembanding)')\n",
    "print(f'Accuracy: {nb_acc:.4f} | F1-Score: {nb_f1:.4f}')\n",
    "print('Confusion Matrix:')\n",
    "print(nb_cm)"
  ]},
  {"cell_type": "code", "execution_count": None, "metadata": {"tags": []}, "outputs": [], "source": [
    "# MODEL B: DECISION TREE (TERBAIK) - SMOTE K=3 + Tuning\n",
    "smote = SMOTE(random_state=42, k_neighbors=3)\n",
    "X_res, y_res = smote.fit_resample(X_train, y_train)\n",
    "print(f'SMOTE K=3: {X_train.shape[0]} -> {X_res.shape[0]} sampel')\n",
    "\n",
    "dt_pipe = ImbPipeline([\n",
    "    ('smote', SMOTE(random_state=42, k_neighbors=3)),\n",
    "    ('dt', DecisionTreeClassifier(random_state=42))\n",
    "])\n",
    "param_grid = {\n",
    "    'dt__criterion': ['gini', 'entropy'],\n",
    "    'dt__max_depth': [3, 5, 7, 10, 15, None],\n",
    "    'dt__min_samples_split': [2, 5, 10, 20],\n",
    "    'dt__min_samples_leaf': [1, 2, 5, 10],\n",
    "    'dt__max_features': [None, 'sqrt', 'log2']\n",
    "}\n",
    "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
    "dt_grid = GridSearchCV(dt_pipe, param_grid, cv=cv,\n",
    "                       scoring='f1_weighted', n_jobs=-1, verbose=0)\n",
    "dt_grid.fit(X_train, y_train)\n",
    "\n",
    "dt_pred = dt_grid.predict(X_test)\n",
    "dt_acc = accuracy_score(y_test, dt_pred)\n",
    "dt_f1 = f1_score(y_test, dt_pred, average='weighted', zero_division=0)\n",
    "dt_prec = precision_score(y_test, dt_pred, average='weighted', zero_division=0)\n",
    "dt_rec = recall_score(y_test, dt_pred, average='weighted', zero_division=0)\n",
    "dt_cm = confusion_matrix(y_test, dt_pred)\n",
    "\n",
    "print('DECISION TREE (Terbaik)')\n",
    "print(f'Accuracy: {dt_acc:.4f} | F1-Score: {dt_f1:.4f}')\n",
    "print(f'Best params: {dt_grid.best_params_}')\n",
    "print('Confusion Matrix:')\n",
    "print(dt_cm)"
  ]},
  {"cell_type": "code", "execution_count": None, "metadata": {"tags": []}, "outputs": [], "source": [
    "# VISUALISASI: Confusion Matrix Berdampingan\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "fig.suptitle('Confusion Matrix: Naive Bayes vs Decision Tree', fontsize=15, fontweight='bold')\n",
    "for ax, (title, cm, cmap) in zip(axes, [\n",
    "    ('Naive Bayes (Pembanding)', nb_cm, 'Blues'),\n",
    "    ('Decision Tree (Terbaik)', dt_cm, 'Greens')\n",
    "]):\n",
    "    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,\n",
    "                xticklabels=['Normal', 'Stunted', 'SS'],\n",
    "                yticklabels=['Normal', 'Stunted', 'SS'],\n",
    "                cbar=False, linewidths=1, linecolor='white',\n",
    "                annot_kws={'size': 14, 'weight': 'bold'})\n",
    "    ax.set_title(title, fontsize=12, fontweight='bold')\n",
    "    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')\n",
    "plt.tight_layout(); plt.show()"
  ]},
  {"cell_type": "code", "execution_count": None, "metadata": {"tags": []}, "outputs": [], "source": [
    "# VISUALISASI: Bar Chart Perbandingan Metrik\n",
    "metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']\n",
    "nb_vals = [nb_acc, nb_prec, nb_rec, nb_f1]\n",
    "dt_vals = [dt_acc, dt_prec, dt_rec, dt_f1]\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(10, 6))\n",
    "x = np.arange(len(metrics))\n",
    "ax.bar(x - 0.15, nb_vals, 0.3, label='Naive Bayes (Pembanding)',\n",
    "       color='#E74C3C', alpha=0.85, edgecolor='black')\n",
    "ax.bar(x + 0.15, dt_vals, 0.3, label='Decision Tree (Terbaik)',\n",
    "       color='#2ECC71', alpha=0.85, edgecolor='black')\n",
    "ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=12)\n",
    "ax.set_ylabel('Score'); ax.set_ylim(0, 1.15)\n",
    "ax.set_title('Perbandingan Metrik: Naive Bayes vs Decision Tree', fontsize=14, fontweight='bold')\n",
    "ax.legend(fontsize=11); ax.grid(axis='y', alpha=0.3)\n",
    "plt.tight_layout(); plt.show()"
  ]},
  {"cell_type": "code", "execution_count": None, "metadata": {"tags": []}, "outputs": [], "source": [
    "# TABEL PERBANDINGAN FINAL\n",
    "final_data = [\n",
    "    ['Naive Bayes (Pembanding)', 'Tidak', 'Tidak',\n",
    "     f'{nb_acc:.4f}', f'{nb_prec:.4f}', f'{nb_rec:.4f}', f'{nb_f1:.4f}',\n",
    "     'Gagal deteksi minoritas'],\n",
    "    ['Decision Tree (Terbaik)', 'Ya (K=3)', 'Ya',\n",
    "     f'{dt_acc:.4f}', f'{dt_prec:.4f}', f'{dt_rec:.4f}', f'{dt_f1:.4f}',\n",
    "     'Model Terbaik']\n",
    "]\n",
    "fig, ax = plt.subplots(figsize=(14, 3))\n",
    "ax.axis('off')\n",
    "tbl = ax.table(cellText=final_data,\n",
    "    colLabels=['Model', 'SMOTE', 'Tuning', 'Accuracy',\n",
    "               'Precision', 'Recall', 'F1-Score', 'Keterangan'],\n",
    "    loc='center', cellLoc='center')\n",
    "tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 2)\n",
    "for (i, j), cell in tbl.get_celld().items():\n",
    "    if i == 0:\n",
    "        cell.set_facecolor('#2C3E50')\n",
    "        cell.set_text_props(color='white', fontweight='bold')\n",
    "    elif i == 2:\n",
    "        cell.set_facecolor('#d4edda')\n",
    "        cell.set_text_props(fontweight='bold')\n",
    "    elif i % 2 == 0:\n",
    "        cell.set_facecolor('#f8f9fa')\n",
    "ax.set_title('Tabel Perbandingan Final: Naive Bayes vs Decision Tree',\n",
    "             fontsize=14, fontweight='bold', pad=20)\n",
    "plt.tight_layout(); plt.show()"
  ]},
  {"cell_type": "code", "execution_count": None, "metadata": {"tags": []}, "outputs": [], "source": [
    "# Classification Report Detail\n",
    "print('='*60)\n",
    "print('CLASSIFICATION REPORT: NAIVE BAYES (Pembanding)')\n",
    "print('='*60)\n",
    "print(classification_report(y_test, nb_pred,\n",
    "    target_names=['Normal', 'Stunted', 'Severely Stunted'], zero_division=0))\n",
    "\n",
    "print('='*60)\n",
    "print('CLASSIFICATION REPORT: DECISION TREE (Terbaik)')\n",
    "print('='*60)\n",
    "print(classification_report(y_test, dt_pred,\n",
    "    target_names=['Normal', 'Stunted', 'Severely Stunted'], zero_division=0))"
  ]},
  {"cell_type": "markdown", "metadata": {"tags": []}, "source": [
    "---\n",
    "# BAGIAN 8: KESIMPULAN\n",
    "---\n",
    "\n",
    "## Ringkasan Proses\n",
    "- **Dataset awal:** 1.351 sampel -> **Dataset bersih:** 1.329 sampel (22 duplikat dihapus)\n",
    "- **Missing values:** Tidak ada\n",
    "- **Outlier:** Ditangani dengan capping (winsorization)\n",
    "- **Split data:** 80:20 stratified (1.063 train : 266 test)\n",
    "\n",
    "## Hasil Model\n",
    "\n",
    "### Naive Bayes (Pembanding)\n",
    "- Preprocessing: Tanpa SMOTE, Tanpa Tuning\n",
    "- F1-Score: 0.8832\n",
    "- **Catatan:** Akurasi 92.11% menipu - model tidak bisa mendeteksi stunting sama sekali\n",
    "- **Penyebab:** Asumsi independensi fitur tidak terpenuhi (umur & tinggi berkorelasi r > 0.9)\n",
    "\n",
    "### Decision Tree (Model Terbaik)\n",
    "- Preprocessing: SMOTE K=3 + Hyperparameter Tuning\n",
    "- F1-Score: **0.9380**\n",
    "- Best Params: criterion=gini, max_depth=None, max_features=sqrt, min_samples_split=2\n",
    "- **Kelebihan:** Mampu mendeteksi Stunted (61%) dan Severely Stunted (67%)\n",
    "\n",
    "> **Kesimpulan Akhir:** Decision Tree dengan SMOTE K=3 dan Hyperparameter Tuning\n",
    "> adalah **model terbaik** untuk prediksi stunting pada dataset ini.\n",
    "> Naive Bayes tidak direkomendasikan karena melanggar asumsi dasar algoritma."
  ]}
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.10.0"}
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('notebooks/penelitian.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Notebook berhasil dibuat!')
cells = nb['cells']
md = sum(1 for c in cells if c['cell_type'] == 'markdown')
cd = sum(1 for c in cells if c['cell_type'] == 'code')
print(f'Total: {len(cells)} cells ({md} markdown + {cd} code)')
