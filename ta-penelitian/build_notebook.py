import json

nb = {
 "cells": [
  {"cell_type": "markdown", "metadata": {"tags": []}, "source": [
    "# Analisis Dataset Stunting\n",
    "\n",
    "**Proyek:** Prediksi Stunting pada Balita\n",
    "**Dataset:** stunting.csv\n",
    "**Deskripsi:** Dataset berisi data tinggi badan balita berdasarkan umur (bulan) dan jenis kelamin, dengan status stunting (Normal / Stunted / Severely Stunted).\n",
    "\n",
    "---\n",
    "## Tahapan Analisis\n",
    "1. **Setup & Data Loading** - Load dataset dan eksplorasi awal\n",
    "2. **Data Preprocessing** - Pembersihan data, fixing inkonsistensi\n",
    "3. **Handling Missing Values** - Deteksi dan penanganan nilai hilang\n",
    "4. **Outlier Detection & Handling** - Deteksi dan penanganan outlier\n",
    "5. **Export Clean Dataset** - Simpan dataset bersih dan ringkasan\n",
    "---"
  ]},
  {"cell_type": "markdown", "metadata": {"tags": []}, "source": [
    "---\n",
    "# BAGIAN 1: SETUP & DATA LOADING\n",
    "---"
  ]},
  {"cell_type": "code", "execution_count": None, "metadata": {"tags": []}, "outputs": [], "source": [
    "# ============================================================\n",
    "# BAGIAN 1: SETUP & DATA LOADING\n",
    "# Tujuan: Import library, load dataset, dan eksplorasi awal\n",
    "# ============================================================\n",
    "\n",
    "# 1.1 Import Library\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import os\n",
    "import warnings\n",
    "\n",
    "# Buat folder processed jika belum ada\n",
    "os.makedirs('../data/processed', exist_ok=True)\n",
    "\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# Konfigurasi visualisasi\n",
    "plt.rcParams['figure.figsize'] = (10, 6)\n",
    "plt.rcParams['font.size'] = 12\n",
    "sns.set_style('whitegrid')\n",
    "\n",
    "print(\"Library berhasil diimport.\")"
  ]},
  {"cell_type": "code", "execution_count": None, "metadata": {"tags": []}, "outputs": [], "source": [
    "# 1.2 Load Dataset\n",
    "df_raw = pd.read_csv('../data/raw/stunting.csv', sep=';')\n",
    "\n",
    "print(f\"Dataset berhasil dimuat!\")\n",
    "print(f\"Shape: {df_raw.shape[0]} baris, {df_raw.shape[1]} kolom\")"
  ]},
  {"cell_type": "code", "execution_count": None, "metadata": {"tags": []}, "outputs": [], "source": [
    "# 1.3 Eksplorasi Awal\n",
    "print(\"=== 5 Data Pertama ===\")\n",
    "display(df_raw.head())\n",
    "\n",
    "print(\"\\n=== Informasi Dataset ===\")\n",
    "df_raw.info()\n",
    "\n",
    "print(\"\\n=== Statistik Deskriptif ===\")\n",
    "display(df_raw.describe())"
  ]},
  {"cell_type": "code", "execution_count": None, "metadata": {"tags": []}, "outputs": [], "source": [
    "# 1.4 Distribusi Target (Status)\n",
    "print(\"=== Distribusi Status Stunting ===\")\n",
    "status_counts = df_raw['Status'].value_counts()\n",
    "print(status_counts)\n",
    "print(f\"\\nPersentase:\")\n",
    "print(df_raw['Status'].value_counts(normalize=True).mul(100).round(2))\n",
    "\n",
    "# Visualisasi\n",
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
    "\n",
    "status_counts.plot(kind='bar', ax=axes[0], color=['green', 'orange', 'red'])\n",
    "axes[0].set_title('Distribusi Status Stunting')\n",
    "axes[0].set_xlabel('Status')\n",
    "axes[0].set_ylabel('Jumlah')\n",
    "axes[0].tick_params(axis='x', rotation=0)\n",
    "\n",
    "status_counts.plot(kind='pie', ax=axes[1], autopct='%1.1f%%',\n",
    "                   colors=['green', 'orange', 'red'], startangle=90)\n",
    "axes[1].set_title('Proporsi Status Stunting')\n",
    "axes[1].set_ylabel('')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig('../data/processed/distribusi_status_awal.png', dpi=100, bbox_inches='tight')\n",
    "plt.show()\n",
    "print(\"\\nVisualisasi tersimpan: data/processed/distribusi_status_awal.png\")"
  ]},
  {"cell_type": "code", "execution_count": None, "metadata": {"tags": []}, "outputs": [], "source": [
    "# 1.5 Export CSV - Data Awal & Statistik\n",
    "# Simpan data mentah untuk referensi\n",
    "df_raw.to_csv('../data/processed/stunting_00_raw.csv', index=False)\n",
    "print(\"Raw data exported: data/processed/stunting_00_raw.csv\")\n",
    "\n",
    "# Simpan ringkasan statistik\n",
    "summary_stats = df_raw.describe(include='all').round(2)\n",
    "summary_stats.to_csv('../data/processed/stunting_00_summary_stats.csv')\n",
    "print(\"Summary stats exported: data/processed/stunting_00_summary_stats.csv\")\n",
    "\n",
    "# Simpan distribusi status\n",
    "dist_status = df_raw['Status'].value_counts().reset_index()\n",
    "dist_status.columns = ['Status', 'Jumlah']\n",
    "dist_status['Persentase'] = (dist_status['Jumlah'] / dist_status['Jumlah'].sum() * 100).round(2)\n",
    "dist_status.to_csv('../data/processed/stunting_00_distribusi_status.csv', index=False)\n",
    "print(\"Distribusi status exported: data/processed/stunting_00_distribusi_status.csv\")\n",
    "\n",
    "print(f\"\\nTotal {df_raw.shape[0]} sampel, {df_raw.shape[1]} kolom siap diproses.\")"
  ]},
  {"cell_type": "markdown", "metadata": {"tags": []}, "source": [
    "> **Kesimpulan Bagian 1:** Dataset berhasil dimuat. Terlihat ada 1.351 sampel dengan 4 kolom. Distribusi status menunjukkan ketidakseimbangan kelas (class imbalance) dengan mayoritas Normal.\n",
    "\n",
    "---\n",
    "*Bersambung ke Bagian 2: Data Preprocessing...*"
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

print('Notebook created successfully!')
md = sum(1 for c in nb['cells'] if c['cell_type'] == 'markdown')
cd = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
cells_count = len(nb['cells'])
print(f'Cells: {cells_count} total ({md} markdown + {cd} code)')
