#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
======================================================================
GENERATE LAPORAN PDF - Analisis Prediksi Stunting pada Balita
======================================================================
Menggabungkan hasil dari:
1. preprocessing.py  -> data/processed/stunting_clean.csv + ringkasan
2. model_comparison.py -> data/processed/model_comparison/
3. final_model.py     -> data/processed/final_model/

Menghasilkan laporan PDF komprehensif dengan visualisasi dan analisis.
======================================================================
"""

import os, sys, warnings, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from datetime import datetime

# ── Setup ──
os.makedirs('reports', exist_ok=True)
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams.update({'figure.figsize': (10, 6), 'font.size': 12})

# ── Load all results ──
CLEAN = 'data/processed/stunting_clean.csv'
MC_CSV = 'data/processed/model_comparison/hasil_perbandingan_8_algoritma.csv'
FM_CSV = 'data/processed/final_model/hasil_final.csv'
NB_DETAIL = 'data/processed/final_model/detail_naive_bayes.csv'
DT_DETAIL = 'data/processed/final_model/detail_decision_tree.csv'

df_clean = pd.read_csv(CLEAN) if os.path.exists(CLEAN) else None
df_mc = pd.read_csv(MC_CSV) if os.path.exists(MC_CSV) else None
df_final = pd.read_csv(FM_CSV) if os.path.exists(FM_CSV) else None
df_nb = pd.read_csv(NB_DETAIL) if os.path.exists(NB_DETAIL) else None
df_dt = pd.read_csv(DT_DETAIL) if os.path.exists(DT_DETAIL) else None

# ── Tanggal Indonesia ──
bln_names = ['Januari','Februari','Maret','April','Mei','Juni',
             'Juli','Agustus','September','Oktober','November','Desember']
now = datetime.now()
tgl_str = f'{now.day} {bln_names[now.month-1]} {now.year}'

# ── Image helper ──
def img_exists(p):
    return os.path.exists(p)

# ═══════════════════════════════════════════════════════════════
# PDF CLASS
# ═══════════════════════════════════════════════════════════════
class LaporanPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(100, 100, 100)
            self.cell(0, 8, 'Laporan Analisis Prediksi Stunting pada Balita', align='L')
            self.cell(0, 8, f'Halaman {self.page_no()}', align='R', new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(25, 60, 120)
            self.line(10, 16, 200, 16)
            self.ln(5)

    def footer(self):
        if self.page_no() > 1:
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 7)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, f'Dokumen penelitian - {tgl_str}', align='C')

    def chapter_title(self, title, level=1):
        if level == 1:
            self.set_font('Helvetica', 'B', 16)
            self.set_text_color(25, 60, 120)
            self.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(25, 60, 120)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(6)
        elif level == 2:
            self.set_font('Helvetica', 'B', 13)
            self.set_text_color(40, 90, 160)
            self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
            self.ln(3)
        elif level == 3:
            self.set_font('Helvetica', 'B', 11)
            self.set_text_color(60, 60, 60)
            self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
            self.ln(2)

    def text(self, t):
        self.set_x(self.l_margin)
        self.set_font('Helvetica', '', 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 6, t)
        self.ln(2)

    def bullet(self, t, bold=''):
        self.set_x(self.l_margin)
        self.set_font('Helvetica', '', 10)
        self.set_text_color(40, 40, 40)
        prefix = f'- {bold}' if bold else '- '
        self.multi_cell(0, 6, prefix + t, new_x="LMARGIN", new_y="NEXT")

    def bold_text(self, label, value):
        """e.g. bold_text('F1-Score:', '0.9546')"""
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(40, 40, 40)
        w = self.get_string_width(label) + 2
        self.cell(w, 7, label)
        self.set_font('Helvetica', '', 10)
        self.multi_cell(0, 7, value)

    def insert_image(self, path, w=170):
        if img_exists(path):
            self.image(path, x=20, w=w)
            self.ln(4)
        else:
            self.set_font('Helvetica', 'I', 9)
            self.set_text_color(150, 150, 150)
            self.cell(0, 6, f'[Gambar: {os.path.basename(path)} tidak ditemukan]',
                      new_x="LMARGIN", new_y="NEXT")
            self.ln(2)

    def mini_img(self, path, w=85):
        """Insert two mini images side by side."""
        if img_exists(path):
            self.image(path, x=15, w=w)
        else:
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(150, 150, 150)
            self.cell(85, 5, f'[Missing: {os.path.basename(path)}]')

    def table(self, headers, data, widths=None):
        if widths is None:
            widths = [190 / len(headers)] * len(headers)
        # Header row
        self.set_font('Helvetica', 'B', 9)
        self.set_fill_color(25, 60, 120)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(widths[i], 8, str(h), border=1, fill=True, align='C')
        self.ln()
        # Data rows
        for ri, row in enumerate(data):
            is_best = (ri == 0)
            if is_best:
                self.set_fill_color(212, 237, 218)
                self.set_text_color(0, 80, 0)
                self.set_font('Helvetica', 'B', 9)
            elif ri % 2:
                self.set_fill_color(245, 245, 245)
                self.set_text_color(40, 40, 40)
                self.set_font('Helvetica', '', 9)
            else:
                self.set_fill_color(255, 255, 255)
                self.set_text_color(40, 40, 40)
                self.set_font('Helvetica', '', 9)
            for i, v in enumerate(row):
                self.cell(widths[i], 7, str(v), border=1, fill=True, align='C')
            self.ln()

    def verdict_box(self, text, color='green'):
        colors = {
            'green': (0, 100, 0),
            'blue': (0, 60, 120),
            'orange': (180, 100, 0),
            'red': (180, 0, 0)
        }
        c = colors.get(color, colors['blue'])
        self.set_fill_color(c[0], c[1], c[2])
        self.set_text_color(255, 255, 255)
        self.set_font('Helvetica', 'B', 10)
        self.cell(0, 8, f'  {text}', border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(40, 40, 40)
        self.ln(3)



# ═══════════════════════════════════════════════════════════════
# BUILD PDF
# ═══════════════════════════════════════════════════════════════
pdf = LaporanPDF('P', 'mm', 'A4')
pdf.set_auto_page_break(auto=True, margin=20)

# ═══════════════════════════════════════════════════════════════
# COVER PAGE
# ═══════════════════════════════════════════════════════════════
pdf.add_page()
pdf.ln(40)
pdf.set_font('Helvetica', 'B', 28)
pdf.set_text_color(25, 60, 120)
pdf.cell(0, 15, 'LAPORAN ANALISIS', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 15, 'PREDIKSI STUNTING PADA BALITA', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(8)
pdf.set_draw_color(25, 60, 120)
pdf.set_line_width(0.8)
pdf.line(40, pdf.get_y(), 170, pdf.get_y())
pdf.ln(10)
pdf.set_font('Helvetica', '', 13)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 8, 'Menggunakan 8 Algoritma Klasifikasi', align='C', new_x="LMARGIN", new_y="NEXT")
if df_clean is not None:
    pdf.cell(0, 8, f'{len(df_clean)} Sampel | 4 Variabel | 3 Kelas Status', align='C',
             new_x="LMARGIN", new_y="NEXT")
pdf.ln(6)
pdf.cell(0, 8, 'Naive Bayes (Pembanding) vs Decision Tree (Terbaik)', align='C',
         new_x="LMARGIN", new_y="NEXT")
pdf.ln(25)
pdf.set_font('Helvetica', '', 10)
pdf.set_text_color(100, 100, 100)
pdf.cell(0, 8, f'Diproduksi: {tgl_str}', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 8, 'Tugas Akhir - Program Studi ...', align='C', new_x="LMARGIN", new_y="NEXT")

# ═══════════════════════════════════════════════════════════════
# BAB 1: PENDAHULUAN
# ═══════════════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title('BAB 1: PENDAHULUAN')

pdf.chapter_title('1.1 Latar Belakang', 2)
pdf.text(
    'Stunting adalah kondisi gagal tumbuh pada balita akibat kekurangan gizi kronis, '
    'yang ditandai dengan tinggi badan di bawah standar seusianya. Deteksi dini stunting '
    'sangat penting untuk mencegah dampak jangka panjang pada perkembangan kognitif dan '
    'produktivitas anak. Penelitian ini menggunakan pendekatan machine learning untuk '
    'membangun model prediktif yang dapat mengklasifikasikan status stunting balita '
    'berdasarkan data antropometri (umur, tinggi, dan jenis kelamin).')

pdf.chapter_title('1.2 Tujuan Penelitian', 2)
pdf.bullet('Melakukan preprocessing pada dataset stunting (cleaning, handling missing values, outlier).')
pdf.bullet('Membandingkan performa 8 algoritma klasifikasi untuk prediksi stunting.')
pdf.bullet('Menentukan algoritma terbaik berdasarkan metrik evaluasi (F1-Score, Precision, Recall).')
pdf.bullet('Menganalisis Naive Bayes (pembanding) vs Decision Tree (terbaik) secara mendalam.')

pdf.chapter_title('1.3 Dataset', 2)
pdf.text(
    'Dataset stunting.csv berisi data antropometri balita dengan 4 variabel: '
    'umur_bulan (usia dalam bulan), tinggi (tinggi badan dalam cm), jk (jenis kelamin), '
    'dan status (target: Normal, Stunted, Severely Stunted). Total 1.351 sampel awal '
    'yang setelah preprocessing menjadi 1.329 sampel bersih.')

# ═══════════════════════════════════════════════════════════════
# BAB 2: DATA PREPROCESSING
# ═══════════════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title('BAB 2: DATA PREPROCESSING')

pdf.chapter_title('2.1 Struktur Dataset Awal', 2)
pdf.table(
    ['Variabel', 'Tipe Data', 'Deskripsi'],
    [
        ['umur_bulan', 'Float64', 'Umur balita dalam bulan (0-60 bln)'],
        ['tinggi', 'Float64', 'Tinggi badan dalam cm'],
        ['jk', 'Object', 'Jenis Kelamin (L/P)'],
        ['status', 'Object', 'Status Stunting (TARGET): Normal, Stunted, Severely Stunted'],
    ],
    [35, 25, 120]
)

pdf.chapter_title('2.2 Distribusi Target Awal', 2)
if df_clean is not None:
    sc = df_clean['status'].value_counts()
    n = len(df_clean)
    pdf.table(
        ['Status', 'Jumlah', 'Persentase'],
        [[s, str(c), f'{c/n*100:.2f}%'] for s, c in sc.items()],
        [70, 50, 70]
    )
pdf.text(
    'Terlihat ketidakseimbangan kelas (class imbalance) yang signifikan: '
    'Normal mendominasi dengan 92.25%, sementara Stunted hanya 6.62% dan '
    'Severely Stunted 1.13%. Ini menjadi tantangan utama dalam pemodelan '
    'karena model cenderung bias terhadap kelas mayoritas.')
pdf.insert_image('data/processed/distribusi_status_awal.png')

pdf.chapter_title('2.3 Tahapan Preprocessing', 2)
pdf.chapter_title('Standardisasi Kolom', 3)
pdf.bullet('Nama kolom distandarisasi ke lowercase: Tinggi -> tinggi, JK -> jk, Status -> status.')
pdf.bullet('Status: spasi dihapus dan kapitalisasi distandarisasi dengan str.title().')
pdf.bullet('JK: L/P dipetakan ke Laki-Laki/Perempuan menggunakan dictionary mapping.')

pdf.chapter_title('Penghapusan Duplikat', 3)
pdf.text('22 baris duplikat ditemukan dan dihapus. Dataset berkurang dari 1.351 menjadi 1.329 sampel.')

pdf.chapter_title('Handling Missing Values', 3)
pdf.text('Tidak ditemukan missing values dalam dataset. Namun handler telah disiapkan '
         '(median untuk numerik, modus untuk kategorikal) untuk mengantisipasi data baru.')

pdf.chapter_title('Outlier Detection & Handling', 3)
pdf.text('Outlier dideteksi menggunakan 2 metode:')
pdf.bullet('IQR (Interquartile Range): 1 outlier pada kolom tinggi')
pdf.bullet('Z-Score (threshold |Z| > 3): 2 outlier pada kolom tinggi')
pdf.text('Penanganan: Capping/Winsorization - mengganti nilai outlier dengan batas IQR '
         '(lower/upper bound). Pendekatan ini lebih baik daripada menghapus data karena '
         'mempertahankan jumlah sampel minoritas yang sangat terbatas.')
pdf.insert_image('data/processed/perbandingan_outlier.png')

pdf.insert_image('data/processed/scatter_umur_tinggi.png')
pdf.text(
    'Scatter plot menunjukkan hubungan positif yang kuat antara umur dan tinggi badan. '
    'Titik berwarna merah (Severely Stunted) dan oranye (Stunted) umumnya berada di '
    'bawah klaster hijau (Normal), mengkonfirmasi bahwa tinggi badan di bawah standar '
    'pada umur tertentu adalah indikator utama stunting.')

# ═══════════════════════════════════════════════════════════════
# BAB 3: PERBANDINGAN 8 ALGORITMA
# ═══════════════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title('BAB 3: PERBANDINGAN 8 ALGORITMA KLASIFIKASI')

pdf.chapter_title('3.1 Metodologi', 2)
pdf.bullet('Split data 80:20 dengan stratified sampling (menjaga proporsi kelas)')
pdf.bullet('Standard Scaling untuk model sensitif skala (LR, SVM, KNN)')
pdf.bullet('SMOTE K=3 untuk menyeimbangkan data latih')
pdf.bullet('Evaluasi: Accuracy, Precision, Recall, F1-Score (weighted), ROC AUC')

pdf.text('8 algoritma yang dibandingkan:')
algos = [
    'Logistic Regression - Model linear untuk klasifikasi multikelas',
    'Naive Bayes - Teorema Bayes dengan asumsi independensi fitur',
    'Decision Tree - Struktur pohon keputusan berbasis aturan',
    'Random Forest - Ensemble bagging dari banyak Decision Trees',
    'SVM (RBF Kernel) - Mencari hyperplane pemisah optimal',
    'K-Nearest Neighbors - Klasifikasi berdasarkan kemiripan tetangga',
    'XGBoost - Gradient Boosting optimized dengan regularisasi',
    'Gradient Boosting - Ensemble sequential koreksi kesalahan',
]
for a in algos:
    pdf.bullet(a)

pdf.chapter_title('3.2 Hasil Lengkap Perbandingan', 2)
if df_mc is not None:
    # Filter top 2 per SMOTE scenario
    df_no = df_mc[df_mc['SMOTE'] == 'Tidak'].sort_values('F1-Score', ascending=False)
    df_yes = df_mc[df_mc['SMOTE'] == 'Ya (K=3)'].sort_values('F1-Score', ascending=False)

    pdf.chapter_title('Skenario A: Tanpa SMOTE (Data Imbalance Asli)', 3)
    rows_no = [[str(i+1), r['Model'], f"{r['Accuracy']:.4f}",
                f"{r['Precision']:.4f}", f"{r['Recall']:.4f}", f"{r['F1-Score']:.4f}"]
               for i, (_, r) in enumerate(df_no.iterrows())]
    pdf.table(['Rank', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
              rows_no, [12, 55, 30, 31, 31, 31])

    pdf.ln(3)
    pdf.chapter_title('Skenario B: Dengan SMOTE K=3 (Data Seimbang)', 3)
    rows_yes = [[str(i+1), r['Model'], f"{r['Accuracy']:.4f}",
                 f"{r['Precision']:.4f}", f"{r['Recall']:.4f}", f"{r['F1-Score']:.4f}"]
                for i, (_, r) in enumerate(df_yes.iterrows())]
    pdf.table(['Rank', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
              rows_yes, [12, 55, 30, 31, 31, 31])

pdf.chapter_title('3.3 Visualisasi Perbandingan', 2)
pdf.insert_image('data/processed/model_comparison/perbandingan_f1_8_algoritma.png')
pdf.text(
    'Grafik batang di atas membandingkan F1-Score 8 algoritma dalam dua skenario. '
    'Terlihat bahwa ensemble methods (Random Forest, Gradient Boosting) mendominasi '
    'di kedua skenario. Naive Bayes menunjukkan performa yang sangat buruk setelah '
    'SMOTE karena asumsi independensi fitur tidak terpenuhi.')

pdf.insert_image('data/processed/model_comparison/heatmap_metrik_8_algoritma.png')
pdf.text(
    'Heatmap metrik memperkuat temuan: Random Forest dan Gradient Boosting konsisten '
    'unggul di semua metrik. SVM dan Logistic Regression menunjukkan performa menengah. '
    'Naive Bayes memiliki warna paling gelap (skor rendah).')

pdf.insert_image('data/processed/model_comparison/confusion_matrix_top4.png')
pdf.text(
    'Confusion matrix 4 model terbaik (dengan SMOTE): Random Forest, Gradient Boosting, '
    'Decision Tree, dan KNN semuanya mampu mendeteksi kelas Stunted dan Severely Stunted '
    'dengan baik. Jumlah false positive untuk kelas Normal minimal.')

pdf.insert_image('data/processed/model_comparison/smote_effect_scatter.png')
pdf.text(
    'Scatter plot ini menunjukkan efek SMOTE K=3 pada setiap algoritma. '
    'Titik di atas garis diagonal berarti SMOTE meningkatkan performa. '
    'Terlihat bahwa KNN, SVM, dan Decision Tree mendapat manfaat terbesar dari SMOTE. '
    'Naive Bayes justru mengalami penurunan drastis. Random Forest dan XGBoost '
    'relatif stabil di kedua skenario.')

pdf.insert_image('data/processed/model_comparison/ranking_table_8_algoritma.png')

# ═══════════════════════════════════════════════════════════════
# BAB 4: ANALISIS DETAIL PER ALGORITMA
# ═══════════════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title('BAB 4: ANALISIS DETAIL PER ALGORITMA')

# --- Random Forest ---
pdf.chapter_title('4.1 Random Forest (Terbaik) - F1: 0.9546', 2)
pdf.text(
    'Random Forest adalah ensemble method yang menggabungkan banyak Decision Trees '
    'melalui teknik bagging (Bootstrap Aggregating). Setiap tree dilatih pada subset '
    'data acak, dan hasil prediksi ditentukan melalui voting mayoritas.')
pdf.bold_text('Mengapa Random Forest Terbaik:', '')
pdf.bullet('Mengurangi overfitting dengan merata-rata banyak trees.')
pdf.bullet('Menangkap hubungan non-linear antara umur, tinggi, dan status stunting.')
pdf.bullet('Memberikan feature importance untuk interpretasi.')
pdf.bullet('Paling stabil: F1-Score tinggi di kedua skenario (dengan/tanpa SMOTE).')
pdf.verdict_box('VERDIK: SANGAT DIREKOMENDASIKAN - Model utama untuk screening stunting', 'green')

# --- Gradient Boosting ---
pdf.chapter_title('4.2 Gradient Boosting - F1: 0.9533', 2)
pdf.text(
    'Gradient Boosting adalah ensemble sequential yang membangun trees secara bertahap. '
    'Setiap tree baru berusaha memperbaiki kesalahan prediksi tree sebelumnya. '
    'Berbeda dengan Random Forest yang melatih trees secara paralel.')
pdf.bold_text('Mengapa Gradient Boosting Unggul:', '')
pdf.bullet('Performa hampir setara Random Forest (selisih hanya 0.0013 F1).')
pdf.bullet('Cocok jika prioritas utama adalah akurasi tanpa perlu bagging.')
pdf.bullet('Lebih sensitif terhadap hyperparameter tuning dibanding Random Forest.')
pdf.verdict_box('VERDIK: SANGAT BAIK - Alternatif utama jika Random Forest tidak tersedia', 'green')

# --- Decision Tree ---
pdf.chapter_title('4.3 Decision Tree - F1: 0.9483', 2)
pdf.text(
    'Decision Tree adalah model berbasis aturan yang membagi data secara rekursif '
    'berdasarkan threshold fitur. Hasilnya adalah struktur pohon yang mudah '
    'diinterpretasikan oleh manusia.')
pdf.bold_text('Mengapa Decision Tree Cocok:', '')
pdf.bullet('Aturan keputusan mudah dipahami: "Jika tinggi < X cm pada umur Y bulan -> Stunted".')
pdf.bullet('Threshold alami pada data stunting (hubungan tinggi vs umur) sangat cocok dengan Decision Tree.')
pdf.bullet('Tidak sensitif terhadap korelasi fitur (berbeda dengan Naive Bayes).')
pdf.bullet('Paling mudah dijelaskan ke tenaga kesehatan non-teknis.')
pdf.verdict_box('VERDIK: TERBAIK UNTUK INTERPRETABILITAS - Ideal untuk aplikasi klinis', 'blue')

# --- XGBoost ---
pdf.chapter_title('4.4 XGBoost - F1: 0.9425 (Tanpa SMOTE: 0.9551)', 2)
pdf.text(
    'XGBoost adalah versi optimized dari Gradient Boosting dengan tambahan regularisasi '
    'untuk mengurangi overfitting. Menjadi state-of-the-art untuk dataset tabular.')
pdf.bold_text('Analisis:', '')
pdf.bullet('Terbaik TANPA SMOTE (F1: 0.9551) - unggul pada data imbalance asli.')
pdf.bullet('Regularisasi membuatnya lebih robust terhadap noise.')
pdf.bullet('Performa sedikit menurun DENGAN SMOTE (F1: 0.9425) karena data sintetis.')
pdf.bullet('Unggul signifikan di dataset > 10.000 sampel - potensi lebih besar dengan data tambahan.')
pdf.verdict_box('VERDIK: BAIK - Ideal jika tidak menggunakan SMOTE', 'blue')

# --- KNN ---
pdf.chapter_title('4.5 K-Nearest Neighbors - F1: 0.9493', 2)
pdf.text(
    'KNN (K-Nearest Neighbors) mengklasifikasikan data berdasarkan label mayoritas dari '
    'K tetangga terdekat dalam ruang fitur. Semakin kecil K, semakin fleksibel boundary.')
pdf.bold_text('Analisis:', '')
pdf.bullet('Peningkatan besar setelah SMOTE (+0.0134) - data sintetis membuat cluster lebih rapat.')
pdf.bullet('Prinsip KNN sangat relevan: balita dengan stunting cenderung berkelompok.')
pdf.bullet('Sensitif terhadap skala fitur - StandardScaler penting digunakan.')
pdf.bullet('Kurang efektif jika data sangat jarang di beberapa region umur.')
pdf.verdict_box('VERDIK: BAIK - Alternatif solid dengan SMOTE', 'blue')

# --- SVM ---
pdf.chapter_title('4.6 Support Vector Machine - F1: 0.9265', 2)
pdf.text(
    'SVM dengan kernel RBF mencari hyperplane non-linear terbaik yang memisahkan kelas '
    'dengan margin maksimal. Cocok untuk dataset dengan dimensi rendah hingga menengah.')
pdf.bold_text('Analisis:', '')
pdf.bullet('Performa meningkat setelah SMOTE (+0.0433) - data seimbang membantu margin SVM.')
pdf.bullet('ROC AUC tertinggi (0.9962) - sangat baik dalam membedakan kelas.')
pdf.bullet('Hyperparameter tuning (C, gamma) bisa meningkatkan F1-Score lebih lanjut.')
pdf.bullet('Kernel RBF mampu menangkap boundary non-linear umur vs tinggi.')
pdf.verdict_box('VERDIK: CUKUP BAIK - Potensi lebih tinggi dengan tuning lanjutan', 'orange')

# --- Logistic Regression ---
pdf.chapter_title('4.7 Logistic Regression - F1: 0.8898', 2)
pdf.text(
    'Logistic Regression memodelkan probabilitas kelas menggunakan fungsi sigmoid. '
    'Model linear yang sederhana, cepat, dan mudah diinterpretasi.')
pdf.bold_text('Analisis:', '')
pdf.bullet('Performa menengah - asumsi linearitas tidak sepenuhnya terpenuhi.')
pdf.bullet('Hubungan umur/tinggi dengan stunting bersifat non-linear.')
pdf.bullet('Menurun drastis setelah SMOTE (F1: 0.8713) - data sintetis mengganggu linearitas.')
pdf.bullet('Cocok sebagai baseline model.')
pdf.verdict_box('VERDIK: KURANG OPTIMAL - Terlalu sederhana untuk data stunting', 'orange')

# --- Naive Bayes ---
pdf.chapter_title('4.8 Naive Bayes - F1: 0.5505 (PALING BURUK)', 2)
pdf.text(
    'Naive Bayes menggunakan teorema Bayes dengan asumsi kuat bahwa semua fitur '
    'independen satu sama lain. Meskipun sederhana dan cepat, asumsi ini jarang '
    'terpenuhi dalam data dunia nyata.')
pdf.bold_text('Mengapa Naive Bayes Gagal Total:', '')
pdf.bullet('Asumsi independensi fitur DILANGGAR: umur_bulan dan tinggi berkorelasi sangat kuat (r > 0.9).')
pdf.bullet('Tanpa SMOTE: Akurasi tinggi (92.11%) tapi TIDAK BISA mendeteksi stunting sama sekali!')
pdf.bullet('Ini adalah contoh klasik Accuracy Paradox: akurasi tinggi tapi model tidak berguna.')
pdf.bullet('Confusion matrix: semua sampel diprediksi Normal - Precision=Recall=0 untuk Stunted & SS.')
pdf.bullet('Setelah SMOTE: Performa ambruk (F1: 0.5505) karena data sintetis memperparah pelanggaran asumsi.')
pdf.bullet('Distribusi data sintetis dari SMOTE tidak sesuai dengan asumsi Gaussian NB.')
pdf.verdict_box('VERDIK: TIDAK DIREKOMENDASIKAN - Sama sekali tidak cocok untuk data antropometri', 'red')

# SMOTE effect summary table
pdf.chapter_title('4.9 Efektivitas SMOTE K=3 pada Setiap Algoritma', 2)
if df_mc is not None:
    smote_data = []
    for model_name in df_mc['Model'].unique():
        no_row = df_mc[(df_mc['Model'] == model_name) & (df_mc['SMOTE'] == 'Tidak')]
        yes_row = df_mc[(df_mc['Model'] == model_name) & (df_mc['SMOTE'] == 'Ya (K=3)')]
        if len(no_row) > 0 and len(yes_row) > 0:
            f1_no = no_row['F1-Score'].values[0]
            f1_yes = yes_row['F1-Score'].values[0]
            delta = f1_yes - f1_no
            arrow = '+' if delta > 0 else '-'
            smote_data.append([model_name, f'{f1_no:.4f}', f'{f1_yes:.4f}',
                               f'{delta:+.4f}', f'{arrow}'])
    smote_data.sort(key=lambda x: float(x[3]), reverse=True)
    pdf.table(['Model', 'F1 Tanpa SMOTE', 'F1 Dengan SMOTE', 'Delta', 'Efek'],
              smote_data, [55, 45, 45, 35, 10])
    pdf.text(
        'Kesimpulan: SMOTE efektif untuk Decision Tree, KNN, dan SVM. '
        'Random Forest dan XGBoost relatif stabil. '
        'Naive Bayes dan Logistic Regression justru menurun drastis.')

# ═══════════════════════════════════════════════════════════════
# BAB 5: FINAL MODEL - NB vs DT
# ═══════════════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title('BAB 5: FINAL MODEL - NAIVE BAYES vs DECISION TREE')

pdf.chapter_title('5.1 Konfigurasi Model', 2)
pdf.table(
    ['Aspek', 'Naive Bayes (Pembanding)', 'Decision Tree (Terbaik)'],
    [
        ['SMOTE', 'Tidak', 'Ya (K=3)'],
        ['Hyperparameter Tuning', 'Tidak', 'Ya (GridSearchCV)'],
        ['Parameter Grid', '-', '576 kombinasi'],
        ['CV Fold', '-', '5-Fold Stratified'],
        ['Scoring', '-', 'F1-Weighted'],
    ],
    [45, 72, 72]
)

pdf.chapter_title('5.2 Hasil Hyperparameter Tuning Decision Tree', 2)
pdf.text('GridSearchCV dengan 576 kombinasi parameter menghasilkan konfigurasi optimal:')
pdf.bullet('criterion: gini (sedikit lebih baik dari entropy)')
pdf.bullet('max_depth: None (tanpa batasan - pohon tumbuh penuh)')
pdf.bullet('max_features: sqrt (mengurangi overfitting)')
pdf.bullet('min_samples_leaf: 1 (leaf node minimal 1 sampel)')
pdf.bullet('min_samples_split: 2 (split minimal 2 sampel)')
pdf.insert_image('data/processed/final_model/dt_tuning_top_params.png')
pdf.text(
    'Grafik di atas menunjukkan 15 kombinasi parameter terbaik. '
    'Parameter max_depth=None terpilih karena data stunting membutuhkan '
    'pohon yang cukup dalam untuk menangkap pola pertumbuhan balita. '
    'max_features=sqrt membantu mengurangi overfitting.')

pdf.chapter_title('5.3 Perbandingan Confusion Matrix', 2)
pdf.insert_image('data/processed/final_model/confusion_matrix_comparison.png')
pdf.text(
    'Confusion matrix menunjukkan perbedaan sangat mencolok:\n'
    '- Naive Bayes (kiri): Semua sampel diprediksi Normal. Tidak ada satupun '
    'sampel Stunted atau Severely Stunted yang terdeteksi. Model ini TIDAK BERGUNA '
    'untuk deteksi stunting meskipun akurasi 92%.\n'
    '- Decision Tree (kanan): 11 dari 18 Stunted terdeteksi (61.1% recall) dan '
    '2 dari 3 Severely Stunted terdeteksi (66.7% recall). False positive minimal.')

pdf.chapter_title('5.4 Perbandingan Metrik', 2)
pdf.insert_image('data/processed/final_model/metric_comparison.png')
pdf.text(
    'Grafik menunjukkan fenomena Accuracy Paradox: Naive Bayes memiliki akurasi '
    '92% namun Precision, Recall, dan F1-Score untuk kelas minoritas = 0. '
    'Decision Tree unggul di semua metrik dengan F1-Score 0.9380.')

pdf.chapter_title('5.5 Efek SMOTE K=3', 2)
pdf.insert_image('data/processed/final_model/smote_effect.png')
pdf.text(
    'SMOTE K=3 berhasil menyeimbangkan data latih: dari 981 Normal, 70 Stunted, '
    '12 Severely Stunted menjadi 981 sampel per kelas (total 2.943). '
    'Data sintetis Stunted dan Severely Stunted dibuat dengan interpolasi K=3 '
    'tetangga terdekat. Ini memungkinkan Decision Tree mempelajari pola kelas minoritas.')

pdf.chapter_title('5.6 Tabel Perbandingan Final', 2)
pdf.insert_image('data/processed/final_model/final_comparison_table.png')

pdf.chapter_title('5.7 Classification Report Detail', 2)

if df_nb is not None:
    pdf.chapter_title('Naive Bayes (Pembanding)', 3)
    rows_nb = [[r['Kelas'], f"{r['Precision']:.4f}", f"{r['Recall']:.4f}",
                f"{r['F1-Score']:.4f}", str(r['Support'])]
               for _, r in df_nb.iterrows()]
    pdf.table(['Kelas', 'Precision', 'Recall', 'F1-Score', 'Support'], rows_nb,
              [50, 35, 35, 35, 35])
    pdf.text(
        'Perhatikan: Precision=Recall=F1=0 untuk Stunted dan Severely Stunted. '
        'Model hanya memprediksi semua sampel sebagai Normal.')

if df_dt is not None:
    pdf.ln(3)
    pdf.chapter_title('Decision Tree (Terbaik)', 3)
    rows_dt = [[r['Kelas'], f"{r['Precision']:.4f}", f"{r['Recall']:.4f}",
                f"{r['F1-Score']:.4f}", str(r['Support'])]
               for _, r in df_dt.iterrows()]
    pdf.table(['Kelas', 'Precision', 'Recall', 'F1-Score', 'Support'], rows_dt,
              [50, 35, 35, 35, 35])
    pdf.text(
        'Decision Tree menunjukkan keseimbangan yang baik:\n'
        '- Normal: Precision 0.97, Recall 0.96 - sangat akurat.\n'
        '- Stunted: Recall 0.61 - 61% terdeteksi, perlu peningkatan data.\n'
        '- Severely Stunted: Precision 1.0, Recall 0.67 - semua prediksi SS benar.')

pdf.insert_image('data/processed/final_model/radar_chart.png')
pdf.text(
    'Radar chart memberikan visualisasi holistik perbandingan model. '
    'Decision Tree (hijau) mengisi area yang jauh lebih luas dibandingkan '
    'Naive Bayes (merah), terutama pada dimensi Recall dan Precision. '
    'Ini mengkonfirmasi superioritas Decision Tree secara multidimensi.')

# ═══════════════════════════════════════════════════════════════
# BAB 6: KESIMPULAN & REKOMENDASI
# ═══════════════════════════════════════════════════════════════
pdf.add_page()
pdf.chapter_title('BAB 6: KESIMPULAN DAN REKOMENDASI')

pdf.chapter_title('6.1 Kesimpulan', 2)

pdf.text('Berdasarkan analisis komprehensif yang telah dilakukan, berikut kesimpulan utama:')

pdf.chapter_title('Temuan Data', 3)
pdf.bullet('Dataset bersih: 1.329 sampel (22 duplikat dihapus, 1.351 -> 1.329).')
pdf.bullet('Tidak ada missing values - kualitas data baik.')
pdf.bullet('1 outlier pada kolom tinggi ditangani dengan Capping tanpa menghapus data.')
pdf.bullet('Distribusi target sangat timpang: Normal 92.2%, Stunted 6.6%, Severely Stunted 1.1%.')

pdf.chapter_title('Temuan Model', 3)
pdf.bullet('Random Forest adalah model terbaik (F1-Score: 0.9546) - stabil di kedua skenario.')
pdf.bullet('Decision Tree dengan SMOTE K=3 + Tuning unggul untuk interpretability (F1: 0.9380).')
pdf.bullet('Naive Bayes gagal total - tidak dapat mendeteksi stunting sama sekali.')
pdf.bullet('SMOTE efektif untuk Decision Tree, KNN, SVM - tidak efektif untuk Naive Bayes.')

pdf.chapter_title('Fenomena Penting', 3)
pdf.bullet('Accuracy Paradox: Naive Bayes akurasi 92% tapi recall Stunted = 0%.')
pdf.bullet('Korelasi fitur umur_bulan dan tinggi (r > 0.9) membuat Naive Bayes tidak cocok.')
pdf.bullet('Ensemble methods (RF, GB, XGB) paling robust terhadap imbalance data.')

pdf.chapter_title('6.2 Rekomendasi Implementasi', 2)
pdf.text('Berdasarkan hasil penelitian, berikut rekomendasi implementasi:')

pdf.chapter_title('Rekomendasi Model', 3)
pdf.bullet('Gunakan Random Forest sebagai primary classifier untuk screening stunting.',
           bold='1. ')
pdf.bullet('Gunakan Decision Tree sebagai model interpretable untuk penjelasan ke tenaga kesehatan.',
           bold='2. ')
pdf.bullet('Implementasi SMOTE K=3 untuk menyeimbangkan data sebelum training.',
           bold='3. ')
pdf.bullet('Evaluasi model secara berkala dengan data baru untuk validasi.',
           bold='4. ')

pdf.chapter_title('Rekomendasi Data', 3)
pdf.bullet('Tambah data kelas Severely Stunted (saat ini hanya 15 sampel).',
           bold='1. ')
pdf.bullet('Kumpulkan fitur tambahan: status gizi, riwayat ASI, sanitasi, pendapatan.',
           bold='2. ')
pdf.bullet('Validasi model dengan data dari lokasi/daerah berbeda.',
           bold='3. ')

pdf.chapter_title('Rekomendasi Pengembangan', 3)
pdf.bullet('Bangun aplikasi web/mobile screening stunting menggunakan model Decision Tree.',
           bold='1. ')
pdf.bullet('Kembangkan dashboard monitoring untuk tenaga kesehatan puskesmas.',
           bold='2. ')
pdf.bullet('Integrasikan dengan sistem data kesehatan nasional (e-PPGBM).',
           bold='3. ')

pdf.chapter_title('6.3 Keterbatasan Penelitian', 2)
pdf.bullet('Ukuran dataset terbatas (1.329 sampel) - generalisasi perlu validasi lanjutan.')
pdf.bullet('Kelas Severely Stunted sangat sedikit (15 sampel) - SMOTE membantu tapi terbatas.')
pdf.bullet('Hanya menggunakan 3 fitur (umur, tinggi, JK) - banyak faktor stunting belum dipertimbangkan.')
pdf.bullet('Data berasal dari satu sumber/lokasi - perlu validasi lintas daerah.')
pdf.bullet('Hyperparameter tuning terbatas - tuning lebih ekstensif bisa meningkatkan performa.')

pdf.chapter_title('6.4 Penutup', 2)
pdf.text(
    'Penelitian ini berhasil membangun model prediksi stunting pada balita dengan '
    'Random Forest sebagai model terbaik (F1-Score: 0.9546). Decision Tree dengan '
    'SMOTE K=3 dan hyperparameter tuning menjadi pilihan terbaik untuk interpretability '
    '(F1-Score: 0.9380). Model ini diharapkan dapat membantu tenaga kesehatan dalam '
    'melakukan deteksi dini stunting sehingga intervensi dapat dilakukan lebih cepat.')

pdf.ln(8)
pdf.set_font('Helvetica', 'I', 10)
pdf.set_text_color(100, 100, 100)
pdf.cell(0, 8, '-- Laporan diproduksi secara otomatis berdasarkan hasil analisis data --',
         align='C', new_x="LMARGIN", new_y="NEXT")

# ═══════════════════════════════════════════════════════════════
# SAVE PDF
# ═══════════════════════════════════════════════════════════════
out_path = 'reports/Laporan_Analisis_Prediksi_Stunting.pdf'
pdf.output(out_path)
print(f'=' * 60)
print(f'PDF BERHASIL DIBUAT!')
print(f'  File: {out_path}')
print(f'  Halaman: {pdf.page_no()}')
print(f'  Ukuran: {os.path.getsize(out_path) / 1024:.1f} KB' if os.path.exists(out_path) else '')
print(f'=' * 60)
