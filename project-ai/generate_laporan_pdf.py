#!/usr/bin/env python3
"""
GENERATE PDF LAPORAN - Training Importance Fitur Survival_Prediction
====================================================================
Membuat PDF laporan lengkap dari hasil training di folder Training_Importance_Fitur/
"""

import os
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, ListFlowable, ListItem
)
from reportlab.lib import colors
import pandas as pd

# ================================================================
# LOAD DATA
# ================================================================
base_dir = 'Training_Importance_Fitur'

# Feature Importance
fi_df = pd.read_csv(f'{base_dir}/feature_importance_rf.csv')
mi_df = pd.read_csv(f'{base_dir}/feature_importance_mutual_info.csv')

# Model Results
results_df = pd.read_csv(f'{base_dir}/model_comparison_results.csv', index_col=0)

# Selected Features
selected_df = pd.read_csv(f'{base_dir}/selected_features.csv')

# Prediction Table
pred_df = pd.read_csv(f'{base_dir}/prediction_table.csv')

# ================================================================
# SETUP PDF
# ================================================================
output_path = 'Laporan_Training_Importance_Fitur.pdf'

doc = SimpleDocTemplate(
    output_path,
    pagesize=A4,
    topMargin=2*cm,
    bottomMargin=2*cm,
    leftMargin=2.5*cm,
    rightMargin=2.5*cm
)

styles = getSampleStyleSheet()

# Custom styles
title_style = ParagraphStyle(
    'CustomTitle',
    parent=styles['Title'],
    fontSize=22,
    leading=28,
    spaceAfter=6,
    textColor=HexColor('#1a1a2e'),
    alignment=TA_CENTER,
    fontName='Helvetica-Bold'
)

subtitle_style = ParagraphStyle(
    'CustomSubtitle',
    parent=styles['Normal'],
    fontSize=13,
    leading=18,
    spaceAfter=20,
    textColor=HexColor('#444444'),
    alignment=TA_CENTER,
    fontName='Helvetica'
)

h1_style = ParagraphStyle(
    'Heading1Custom',
    parent=styles['Heading1'],
    fontSize=16,
    leading=22,
    spaceBefore=20,
    spaceAfter=10,
    textColor=HexColor('#1a1a2e'),
    fontName='Helvetica-Bold',
    borderWidth=0,
    borderPadding=0,
)

h2_style = ParagraphStyle(
    'Heading2Custom',
    parent=styles['Heading2'],
    fontSize=13,
    leading=18,
    spaceBefore=15,
    spaceAfter=8,
    textColor=HexColor('#2c3e50'),
    fontName='Helvetica-Bold'
)

body_style = ParagraphStyle(
    'BodyCustom',
    parent=styles['Normal'],
    fontSize=10,
    leading=14,
    spaceAfter=6,
    alignment=TA_JUSTIFY,
    fontName='Helvetica'
)

bullet_style = ParagraphStyle(
    'BulletCustom',
    parent=body_style,
    leftIndent=20,
    bulletIndent=10,
    spaceBefore=2,
    spaceAfter=2,
)

code_style = ParagraphStyle(
    'CodeStyle',
    parent=styles['Normal'],
    fontSize=9,
    leading=12,
    fontName='Courier',
    leftIndent=10,
    spaceAfter=4,
    textColor=HexColor('#333333'),
)

header_cell = ParagraphStyle(
    'HeaderCell',
    fontSize=9,
    leading=12,
    fontName='Helvetica-Bold',
    textColor=white,
    alignment=TA_CENTER,
)

cell_style = ParagraphStyle(
    'CellStyle',
    fontSize=9,
    leading=11,
    fontName='Helvetica',
    alignment=TA_CENTER,
)

cell_left = ParagraphStyle(
    'CellLeft',
    fontSize=9,
    leading=11,
    fontName='Helvetica',
    alignment=TA_LEFT,
)

# ================================================================
# BUILD PDF CONTENT
# ================================================================
elements = []

# ---- COVER / TITLE ----
elements.append(Spacer(1, 3*cm))
elements.append(Paragraph("LAPORAN TRAINING", title_style))
elements.append(Paragraph("IMPORTANCE FITUR - SURVIVAL PREDICTION", title_style))
elements.append(Spacer(1, 0.5*cm))
elements.append(Paragraph(
    "Kanker Kolorektal | SMOTE K=5 | 4 Algoritma ML",
    subtitle_style
))
elements.append(Spacer(1, 0.3*cm))
elements.append(Paragraph(
    "_______________________________________________",
    ParagraphStyle('Line', parent=styles['Normal'], alignment=TA_CENTER, textColor=HexColor('#cccccc'))
))
elements.append(Spacer(1, 0.5*cm))
elements.append(Paragraph(
    "Analisis fitur importance untuk memprediksi Survival_Prediction<br/>"
    "menggunakan 4 algoritma machine learning dengan SMOTE oversampling",
    body_style
))
elements.append(Spacer(1, 1.5*cm))

# Info box
info_data = [
    ['Target', 'Survival_Prediction (Yes/No)'],
    ['Total Dataset', '167,497 baris'],
    ['Sample', '30,000 baris (stratified)'],
    ['Fitur Awal', '46 fitur (setelah One-Hot Encoding)'],
    ['Fitur Terpilih', f'{len(selected_df)} fitur (intersection RF + MI)'],
    ['SMOTE', 'K=5 (balancing classes)'],
    ['Train/Test Split', '80/20'],
    ['Setelah SMOTE', '28,782 baris (50:50 ratio)'],
    ['Algoritma', 'SVM, XGBoost, Random Forest, KNN, Baseline'],
]

info_table = Table(info_data, colWidths=[5*cm, 10*cm])
info_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (0, -1), HexColor('#1a1a2e')),
    ('TEXTCOLOR', (0, 0), (0, -1), white),
    ('BACKGROUND', (1, 0), (1, -1), HexColor('#f0f0f5')),
    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
    ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
    ('FONTSIZE', (0, 0), (-1, -1), 10),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('TOPPADDING', (0, 0), (-1, -1), 5),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ('LEFTPADDING', (0, 0), (-1, -1), 10),
    ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
]))
elements.append(info_table)

# ---- PAGE 2: FEATURE IMPORTANCE ----
elements.append(PageBreak())
elements.append(Paragraph("1. FEATURE IMPORTANCE ANALYSIS", h1_style))
elements.append(Paragraph(
    "Feature Importance dihitung menggunakan dua metode untuk menentukan fitur "
    "yang paling berpengaruh terhadap target <b>Survival_Prediction</b>.",
    body_style
))
elements.append(Spacer(1, 0.3*cm))

# 1a. Random Forest Importance
elements.append(Paragraph("1.1 Random Forest Feature Importance (Top 15)", h2_style))
elements.append(Paragraph(
    "Random Forest mengukur seberapa sering suatu fitur digunakan untuk split "
    "di seluruh decision tree. Semakin tinggi nilai, semakin penting fitur tersebut.",
    body_style
))

rf_top = fi_df.head(15)
rf_table_data = [['No', 'Fitur', 'Importance', '%']]
for i, (_, row) in enumerate(rf_top.iterrows()):
    rf_table_data.append([
        str(i+1),
        row['Feature'],
        f"{row['Importance']:.4f}",
        f"{row['Importance']*100:.2f}%"
    ])

rf_table = Table(rf_table_data, colWidths=[1*cm, 10*cm, 2.5*cm, 2*cm])
rf_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a1a2e')),
    ('TEXTCOLOR', (0, 0), (-1, 0), white),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 9),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('ALIGN', (1, 0), (1, -1), 'LEFT'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('TOPPADDING', (0, 0), (-1, -1), 4),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#f8f9fa'), white]),
]))
elements.append(rf_table)

elements.append(Spacer(1, 0.3*cm))

# 1b. Mutual Information
elements.append(Paragraph("1.2 Mutual Information Score (Top 15)", h2_style))
elements.append(Paragraph(
    "Mutual Information mengukur ketergantungan antara fitur dan target. "
    "Nilai 0 berarti independen (tidak ada informasi bersama).",
    body_style
))

mi_top = mi_df.head(15)
mi_table_data = [['No', 'Fitur', 'MI Score']]
for i, (_, row) in enumerate(mi_top.iterrows()):
    mi_table_data.append([str(i+1), row['Feature'], f"{row['MI_Score']:.4f}"])

mi_table = Table(mi_table_data, colWidths=[1*cm, 10*cm, 3*cm])
mi_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a1a2e')),
    ('TEXTCOLOR', (0, 0), (-1, 0), white),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 9),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('ALIGN', (1, 0), (1, -1), 'LEFT'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('TOPPADDING', (0, 0), (-1, -1), 4),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#f8f9fa'), white]),
]))
elements.append(mi_table)

elements.append(Spacer(1, 0.3*cm))

# Insert feature importance image
img_path = f'{base_dir}/01_feature_importance.png'
if os.path.exists(img_path):
    elements.append(Spacer(1, 0.3*cm))
    img = Image(img_path, width=17*cm, height=7.5*cm)
    elements.append(img)
    elements.append(Paragraph(
        "Gambar 1: Perbandingan Feature Importance (RF vs Mutual Information)",
        ParagraphStyle('Caption', parent=body_style, alignment=TA_CENTER, fontSize=9, textColor=HexColor('#666666'))
    ))

# ---- PAGE 3: SELECTED FEATURES ----
elements.append(PageBreak())
elements.append(Paragraph("2. SELEKSI FITUR TERPILIH", h1_style))
elements.append(Paragraph(
    "Fitur dipilih berdasarkan <b>intersection</b> dari Top 20 RF Importance dan "
    "Top 20 Mutual Information. Hasilnya adalah 10 fitur yang konsisten penting "
    "di kedua metode pengukuran.",
    body_style
))
elements.append(Spacer(1, 0.3*cm))

sel_table_data = [['No', 'Fitur', 'Kategori', 'Keterangan']]

fitur_info = {
    'Alcohol_Consumption_Yes': 'Riwayat konsumsi alkohol / Lifestyle',
    'Cancer_Stage_Regional': 'Stadium kanker regional / Klinis',
    'Early_Detection_Yes': 'Deteksi dini / Screening',
    'Economic_Classification_Developing': 'Klasifikasi negara berkembang / Demografi',
    'Family_History_Yes': 'Riwayat keluarga / Genetik',
    'Gender_M': 'Jenis kelamin pria / Demografi',
    'Incidence_Rate_per_100K': 'Tingkat insidensi / Epidemiology',
    'Insurance_Status_Uninsured': 'Status tidak punya asuransi / Ekonomi',
    'Obesity_BMI_Overweight': 'Obesitas (overweight) / Lifestyle',
    'Screening_History_Regular': 'Riwayat screening rutin / Screening',
}

for i, (_, row) in enumerate(selected_df.iterrows()):
    feat = row['Feature']
    desc = fitur_info.get(feat, '-')
    # Determine category
    num_feats = ['Incidence_Rate_per_100K']
    if feat in num_feats:
        cat = 'Numerik'
    else:
        cat = 'Kategorikal (encoded)'
    sel_table_data.append([str(i+1), feat, cat, desc])

sel_table = Table(sel_table_data, colWidths=[1*cm, 7*cm, 4*cm, 4.5*cm])
sel_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a1a2e')),
    ('TEXTCOLOR', (0, 0), (-1, 0), white),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 9),
    ('ALIGN', (0, 0), (1, -1), 'CENTER'),
    ('ALIGN', (1, 0), (1, -1), 'LEFT'),
    ('ALIGN', (3, 0), (3, -1), 'LEFT'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('TOPPADDING', (0, 0), (-1, -1), 4),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ('LEFTPADDING', (1, 0), (1, -1), 8),
    ('LEFTPADDING', (3, 0), (3, -1), 8),
    ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#f8f9fa'), white]),
]))
elements.append(sel_table)

elements.append(Spacer(1, 0.3*cm))
elements.append(Paragraph(
    "<b>Catatan:</b> 5 fitur numerik (Age, Tumor_Size_mm, Healthcare_Costs, "
    "Mortality_Rate_per_100K) memiliki RF importance tinggi (~0.08–0.13) namun tidak "
    "lolos intersection karena MI Score rendah. Ini menunjukkan hubungan non-linear "
    "yang kompleks dengan target.",
    ParagraphStyle('Note', parent=body_style, fontSize=9, textColor=HexColor('#666666'),
                   leftIndent=10, borderWidth=1, borderColor=HexColor('#ddd'),
                   borderPadding=8, backColor=HexColor('#fffbe6'))
))

# ---- PAGE 4: TARGET & SMOTE ----
elements.append(PageBreak())
elements.append(Paragraph("3. TARGET VARIABLE & IMBALANCE HANDLING", h1_style))

elements.append(Paragraph("3.1 Target: Survival_Prediction", h2_style))
elements.append(Paragraph(
    "Target <b>Survival_Prediction</b> adalah variabel biner yang menunjukkan "
    "prediksi kelangsungan hidup pasien kanker kolorektal:",
    body_style
))
elements.append(Spacer(1, 0.2*cm))

# Target distribution
target_data = [
    ['Kelas', 'Jumlah', 'Persentase', 'Kode'],
    ['Yes (Hidup)', '17,989', '60.0%', '1'],
    ['No (Meninggal)', '12,011', '40.0%', '0'],
    ['Total', '30,000', '100%', '-'],
]
target_table = Table(target_data, colWidths=[4*cm, 3.5*cm, 3*cm, 2*cm])
target_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a1a2e')),
    ('TEXTCOLOR', (0, 0), (-1, 0), white),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 10),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('TOPPADDING', (0, 0), (-1, -1), 5),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
    ('ROWBACKGROUNDS', (0, 1), (-1, -2), [HexColor('#d4edda'), HexColor('#f8d7da')]),
    ('BACKGROUND', (0, -1), (-1, -1), HexColor('#e3f2fd')),
    ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
]))
elements.append(target_table)

elements.append(Spacer(1, 0.3*cm))

# SMOTE explanation
elements.append(Paragraph("3.2 SMOTE Oversampling (K=5)", h2_style))
elements.append(Paragraph(
    "<b>SMOTE (Synthetic Minority Oversampling Technique)</b> adalah teknik "
    "untuk menangani data tidak seimbang dengan membuat sampel sintetis dari "
    "kelas minoritas. Parameter K=5 berarti menggunakan 5 tetangga terdekat "
    "untuk menginterpolasi sampel baru.",
    body_style
))
elements.append(Spacer(1, 0.2*cm))

smote_data = [
    ['Tahap', 'Yes (Hidup)', 'No (Meninggal)', 'Total', 'Ratio'],
    ['Sebelum SMOTE', '14,391', '9,609', '24,000', '60:40'],
    ['Setelah SMOTE (K=5)', '14,391', '14,391', '28,782', '50:50'],
]
smote_table = Table(smote_data, colWidths=[4*cm, 3*cm, 3.5*cm, 2.5*cm, 2.5*cm])
smote_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a1a2e')),
    ('TEXTCOLOR', (0, 0), (-1, 0), white),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 10),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('TOPPADDING', (0, 0), (-1, -1), 5),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#f8f9fa'), HexColor('#e8f5e9')]),
]))
elements.append(smote_table)

elements.append(Spacer(1, 0.3*cm))
elements.append(Paragraph(
    "<b>Keuntungan SMOTE:</b> Menyeimbangkan kelas tanpa kehilangan data, "
    "mengurangi bias model terhadap kelas mayoritas, dan menghasilkan "
    "sampel sintetis yang realistis.",
    body_style
))

# ---- PAGE 5: MODEL TRAINING ----
elements.append(PageBreak())
elements.append(Paragraph("4. TRAINING 4 ALGORITMA MACHINE LEARNING", h1_style))
elements.append(Paragraph(
    "Empat algoritma klasifikasi dilatih menggunakan fitur terpilih dengan SMOTE K=5. "
    "SVM dan KNN menggunakan data terstandarisasi (StandardScaler), sementara "
    "XGBoost dan Random Forest menggunakan data mentah (tree-based tidak sensitif skala).",
    body_style
))
elements.append(Spacer(1, 0.3*cm))

# Algorithm descriptions
algo_info = [
    ["<b>SVM</b> (Support Vector Machine)",
     "Mencari hyperplane terbaik yang memisahkan kelas. Menggunakan LinearSVC "
     "dengan class_weight='balanced' dan CalibratedClassifierCV untuk probabilitas."],
    ["<b>XGBoost</b> (Extreme Gradient Boosting)",
     "Algoritma boosting berbasis tree yang sangat efisien. Menggunakan 100 estimator, "
     "max_depth=6, learning_rate=0.1."],
    ["<b>Random Forest</b>",
     "Ensemble dari 50 decision tree dengan max_depth=8. Menggunakan bootstrap "
     "aggregation untuk mengurangi overfitting."],
    ["<b>KNN</b> (K-Nearest Neighbors)",
     "Klasifikasi berdasarkan jarak ke K=5 tetangga terdekat. Menggunakan "
     "weights='distance' untuk memberikan bobot lebih pada tetangga terdekat."],
]

for algo_name, algo_desc in algo_info:
    elements.append(Paragraph(f"<b>{algo_name}</b>", h2_style))
    elements.append(Paragraph(algo_desc, body_style))
    elements.append(Spacer(1, 0.1*cm))

# ---- PAGE 6: RESULTS ----
elements.append(PageBreak())
elements.append(Paragraph("5. HASIL PERBANDINGAN KINERJA", h1_style))

elements.append(Paragraph("5.1 Metrik Evaluasi", h2_style))
elements.append(Paragraph(
    "Model dievaluasi menggunakan 5 metrik utama pada data test (6,000 sampel):",
    body_style
))
elements.append(Spacer(1, 0.2*cm))

# Metrics explanation
metrics_desc = [
    "<b>Accuracy:</b> Proporsi prediksi benar dari total prediksi.",
    "<b>Precision:</b> Proporsi prediksi positif yang benar-benar positif.",
    "<b>Recall:</b> Proporsi data positif yang berhasil diprediksi positif.",
    "<b>F1-Score:</b> Harmonic mean dari Precision dan Recall.",
    "<b>ROC-AUC:</b> Area Under ROC Curve - kemampuan model membedakan kelas.",
]
for md in metrics_desc:
    elements.append(Paragraph(f"• {md}", bullet_style))

elements.append(Spacer(1, 0.3*cm))

# Performance Table
elements.append(Paragraph("5.2 Perbandingan Kinerja Model", h2_style))

perf_data = [['Rank', 'Algoritma', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Time']]
for i, (idx, row) in enumerate(results_df.iterrows()):
    perf_data.append([
        str(i+1),
        idx,
        f"{row['Accuracy']:.4f}",
        f"{row['Precision']:.4f}",
        f"{row['Recall']:.4f}",
        f"{row['F1-Score']:.4f}",
        f"{row['ROC-AUC']:.4f}",
        f"{row['Training_Time']:.2f}s"
    ])

perf_table = Table(perf_data, colWidths=[1*cm, 3*cm, 2*cm, 2.2*cm, 2*cm, 2*cm, 2*cm, 1.8*cm])

# Build style commands safely (no None values)
style_cmds = [
    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a1a2e')),
    ('TEXTCOLOR', (0, 0), (-1, 0), white),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 9),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('ALIGN', (1, 0), (1, -1), 'LEFT'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('TOPPADDING', (0, 0), (-1, -1), 5),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
]

# Color rows
for i in range(1, len(perf_data)):
    if i == 1:
        bg = HexColor('#d4edda')  # Best - green
    elif i == 2:
        bg = HexColor('#e8f5e9')  # Second - light green
    else:
        bg = HexColor('#f8f9fa')
    style_cmds.append(('BACKGROUND', (0, i), (-1, i), bg))
    if i <= 2:
        style_cmds.append(('FONTNAME', (1, i), (1, i), 'Helvetica-Bold'))

perf_table.setStyle(TableStyle(style_cmds))

elements.append(perf_table)

elements.append(Spacer(1, 0.2*cm))

# Best model highlight
best_model = results_df.index[0]
best_auc = results_df['ROC-AUC'].iloc[0]
elements.append(Paragraph(
    f"🏆 <b>Model Terbaik: {best_model}</b> dengan ROC-AUC = {best_auc:.4f}",
    ParagraphStyle('BestModel', parent=body_style, fontSize=11, textColor=HexColor('#1a7a2e'),
                   leftIndent=10, borderWidth=1, borderColor=HexColor('#2ecc71'),
                   borderPadding=8, backColor=HexColor('#e8f5e9'), fontName='Helvetica-Bold')
))

# ---- CONFUSION MATRICES ----
elements.append(PageBreak())
elements.append(Paragraph("6. CONFUSION MATRIX", h1_style))
elements.append(Paragraph(
    "Confusion Matrix menunjukkan perbandingan antara prediksi model dengan "
    "nilai aktual untuk setiap algoritma.",
    body_style
))
elements.append(Spacer(1, 0.3*cm))

# Insert confusion matrix images for each model
cm_files = ['03_cm_SVM.png', '03_cm_XGBoost.png', '03_cm_Random_Forest.png', '03_cm_KNN.png']
for cm_file in cm_files:
    cm_path = f'{base_dir}/{cm_file}'
    if os.path.exists(cm_path):
        elements.append(Paragraph(f"<b>{cm_file.replace('03_cm_', '').replace('.png', '')}</b>", h2_style))
        img = Image(cm_path, width=12*cm, height=10*cm)
        elements.append(img)
        elements.append(Spacer(1, 0.3*cm))

# All confusion matrices
all_cm_path = f'{base_dir}/04_all_confusion_matrices.png'
if os.path.exists(all_cm_path):
    elements.append(PageBreak())
    elements.append(Paragraph("6.1 Semua Confusion Matrix (1 Gambar)", h2_style))
    img = Image(all_cm_path, width=17*cm, height=10*cm)
    elements.append(img)

# ROC Curves
roc_path = f'{base_dir}/05_roc_curves.png'
if os.path.exists(roc_path):
    elements.append(PageBreak())
    elements.append(Paragraph("7. ROC CURVES", h1_style))
    elements.append(Paragraph(
        "ROC Curve menggambarkan trade-off antara True Positive Rate dan "
        "False Positive Rate. Semakin dekat kurva ke sudut kiri atas, "
        "semakin baik performa model.",
        body_style
    ))
    elements.append(Spacer(1, 0.3*cm))
    img = Image(roc_path, width=14*cm, height=11*cm)
    elements.append(img)

# ---- PERFORMANCE TABLE IMAGE ----
perf_img_path = f'{base_dir}/06_performance_table.png'
if os.path.exists(perf_img_path):
    elements.append(PageBreak())
    elements.append(Paragraph("8. TABEL PERBANDINGAN KINERJA", h1_style))
    img = Image(perf_img_path, width=17*cm, height=6*cm)
    elements.append(img)

# Performance charts
charts_path = f'{base_dir}/07_performance_charts.png'
if os.path.exists(charts_path):
    elements.append(Spacer(1, 0.3*cm))
    elements.append(Paragraph("8.1 Grafik Perbandingan", h2_style))
    img = Image(charts_path, width=17*cm, height=12*cm)
    elements.append(img)

# CM Summary table
cm_summary_path = f'{base_dir}/08_cm_summary_table.png'
if os.path.exists(cm_summary_path):
    elements.append(PageBreak())
    elements.append(Paragraph("9. RINGKASAN CONFUSION MATRIX", h1_style))
    elements.append(Paragraph(
        "Ringkasan nilai True Negative (TN), False Positive (FP), "
        "False Negative (FN), dan True Positive (TP) untuk setiap model.",
        body_style
    ))
    elements.append(Spacer(1, 0.3*cm))
    img = Image(cm_summary_path, width=16*cm, height=5*cm)
    elements.append(img)

# ---- CONCLUSION ----
elements.append(PageBreak())
elements.append(Paragraph("10. KESIMPULAN & REKOMENDASI", h1_style))

elements.append(Paragraph("10.1 Temuan Utama", h2_style))

findings = [
    "<b>AUC ~0.5:</b> Semua model menghasilkan ROC-AUC sekitar 0.5, yang menunjukkan "
    "bahwa fitur yang tersedia tidak memiliki sinyal prediktif yang cukup kuat "
    "untuk memprediksi Survival_Prediction secara akurat.",

    "<b>XGBoost terbaik:</b> XGBoost menjadi model dengan performa terbaik "
    "(ROC-AUC = 0.5086), meskipun masih mendekati prediksi acak.",

    "<b>Feature Importance:</b> Dari 46 fitur awal, hanya 10 fitur yang lolos "
    "intersection RF + MI, menunjukkan bahwa sebagian besar fitur memiliki "
    "hubungan yang sangat lemah dengan target.",

    "<b>SMOTE efektif:</b> SMOTE K=5 berhasil menyeimbangkan kelas dari "
    "rasio 60:40 menjadi 50:50 dengan menambah 4,782 sampel sintetis.",
]
for f in findings:
    elements.append(Paragraph(f"• {f}", bullet_style))
    elements.append(Spacer(1, 0.15*cm))

elements.append(Spacer(1, 0.5*cm))
elements.append(Paragraph("10.2 Rekomendasi", h2_style))

recommendations = [
    "<b>Dataset tambahan:</b> Cari fitur klinis yang lebih relevan seperti "
    "TNM staging detail, biomarker (CEA, CA19-9), dan data genetik.",

    "<b>Target berbeda:</b> Survival_5_years mungkin lebih prediktif "
    "daripada Survival_Prediction karena memiliki sinyal yang lebih kuat.",

    "<b>Deep Learning:</b> Neural network dengan arsitektur yang lebih kompleks "
    "mungkin bisa menangkap pola non-linear yang tidak terdeteksi.",

    "<b>Validasi data:</b> Periksa kembali proses labeling Survival_Prediction "
    "karena dataset ini mungkin sintetis (0 missing values + korelasi sangat rendah).",
]
for r in recommendations:
    elements.append(Paragraph(f"• {r}", bullet_style))
    elements.append(Spacer(1, 0.15*cm))

elements.append(Spacer(1, 1*cm))
elements.append(Paragraph(
    "— End of Report —",
    ParagraphStyle('End', parent=body_style, alignment=TA_CENTER, fontSize=11,
                   textColor=HexColor('#999999'), fontName='Helvetica-Bold')
))

# ================================================================
# BUILD PDF
# ================================================================
doc.build(elements)
print(f"✅ PDF berhasil dibuat: {output_path}")
print(f"   {os.path.getsize(output_path) / 1024:.1f} KB")
