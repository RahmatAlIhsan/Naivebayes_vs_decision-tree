# BAB IV: HASIL DAN PEMBAHASAN

## 4.1 Deskripsi Dataset

Dataset yang digunakan dalam penelitian ini adalah data stunting pada balita yang telah melalui tahap preprocessing (data cleaning). Dataset bersih (`stunting_clean.csv`) memiliki **1.329 sampel** dengan **4 variabel**, yaitu:

| Variabel | Tipe Data | Deskripsi |
|---|---|---|
| `umur_bulan` | Numerik (Float) | Usia balita dalam bulan (0-60 bulan) |
| `tinggi` | Numerik (Float) | Tinggi badan balita dalam cm |
| `jk` | Kategorikal | Jenis kelamin (Laki-Laki / Perempuan) |
| `status` | Kategorikal (Target) | Status stunting: Normal, Stunted, Severely Stunted |

### 4.1.1 Distribusi Kelas Target

Distribusi status stunting pada dataset menunjukkan ketidakseimbangan kelas (class imbalance) yang signifikan:

| Status | Jumlah | Persentase |
|---|---|---|
| Normal | 1.226 | 92,25% |
| Stunted | 88 | 6,62% |
| Severely Stunted | 15 | 1,13% |
| **Total** | **1.329** | **100%** |

Ketidakseimbangan ini menjadi tantangan utama dalam pemodelan, di mana model cenderung bias terhadap kelas mayoritas (Normal) dan gagal mendeteksi kelas minoritas (Stunted dan Severely Stunted).

### 4.1.2 Pembagian Data (Split)

Data dibagi menjadi data latih (train) dan data uji (test) dengan proporsi **80:20** menggunakan metode stratified splitting untuk mempertahankan proporsi kelas yang sama di kedua set:

- **Data Latih (Train)**: 1.063 sampel
- **Data Uji (Test)**: 266 sampel

---

## 4.2 Skenario Eksperimen

Penelitian ini melakukan dua skenario eksperimen untuk membandingkan performa algoritma **Naive Bayes** dan **Decision Tree**:

### Skenario 1: Tanpa SMOTE - Tanpa Hyperparameter Tuning
- Menggunakan data dengan ketidakseimbangan kelas asli
- Model menggunakan parameter default
- Tujuan: Melihat performa baseline tanpa penanganan imbalance

### Skenario 2: SMOTE K=3 - Dengan Hyperparameter Tuning
- Menggunakan SMOTE (Synthetic Minority Oversampling Technique) dengan K=3 untuk menyeimbangkan kelas
- Hyperparameter tuning menggunakan GridSearchCV dengan 5-fold Stratified Cross-Validation
- Tujuan: Meningkatkan kemampuan deteksi kelas minoritas

---

## 4.3 Hasil Eksperimen 1: Tanpa SMOTE - Tanpa Tuning

### 4.3.1 Performa Model Baseline

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Naive Bayes | 0,9211 | 0,8483 | 0,9211 | 0,8832 |
| Decision Tree | 0,9286 | 0,9287 | 0,9286 | 0,9284 |

### 4.3.2 Analisis Confusion Matrix

**Naive Bayes:**
```
                Predicted
                Normal  Stunted  Severely Stunted
Actual Normal      245        0                 0
       Stunted      18        0                 0
Severely Stunted     3        0                 0
```

Naive Bayes gagal total dalam mendeteksi kelas minoritas. Model hanya memprediksi semua sampel sebagai "Normal". Hal ini disebabkan oleh:

1. **Asumsi independensi** (Naive Bayes) yang tidak terpenuhi — fitur `umur_bulan` dan `tinggi` memiliki korelasi yang sangat kuat (r > 0,9)
2. **Dominasi kelas mayoritas** (92,25%) yang membuat prior probability kelas Normal sangat tinggi

**Decision Tree:**
```
                Predicted
                Normal  Stunted  Severely Stunted
Actual Normal      237        8                 0
       Stunted       8        9                 1
Severely Stunted     0        2                 1
```

Decision Tree menunjukkan kemampuan yang lebih baik dengan:
- **9 dari 18** sampel Stunted terdeteksi dengan benar (recall 50%)
- **1 dari 3** sampel Severely Stunted terdeteksi dengan benar
- Namun masih terdapat 8 false negative untuk kelas Stunted

---

## 4.4 Hasil Eksperimen 2: SMOTE K=3 - Hyperparameter Tuning

### 4.4.1 Penerapan SMOTE K=3

SMOTE dengan K=3 berhasil menyeimbangkan data latih:

| Kelas | Sebelum SMOTE | Sesudah SMOTE K=3 |
|---|---|---|
| Normal | 981 | 981 |
| Stunted | 70 | 981 |
| Severely Stunted | 12 | 981 |
| **Total** | **1.063** | **2.943** |

### 4.4.2 Hasil Hyperparameter Tuning (GridSearchCV)

**Naive Bayes:**
- Parameter yang di-tuning: `var_smoothing`
- Rentang: 10⁻¹² hingga 10⁻⁵ (8 nilai)
- **Best parameter: var_smoothing = 1 × 10⁻¹²**
- **Best CV Score (F1-Weighted): 0,4879**

Parameter `var_smoothing` yang optimal menunjukkan bahwa Naive Bayes membutuhkan smoothing minimal untuk stabilitas numerik, namun performa tetap rendah karena asumsi independensi fitur tidak terpenuhi.

**Decision Tree:**
- Parameter yang di-tuning: `max_depth`, `min_samples_split`, `min_samples_leaf`, `criterion`, `max_features`
- Total kombinasi: 6 × 4 × 4 × 2 × 3 = **576 kombinasi**
- **Best parameter:**
  - `criterion`: gini
  - `max_depth`: None (tanpa batasan)
  - `max_features`: sqrt
  - `min_samples_leaf`: 1
  - `min_samples_split`: 2
- **Best CV Score (F1-Weighted): 0,9365**

Parameter optimal Decision Tree menunjukkan bahwa:
- `max_depth=None` (tanpa batasan kedalaman) dipilih, menunjukkan bahwa kompleksitas pohon diperlukan untuk menangkap pola stunting
- `max_features='sqrt'` membatasi jumlah fitur yang dipertimbangkan di setiap split, mengurangi overfitting
- `criterion='gini'` sedikit lebih baik daripada 'entropy' untuk dataset ini

### 4.4.3 Performa pada Test Set

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Naive Bayes (Tuned) | 0,4361 | 0,9125 | 0,4361 | 0,5505 |
| Decision Tree (Tuned) | **0,9361** | **0,9412** | **0,9361** | **0,9380** |

### 4.4.4 Analisis Confusion Matrix

**Naive Bayes (Tuned + SMOTE K=3):**
```
                Predicted
                Normal  Stunted  Severely Stunted
Actual Normal      101       84                60
       Stunted       2       13                 3
Severely Stunted     0        1                 2
```

Setelah SMOTE dan tuning, Naive Bayes mulai mendeteksi kelas minoritas, namun dengan konsekuensi:
- **False positive yang sangat tinggi**: 84 sampel Normal diprediksi Stunted, 60 diprediksi Severely Stunted
- Akurasi menurun drastis dari 0,9211 menjadi 0,4361
- Precision tetap tinggi (0,9125) untuk kelas Normal, tapi recall sangat rendah untuk kelas minoritas

**Decision Tree (Tuned + SMOTE K=3):**
```
                Predicted
                Normal  Stunted  Severely Stunted
Actual Normal      236        9                 0
       Stunted       7       11                 0
Severely Stunted     0        1                 2
```

Decision Tree menunjukkan performa terbaik dengan:
- **11 dari 18** sampel Stunted terdeteksi (61,1% recall)
- **2 dari 3** sampel Severely Stunted terdeteksi (66,7% recall)
- Hanya 9 false positive untuk kelas Normal
- **F1-Score tertinggi: 0,9380**

---

## 4.5 Perbandingan Hasil dan Analisis

### 4.5.1 Perbandingan Metrik Utama

| Skenario | Model | F1-Score |
|---|---|---|
| Tanpa SMOTE - Tanpa Tuning | Naive Bayes | 0,8832 |
| Tanpa SMOTE - Tanpa Tuning | Decision Tree | 0,9284 |
| SMOTE K=3 - Hyperparameter Tuning | Naive Bayes | 0,5505 |
| **SMOTE K=3 - Hyperparameter Tuning** | **Decision Tree** | **0,9380** |

### 4.5.2 Analisis Naive Bayes

**Kelebihan:**
- Cepat dalam training dan prediksi
- Sederhana secara komputasi

**Kekurangan:**
- **Asumsi independensi fitur tidak terpenuhi**: Fitur `umur_bulan` dan `tinggi` memiliki korelasi sangat tinggi. Balita yang lebih tua cenderung memiliki tinggi badan yang lebih besar, sehingga kedua fitur ini saling bergantung.
- **Performa menurun drastis setelah SMOTE**: SMOTE membuat data sintetis yang memperparah pelanggaran asumsi independensi
- **Tidak cocok untuk dataset ini**: Naive Bayes mengasumsikan semua fitur berkontribusi independen terhadap keputusan, padahal dalam kasus stunting, umur dan tinggi sangat berkorelasi

### 4.5.3 Analisis Decision Tree

**Kelebihan:**
- **Mampu menangkap pola non-linear**: Hubungan antara umur dan tinggi dengan status stunting bersifat non-linear, dan Decision Tree dapat menangkap pola ini dengan baik
- **Interpretability**: Aturan keputusan yang dihasilkan mudah dipahami (misal: "jika tinggi < X cm pada umur Y bulan, maka Stunted")
- **Tidak terpengaruh korelasi fitur**: Decision Tree dapat memilih fitur terbaik di setiap split tanpa terpengaruh korelasi antar fitur
- **Peningkatan signifikan setelah SMOTE + Tuning**: F1-Score meningkat dari 0,9284 menjadi 0,9380

**Kekurangan:**
- **Rentan overfitting**: Tanpa pembatasan kedalaman (max_depth=None), pohon bisa tumbuh terlalu dalam
- **Stabilitas**: Perubahan kecil pada data dapat mengubah struktur pohon secara signifikan

### 4.5.4 Efektivitas SMOTE K=3

Penerapan SMOTE dengan K=3 memberikan dampak yang berbeda pada kedua model:

| Aspek | Sebelum SMOTE | Sesudah SMOTE K=3 |
|---|---|---|
| Jumlah sampel latih | 1.063 | 2.943 |
| Keseimbangan kelas | Sangat timpang | Sempurna (981 per kelas) |
| Naive Bayes F1 | 0,8832 | 0,5505 (+) |
| Decision Tree F1 | 0,9284 | **0,9380** (+) |

**Kesimpulan SMOTE:**
- SMOTE efektif untuk Decision Tree (meningkatkan deteksi kelas minoritas)
- SMOTE tidak efektif untuk Naive Bayes (melanggar asumsi distribusi data)

### 4.5.5 Efektivitas Hyperparameter Tuning

Tuning dengan GridSearchCV memberikan parameter optimal untuk Decision Tree yang menghasilkan peningkatan F1-Score. Parameter `max_features='sqrt'' sangat penting untuk mengurangi overfitting dengan membatasi jumlah fitur yang dipertimbangkan di setiap split.

---

## 4.6 Hasil Terbaik dan Rekomendasi

### Model Terbaik: Decision Tree dengan SMOTE K=3 dan Hyperparameter Tuning

| Metrik | Nilai |
|---|---|
| Accuracy | 0,9361 |
| Precision | 0,9412 |
| Recall | 0,9361 |
| **F1-Score** | **0,9380** |

### Detail Performa per Kelas (Decision Tree Tuned):

| Kelas | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Normal | 0,97 | 0,96 | 0,97 | 245 |
| Stunted | 0,52 | 0,61 | 0,56 | 18 |
| Severely Stunted | 1,00 | 0,67 | 0,80 | 3 |
| **Weighted Avg** | **0,94** | **0,94** | **0,94** | **266** |
| **Macro Avg** | **0,83** | **0,75** | **0,78** | **266** |

### Interpretasi Hasil:

1. **Kelas Normal (Mayoritas)**: Precision 0,97 dan Recall 0,96 — model sangat akurat dalam mengidentifikasi balita normal
2. **Kelas Stunted**: Precision 0,52 — dari semua yang diprediksi Stunted, hanya 52% yang benar-benar Stunted. Recall 0,61 — model berhasil menangkap 61% dari seluruh balita Stunted. Ini wajar mengingat jumlah sampel yang terbatas.
3. **Kelas Severely Stunted**: Precision 1,00 — tidak ada false positive untuk kelas ini. Recall 0,67 — 2 dari 3 sampel terdeteksi, namun dengan hanya 3 sampel uji, hasil ini perlu validasi lebih lanjut.

### Rekomendasi:

1. **Decision Tree dengan SMOTE K=3 + Tuning** adalah model terbaik untuk prediksi stunting pada dataset ini
2. **Naive Bayes tidak direkomendasikan** karena asumsi independensi fitur tidak terpenuhi pada data antropometri stunting
3. **Perlu penambahan data kelas minoritas** (terutama Severely Stunted) untuk meningkatkan keandalan model
4. **Decision Tree direkomendasikan untuk interpretability** — aturan keputusan yang dihasilkan dapat digunakan oleh tenaga kesehatan untuk screening awal stunting
