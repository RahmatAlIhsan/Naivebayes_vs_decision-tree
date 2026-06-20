#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) - Colorectal Cancer Dataset
=============================================================
Analisis komprehensif dari dataset kanker kolorektal.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD & PREPROCESSING
# ============================================================
print("=" * 70)
print("EXPLORATORY DATA ANALYSIS - COLORECTAL CANCER DATASET")
print("=" * 70)

df = pd.read_csv('colorectal_cancer_dataset_asli.csv', sep=';')
print(f"\nDataset shape: {df.shape}")
print(f"Jumlah baris: {df.shape[0]:,}")
print(f"Jumlah kolom: {df.shape[1]}")

# Drop Patient_ID karena tidak informatif untuk analisis
df_analysis = df.drop(columns=['Patient_ID'])

# Konversi kolom numerik
num_cols = ['Age', 'Tumor_Size_mm', 'Healthcare_Costs', 'Incidence_Rate_per_100K', 'Mortality_Rate_per_100K']
cat_cols = [c for c in df_analysis.columns if c not in num_cols]

# Buat folder output
output_dir = 'eda_output'
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# 2. DATA QUALITY CHECK
# ============================================================
print("\n" + "=" * 70)
print("2. DATA QUALITY CHECK")
print("=" * 70)

print("\n--- Missing Values ---")
missing = df_analysis.isnull().sum()
missing_pct = (missing / len(df_analysis)) * 100
missing_df = pd.DataFrame({'Missing': missing, 'Percentage (%)': missing_pct.round(2)})
print(missing_df[missing_df['Missing'] > 0])

print("\n--- Duplicated Rows ---")
print(f"Duplikat: {df_analysis.duplicated().sum()} baris")

print("\n--- Data Types ---")
print(df_analysis.dtypes.to_string())

# ============================================================
# 3. UNIVARIATE ANALYSIS - NUMERIC
# ============================================================
print("\n" + "=" * 70)
print("3. UNIVARIATE ANALYSIS - NUMERIC VARIABLES")
print("=" * 70)

print("\n--- Numeric Summary Statistics ---")
print(df_analysis[num_cols].describe().round(2).to_string())

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
axes = axes.flatten()

num_plot_cols = num_cols  # Exclude Patient_ID

for i, col in enumerate(num_plot_cols):
    ax = axes[i]
    # Histogram dengan KDE
    sns.histplot(df_analysis[col], kde=True, bins=50, color='steelblue', ax=ax)
    ax.set_title(f'Distribusi {col}', fontsize=12, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Frekuensi')
    # Tambahkan garis mean dan median
    mean_val = df_analysis[col].mean()
    median_val = df_analysis[col].median()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.1f}')
    ax.axvline(median_val, color='green', linestyle=':', linewidth=1.5, label=f'Median: {median_val:.1f}')
    ax.legend(fontsize=8)

# Boxplot untuk semua numerik
ax = axes[5]
df_box = df_analysis[num_plot_cols].melt(var_name='Variable', value_name='Value')
sns.boxplot(data=df_box, x='Variable', y='Value', palette='Set2', ax=ax)
ax.set_title('Boxplot Variabel Numerik', fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(f'{output_dir}/01_numeric_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  => Saved: {output_dir}/01_numeric_distributions.png")

# ============================================================
# 4. UNIVARIATE ANALYSIS - CATEGORICAL
# ============================================================
print("\n" + "=" * 70)
print("4. UNIVARIATE ANALYSIS - CATEGORICAL VARIABLES")
print("=" * 70)

# Pilih kategorikal yang paling penting untuk divisualisasikan
key_cat_cols = ['Gender', 'Cancer_Stage', 'Country', 'Treatment_Type',
                'Survival_5_years', 'Mortality', 'Economic_Classification',
                'Obesity_BMI', 'Smoking_History', 'Alcohol_Consumption',
                'Diabetes', 'Genetic_Mutation', 'Early_Detection', 'Insurance_Status']

# Split into multiple pages
cat_per_page = 4
num_pages = (len(key_cat_cols) + cat_per_page - 1) // cat_per_page

for page in range(num_pages):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    start_idx = page * cat_per_page
    end_idx = min(start_idx + cat_per_page, len(key_cat_cols))
    
    for i, col in enumerate(key_cat_cols[start_idx:end_idx]):
        ax = axes[i]
        value_counts = df_analysis[col].value_counts()
        colors = plt.cm.Set2(np.linspace(0, 1, len(value_counts)))
        
        bars = ax.barh(range(len(value_counts)), value_counts.values, color=colors)
        ax.set_yticks(range(len(value_counts)))
        ax.set_yticklabels(value_counts.index, fontsize=10)
        ax.set_title(f'Distribusi {col}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Jumlah')
        
        # Tambahkan label
        for bar, val in zip(bars, value_counts.values):
            ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2,
                    f'{val:,}', ha='left', va='center', fontsize=9)
    
    # Hide unused subplots
    for j in range(end_idx - start_idx, 4):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_categorical_distributions_p{page+1}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  => Saved: {output_dir}/02_categorical_distributions_p{page+1}.png")

# Print frequency tables for all categorical variables
print("\n--- Frequency Tables ---")
for col in cat_cols:
    print(f"\n{col}:")
    vc = df_analysis[col].value_counts()
    vc_pct = (vc / len(df_analysis) * 100).round(2)
    freq_df = pd.DataFrame({'Count': vc, 'Percentage (%)': vc_pct})
    print(freq_df.to_string())

# ============================================================
# 5. BIVARIATE ANALYSIS - NUMERIC vs TARGET (Survival_5_years)
# ============================================================
print("\n" + "=" * 70)
print("5. BIVARIATE ANALYSIS - NUMERIC vs SURVIVAL")
print("=" * 70)

target = 'Survival_5_years'

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, col in enumerate(num_plot_cols):
    ax = axes[i]
    for surv_val in df_analysis[target].unique():
        subset = df_analysis[df_analysis[target] == surv_val][col]
        sns.kdeplot(subset, label=f'{target}={surv_val}', ax=ax, linewidth=2)
    ax.set_title(f'{col} by {target}', fontsize=11, fontweight='bold')
    ax.set_xlabel(col)
    ax.legend()

# Group stats
ax = axes[5]
group_stats = df_analysis.groupby(target)[num_plot_cols].mean().T
group_stats.plot(kind='bar', ax=ax, colormap='Set2', width=0.8)
ax.set_title('Rata-rata Variabel Numerik by Survival', fontsize=11, fontweight='bold')
ax.set_ylabel('Rata-rata')
ax.legend(title=target)
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(f'{output_dir}/03_numeric_by_survival.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  => Saved: {output_dir}/03_numeric_by_survival.png")

# ============================================================
# 6. BIVARIATE ANALYSIS - CATEGORICAL vs TARGET
# ============================================================
print("\n" + "=" * 70)
print("6. BIVARIATE ANALYSIS - CATEGORICAL vs SURVIVAL")
print("=" * 70)

# Pilih kategorikal yang relevan untuk cross-tabulation
cross_cols = ['Gender', 'Cancer_Stage', 'Country', 'Treatment_Type',
              'Smoking_History', 'Alcohol_Consumption', 'Obesity_BMI',
              'Diabetes', 'Genetic_Mutation', 'Early_Detection',
              'Economic_Classification', 'Healthcare_Access', 'Insurance_Status']

cross_per_page = 4
num_cross_pages = (len(cross_cols) + cross_per_page - 1) // cross_per_page

for page in range(num_cross_pages):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    start_idx = page * cross_per_page
    end_idx = min(start_idx + cross_per_page, len(cross_cols))
    
    for i, col in enumerate(cross_cols[start_idx:end_idx]):
        ax = axes[i]
        ct = pd.crosstab(df_analysis[col], df_analysis[target], normalize='index') * 100
        ct.plot(kind='barh', stacked=True, ax=ax, colormap='RdYlGn', width=0.85)
        ax.set_title(f'{target} vs {col}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Persentase (%)')
        ax.set_ylabel(col)
        ax.legend(loc='lower right', title=target)
        
        # Tambahkan persentase pada bar
        for j, (idx, row) in enumerate(ct.iterrows()):
            cumsum = 0
            for k, val in enumerate(row):
                if val > 5:  # Hanya tampilkan jika > 5%
                    ax.text(cumsum + val/2, j, f'{val:.1f}%',
                            ha='center', va='center', fontsize=7, fontweight='bold', color='white')
                cumsum += val
    
    for j in range(end_idx - start_idx, 4):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_categorical_vs_survival_p{page+1}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  => Saved: {output_dir}/04_categorical_vs_survival_p{page+1}.png")

# Chi-square test untuk hubungan kategorikal
print("\n--- Chi-Square Test (Categorical vs Survival_5_years) ---")
from scipy.stats import chi2_contingency

chi_results = []
for col in cross_cols:
    ct = pd.crosstab(df_analysis[col], df_analysis[target])
    chi2, p, dof, expected = chi2_contingency(ct)
    chi_results.append({'Variable': col, 'Chi2': round(chi2, 2), 'p-value': p, 'Signifikan': 'Ya' if p < 0.05 else 'Tidak'})

chi_df = pd.DataFrame(chi_results)
print(chi_df.to_string(index=False))

# ============================================================
# 7. KORELASI MATRIKS (NUMERIC)
# ============================================================
print("\n" + "=" * 70)
print("7. KORELASI MATRIKS")
print("=" * 70)

fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = df_analysis[num_plot_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', 
            center=0, vmin=-1, vmax=1, mask=mask,
            square=True, linewidths=0.5, ax=ax,
            annot_kws={'fontsize': 10})
ax.set_title('Korelasi Antar Variabel Numerik', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/05_correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  => Saved: {output_dir}/05_correlation_matrix.png")

print("\n--- Correlation Values ---")
print(corr_matrix.round(4).to_string())

# ============================================================
# 8. CANCER STAGE & TREATMENT ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("8. CANCER STAGE & TREATMENT ANALYSIS")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Cancer Stage vs Survival
ax = axes[0]
ct_stage = pd.crosstab(df_analysis['Cancer_Stage'], df_analysis[target], normalize='index') * 100
ct_stage.plot(kind='bar', ax=ax, colormap='RdYlGn', width=0.8)
ax.set_title('Survival Rate by Cancer Stage', fontsize=12, fontweight='bold')
ax.set_xlabel('Cancer Stage')
ax.set_ylabel('Persentase (%)')
ax.legend(title=target)
ax.tick_params(axis='x', rotation=0)

# Cancer Stage vs Treatment Type
ax = axes[1]
ct_treat = pd.crosstab(df_analysis['Treatment_Type'], df_analysis['Cancer_Stage'], normalize='index') * 100
ct_treat.plot(kind='bar', ax=ax, colormap='Set2', width=0.8)
ax.set_title('Treatment Distribution by Cancer Stage', fontsize=12, fontweight='bold')
ax.set_xlabel('Treatment Type')
ax.set_ylabel('Persentase (%)')
ax.legend(title='Cancer Stage')
ax.tick_params(axis='x', rotation=15)

# Tumor Size by Cancer Stage
ax = axes[2]
sns.boxplot(data=df_analysis, x='Cancer_Stage', y='Tumor_Size_mm', 
            palette='Set2', ax=ax, order=sorted(df_analysis['Cancer_Stage'].unique()))
ax.set_title('Tumor Size by Cancer Stage', fontsize=12, fontweight='bold')
ax.set_xlabel('Cancer Stage')
ax.set_ylabel('Tumor Size (mm)')

plt.tight_layout()
plt.savefig(f'{output_dir}/06_stage_treatment_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  => Saved: {output_dir}/06_stage_treatment_analysis.png")

# ============================================================
# 9. RISK FACTORS ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("9. RISK FACTORS ANALYSIS")
print("=" * 70)

risk_factors = ['Smoking_History', 'Alcohol_Consumption', 'Obesity_BMI', 
                'Diabetes', 'Genetic_Mutation', 'Family_History',
                'Diet_Risk', 'Physical_Activity', 'Inflammatory_Bowel_Disease']

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.flatten()

for i, factor in enumerate(risk_factors):
    ax = axes[i]
    ct = pd.crosstab(df_analysis[factor], df_analysis[target], normalize='index') * 100
    ct.plot(kind='barh', stacked=True, ax=ax, colormap='RdYlGn', width=0.8)
    ax.set_title(f'{target} by {factor}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Persentase (%)')
    ax.set_ylabel(factor)
    ax.legend(loc='lower right', title=target, fontsize=7)

plt.tight_layout()
plt.savefig(f'{output_dir}/07_risk_factors_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  => Saved: {output_dir}/07_risk_factors_analysis.png")

# ============================================================
# 10. DEMOGRAPHIC ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("10. DEMOGRAPHIC ANALYSIS")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# Age distribution by Gender
ax = axes[0]
sns.histplot(data=df_analysis, x='Age', hue='Gender', kde=True, 
             bins=50, palette='Set2', alpha=0.6, ax=ax)
ax.set_title('Age Distribution by Gender', fontsize=12, fontweight='bold')

# Age by Survival
ax = axes[1]
sns.boxplot(data=df_analysis, x=target, y='Age', palette='Set2', ax=ax)
ax.set_title('Age Distribution by Survival', fontsize=12, fontweight='bold')

# Country distribution
ax = axes[2]
country_counts = df_analysis['Country'].value_counts().head(15)
colors = plt.cm.Set3(np.linspace(0, 1, len(country_counts)))
ax.barh(range(len(country_counts)), country_counts.values, color=colors)
ax.set_yticks(range(len(country_counts)))
ax.set_yticklabels(country_counts.index, fontsize=8)
ax.set_title('Top 15 Countries', fontsize=12, fontweight='bold')
ax.set_xlabel('Count')

# Survival by Country (Top 15)
ax = axes[3]
top_countries = df_analysis['Country'].value_counts().head(15).index
ct_country = pd.crosstab(df_analysis['Country'], df_analysis[target], normalize='index') * 100
ct_country = ct_country.loc[ct_country.index.isin(top_countries)]
ct_country.plot(kind='barh', stacked=True, ax=ax, colormap='RdYlGn', width=0.85)
ax.set_title('Survival Rate by Country (Top 15)', fontsize=11, fontweight='bold')
ax.set_xlabel('Persentase (%)')
ax.legend(loc='lower right', title=target)

# Economic Classification
ax = axes[4]
ct_econ = pd.crosstab(df_analysis['Economic_Classification'], df_analysis[target], normalize='index') * 100
ct_econ.plot(kind='barh', stacked=True, ax=ax, colormap='RdYlGn', width=0.8)
ax.set_title(f'{target} by Economic Classification', fontsize=11, fontweight='bold')
ax.set_xlabel('Persentase (%)')
ax.legend(loc='lower right', title=target)

# Urban vs Rural
ax = axes[5]
ct_urban = pd.crosstab(df_analysis['Urban_or_Rural'], df_analysis[target], normalize='index') * 100
ct_urban.plot(kind='barh', stacked=True, ax=ax, colormap='RdYlGn', width=0.8)
ax.set_title(f'{target} by Urban/Rural', fontsize=11, fontweight='bold')
ax.set_xlabel('Persentase (%)')
ax.legend(loc='lower right', title=target)

plt.tight_layout()
plt.savefig(f'{output_dir}/08_demographic_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  => Saved: {output_dir}/08_demographic_analysis.png")

# ============================================================
# 11. MORTALITY ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("11. MORTALITY ANALYSIS")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Mortality Rate per 100K by Economic Classification
ax = axes[0]
sns.boxplot(data=df_analysis, x='Economic_Classification', y='Mortality_Rate_per_100K',
            palette='Set2', ax=ax, order=sorted(df_analysis['Economic_Classification'].unique()))
ax.set_title('Mortality Rate by Economic Classification', fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=15)

# Incidence vs Mortality Rate
ax = axes[1]
sns.scatterplot(data=df_analysis.sample(min(10000, len(df_analysis))), 
                x='Incidence_Rate_per_100K', y='Mortality_Rate_per_100K',
                hue='Economic_Classification', alpha=0.5, palette='Set2', ax=ax)
ax.set_title('Incidence vs Mortality Rate', fontsize=12, fontweight='bold')

# Healthcare Costs Analysis
ax = axes[2]
sns.boxplot(data=df_analysis, x=target, y='Healthcare_Costs', palette='Set2', ax=ax)
ax.set_title('Healthcare Costs by Survival', fontsize=12, fontweight='bold')
ax.set_ylabel('Healthcare Costs ($)')

plt.tight_layout()
plt.savefig(f'{output_dir}/09_mortality_cost_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  => Saved: {output_dir}/09_mortality_cost_analysis.png")

# ============================================================
# 12. HEALTHCARE ACCESS & INSURANCE
# ============================================================
print("\n" + "=" * 70)
print("12. HEALTHCARE ACCESS & INSURANCE ANALYSIS")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

ax = axes[0]
ct_access = pd.crosstab(df_analysis['Healthcare_Access'], df_analysis[target], normalize='index') * 100
ct_access.plot(kind='barh', stacked=True, ax=ax, colormap='RdYlGn', width=0.8)
ax.set_title(f'{target} by Healthcare Access', fontsize=12, fontweight='bold')
ax.set_xlabel('Persentase (%)')
ax.legend(loc='lower right', title=target)

ax = axes[1]
ct_ins = pd.crosstab(df_analysis['Insurance_Status'], df_analysis[target], normalize='index') * 100
ct_ins.plot(kind='barh', stacked=True, ax=ax, colormap='RdYlGn', width=0.8)
ax.set_title(f'{target} by Insurance Status', fontsize=12, fontweight='bold')
ax.set_xlabel('Persentase (%)')
ax.legend(loc='lower right', title=target)

ax = axes[2]
ct_ei = pd.crosstab([df_analysis['Healthcare_Access'], df_analysis['Insurance_Status']], 
                     df_analysis['Early_Detection'], normalize='index') * 100
ct_ei_simple = ct_ei.groupby(level=0).first()
ct_ei_simple.plot(kind='barh', stacked=True, ax=ax, colormap='Set2', width=0.8)
ax.set_title('Early Detection by Access & Insurance', fontsize=11, fontweight='bold')
ax.set_xlabel('Persentase (%)')
ax.legend(title='Early Detection', loc='lower right')

plt.tight_layout()
plt.savefig(f'{output_dir}/10_healthcare_access_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  => Saved: {output_dir}/10_healthcare_access_analysis.png")

# ============================================================
# 13. EARLY DETECTION & SCREENING
# ============================================================
print("\n" + "=" * 70)
print("13. EARLY DETECTION & SCREENING ANALYSIS")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

ax = axes[0]
ct_ed = pd.crosstab(df_analysis['Early_Detection'], df_analysis[target], normalize='index') * 100
ct_ed.plot(kind='barh', stacked=True, ax=ax, colormap='RdYlGn', width=0.8)
ax.set_title(f'{target} by Early Detection', fontsize=12, fontweight='bold')
ax.set_xlabel('Persentase (%)')
ax.legend(loc='lower right', title=target)

ax = axes[1]
ct_sc = pd.crosstab(df_analysis['Screening_History'], df_analysis['Cancer_Stage'], normalize='index') * 100
ct_sc.plot(kind='barh', stacked=True, ax=ax, colormap='Set2', width=0.8)
ax.set_title('Cancer Stage by Screening History', fontsize=12, fontweight='bold')
ax.set_xlabel('Persentase (%)')
ax.legend(title='Cancer Stage', loc='lower right')

ax = axes[2]
ct_sd = pd.crosstab(df_analysis['Screening_History'], df_analysis[target], normalize='index') * 100
ct_sd.plot(kind='barh', stacked=True, ax=ax, colormap='RdYlGn', width=0.8)
ax.set_title(f'{target} by Screening History', fontsize=12, fontweight='bold')
ax.set_xlabel('Persentase (%)')
ax.legend(loc='lower right', title=target)

plt.tight_layout()
plt.savefig(f'{output_dir}/11_early_detection_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  => Saved: {output_dir}/11_early_detection_analysis.png")

# ============================================================
# 14. SURVIVAL PREDICTION TARGET ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("14. SURVIVAL PREDICTION TARGET ANALYSIS")
print("=" * 70)

# Survival Prediction breakdown
print("\n--- Survival Prediction Distribution ---")
print(df_analysis['Survival_Prediction'].value_counts())
print("\n--- Survival Prediction vs Actual Survival ---")
ct_pred = pd.crosstab(df_analysis['Survival_Prediction'], df_analysis['Survival_5_years'])
print(ct_pred)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ct_pred_pct = pd.crosstab(df_analysis['Survival_Prediction'], df_analysis[target], normalize='index') * 100
ct_pred_pct.plot(kind='barh', stacked=True, ax=ax, colormap='RdYlGn', width=0.8)
ax.set_title('Actual Survival by Prediction', fontsize=12, fontweight='bold')
ax.set_xlabel('Persentase (%)')
ax.legend(title=target)

ax = axes[1]
df_analysis['Survival_Prediction'].value_counts().plot(kind='pie', autopct='%1.1f%%', 
                                                        colors=plt.cm.Set3(np.linspace(0, 1, 5)),
                                                        ax=ax, explode=[0.05]*5)
ax.set_title('Survival Prediction Distribution', fontsize=12, fontweight='bold')
ax.set_ylabel('')

plt.tight_layout()
plt.savefig(f'{output_dir}/12_survival_prediction_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  => Saved: {output_dir}/12_survival_prediction_analysis.png")

# ============================================================
# RINGKASAN EDA
# ============================================================
print("\n" + "=" * 70)
print("RINGKASAN EDA - KEY INSIGHTS")
print("=" * 70)

insights = []

# Overall survival rate
survival_rate = (df_analysis[target].value_counts(normalize=True) * 100)
insights.append(f"1. Survival Rate: {survival_rate.get('Yes', 0):.2f}% survived, {survival_rate.get('No', 0):.2f}% tidak")

# Age
insights.append(f"2. Rata-rata usia pasien: {df_analysis['Age'].mean():.1f} tahun (std: {df_analysis['Age'].std():.1f})")

# Cancer Stage distribution
stage_dist = df_analysis['Cancer_Stage'].value_counts(normalize=True) * 100
stage_str = ', '.join([f"{k}: {v:.1f}%" for k, v in stage_dist.items()])
insights.append(f"3. Distribusi Cancer Stage: {stage_str}")

# Gender distribution
gender_dist = df_analysis['Gender'].value_counts(normalize=True) * 100
gender_str = ', '.join([f"{k}: {v:.1f}%" for k, v in gender_dist.items()])
insights.append(f"4. Distribusi Gender: {gender_str}")

# Top risk factors (those with lowest survival)
for factor in risk_factors[:5]:
    ct = pd.crosstab(df_analysis[factor], df_analysis[target], normalize='index') * 100
    if 'Yes' in ct.columns:
        worst = ct['Yes'].idxmin()
        best = ct['Yes'].idxmax()
        insights.append(f"5. {factor}: Survival tertinggi pada '{best}', terendah pada '{worst}'")

# Early detection impact
ed_ct = pd.crosstab(df_analysis['Early_Detection'], df_analysis[target], normalize='index') * 100
if 'Yes' in ed_ct.columns:
    insights.append(f"6. Early Detection sangat {'meningkatkan' if ed_ct.loc['Yes', 'Yes'] > ed_ct.loc['No', 'Yes'] else 'mempengaruhi'} survival (Yes: {ed_ct.loc['Yes', 'Yes']:.1f}% vs No: {ed_ct.loc['No', 'Yes']:.1f}%)")

# Economic impact
econ_ct = pd.crosstab(df_analysis['Economic_Classification'], df_analysis[target], normalize='index') * 100
if 'Yes' in econ_ct.columns:
    best_econ = econ_ct['Yes'].idxmax()
    worst_econ = econ_ct['Yes'].idxmin()
    insights.append(f"7. Economic Classification: Survival tertinggi pada '{best_econ}', terendah pada '{worst_econ}'")

# Tumor size
insights.append(f"8. Rata-rata Tumor Size: {df_analysis['Tumor_Size_mm'].mean():.1f} mm")

# Healthcare costs
insights.append(f"9. Rata-rata Healthcare Costs: ${df_analysis['Healthcare_Costs'].mean():,.0f}")

# Country analysis
top_country = df_analysis['Country'].value_counts().index[0]
insights.append(f"10. Negara dengan data terbanyak: {top_country}")

print("\n".join(insights))

print("\n" + "=" * 70)
print(f"EDA SELESAI! Semua output tersimpan di folder: '{output_dir}/'")
print("=" * 70)
print("\nGenerated plots:")
for f in sorted(os.listdir(output_dir)):
    print(f"  - {f}")
