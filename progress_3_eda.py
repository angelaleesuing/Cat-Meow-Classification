# ====================================================================
# PROJECT PROGRESS 3: EXPLORATORY DATA ANALYSIS (EDA)
# Cat Sound Classification - Bioinformatics Project
# ====================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ====================================================================
# LOAD PROCESSED DATA
# ====================================================================

print("="*70)
print("EXPLORATORY DATA ANALYSIS")
print("="*70)

# Load features and metadata
features = np.load('cat_features.npy')
metadata = pd.read_csv('cat_metadata.csv')

print(f"\n✓ Loaded data:")
print(f"   • Features shape: {features.shape}")
print(f"   • Metadata shape: {metadata.shape}")

# Combine for analysis
df = metadata.copy()
df['features'] = list(features)

# Feature names
feature_names = (
    [f'MFCC_{i}' for i in range(1, 14)] + 
    [f'Chroma_{i}' for i in range(1, 13)] + 
    [f'Contrast_{i}' for i in range(1, 8)]
)

# Create DataFrame with individual features
features_df = pd.DataFrame(features, columns=feature_names)
full_df = pd.concat([metadata, features_df], axis=1)

# ====================================================================
# 1. DESCRIPTIVE STATISTICS
# ====================================================================

print("\n" + "="*70)
print("1. DESCRIPTIVE STATISTICS")
print("="*70)

# Overall statistics
print("\n📊 Feature Statistics:")
print(features_df.describe().round(4))

# Statistics by context
print("\n📊 Context Distribution:")
context_counts = metadata['context_label'].value_counts()
print(context_counts)
print(f"\nProportions:")
print((context_counts / len(metadata) * 100).round(2))

# Statistics by breed
print("\n📊 Breed Distribution:")
breed_counts = metadata['breed_label'].value_counts()
print(breed_counts)
print(f"\nProportions:")
print((breed_counts / len(metadata) * 100).round(2))

# Visualize distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Context distribution
axes[0].pie(context_counts, labels=context_counts.index, autopct='%1.1f%%', 
            startangle=90, colors=sns.color_palette('Set2'))
axes[0].set_title('Distribution by Emotional Context', fontsize=14, fontweight='bold')

# Breed distribution
axes[1].pie(breed_counts, labels=breed_counts.index, autopct='%1.1f%%',
            startangle=90, colors=sns.color_palette('Set3'))
axes[1].set_title('Distribution by Breed', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('eda_distributions.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: eda_distributions.png")
plt.show()

# ====================================================================
# 2. FEATURE DISTRIBUTIONS
# ====================================================================

print("\n" + "="*70)
print("2. FEATURE DISTRIBUTIONS")
print("="*70)

# Plot histograms for key features
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()

key_features = ['MFCC_1', 'MFCC_2', 'Chroma_1', 'Chroma_2', 'Contrast_1', 'Contrast_2']

for i, feat in enumerate(key_features):
    axes[i].hist(features_df[feat], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[i].set_title(f'{feat} Distribution', fontweight='bold')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
    axes[i].axvline(features_df[feat].mean(), color='red', linestyle='--', 
                    label=f'Mean: {features_df[feat].mean():.2f}')
    axes[i].legend()

plt.tight_layout()
plt.savefig('eda_feature_histograms.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: eda_feature_histograms.png")
plt.show()

# ====================================================================
# 3. GROUPING AND COMPARISON
# ====================================================================

print("\n" + "="*70)
print("3. BASIC GROUPING ANALYSIS")
print("="*70)

# Group by context
grouped_context = full_df.groupby('context_label')[feature_names].mean()
print("\n📊 Mean Features by Context:")
print(grouped_context.round(4))

# Group by breed
grouped_breed = full_df.groupby('breed_label')[feature_names].mean()
print("\n📊 Mean Features by Breed:")
print(grouped_breed.round(4))

# Visualize grouped means
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot for contexts
grouped_context[['MFCC_1', 'MFCC_2', 'MFCC_3', 'Chroma_1', 'Contrast_1']].T.plot(
    kind='bar', ax=axes[0], color=sns.color_palette('Set2', 3), width=0.8)
axes[0].set_title('Mean Feature Values by Context', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Features')
axes[0].set_ylabel('Mean Value')
axes[0].legend(title='Context', bbox_to_anchor=(1.05, 1))
axes[0].grid(axis='y', alpha=0.3)

# Plot for breeds
grouped_breed[['MFCC_1', 'MFCC_2', 'MFCC_3', 'Chroma_1', 'Contrast_1']].T.plot(
    kind='bar', ax=axes[1], color=sns.color_palette('Set3', 2), width=0.8)
axes[1].set_title('Mean Feature Values by Breed', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Features')
axes[1].set_ylabel('Mean Value')
axes[1].legend(title='Breed', bbox_to_anchor=(1.05, 1))
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('eda_grouped_means.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: eda_grouped_means.png")
plt.show()

# ====================================================================
# 4. ANOVA (Analysis of Variance)
# ====================================================================

print("\n" + "="*70)
print("4. ANOVA - FEATURE SIGNIFICANCE")
print("="*70)

# Perform ANOVA for each feature across contexts
anova_results = []

contexts = full_df['context_label'].unique()

for feat in feature_names:
    groups = [full_df[full_df['context_label'] == ctx][feat].values 
              for ctx in contexts]
    
    f_stat, p_value = f_oneway(*groups)
    
    anova_results.append({
        'Feature': feat,
        'F-statistic': f_stat,
        'p-value': p_value,
        'Significant': 'Yes' if p_value < 0.05 else 'No'
    })

anova_df = pd.DataFrame(anova_results)
anova_df = anova_df.sort_values('p-value')

print("\n📊 ANOVA Results (Top 10 Most Significant Features):")
print(anova_df.head(10).to_string(index=False))

# Visualize ANOVA results
fig, ax = plt.subplots(figsize=(12, 8))
colors = ['green' if sig == 'Yes' else 'red' for sig in anova_df['Significant']]

ax.barh(anova_df['Feature'], -np.log10(anova_df['p-value']), color=colors, alpha=0.7)
ax.axvline(-np.log10(0.05), color='blue', linestyle='--', 
           label='Significance threshold (p=0.05)')
ax.set_xlabel('-log10(p-value)', fontsize=12)
ax.set_ylabel('Features', fontsize=12)
ax.set_title('Feature Significance Across Contexts (ANOVA)', 
             fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('eda_anova_results.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: eda_anova_results.png")
plt.show()

# ====================================================================
# 5. CORRELATION ANALYSIS
# ====================================================================

print("\n" + "="*70)
print("5. CORRELATION ANALYSIS")
print("="*70)

# Calculate correlation matrix
correlation_matrix = features_df.corr()

print("\n📊 Correlation Matrix (sample):")
print(correlation_matrix.iloc[:5, :5].round(3))

# Find highly correlated features
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.7:
            high_corr_pairs.append({
                'Feature 1': correlation_matrix.columns[i],
                'Feature 2': correlation_matrix.columns[j],
                'Correlation': correlation_matrix.iloc[i, j]
            })

if high_corr_pairs:
    print("\n⚠️  Highly Correlated Feature Pairs (|r| > 0.7):")
    print(pd.DataFrame(high_corr_pairs).to_string(index=False))
else:
    print("\n✓ No highly correlated features found (|r| > 0.7)")

# Visualize correlation heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            annot=False)
plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('eda_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: eda_correlation_heatmap.png")
plt.show()

# ====================================================================
# 6. FEATURE COMPARISON BY CONTEXT (BOX PLOTS)
# ====================================================================

print("\n" + "="*70)
print("6. FEATURE COMPARISON BY CONTEXT")
print("="*70)

# Select top 4 significant features from ANOVA
top_features = anova_df.head(4)['Feature'].tolist()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for i, feat in enumerate(top_features):
    sns.boxplot(data=full_df, x='context_label', y=feat, 
                palette='Set2', ax=axes[i])
    axes[i].set_title(f'{feat} by Context', fontweight='bold')
    axes[i].set_xlabel('Context')
    axes[i].set_ylabel('Value')
    axes[i].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('eda_context_boxplots.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: eda_context_boxplots.png")
plt.show()

# ====================================================================
# SUMMARY REPORT
# ====================================================================

print("\n" + "="*70)
print("EDA SUMMARY REPORT")
print("="*70)

print(f"""
📋 Dataset Overview:
   • Total samples: {len(full_df)}
   • Total features: {len(feature_names)}
   • Contexts: {full_df['context_label'].nunique()} ({', '.join(contexts)})
   • Breeds: {full_df['breed_label'].nunique()}

📊 Key Findings:
   1. Class Balance:
      - Most common context: {context_counts.idxmax()} ({context_counts.max()} samples)
      - Least common context: {context_counts.idxmin()} ({context_counts.min()} samples)
   
   2. Feature Significance:
      - Significant features (p < 0.05): {(anova_df['p-value'] < 0.05).sum()}/{len(feature_names)}
      - Most discriminative: {anova_df.iloc[0]['Feature']}
   
   3. Feature Relationships:
      - Highly correlated pairs: {len(high_corr_pairs)}
   
   4. Data Quality:
      - Missing values: {full_df.isnull().sum().sum()}
      - Feature range: [{features_df.min().min():.4f}, {features_df.max().max():.4f}]

✅ Ready for Model Development!
""")

print("\n" + "="*70)
print("EDA COMPLETE - Generated 5 visualization files")
print("="*70)