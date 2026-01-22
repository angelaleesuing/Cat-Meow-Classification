# ====================================================================
# PROJECT PROGRESS 5: MODEL EVALUATION & REFINEMENT
# Cat Sound Classification - Bioinformatics Project
# ====================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, GridSearchCV, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize
import joblib
import warnings
warnings.filterwarnings('ignore')

# ====================================================================
# LOAD DATA AND MODEL
# ====================================================================

print("="*70)
print("MODEL EVALUATION & REFINEMENT")
print("="*70)

# Load data
features = np.load('cat_features.npy')
metadata = pd.read_csv('cat_metadata.csv')

# Load trained model
rf_model = joblib.load('cat_classifier_model.pkl')
scaler = joblib.load('cat_scaler.pkl')
le_context = joblib.load('cat_label_encoder.pkl')

print(f"\n✓ Loaded model and data")
print(f"   • Dataset: {features.shape[0]} samples")
print(f"   • Model: Random Forest with {rf_model.n_estimators} trees")

# Prepare data
from sklearn.model_selection import train_test_split

y_context = le_context.transform(metadata['context_label'])
X_train, X_test, y_train, y_test = train_test_split(
    features, y_context, test_size=0.2, random_state=42, stratify=y_context
)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ====================================================================
# 1. COMPREHENSIVE MODEL EVALUATION
# ====================================================================

print("\n" + "="*70)
print("1. MODEL EVALUATION METRICS")
print("="*70)

# Predictions
y_pred = rf_model.predict(X_test_scaled)
y_pred_proba = rf_model.predict_proba(X_test_scaled)

# Classification Report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=le_context.classes_,
                          digits=4))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le_context.classes_,
            yticklabels=le_context.classes_,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Actual Class', fontsize=12)
plt.xlabel('Predicted Class', fontsize=12)
plt.tight_layout()
plt.savefig('eval_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: eval_confusion_matrix.png")
plt.show()

# Calculate per-class metrics
print("\n📊 Per-Class Performance:")
for i, class_name in enumerate(le_context.classes_):
    class_mask = (y_test == i)
    class_acc = accuracy_score(y_test[class_mask], y_pred[class_mask])
    print(f"   • {class_name}: {class_acc*100:.2f}% accuracy")

# ====================================================================
# 2. CROSS-VALIDATION
# ====================================================================

print("\n" + "="*70)
print("2. CROSS-VALIDATION (K=5)")
print("="*70)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(rf_model, scaler.transform(features), y_context, 
                            cv=5, scoring='accuracy', n_jobs=-1)

print(f"\n📊 Cross-Validation Results:")
print(f"   • Fold Scores: {[f'{score*100:.2f}%' for score in cv_scores]}")
print(f"   • Mean Accuracy: {cv_scores.mean()*100:.2f}% (±{cv_scores.std()*100:.2f}%)")
print(f"   • Min Score: {cv_scores.min()*100:.2f}%")
print(f"   • Max Score: {cv_scores.max()*100:.2f}%")

# Visualize CV scores
plt.figure(figsize=(10, 6))
plt.bar(range(1, 6), cv_scores * 100, color='steelblue', alpha=0.7, edgecolor='black')
plt.axhline(cv_scores.mean() * 100, color='red', linestyle='--', 
            label=f'Mean: {cv_scores.mean()*100:.2f}%', linewidth=2)
plt.xlabel('Fold', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('5-Fold Cross-Validation Scores', fontsize=14, fontweight='bold')
plt.ylim(0, 100)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('eval_cross_validation.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: eval_cross_validation.png")
plt.show()

# ====================================================================
# 3. OVER-FITTING AND UNDER-FITTING ANALYSIS
# ====================================================================

print("\n" + "="*70)
print("3. OVER-FITTING / UNDER-FITTING ANALYSIS")
print("="*70)

# Calculate training and test accuracy
train_acc = rf_model.score(X_train_scaled, y_train)
test_acc = rf_model.score(X_test_scaled, y_test)

print(f"\n📊 Training vs Testing Performance:")
print(f"   • Training Accuracy: {train_acc*100:.2f}%")
print(f"   • Testing Accuracy: {test_acc*100:.2f}%")
print(f"   • Gap: {(train_acc - test_acc)*100:.2f}%")

# Interpretation
if train_acc - test_acc > 0.1:
    print(f"\n⚠️  Model shows signs of OVERFITTING")
    print(f"   → Consider regularization or reducing model complexity")
elif test_acc < 0.7:
    print(f"\n⚠️  Model shows signs of UNDERFITTING")
    print(f"   → Consider increasing model complexity or more features")
else:
    print(f"\n✓ Model shows good generalization")

# Learning Curves
print(f"\n📊 Generating Learning Curves...")

train_sizes, train_scores, val_scores = learning_curve(
    rf_model, scaler.transform(features), y_context,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='accuracy', n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                 alpha=0.2, color='blue')
plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                 alpha=0.2, color='red')
plt.xlabel('Training Set Size', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Learning Curves', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('eval_learning_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved: eval_learning_curves.png")
plt.show()

# ====================================================================
# 4. RIDGE REGRESSION (for comparison)
# ====================================================================

print("\n" + "="*70)
print("4. RIDGE CLASSIFIER (Regularization)")
print("="*70)

# Train Ridge Classifier with different alpha values
alphas = [0.1, 1.0, 10.0, 100.0]
ridge_results = []

for alpha in alphas:
    ridge = RidgeClassifier(alpha=alpha, random_state=42)
    ridge.fit(X_train_scaled, y_train)
    
    train_score = ridge.score(X_train_scaled, y_train)
    test_score = ridge.score(X_test_scaled, y_test)
    
    ridge_results.append({
        'Alpha': alpha,
        'Train Accuracy': train_score,
        'Test Accuracy': test_score,
        'Gap': train_score - test_score
    })
    
    print(f"\n📊 Ridge (alpha={alpha}):")
    print(f"   • Train: {train_score*100:.2f}%")
    print(f"   • Test: {test_score*100:.2f}%")
    print(f"   • Gap: {(train_score - test_score)*100:.2f}%")

ridge_df = pd.DataFrame(ridge_results)

# Compare Ridge vs Random Forest
print(f"\n📊 Model Comparison:")
print(f"   • Random Forest Test Accuracy: {test_acc*100:.2f}%")
print(f"   • Best Ridge Test Accuracy: {ridge_df['Test Accuracy'].max()*100:.2f}%")

# Visualize Ridge results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Alpha vs Accuracy
axes[0].plot(ridge_df['Alpha'], ridge_df['Train Accuracy'] * 100, 
             'o-', label='Train', linewidth=2, markersize=8)
axes[0].plot(ridge_df['Alpha'], ridge_df['Test Accuracy'] * 100, 
             's-', label='Test', linewidth=2, markersize=8)
axes[0].set_xscale('log')
axes[0].set_xlabel('Alpha (Regularization)', fontsize=12)
axes[0].set_ylabel('Accuracy (%)', fontsize=12)
axes[0].set_title('Ridge Classifier: Alpha vs Accuracy', fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Overfitting gap
axes[1].bar(ridge_df['Alpha'].astype(str), ridge_df['Gap'] * 100, 
            color='coral', alpha=0.7, edgecolor='black')
axes[1].axhline(0, color='black', linestyle='-', linewidth=0.8)
axes[1].set_xlabel('Alpha', fontsize=12)
axes[1].set_ylabel('Train-Test Gap (%)', fontsize=12)
axes[1].set_title('Ridge Classifier: Overfitting Gap', fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('eval_ridge_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: eval_ridge_comparison.png")
plt.show()

# ====================================================================
# 5. GRID SEARCH (Hyperparameter Tuning)
# ====================================================================

print("\n" + "="*70)
print("5. GRID SEARCH - HYPERPARAMETER TUNING")
print("="*70)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print(f"\n⚙️  Grid Search Configuration:")
print(f"   • Total combinations: {np.prod([len(v) for v in param_grid.values()])}")
print(f"   • Cross-validation folds: 3")

# Perform Grid Search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

print(f"\n🔍 Running Grid Search (this may take a while)...")
grid_search.fit(X_train_scaled, y_train)

print(f"\n✓ Grid Search Complete!")
print(f"\n📊 Best Parameters:")
for param, value in grid_search.best_params_.items():
    print(f"   • {param}: {value}")

print(f"\n📊 Best Cross-Validation Score: {grid_search.best_score_*100:.2f}%")

# Evaluate best model
best_model = grid_search.best_estimator_
best_train_acc = best_model.score(X_train_scaled, y_train)
best_test_acc = best_model.score(X_test_scaled, y_test)

print(f"\n📊 Best Model Performance:")
print(f"   • Training Accuracy: {best_train_acc*100:.2f}%")
print(f"   • Testing Accuracy: {best_test_acc*100:.2f}%")
print(f"   • Improvement: {(best_test_acc - test_acc)*100:.2f}%")

# Grid search results visualization
results_df = pd.DataFrame(grid_search.cv_results_)
top_results = results_df.nlargest(10, 'mean_test_score')[
    ['params', 'mean_test_score', 'std_test_score']
]

print(f"\n📊 Top 5 Configurations:")
for i, row in top_results.head().iterrows():
    print(f"\n   {i+1}. Score: {row['mean_test_score']*100:.2f}% (±{row['std_test_score']*100:.2f}%)")
    print(f"      Params: {row['params']}")

# ====================================================================
# 6. MODEL REFINEMENT SUMMARY
# ====================================================================

print("\n" + "="*70)
print("6. MODEL REFINEMENT SUMMARY")
print("="*70)

models_comparison = pd.DataFrame({
    'Model': [
        'Baseline Random Forest',
        'Ridge Classifier (best)',
        'Tuned Random Forest'
    ],
    'Test Accuracy': [
        test_acc * 100,
        ridge_df['Test Accuracy'].max() * 100,
        best_test_acc * 100
    ],
    'Train-Test Gap': [
        (train_acc - test_acc) * 100,
        ridge_df.loc[ridge_df['Test Accuracy'].idxmax(), 'Gap'] * 100,
        (best_train_acc - best_test_acc) * 100
    ]
})

print(f"\n📊 Model Comparison:")
print(models_comparison.to_string(index=False))

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy comparison
axes[0].bar(models_comparison['Model'], models_comparison['Test Accuracy'],
            color=['steelblue', 'coral', 'green'], alpha=0.7, edgecolor='black')
axes[0].set_ylabel('Test Accuracy (%)', fontsize=12)
axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylim(0, 100)
axes[0].tick_params(axis='x', rotation=15)
axes[0].grid(axis='y', alpha=0.3)

# Gap comparison
axes[1].bar(models_comparison['Model'], models_comparison['Train-Test Gap'],
            color=['steelblue', 'coral', 'green'], alpha=0.7, edgecolor='black')
axes[1].set_ylabel('Train-Test Gap (%)', fontsize=12)
axes[1].set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
axes[1].axhline(0, color='black', linestyle='-', linewidth=0.8)
axes[1].tick_params(axis='x', rotation=15)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('eval_model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: eval_model_comparison.png")
plt.show()

# ====================================================================
# 7. SAVE BEST MODEL
# ====================================================================

print("\n" + "="*70)
print("7. SAVING BEST MODEL")
print("="*70)

# Save best model
joblib.dump(best_model, 'cat_classifier_best_model.pkl')
print("\n✓ Saved: cat_classifier_best_model.pkl")

# ====================================================================
# FINAL REPORT
# ====================================================================

print("\n" + "="*70)
print("FINAL EVALUATION REPORT")
print("="*70)

print(f"""
📋 Project Summary:
   • Dataset: {len(features)} cat vocalizations
   • Classes: {len(le_context.classes_)} emotional contexts
   • Features: {features.shape[1]} acoustic features

📊 Final Model Performance:
   • Model: Random Forest (Tuned)
   • Test Accuracy: {best_test_acc*100:.2f}%
   • Cross-Validation: {cv_scores.mean()*100:.2f}% (±{cv_scores.std()*100:.2f}%)
   • Best Parameters: {grid_search.best_params_}

✅ Key Achievements:
   1. Successfully preprocessed and cleaned audio data
   2. Extracted meaningful acoustic features (MFCC, Chroma, Spectral Contrast)
   3. Achieved {best_test_acc*100:.1f}% classification accuracy
   4. Identified most important features for cat emotion detection
   5. Optimized model through hyperparameter tuning
   6. Addressed overfitting through regularization and cross-validation

📁 Generated Files:
   • Model files: cat_classifier_best_model.pkl
   • Visualizations: 6 PNG files
   • Data files: cat_features.npy, cat_metadata.csv

🎯 Next Steps:
   • Deploy model for real-time classification
   • Collect more data to improve generalization
   • Explore deep learning approaches (CNN for raw audio)
   • Implement real-time prediction system
""")

print("="*70)
print("PROJECT COMPLETE!")
print("="*70)