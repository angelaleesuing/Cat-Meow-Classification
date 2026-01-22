# ====================================================================
# PROJECT PROGRESS 4: MODEL DEVELOPMENT
# Cat Sound Classification - Bioinformatics Project
# ====================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

# ====================================================================
# LOAD DATA
# ====================================================================

print("="*70)
print("MODEL DEVELOPMENT")
print("="*70)

features = np.load('cat_features.npy')
metadata = pd.read_csv('cat_metadata.csv')

print(f"\n✓ Loaded data: {features.shape[0]} samples, {features.shape[1]} features")

# Prepare target variable
le_context = LabelEncoder()
y_context = le_context.fit_transform(metadata['context_label'])

print(f"✓ Target classes: {le_context.classes_}")

# ====================================================================
# 1. DATA SPLITTING
# ====================================================================

print("\n" + "="*70)
print("1. TRAIN-TEST SPLIT")
print("="*70)

X_train, X_test, y_train, y_test = train_test_split(
    features, y_context, test_size=0.2, random_state=42, stratify=y_context
)

print(f"\n📊 Data Split:")
print(f"   • Training set: {X_train.shape[0]} samples")
print(f"   • Testing set: {X_test.shape[0]} samples")
print(f"   • Train proportion: {X_train.shape[0]/features.shape[0]*100:.1f}%")

# Check class distribution
print(f"\n📊 Class Distribution:")
for i, class_name in enumerate(le_context.classes_):
    train_count = np.sum(y_train == i)
    test_count = np.sum(y_test == i)
    print(f"   • {class_name}: Train={train_count}, Test={test_count}")

# ====================================================================
# 2. SIMPLE LINEAR REGRESSION (for demonstration)
# ====================================================================

print("\n" + "="*70)
print("2. SIMPLE LINEAR REGRESSION")
print("="*70)

# Use first MFCC feature to predict second MFCC (just for demonstration)
X_simple = X_train[:, 0].reshape(-1, 1)  # MFCC_1
y_simple = X_train[:, 1]  # MFCC_2

X_simple_test = X_test[:, 0].reshape(-1, 1)
y_simple_test = X_test[:, 1]

# Fit simple linear regression
simple_lr = LinearRegression()
simple_lr.fit(X_simple, y_simple)

# Predictions
y_pred_simple = simple_lr.predict(X_simple_test)

# Metrics
mse_simple = mean_squared_error(y_simple_test, y_pred_simple)
r2_simple = r2_score(y_simple_test, y_pred_simple)

print(f"\n📊 Simple Linear Regression (MFCC_1 → MFCC_2):")
print(f"   • Coefficient: {simple_lr.coef_[0]:.4f}")
print(f"   • Intercept: {simple_lr.intercept_:.4f}")
print(f"   • R² Score: {r2_simple:.4f}")
print(f"   • MSE: {mse_simple:.4f}")

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X_simple_test, y_simple_test, alpha=0.5, label='Actual', s=50)
plt.plot(X_simple_test, y_pred_simple, 'r-', linewidth=2, label='Predicted')
plt.xlabel('MFCC_1 (Feature 1)', fontsize=12)
plt.ylabel('MFCC_2 (Feature 2)', fontsize=12)
plt.title('Simple Linear Regression: Feature Relationship', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('model_simple_regression.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: model_simple_regression.png")
plt.show()

# ====================================================================
# 3. MULTIPLE LINEAR REGRESSION
# ====================================================================

print("\n" + "="*70)
print("3. MULTIPLE LINEAR REGRESSION")
print("="*70)

# Use all features to predict a continuous target
# For demo: predict MFCC_13 from other MFCCs
X_multi = X_train[:, :12]  # First 12 MFCCs
y_multi = X_train[:, 12]   # 13th MFCC

X_multi_test = X_test[:, :12]
y_multi_test = X_test[:, 12]

# Fit multiple linear regression
multi_lr = LinearRegression()
multi_lr.fit(X_multi, y_multi)

# Predictions
y_pred_multi = multi_lr.predict(X_multi_test)

# Metrics
mse_multi = mean_squared_error(y_multi_test, y_pred_multi)
r2_multi = r2_score(y_multi_test, y_pred_multi)

print(f"\n📊 Multiple Linear Regression (MFCC_1-12 → MFCC_13):")
print(f"   • R² Score: {r2_multi:.4f}")
print(f"   • MSE: {mse_multi:.4f}")
print(f"   • RMSE: {np.sqrt(mse_multi):.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': [f'MFCC_{i+1}' for i in range(12)],
    'Coefficient': multi_lr.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print(f"\n📊 Top 5 Important Features:")
print(feature_importance.head().to_string(index=False))

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(y_multi_test, y_pred_multi, alpha=0.6, s=50)
plt.plot([y_multi_test.min(), y_multi_test.max()], 
         [y_multi_test.min(), y_multi_test.max()], 
         'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual MFCC_13', fontsize=12)
plt.ylabel('Predicted MFCC_13', fontsize=12)
plt.title('Multiple Linear Regression: Actual vs Predicted', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('model_multiple_regression.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: model_multiple_regression.png")
plt.show()

# ====================================================================
# 4. POLYNOMIAL REGRESSION & PIPELINES
# ====================================================================

print("\n" + "="*70)
print("4. POLYNOMIAL REGRESSION")
print("="*70)

# Create polynomial features pipeline
degrees = [1, 2, 3]
poly_results = []

for degree in degrees:
    # Create pipeline
    poly_pipeline = Pipeline([
        ('poly_features', PolynomialFeatures(degree=degree)),
        ('linear_regression', LinearRegression())
    ])
    
    # Fit
    poly_pipeline.fit(X_simple, y_simple)
    
    # Predict
    y_pred_poly = poly_pipeline.predict(X_simple_test)
    
    # Metrics
    mse_poly = mean_squared_error(y_simple_test, y_pred_poly)
    r2_poly = r2_score(y_simple_test, y_pred_poly)
    
    poly_results.append({
        'Degree': degree,
        'R²': r2_poly,
        'MSE': mse_poly
    })
    
    print(f"\n📊 Polynomial Degree {degree}:")
    print(f"   • R² Score: {r2_poly:.4f}")
    print(f"   • MSE: {mse_poly:.4f}")

# Compare polynomial models
poly_df = pd.DataFrame(poly_results)
print(f"\n📊 Polynomial Regression Comparison:")
print(poly_df.to_string(index=False))

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, degree in enumerate(degrees):
    poly_pipeline = Pipeline([
        ('poly_features', PolynomialFeatures(degree=degree)),
        ('linear_regression', LinearRegression())
    ])
    poly_pipeline.fit(X_simple, y_simple)
    
    X_plot = np.linspace(X_simple.min(), X_simple.max(), 100).reshape(-1, 1)
    y_plot = poly_pipeline.predict(X_plot)
    
    axes[idx].scatter(X_simple_test, y_simple_test, alpha=0.5, s=30, label='Actual')
    axes[idx].plot(X_plot, y_plot, 'r-', linewidth=2, label=f'Degree {degree}')
    axes[idx].set_xlabel('MFCC_1')
    axes[idx].set_ylabel('MFCC_2')
    axes[idx].set_title(f'Polynomial Degree {degree}\n(R²={poly_df.iloc[idx]["R²"]:.3f})', 
                        fontweight='bold')
    axes[idx].legend()
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('model_polynomial_regression.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: model_polynomial_regression.png")
plt.show()

# ====================================================================
# 5. CLASSIFICATION MODEL (Random Forest)
# ====================================================================

print("\n" + "="*70)
print("5. RANDOM FOREST CLASSIFICATION (Main Model)")
print("="*70)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_train = rf_model.predict(X_train_scaled)
y_pred_test = rf_model.predict(X_test_scaled)

# Accuracy scores
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

print(f"\n📊 Random Forest Performance:")
print(f"   • Training Accuracy: {train_acc*100:.2f}%")
print(f"   • Testing Accuracy: {test_acc*100:.2f}%")
print(f"   • Overfitting Gap: {(train_acc - test_acc)*100:.2f}%")

# Feature importance
feature_names = [f'Feature_{i+1}' for i in range(features.shape[1])]
feature_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\n📊 Top 10 Most Important Features:")
print(feature_imp_df.head(10).to_string(index=False))

# Visualization - Feature Importance
plt.figure(figsize=(12, 8))
top_features = feature_imp_df.head(15)
plt.barh(top_features['Feature'], top_features['Importance'], color='steelblue')
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Top 15 Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('model_feature_importance.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: model_feature_importance.png")
plt.show()

# ====================================================================
# 6. IN-SAMPLE EVALUATION (R² and MSE)
# ====================================================================

print("\n" + "="*70)
print("6. IN-SAMPLE EVALUATION")
print("="*70)

# For regression metrics, we'll use probability predictions
y_prob_train = rf_model.predict_proba(X_train_scaled)
y_prob_test = rf_model.predict_proba(X_test_scaled)

# Calculate R² for each class (one-vs-rest approach)
from sklearn.preprocessing import label_binarize

y_train_bin = label_binarize(y_train, classes=[0, 1, 2])
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

r2_scores = []
mse_scores = []

for i, class_name in enumerate(le_context.classes_):
    r2 = r2_score(y_test_bin[:, i], y_prob_test[:, i])
    mse = mean_squared_error(y_test_bin[:, i], y_prob_test[:, i])
    
    r2_scores.append(r2)
    mse_scores.append(mse)
    
    print(f"\n📊 {class_name}:")
    print(f"   • R² Score: {r2:.4f}")
    print(f"   • MSE: {mse:.4f}")

print(f"\n📊 Overall Metrics:")
print(f"   • Average R²: {np.mean(r2_scores):.4f}")
print(f"   • Average MSE: {np.mean(mse_scores):.4f}")

# ====================================================================
# 7. PREDICTION AND DECISION MAKING
# ====================================================================

print("\n" + "="*70)
print("7. PREDICTION AND DECISION MAKING")
print("="*70)

# Make predictions on test set
predictions = rf_model.predict(X_test_scaled)
probabilities = rf_model.predict_proba(X_test_scaled)

# Show sample predictions
print(f"\n📊 Sample Predictions (First 10):")
print(f"{'Actual':<25} {'Predicted':<25} {'Confidence':<15}")
print("-" * 65)

for i in range(min(10, len(y_test))):
    actual = le_context.classes_[y_test[i]]
    predicted = le_context.classes_[predictions[i]]
    confidence = probabilities[i].max() * 100
    
    symbol = "✓" if actual == predicted else "✗"
    print(f"{symbol} {actual:<23} {predicted:<23} {confidence:>6.2f}%")

# Decision making metrics
correct_predictions = np.sum(predictions == y_test)
print(f"\n📊 Decision Making Summary:")
print(f"   • Correct predictions: {correct_predictions}/{len(y_test)}")
print(f"   • Accuracy: {test_acc*100:.2f}%")
print(f"   • High confidence (>80%): {np.sum(probabilities.max(axis=1) > 0.8)}/{len(y_test)}")
print(f"   • Low confidence (<50%): {np.sum(probabilities.max(axis=1) < 0.5)}/{len(y_test)}")

# ====================================================================
# SAVE MODEL
# ====================================================================

print("\n" + "="*70)
print("SAVING MODEL")
print("="*70)

import joblib

# Save model and scaler
joblib.dump(rf_model, 'cat_classifier_model.pkl')
joblib.dump(scaler, 'cat_scaler.pkl')
joblib.dump(le_context, 'cat_label_encoder.pkl')

print("\n✓ Saved model files:")
print("   • cat_classifier_model.pkl")
print("   • cat_scaler.pkl")
print("   • cat_label_encoder.pkl")

print("\n" + "="*70)
print("MODEL DEVELOPMENT COMPLETE!")
print("="*70)