# Project 03 — Classification Baseline Model
**Difficulty:** ⭐⭐⭐ Intermediate  
**Estimated Time:** 4-5 hours  
**Prerequisites:** Chapters 01-04, Projects 01-02

---

## 📋 Project Description

Build your FIRST machine learning classification model! Learn to prepare data, train models, evaluate performance, and interpret results. This is where data science becomes machine learning!

**What you'll learn:**
- Train/test split to avoid overfitting
- Feature scaling and encoding
- Training classification models (Logistic Regression, Random Forest)
- Evaluating models (accuracy, precision, recall, ROC-AUC)
- Cross-validation for robust evaluation
- Basic hyperparameter tuning

---

## 🎯 Objectives

- [ ] Split data into train and test sets
- [ ] Scale numeric features and encode categorical features
- [ ] Train at least 2 classification models
- [ ] Evaluate models using multiple metrics
- [ ] Perform cross-validation
- [ ] Save the best model

---

## 📊 Suggested Datasets

### Option 1: Titanic (RECOMMENDED)
- **Link:** https://www.kaggle.com/c/titanic
- **Target:** Survived (0 or 1)
- **Type:** Binary classification

### Option 2: Breast Cancer Wisconsin
- **Link:** https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
- **Target:** Diagnosis (M or B)
- **Type:** Binary classification

---

## 🛠️ Tools & Libraries

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import joblib  # For saving models
```

---

## 📝 Step-by-Step Tasks

### Task 1: Load and Prepare Data

```python
# TODO: Load your cleaned data
df = pd.read_csv('cleaned_data.csv')

# TODO: Separate features (X) and target (y)
# Drop ID columns and target column
X = df.drop(['PassengerId', 'Survived'], axis=1)  # Adjust column names
y = df['Survived']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target distribution:\n{y.value_counts()}")
```

**Hints:**
- X = features (input variables)
- y = target (what you want to predict)
- Check for class imbalance in y

---

### Task 2: Handle Categorical Variables

```python
# TODO: Identify categorical columns
cat_cols = X.select_dtypes(include=['object', 'category']).columns
print("Categorical columns:", cat_cols.tolist())

# TODO: Encode categorical variables
from sklearn.preprocessing import LabelEncoder

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

print("After encoding:\n", X.head())
```

**Hints:**
- LabelEncoder for binary columns (Sex: male/female → 0/1)
- For multi-category columns, consider OneHotEncoder
- Always encode before scaling!

---

### Task 3: Train/Test Split

```python
# TODO: Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Train target distribution:\n{y_train.value_counts()}")
```

**Hints:**
- `test_size=0.2` means 20% for testing
- `random_state=42` for reproducibility
- `stratify=y` keeps class proportions same in train/test
- NEVER look at test set until final evaluation!

---

### Task 4: Scale Features

```python
# TODO: Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use same scaler, don't fit again!

print("Scaled train set shape:", X_train_scaled.shape)
```

**Hints:**
- StandardScaler: (value - mean) / std
- Fit on train set ONLY
- Transform both train and test
- Scaling helps models like Logistic Regression

---

### Task 5: Train Models

```python
# TODO: Train Logistic Regression
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
print("Logistic Regression trained!")

# TODO: Train Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_scaled, y_train)
print("Random Forest trained!")
```

**Hints:**
- Start with default hyperparameters
- `max_iter=1000` prevents convergence warnings
- `n_estimators=100` means 100 trees in forest

---

### Task 6: Make Predictions

```python
# TODO: Predict on test set
y_pred_lr = log_reg.predict(X_test_scaled)
y_pred_rf = rf_clf.predict(X_test_scaled)

# TODO: Get probability predictions for ROC-AUC
y_pred_proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]
y_pred_proba_rf = rf_clf.predict_proba(X_test_scaled)[:, 1]

print("Predictions made!")
```

---

### Task 7: Evaluate Models

```python
# TODO: Calculate metrics for Logistic Regression
print("===== Logistic Regression =====")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_lr):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_lr):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_lr):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_lr):.4f}")

# TODO: Confusion matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
print("Confusion Matrix:")
print(cm_lr)

# TODO: Repeat for Random Forest
print("\n===== Random Forest =====")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_rf):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_rf):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_rf):.4f}")

cm_rf = confusion_matrix(y_test, y_pred_rf)
print("Confusion Matrix:")
print(cm_rf)
```

**Metric Guide:**
- **Accuracy:** Overall correct predictions (use when classes balanced)
- **Precision:** Of predicted positives, how many are actually positive? (minimize false positives)
- **Recall:** Of actual positives, how many did we catch? (minimize false negatives)
- **F1-Score:** Harmonic mean of precision and recall
- **ROC-AUC:** Area under ROC curve (0.5=random, 1.0=perfect)

**Confusion Matrix:**
```
             Predicted
             0    1
Actual  0  [TN  FP]
        1  [FN  TP]
```

---

### Task 8: Cross-Validation

```python
# TODO: Perform 5-fold cross-validation
cv_scores_lr = cross_val_score(log_reg, X_train_scaled, y_train, cv=5, scoring='accuracy')
cv_scores_rf = cross_val_score(rf_clf, X_train_scaled, y_train, cv=5, scoring='accuracy')

print(f"Logistic Regression CV scores: {cv_scores_lr}")
print(f"Mean CV Accuracy: {cv_scores_lr.mean():.4f} (+/- {cv_scores_lr.std():.4f})")

print(f"\nRandom Forest CV scores: {cv_scores_rf}")
print(f"Mean CV Accuracy: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std():.4f})")
```

**Hints:**
- Cross-validation gives more robust estimate
- 5-fold means split train data into 5 parts, train on 4, validate on 1
- Lower std = more stable model

---

### Task 9: Feature Importance (Random Forest)

```python
# TODO: Get feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_clf.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 Important Features:")
print(feature_importance.head(10))

# TODO: Visualize
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'][:10], feature_importance['importance'][:10])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()
```

---

### Task 10: Save Best Model

```python
# TODO: Choose best model based on metrics
best_model = rf_clf  # Or log_reg if it performed better

# TODO: Save model
joblib.dump(best_model, 'best_classification_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("Model and scaler saved!")

# TODO: Test loading
loaded_model = joblib.load('best_classification_model.joblib')
loaded_scaler = joblib.load('scaler.joblib')
print("Model loaded successfully!")
```

---

## ✅ Success Criteria

- [ ] Data split into train/test correctly
- [ ] Categorical variables encoded
- [ ] Features scaled
- [ ] At least 2 models trained
- [ ] All metrics calculated and interpreted
- [ ] Cross-validation performed
- [ ] Best model saved

---

## 🎓 Bonus Challenges

1. Try GridSearchCV for hyperparameter tuning
2. Plot ROC curve
3. Try additional models (SVM, KNN, Decision Tree)
4. Create a pipeline combining preprocessing and modeling
5. Handle class imbalance with SMOTE or class weights

---

## 🐛 Common Errors & Solutions

### "ValueError: could not convert string to float"
**Solution:** Encode categorical variables before training.

### "Data leakage warning"
**Solution:** Fit scaler/encoder on train set only, then transform test set.

### Low accuracy on test set (overfitting)
**Solution:** Try simpler model, more data, or regularization.

---

## 📚 Resources

- scikit-learn docs: https://scikit-learn.org/stable/supervised_learning.html
- Chapter 04: scikit-learn Machine Learning
- Confusion matrix guide: https://en.wikipedia.org/wiki/Confusion_matrix

---

**Next:** Project 04 - Regression Baseline 🚀
