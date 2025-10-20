# Project 04 — Regression Baseline Model
**Difficulty:** ⭐⭐⭐ Intermediate  
**Estimated Time:** 3-4 hours  
**Prerequisites:** Chapters 01-04, Projects 01-03

---

## 📋 Description
Build regression models to predict continuous values (prices, scores, etc). Learn metrics like MSE, RMSE, R².

## 🎯 Objectives
- [ ] Preprocess data for regression
- [ ] Train Linear Regression and Random Forest Regressor
- [ ] Evaluate with MSE, RMSE, MAE, R²
- [ ] Perform cross-validation
- [ ] Interpret model coefficients

## 📊 Datasets
- **House Prices:** https://www.kaggle.com/c/house-prices-advanced-regression-techniques (RECOMMENDED)
- **Boston Housing:** Built-in scikit-learn

## 🛠️ Libraries
```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
```

## 📝 Tasks

### 1. Load and Split Data
```python
X = df.drop('target_column', axis=1)
y = df['target_column']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 2. Train Models
```python
# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
```

### 3. Evaluate
```python
y_pred = lr.predict(X_test_scaled)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
```

**Metric Guide:**
- **MAE:** Average absolute error (easy to interpret)
- **MSE:** Squared errors (penalizes large errors more)
- **RMSE:** Square root of MSE (same units as target)
- **R²:** 0 to 1, higher is better (1 = perfect fit)

### 4. Visualize Predictions
```python
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()
```

### 5. Feature Engineering
```python
# Try polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train_scaled)
```

## ✅ Success Criteria
- [ ] R² > 0.7 on test set
- [ ] RMSE interpretable and reasonable
- [ ] Residual plot shows no patterns
- [ ] At least 2 models compared

## 🎓 Bonus
- Try Ridge and Lasso regression
- Feature selection with RFE
- Plot residuals to check for patterns

**Next:** Project 05 - Clustering & PCA 🚀
