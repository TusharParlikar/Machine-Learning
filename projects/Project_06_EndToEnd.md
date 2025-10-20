# Project 06 — End-to-End Mini-Project
**Difficulty:** ⭐⭐⭐⭐ Advanced  
**Estimated Time:** 6-8 hours  
**Prerequisites:** All chapters and Projects 01-05

---

## 📋 Description
**The CAPSTONE project!** Bring together everything you've learned: data cleaning, visualization, modeling, and presentation. Build a complete ML pipeline from raw data to final report.

## 🎯 Objectives
- [ ] Complete EDA with visualizations
- [ ] Build preprocessing pipeline
- [ ] Train and compare multiple models
- [ ] Create comprehensive evaluation report
- [ ] Document all decisions and insights

## 📊 Choose Your Own Dataset!
Pick one that interests you:
- **Titanic** (classification): https://www.kaggle.com/c/titanic
- **House Prices** (regression): https://www.kaggle.com/c/house-prices-advanced-regression-techniques
- **Credit Card Fraud** (imbalanced): https://www.kaggle.com/mlg-ulb/creditcardfraud
- **Any Kaggle dataset** you find interesting!

---

## 📝 Full Workflow

### Phase 1: Problem Definition (30 min)
```markdown
# TODO: Answer these questions in a markdown file

1. What is the problem you're solving?
2. What is the target variable?
3. Is this classification or regression?
4. What metrics will you use?
5. What would be a "good" performance?
```

### Phase 2: Data Loading & EDA (1-2 hours)
```python
# TODO: Load data
df = pd.read_csv('your_dataset.csv')

# TODO: Basic inspection
print(df.shape)
print(df.info())
print(df.describe())
print(df.head(10))

# TODO: Check target distribution
print(df['target'].value_counts())

# TODO: Check missing values
print(df.isnull().sum())

# TODO: Create at least 5 visualizations
# - Distribution of target
# - Distribution of key features (histograms)
# - Correlation heatmap
# - Scatter plots of relationships
# - Box plots by category
```

### Phase 3: Data Cleaning (1 hour)
```python
# TODO: Handle missing values
# Document your strategy for each column

# TODO: Handle outliers
# Use box plots and z-scores

# TODO: Fix data types
# Convert to appropriate dtypes

# TODO: Feature engineering
# Create at least 3 new features
```

### Phase 4: Preprocessing Pipeline (1 hour)
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# TODO: Define numeric and categorical columns
numeric_features = [...]
categorical_features = [...]

# TODO: Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
```

### Phase 5: Model Training & Comparison (2 hours)
```python
# TODO: Train at least 3 different models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

# TODO: Compare models with cross-validation
results = {}
for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    results[name] = scores
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# TODO: Choose best model and tune hyperparameters
# Use GridSearchCV
```

### Phase 6: Final Evaluation (1 hour)
```python
# TODO: Train final model on full train set
best_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', best_model)
])
best_pipeline.fit(X_train, y_train)

# TODO: Evaluate on test set
y_pred = best_pipeline.predict(X_test)
y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]

# TODO: Calculate all metrics
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

print("Test Set Performance:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

# TODO: Plot confusion matrix
import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# TODO: Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('roc_curve.png', dpi=150)
plt.show()

# TODO: Feature importance (if tree-based model)
# TODO: Analyze misclassifications
```

### Phase 7: Documentation & Report (1-2 hours)
```markdown
# TODO: Create a comprehensive README.md with:

## Project Title
Brief one-sentence description

## Problem Statement
What are you trying to solve?

## Dataset
- Source
- Size
- Features description
- Target variable

## Methodology
1. Data cleaning decisions
2. Feature engineering
3. Models tried
4. Best model selection

## Results
- Final metrics
- Confusion matrix
- ROC curve
- Feature importance

## Key Insights
- What did you learn about the data?
- What features are most important?
- What would you do differently next time?

## How to Run
```bash
python main.py
```

## Files
- `data/`: Raw and cleaned data
- `notebooks/`: Jupyter notebooks
- `models/`: Saved models
- `results/`: Figures and reports
```

### Phase 8: Save Everything
```python
# TODO: Save final model
import joblib
joblib.dump(best_pipeline, 'final_model.joblib')

# TODO: Save preprocessing objects
joblib.dump(preprocessor, 'preprocessor.joblib')

# TODO: Create a prediction function
def predict_new_data(new_data):
    """
    Predict on new data.
    
    Parameters:
    new_data (DataFrame): New data to predict
    
    Returns:
    predictions (array): Predicted labels
    """
    model = joblib.load('final_model.joblib')
    predictions = model.predict(new_data)
    return predictions
```

---

## ✅ Success Criteria

- [ ] Complete EDA with insights documented
- [ ] All missing values handled with justification
- [ ] At least 3 models trained and compared
- [ ] Hyperparameter tuning performed
- [ ] Test set evaluation with multiple metrics
- [ ] Visualizations (confusion matrix, ROC curve, feature importance)
- [ ] Comprehensive README/report written
- [ ] Final model saved and loadable
- [ ] Code is clean, commented, and reproducible

---

## 📊 Suggested Folder Structure

```
your_project/
├── data/
│   ├── raw/
│   │   └── dataset.csv
│   └── processed/
│       └── cleaned_data.csv
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Modeling.ipynb
│   └── 04_Evaluation.ipynb
├── src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── evaluation.py
├── models/
│   ├── final_model.joblib
│   └── preprocessor.joblib
├── results/
│   ├── figures/
│   │   ├── confusion_matrix.png
│   │   ├── roc_curve.png
│   │   └── feature_importance.png
│   └── metrics.txt
├── README.md
└── requirements.txt
```

---

## 🎓 Bonus Challenges

1. Deploy your model as a simple Flask API
2. Create a Streamlit dashboard for predictions
3. Try ensemble methods (stacking, voting)
4. Perform error analysis on misclassified examples
5. Create presentation slides summarizing your project
6. Write a blog post about your findings
7. Submit to Kaggle competition (if applicable)

---

## 🐛 Troubleshooting

### "Pipeline fails during fit"
**Solution:** Check that column names match in preprocessor definition.

### "Memory error during training"
**Solution:** Use smaller dataset subset or simpler model; try `max_samples` parameter.

### "Model overfits badly"
**Solution:** Try regularization, simpler model, more data, or cross-validation.

### "Can't reproduce results"
**Solution:** Set `random_state` everywhere (split, models, cross-validation).

---

## 📚 Resources

- Pipeline tutorial: https://scikit-learn.org/stable/modules/compose.html
- Model evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html
- Kaggle learn: https://www.kaggle.com/learn

---

## 🎯 Final Tips

1. **Start simple**: Get a baseline model working first, then improve
2. **Document everything**: Write down why you made each decision
3. **Version control**: Use git to track changes
4. **Ask questions**: What story does the data tell?
5. **Be proud**: This is your portfolio piece!

---

**Congratulations!** 🎉 You've completed all projects. You now have the skills to tackle real-world machine learning problems. Keep practicing and building!

**What's next?**
- Participate in Kaggle competitions
- Build projects on topics you're passionate about
- Learn deep learning (TensorFlow, PyTorch)
- Study advanced topics (NLP, Computer Vision, Time Series)
- Contribute to open-source ML projects

**You're now a Machine Learning practitioner!** 🚀🎓
