# Project 01 — Data Cleaning & Exploration
**Difficulty:** ⭐ Beginner  
**Estimated Time:** 2-3 hours  
**Prerequisites:** Complete Chapter 01 (NumPy) and Chapter 02 (pandas basics)

---

## 📋 Project Description

Learn to load real-world messy datasets, inspect them, handle missing values, fix data types, and create clean data ready for analysis. This is the FIRST skill every data scientist needs!

**What you'll learn:**
- Loading CSV files with pandas
- Inspecting data structure and finding problems
- Handling missing data (drop vs fill strategies)
- Converting data types (dates, categories, numbers)
- Creating new features from existing columns
- Saving cleaned data

---

## 🎯 Objectives

By the end of this project, you should be able to:
- [ ] Load a CSV file and display basic information
- [ ] Identify missing values and decide how to handle them
- [ ] Convert columns to appropriate data types
- [ ] Create derived features (e.g., family size from name)
- [ ] Save cleaned data to a new CSV file

---

## 📊 Suggested Datasets (Pick ONE to start)

### Option 1: Titanic Dataset (RECOMMENDED FOR BEGINNERS)
- **Link:** https://www.kaggle.com/c/titanic/data
- **Why this one?** Small size, well-documented, common missing values
- **Download:** Download `train.csv` from Kaggle
- **Size:** 891 rows, 12 columns

### Option 2: Housing Prices
- **Link:** https://www.kaggle.com/c/house-prices-advanced-regression-techniques
- **Why this one?** Real-world features, many data types
- **Download:** Download `train.csv`
- **Size:** 1460 rows, 81 columns

---

## 🛠️ Tools & Libraries You'll Use

```python
import pandas as pd
import numpy as np
```

---

## 📝 Step-by-Step Tasks

### Task 1: Load and Inspect the Data
**Goal:** Understand what data you have

```python
# TODO: Load the CSV file
df = pd.read_csv('path/to/your/file.csv')

# TODO: Display first 5 rows
print(df.head())

# TODO: Get summary information
print(df.info())

# TODO: Get statistical summary
print(df.describe())

# TODO: Check shape (rows, columns)
print(f"Shape: {df.shape}")
```

**Hints:**
- Use `df.head()` to see the first few rows
- Use `df.info()` to see column types and missing values
- Use `df.describe()` to see statistics for numeric columns

**Expected Output:**
- You should see column names, data types, and how many non-null values each column has

---

### Task 2: Identify Missing Values
**Goal:** Find which columns have missing data

```python
# TODO: Count missing values per column
missing = df.isnull().sum()
print("Missing values:\n", missing[missing > 0])

# TODO: Calculate percentage of missing values
missing_percent = (df.isnull().sum() / len(df)) * 100
print("Missing percentage:\n", missing_percent[missing_percent > 0])

# TODO: Visualize missing data pattern (optional)
import matplotlib.pyplot as plt
df.isnull().sum().plot(kind='bar')
plt.title('Missing Values by Column')
plt.show()
```

**Hints:**
- `df.isnull().sum()` counts NaN values in each column
- Columns with >50% missing might need to be dropped
- Columns with <5% missing can often be filled

**Questions to ask yourself:**
- Which columns have the most missing values?
- Can I drop these columns, or are they important?
- What's a reasonable way to fill the missing values?

---

### Task 3: Handle Missing Values
**Goal:** Decide strategy for each column with missing data

**Strategy Guide:**
1. **Drop the column** if >50% missing AND not important
2. **Drop rows** if very few rows have missing values
3. **Fill with mean/median** for numeric columns
4. **Fill with mode** for categorical columns
5. **Fill with a placeholder** like "Unknown" or 0

```python
# Example: Drop column with too many missing values
# TODO: Replace 'ColumnName' with actual column
# df = df.drop('ColumnName', axis=1)

# Example: Fill numeric column with median
# TODO: Replace 'Age' with your numeric column
# df['Age'].fillna(df['Age'].median(), inplace=True)

# Example: Fill categorical column with mode
# TODO: Replace 'Embarked' with your categorical column
# df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Example: Drop rows with any missing values (use carefully!)
# df = df.dropna()

# TODO: Verify no missing values remain
print("Missing values after cleaning:", df.isnull().sum().sum())
```

**Hints:**
- For Titanic: Fill Age with median, Embarked with mode, drop Cabin
- Use `inplace=True` to modify the DataFrame directly
- Always check `df.isnull().sum()` after cleaning

---

### Task 4: Fix Data Types
**Goal:** Convert columns to appropriate types

```python
# TODO: Check current data types
print(df.dtypes)

# Example: Convert to category (for columns with few unique values)
# df['Sex'] = df['Sex'].astype('category')
# df['Pclass'] = df['Pclass'].astype('category')

# Example: Convert to datetime (if you have date columns)
# df['Date'] = pd.to_datetime(df['Date'])

# Example: Convert string numbers to numeric
# df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# TODO: Verify types changed
print("Data types after conversion:\n", df.dtypes)
```

**Hints:**
- Use `astype('category')` for columns like gender, class, etc.
- Categories save memory and are faster for grouping
- Use `pd.to_datetime()` for any date/time columns

**Common Fixes:**
- Survived, Pclass, Sex → category
- Age, Fare → float
- PassengerId → int or string (doesn't matter much)

---

### Task 5: Create Derived Features
**Goal:** Engineer new columns from existing data

```python
# Example: Create FamilySize from SibSp and Parch (Titanic)
# TODO: Adapt to your dataset
# df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Example: Extract title from Name column
# TODO: Adapt to your dataset
# df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Example: Create age groups (binning)
# df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 60, 100], 
#                          labels=['Child', 'Teen', 'Adult', 'Senior'])

# TODO: Create at least 2 new features
# Think: What combinations or transformations might be useful?

# TODO: Display new columns
print(df[['OriginalColumn', 'NewFeature']].head())
```

**Hints:**
- FamilySize = number of family members aboard
- Title extraction from name (Mr., Mrs., Miss., etc.)
- Age groups: binning continuous Age into categories
- IsAlone: 1 if FamilySize == 1, else 0

**Think about:**
- What patterns might help predict survival (or your target)?
- Can you combine columns in meaningful ways?

---

### Task 6: Save Cleaned Data
**Goal:** Export your clean DataFrame for future use

```python
# TODO: Save to CSV
df.to_csv('cleaned_data.csv', index=False)
print("Cleaned data saved to 'cleaned_data.csv'")

# TODO: Verify the saved file
df_check = pd.read_csv('cleaned_data.csv')
print("Saved file shape:", df_check.shape)
print("Saved file columns:", df_check.columns.tolist())
```

**Hints:**
- `index=False` prevents saving the row index as a column
- Always load the saved file to verify it worked
- Consider saving a data dictionary (column descriptions)

---

## ✅ Success Criteria

You've completed this project successfully if:
- [ ] You loaded the dataset without errors
- [ ] You identified all columns with missing values
- [ ] You handled missing values using appropriate strategies
- [ ] You converted at least 2 columns to better data types
- [ ] You created at least 2 new derived features
- [ ] Your final dataset has zero missing values (or you documented why you kept some)
- [ ] You saved the cleaned data to CSV
- [ ] You can explain your cleaning decisions

---

## 🎓 Bonus Challenges

If you finish early, try these:
1. Create a `data_cleaning_report.md` documenting your decisions
2. Create visualizations showing before/after data quality
3. Handle outliers (e.g., Age > 100 or Fare = 0)
4. Create interaction features (e.g., Age * Pclass)
5. Standardize column names (lowercase, no spaces)

---

## 🐛 Common Errors & Solutions

### Error: "FileNotFoundError"
**Solution:** Check your file path. Use absolute path or ensure the file is in your working directory.

```python
import os
print("Current directory:", os.getcwd())
```

### Error: "KeyError: 'ColumnName'"
**Solution:** Check column names with `df.columns`. They might have extra spaces or different capitalization.

### Error: "ValueError: cannot convert float NaN to integer"
**Solution:** Fill or drop NaN values before converting to int.

```python
df['Age'] = df['Age'].fillna(0).astype(int)
```

### Warning: "SettingWithCopyWarning"
**Solution:** Use `.loc[]` for assignments or `.copy()` when creating a subset.

```python
df.loc[df['Age'] > 60, 'AgeGroup'] = 'Senior'
```

---

## 📚 Resources

- pandas docs: https://pandas.pydata.org/docs/user_guide/missing_data.html
- Chapter 02: pandas Data Manipulation
- Video: Refer to pandas video in main README

---

## 🎯 Next Steps

After completing this project:
1. Move to **Project 02: Visualization & EDA** to visualize your cleaned data
2. Review Chapter 02 if you struggled with any pandas concepts
3. Try cleaning a different dataset to practice

---

**Remember:** Data cleaning takes 60-80% of a data scientist's time. Master this, and you're ahead of the game! 🚀
