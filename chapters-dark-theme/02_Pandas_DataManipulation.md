# 📊 Chapter 02: pandas - Data Manipulation with Multiple Real Datasets

## 🎯 Learning Objectives
By the end of this chapter, you will:
- ✅ Understand what pandas is and why it's essential for data analysis
- ✅ Master Series and DataFrame objects (core pandas data structures)
- ✅ Apply pandas operations on **3 different real datasets** for maximum practice
- ✅ Perform data cleaning, filtering, grouping, and merging operations
- ✅ Handle missing data and perform aggregations

---

## 📁 Datasets Used (Multiple datasets for maximum practice!)

We'll practice **each concept** on **3 different real-world datasets**:

### 1️⃣ Titanic Dataset (891 passengers)
**Purpose**: Analyze survival rates and passenger demographics
- PassengerId, Name, Sex, Age
- Pclass (ticket class), Fare, Cabin
- Survived (target: 0=No, 1=Yes)

### 2️⃣ Tips Dataset (244 restaurant bills)
**Purpose**: Analyze tipping behavior in restaurants
- total_bill, tip, sex, smoker
- day, time, size (party size)

### 3️⃣ Diamonds Dataset (53,940 diamonds)
**Purpose**: Analyze diamond prices and characteristics
- carat, cut, color, clarity
- depth, table, price
- x, y, z (dimensions)

**Why 3 datasets?** Different data types, sizes, and real-world scenarios help you master pandas deeply!

---

## 📚 Table of Contents
1. [Introduction to pandas](#intro)
**1.5. [pandas Basics - Essential Concepts](#pandas-basics)** ⭐ **Start Here! Learn concepts BEFORE datasets**
2. [Loading Multiple Datasets](#loading)
3. [Series & DataFrame Basics](#basics)
4. [Data Exploration & Inspection](#exploration)
5. [Indexing & Selection](#indexing)
6. [Data Cleaning](#cleaning)
7. [Filtering & Sorting](#filtering)
8. [GroupBy Operations](#groupby)
9. [Merging & Joining](#merging)
10. [Handling Missing Data](#missing)
11. [Practice Exercises](#exercises)
12. [Next Steps: Projects](#projects)

💡 **Tip for Beginners**: Section 1.5 teaches you .shape, .loc, .iloc, .dtypes, and other basics BEFORE we use them with real datasets. Don't skip it!
---
<a id="intro"></a>
## 1️⃣ Introduction to pandas

### 📖 What is pandas?

**pandas** is the most popular Python library for data manipulation and analysis. Built on top of NumPy, it provides:

#### Why pandas Exists:
NumPy is great for numerical arrays, but real-world data has:
- **Mixed data types** (numbers, strings, dates in same table)
- **Row/column labels** (not just numeric indices)
- **Missing values** (NA, NaN, None)
- **Relationships** (merge tables like SQL)

pandas solves these problems with two main data structures:

1. **Series**: 1D labeled array (like a single column)
2. **DataFrame**: 2D labeled table (like Excel spreadsheet or SQL table)

#### What pandas Provides:
- **Read/Write**: CSV, Excel, SQL, JSON, HTML tables
- **Data cleaning**: Handle missing values, duplicates, type conversions
- **Data manipulation**: Filter, sort, group, pivot, merge
- **Time series**: Date/time handling, resampling, rolling windows
- **Statistics**: Aggregations, correlations, statistical tests

#### Real-World Use Cases:
- **Business**: Sales analysis, customer segmentation
- **Finance**: Stock prices, portfolio analysis
- **Healthcare**: Patient records, clinical trials
- **Research**: Scientific data analysis
- **Web**: Log analysis, user behavior

### 🔑 Key Concept: pandas vs NumPy

| Feature | NumPy | pandas |
|---------|-------|--------|
| **Data Structure** | ndarray (n-dimensional) | Series (1D), DataFrame (2D) |
| **Labels** | Integer indices only | Custom row/column labels |
| **Data Types** | Homogeneous (single type) | Heterogeneous (mixed types) |
| **Missing Data** | Limited support | Built-in NA/NaN handling |
| **Best For** | Numerical computations | Data analysis & cleaning |

**Bottom Line**: NumPy for math, pandas for data!
```python
# Import pandas library
# pandas is THE standard library for data manipulation and analysis in Python
# It's built on top of NumPy but adds labels, mixed types, and powerful data operations
import pandas as pd

# Import NumPy for numerical operations
# pandas and NumPy work together - pandas uses NumPy arrays internally
import numpy as np

# Import seaborn for loading sample datasets
# seaborn has built-in datasets we can use for practice
import seaborn as sns

# Check installed versions
# This helps ensure compatibility and troubleshoot issues
print(f"pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"seaborn version: {sns.__version__}")

# Quick demo: Why pandas?
# Let's compare working with a table in pure Python vs pandas
print("\n" + "=" * 70)
print("WHY PANDAS? Quick Comparison")
print("=" * 70)

# Python way: Lists of lists (messy, hard to work with)
python_data = [
    ["Alice", 25, "Engineer", 75000],
    ["Bob", 30, "Doctor", 95000],
    ["Charlie", 35, "Teacher", 55000]
]
print("\nPython list (hard to query):")
print(python_data)
print("To get Bob's salary: python_data[1][3] = ", python_data[1][3])
print("❌ Hard to remember indices, no column names!")

# pandas way: DataFrame (clean, powerful, labeled)
pandas_data = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Job': ['Engineer', 'Doctor', 'Teacher'],
    'Salary': [75000, 95000, 55000]
})
print("\npandas DataFrame (easy to query):")
print(pandas_data)
print("\nTo get Bob's salary: pandas_data.loc[1, 'Salary'] = ", pandas_data.loc[1, 'Salary'])
print("✅ Column names, labeled rows, easy filtering!")

print("\n" + "=" * 70)
print("✅ pandas makes data manipulation 10x easier!")
print("=" * 70)
```

**Output:**

```
pandas version: 2.3.0
NumPy version: 2.3.1
seaborn version: 0.13.2

======================================================================
WHY PANDAS? Quick Comparison
======================================================================

Python list (hard to query):
[['Alice', 25, 'Engineer', 75000], ['Bob', 30, 'Doctor', 95000], ['Charlie', 35, 'Teacher', 55000]]
To get Bob's salary: python_data[1][3] =  95000
❌ Hard to remember indices, no column names!

pandas DataFrame (easy to query):
      Name  Age       Job  Salary
0    Alice   25  Engineer   75000
1      Bob   30    Doctor   95000
2  Charlie   35   Teacher   55000

To get Bob's salary: pandas_data.loc[1, 'Salary'] =  95000
✅ Column names, labeled rows, easy filtering!

======================================================================
✅ pandas makes data manipulation 10x easier!
======================================================================

```

---
<a id="pandas-basics"></a>
## 1.5 pandas Basics - Essential Concepts 🔑

**⚠️ IMPORTANT FOR BEGINNERS**: This section introduces all fundamental pandas concepts **BEFORE** we use them with real datasets. Don't skip this - it will make everything that follows much easier to understand!

---

### 📌 Concept 1: DataFrame & Series (pandas data structures)

pandas has two main data structures:

#### **Series** (1-dimensional)
- Like a single column in Excel
- Has an **index** (row labels) and **values**
- All values must be same data type
- Created with `pd.Series([values])`

#### **DataFrame** (2-dimensional)
- Like an Excel spreadsheet or SQL table
- Has **rows** (index) and **columns**
- Each column is a Series
- Columns can have different data types
- Created with `pd.DataFrame({dict})` or `pd.read_csv()`

**Example:**
```python
# Series (1D)
ages = pd.Series([25, 30, 35], name='Age')
# Result: Single column with 3 values

# DataFrame (2D)
people = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35]
})
# Result: Table with 2 columns, 3 rows
```

---

### 📌 Concept 2: Shape & Size (Understanding dimensions)

Every DataFrame/Series has dimensions:

| Attribute | What it shows | Example |
|-----------|---------------|---------|
| `.shape` | (rows, columns) tuple | `(891, 15)` = 891 rows, 15 columns |
| `.size` | Total elements | 891 × 15 = 13,365 cells |
| `.ndim` | Number of dimensions | 1 for Series, 2 for DataFrame |

**How to read `.shape`:**
- `df.shape` → `(100, 5)` means:
  - `df.shape[0]` = 100 rows
  - `df.shape[1]` = 5 columns

---

### 📌 Concept 3: Column & Index Access

**Access columns** (get a Series):
```python
df['column_name']        # Returns Series
df[['col1', 'col2']]     # Returns DataFrame (note double brackets!)
```

**Access rows by index**:
```python
df.loc[0]                # Row with index label 0
df.iloc[0]               # First row by position (0-based)
```

**Key difference:**
- `df['col']` → Returns **Series** (single column)
- `df[['col']]` → Returns **DataFrame** (one-column table)

---

### 📌 Concept 4: Data Types (dtypes)

Each column has a data type:

| dtype | Description | Example values |
|-------|-------------|----------------|
| `int64` | Integers (whole numbers) | 1, 2, 100, -50 |
| `float64` | Decimals (floating-point) | 3.14, 2.5, -0.5 |
| `object` | Text/strings | "Alice", "male", "NY" |
| `bool` | True/False | True, False |
| `datetime64` | Dates/times | 2023-01-15 |
| `category` | Categorical data | "Low", "Medium", "High" |

**Check dtypes:**
```python
df.dtypes              # All column types
df['age'].dtype        # Single column type
```

---

### 📌 Concept 5: Loading Data

pandas can read many formats:

| Function | Purpose | Example |
|----------|---------|---------|
| `pd.read_csv('file.csv')` | Read CSV files | Most common format |
| `pd.read_excel('file.xlsx')` | Read Excel files | .xlsx, .xls |
| `sns.load_dataset('name')` | Load seaborn sample data | Built-in datasets |
| `pd.read_sql(query, conn)` | Read from database | SQL queries |
| `pd.read_json('file.json')` | Read JSON files | Web APIs |

**Common parameters:**
- `sep=','` - Column delimiter (default comma)
- `header=0` - Which row has column names
- `index_col=0` - Which column to use as index

---

### 📌 Concept 6: Essential Methods

| Method | Purpose | Example Output |
|--------|---------|----------------|
| `.head(n)` | First n rows (default 5) | First 5 rows of data |
| `.tail(n)` | Last n rows (default 5) | Last 5 rows of data |
| `.info()` | DataFrame summary | Column names, types, memory |
| `.describe()` | Statistical summary | mean, std, min, max, etc. |
| `.columns` | Column names | ['age', 'name', 'salary'] |
| `.index` | Row labels | [0, 1, 2, ..., 890] |
| `.values` | NumPy array of data | Underlying array |
| `.memory_usage()` | Memory per column | Bytes used |

---

### 📌 Concept 7: Indexing - loc vs iloc

**Two ways to access data:**

#### `.loc[row, column]` - Label-based (use names)
```python
df.loc[0, 'age']           # Row index 0, column 'age'
df.loc[0:5, 'name']        # Rows 0-5 (INCLUDES 5), column 'name'
df.loc[df['age'] > 30]     # Boolean indexing
```

#### `.iloc[row_num, col_num]` - Position-based (use numbers)
```python
df.iloc[0, 0]              # First row, first column
df.iloc[0:5, 0]            # First 5 rows (EXCLUDES 5), first column
df.iloc[:, 0:3]            # All rows, first 3 columns
```

**Key differences:**
- `.loc` uses **labels** and **INCLUDES** endpoint: `loc[0:5]` includes row 5
- `.iloc` uses **positions** and **EXCLUDES** endpoint: `iloc[0:5]` excludes row 5 (like Python slicing)

---

### 🎯 Quick Reference Summary

| Concept | Syntax | Example |
|---------|--------|---------|
| **Create DataFrame** | `pd.DataFrame({dict})` | `pd.DataFrame({'A': [1,2,3]})` |
| **Get column** | `df['col']` or `df.col` | `titanic['age']` |
| **Get multiple columns** | `df[['col1', 'col2']]` | `titanic[['age', 'fare']]` |
| **Check shape** | `df.shape` | `(891, 15)` means 891 rows, 15 cols |
| **Check types** | `df.dtypes` | `age: int64, name: object` |
| **First rows** | `df.head(n)` | `titanic.head(10)` |
| **Summary** | `df.info()` | Column names, types, nulls |
| **Statistics** | `df.describe()` | mean, std, min, max |
| **Access by label** | `df.loc[row, col]` | `df.loc[0, 'age']` |
| **Access by position** | `df.iloc[row, col]` | `df.iloc[0, 0]` |

---

### 💡 Why This Section Matters

In the sections that follow, you'll see code like:
- `titanic.shape` → Now you know this shows (rows, columns)!
- `df.loc[0, 'age']` → Now you know this gets row 0, column 'age'!
- `df['survived'].value_counts()` → Now you know `df['col']` gets a column!
- `sns.load_dataset('titanic')` → Now you know this loads data into a DataFrame!

**✅ You're now ready to work with real datasets!** Let's practice these concepts below.
```python
# ============================================================================
# HANDS-ON PRACTICE: pandas Basics (Practice BEFORE working with real data!)
# ============================================================================

print("=" * 80)
print("PANDAS BASICS - HANDS-ON PRACTICE")
print("=" * 80)

# ============================================================================
# PRACTICE 1: Creating Series and DataFrame
# ============================================================================
print("\n📌 PRACTICE 1: Creating Series and DataFrame")
print("-" * 80)

# Create a Series (1D - single column)
ages_series = pd.Series([25, 30, 35, 40, 45], name='Age')
print("Series (1D - Single Column):")
print(ages_series)
print(f"Type: {type(ages_series)}")  # pandas.core.series.Series
print(f"Shape: {ages_series.shape}")  # (5,) means 5 elements, 1-dimensional
print(f"Dimension: {ages_series.ndim}D")  # 1
print(f"Size (total elements): {ages_series.size}")  # 5

# Create a DataFrame (2D - table with multiple columns)
people_df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'Age': [25, 30, 35, 40, 45],
    'City': ['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'],
    'Salary': [70000, 80000, 90000, 100000, 110000]
})
print("\n\nDataFrame (2D - Table with multiple columns):")
print(people_df)
print(f"\nType: {type(people_df)}")  # pandas.core.frame.DataFrame
print(f"Shape: {people_df.shape}")  # (5, 4) means 5 rows, 4 columns
print(f"  → Rows: {people_df.shape[0]}")  # 5 rows
print(f"  → Columns: {people_df.shape[1]}")  # 4 columns
print(f"Dimension: {people_df.ndim}D")  # 2
print(f"Size (total cells): {people_df.size}")  # 5 × 4 = 20 cells

# ============================================================================
# PRACTICE 2: Understanding Shape
# ============================================================================
print("\n\n📌 PRACTICE 2: Understanding .shape")
print("-" * 80)

# Create DataFrames of different sizes
small_df = pd.DataFrame({'A': [1, 2, 3]})  # 3 rows, 1 column
medium_df = pd.DataFrame({'X': [1,2,3,4,5], 'Y': [10,20,30,40,50]})  # 5 rows, 2 cols
large_df = pd.DataFrame(np.random.rand(100, 10))  # 100 rows, 10 columns

print(f"Small DataFrame shape: {small_df.shape}")
print(f"  → {small_df.shape[0]} rows × {small_df.shape[1]} column = {small_df.size} cells")

print(f"\nMedium DataFrame shape: {medium_df.shape}")
print(f"  → {medium_df.shape[0]} rows × {medium_df.shape[1]} columns = {medium_df.size} cells")

print(f"\nLarge DataFrame shape: {large_df.shape}")
print(f"  → {large_df.shape[0]} rows × {large_df.shape[1]} columns = {large_df.size} cells")

# ============================================================================
# PRACTICE 3: Column Access (DataFrame → Series)
# ============================================================================
print("\n\n📌 PRACTICE 3: Accessing Columns")
print("-" * 80)

# Single column access → Returns Series
print("Single column access with df['column']:")
name_column = people_df['Name']
print(name_column)
print(f"Type: {type(name_column)}")  # Series
print(f"Shape: {name_column.shape}")  # (5,) = 5 elements, 1D

# Multiple columns access → Returns DataFrame
print("\n\nMultiple columns access with df[['col1', 'col2']]:")
name_age_df = people_df[['Name', 'Age']]
print(name_age_df)
print(f"Type: {type(name_age_df)}")  # DataFrame
print(f"Shape: {name_age_df.shape}")  # (5, 2) = 5 rows, 2 columns

# Single vs Double Brackets
print("\n\nSingle vs Double Brackets:")
print(f"people_df['Age'] → Returns Series, shape = {people_df['Age'].shape}")
print(f"people_df[['Age']] → Returns DataFrame, shape = {people_df[['Age']].shape}")

# ============================================================================
# PRACTICE 4: Data Types (dtypes)
# ============================================================================
print("\n\n📌 PRACTICE 4: Understanding Data Types (dtypes)")
print("-" * 80)

# Create DataFrame with different data types
mixed_types_df = pd.DataFrame({
    'integers': [1, 2, 3, 4, 5],                    # int64
    'floats': [1.1, 2.2, 3.3, 4.4, 5.5],           # float64
    'strings': ['A', 'B', 'C', 'D', 'E'],          # object
    'booleans': [True, False, True, False, True],   # bool
    'dates': pd.date_range('2024-01-01', periods=5) # datetime64
})

print("DataFrame with mixed data types:")
print(mixed_types_df)

print("\n\nData types for each column:")
print(mixed_types_df.dtypes)

print("\n\nMemory usage per column:")
print(mixed_types_df.memory_usage(deep=True))

# Check single column dtype
print(f"\n\nSingle column dtype:")
print(f"  integers column: {mixed_types_df['integers'].dtype}")
print(f"  floats column: {mixed_types_df['floats'].dtype}")
print(f"  strings column: {mixed_types_df['strings'].dtype}")

# ============================================================================
# PRACTICE 5: Essential Methods (.head, .tail, .info, .describe)
# ============================================================================
print("\n\n📌 PRACTICE 5: Essential Methods")
print("-" * 80)

# Create sample DataFrame
sample_df = pd.DataFrame({
    'Product': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'Price': [10, 20, 15, 30, 25, 40, 35, 50, 45, 60],
    'Quantity': [100, 200, 150, 300, 250, 400, 350, 500, 450, 600]
})

print("Method 1: .head() - First n rows (default 5)")
print(sample_df.head())  # First 5 rows

print("\n\nMethod 2: .head(3) - First 3 rows")
print(sample_df.head(3))

print("\n\nMethod 3: .tail() - Last n rows (default 5)")
print(sample_df.tail())  # Last 5 rows

print("\n\nMethod 4: .tail(2) - Last 2 rows")
print(sample_df.tail(2))

print("\n\nMethod 5: .columns - Column names")
print(f"Column names: {list(sample_df.columns)}")

print("\n\nMethod 6: .index - Row labels")
print(f"Row indices: {list(sample_df.index)}")

print("\n\nMethod 7: .info() - DataFrame summary")
sample_df.info()

print("\n\nMethod 8: .describe() - Statistical summary (numerical columns only)")
print(sample_df.describe())

# ============================================================================
# PRACTICE 6: Indexing - .loc vs .iloc
# ============================================================================
print("\n\n📌 PRACTICE 6: Indexing with .loc and .iloc")
print("-" * 80)

# Create sample DataFrame
students_df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'Math': [85, 90, 78, 92, 88],
    'English': [88, 85, 92, 89, 91],
    'Science': [90, 87, 85, 94, 86]
})

print("Sample DataFrame:")
print(students_df)

# .loc - Label-based indexing
print("\n\n.loc[] - Label-based indexing:")
print(f"  students_df.loc[0, 'Name'] = {students_df.loc[0, 'Name']}")
print(f"  students_df.loc[0, 'Math'] = {students_df.loc[0, 'Math']}")
print(f"  students_df.loc[2, 'English'] = {students_df.loc[2, 'English']}")

print("\n  students_df.loc[0:2, 'Name':'Math'] (rows 0-2, columns Name-Math):")
print(students_df.loc[0:2, 'Name':'Math'])  # INCLUDES endpoint (row 2)

# .iloc - Position-based indexing
print("\n\n.iloc[] - Position-based indexing:")
print(f"  students_df.iloc[0, 0] = {students_df.iloc[0, 0]}")  # First row, first column
print(f"  students_df.iloc[0, 1] = {students_df.iloc[0, 1]}")  # First row, second column
print(f"  students_df.iloc[2, 2] = {students_df.iloc[2, 2]}")  # Third row, third column

print("\n  students_df.iloc[0:3, 0:2] (rows 0-2, columns 0-1):")
print(students_df.iloc[0:3, 0:2])  # EXCLUDES endpoint (row 3)

# Difference: loc includes endpoint, iloc excludes
print("\n\n🔑 KEY DIFFERENCE:")
print("  .loc[0:2] includes row 2 (like 0, 1, 2)")
print("  .iloc[0:3] excludes row 3 (like 0, 1, 2) - same as Python slicing!")

# ============================================================================
# PRACTICE 7: Slicing with : (colon)
# ============================================================================
print("\n\n📌 PRACTICE 7: Slicing with : (All rows/columns)")
print("-" * 80)

print("Get all rows, specific columns with .loc[:, columns]:")
print(students_df.loc[:, ['Name', 'Math']])

print("\n\nGet specific rows, all columns with .loc[rows, :]:")
print(students_df.loc[0:2, :])

print("\n\nGet all rows, first 2 columns with .iloc[:, 0:2]:")
print(students_df.iloc[:, 0:2])

print("\n\nGet first 3 rows, all columns with .iloc[0:3, :]:")
print(students_df.iloc[0:3, :])

# ============================================================================
# SUMMARY
# ============================================================================
print("\n\n" + "=" * 80)
print("✅ PRACTICE COMPLETE!")
print("=" * 80)
print("\n🎯 You've practiced:")
print("  1. Creating Series and DataFrames")
print("  2. Understanding .shape (rows, columns)")
print("  3. Accessing columns with df['col'] and df[['col1', 'col2']]")
print("  4. Data types (int64, float64, object, bool, datetime64)")
print("  5. Essential methods (.head, .tail, .info, .describe, .columns, .index)")
print("  6. Indexing with .loc[row, col] (labels) and .iloc[row, col] (positions)")
print("  7. Slicing with : to get all rows or columns")
print("\n📚 Now you're ready to work with REAL datasets (Titanic, Tips, Diamonds)!")
print("=" * 80)
```

---
<a id="loading"></a>
## 2️⃣ Loading Multiple Real Datasets

### 📖 Concept: How pandas Loads Data

pandas can read data from many sources:
- **CSV files**: `pd.read_csv()` - Most common format
- **Excel files**: `pd.read_excel()` - .xlsx, .xls
- **SQL databases**: `pd.read_sql()` - Connect to databases
- **JSON files**: `pd.read_json()` - Web APIs often return JSON
- **HTML tables**: `pd.read_html()` - Scrape tables from websites
- **Clipboard**: `pd.read_clipboard()` - Copy/paste data

### 🔑 Key Parameters for `read_csv()`:
- **`filepath`**: Path to file (local or URL)
- **`sep`**: Delimiter (default=',', can be '\t', '|', etc.)
- **`header`**: Row number for column names (default=0)
- **`names`**: Custom column names
- **`index_col`**: Column to use as row labels
- **`usecols`**: Which columns to load (saves memory)
- **`dtype`**: Specify data types for columns
- **`na_values`**: Additional strings to recognize as NA/NaN
- **`parse_dates`**: Columns to parse as dates

For this tutorial, we'll use seaborn's built-in datasets (already cleaned and ready!).
```python
# Load all 3 datasets for practice
print("=" * 70)
print("LOADING 3 REAL-WORLD DATASETS")
print("=" * 70)

# ============================================================================
# DATASET 1: TITANIC - Survival on the Titanic ship disaster
# ============================================================================
# seaborn.load_dataset() downloads the dataset and returns a pandas DataFrame
# The Titanic dataset is famous for machine learning classification problems
titanic = sns.load_dataset('titanic')

print("\n📊 DATASET 1: Titanic Passenger Records")
print("-" * 70)
print(f"Purpose: Analyze passenger survival in 1912 Titanic disaster")
print(f"Shape: {titanic.shape}")  # (rows, columns)
print(f"  → {titanic.shape[0]} passengers")  # Number of rows
print(f"  → {titanic.shape[1]} features per passenger")  # Number of columns

# Display column names and their data types
# dtypes shows whether columns are numbers (int64, float64) or text (object)
print(f"\nColumns and data types:")
print(titanic.dtypes)

# Display first few rows to see what data looks like
# head() shows first 5 rows by default
print(f"\nFirst 3 passengers:")
print(titanic.head(3))  # Show only 3 rows to save space

# ============================================================================
# DATASET 2: TIPS - Restaurant tipping behavior
# ============================================================================
# This dataset records tips given at a restaurant
# Useful for analyzing relationships between bill amount, tip, and other factors
tips = sns.load_dataset('tips')

print("\n\n🍽️ DATASET 2: Restaurant Tips")
print("-" * 70)
print(f"Purpose: Analyze tipping patterns at restaurants")
print(f"Shape: {tips.shape}")
print(f"  → {tips.shape[0]} restaurant bills")
print(f"  → {tips.shape[1]} features per bill")

print(f"\nColumns and data types:")
print(tips.dtypes)

print(f"\nFirst 3 bills:")
print(tips.head(3))

# ============================================================================
# DATASET 3: DIAMONDS - Diamond prices and characteristics
# ============================================================================
# This dataset has information about diamond quality and prices
# Useful for price prediction and quality analysis
diamonds = sns.load_dataset('diamonds')

print("\n\n💎 DATASET 3: Diamond Characteristics & Prices")
print("-" * 70)
print(f"Purpose: Analyze diamond pricing based on quality features")
print(f"Shape: {diamonds.shape}")
print(f"  → {diamonds.shape[0]:,} diamonds")  # :, adds commas for readability
print(f"  → {diamonds.shape[1]} features per diamond")

print(f"\nColumns and data types:")
print(diamonds.dtypes)

print(f"\nFirst 3 diamonds:")
print(diamonds.head(3))

# ============================================================================
# QUICK COMPARISON
# ============================================================================
print("\n" + "=" * 70)
print("📊 DATASET COMPARISON")
print("=" * 70)

# Create a summary DataFrame to compare our datasets
# This shows how pandas can create DataFrames from dictionaries
comparison = pd.DataFrame({
    'Dataset': ['Titanic', 'Tips', 'Diamonds'],
    'Rows': [titanic.shape[0], tips.shape[0], diamonds.shape[0]],
    'Columns': [titanic.shape[1], tips.shape[1], diamonds.shape[1]],
    'Memory (KB)': [
        titanic.memory_usage(deep=True).sum() / 1024,  # Convert bytes to KB
        tips.memory_usage(deep=True).sum() / 1024,
        diamonds.memory_usage(deep=True).sum() / 1024
    ],
    'Type': ['Classification', 'Regression', 'Regression']
})

# Format the DataFrame for better display
# round(2) rounds to 2 decimal places
comparison['Memory (KB)'] = comparison['Memory (KB)'].round(2)

print(comparison.to_string(index=False))  # index=False hides row numbers

print("\n✅ All 3 datasets loaded successfully!")
print("=" * 70)
```

**Output:**

```
======================================================================
LOADING 3 REAL-WORLD DATASETS
======================================================================

📊 DATASET 1: Titanic Passenger Records
----------------------------------------------------------------------
Purpose: Analyze passenger survival in 1912 Titanic disaster
Shape: (891, 15)
  → 891 passengers
  → 15 features per passenger

Columns and data types:
survived          int64
pclass            int64
sex              object
age             float64
sibsp             int64
parch             int64
fare            float64
embarked         object
class          category
who              object
adult_male         bool
deck           category
embark_town      object
alive            object
alone              bool
dtype: object

First 3 passengers:
   survived  pclass     sex   age  sibsp  parch     fare embarked  class  \
0         0       3    male  22.0      1      0   7.2500        S  Third   
1         1       1  female  38.0      1      0  71.2833        C  First   
2         1       3  female  26.0      0      0   7.9250        S  Third   

     who  adult_male deck  embark_town alive  alone  
0    man        True  NaN  Southampton    no  False  
1  woman       False    C    Cherbourg   yes  False  
2  woman       False  NaN  Southampton   yes   True  


🍽️ DATASET 2: Restaurant Tips
----------------------------------------------------------------------
Purpose: Analyze tipping patterns at restaurants
Shape: (244, 7)
  → 244 restaurant bills
  → 7 features per bill

Columns and data types:
total_bill     float64
tip            float64
sex           category
smoker        category
day           category
time          category
size             int64
dtype: object

First 3 bills:
   total_bill   tip     sex smoker  day    time  size
0       16.99  1.01  Female     No  Sun  Dinner     2
1       10.34  1.66    Male     No  Sun  Dinner     3
2       21.01  3.50    Male     No  Sun  Dinner     3


💎 DATASET 3: Diamond Characteristics & Prices
----------------------------------------------------------------------
Purpose: Analyze diamond pricing based on quality features
Shape: (53940, 10)
  → 53,940 diamonds
  → 10 features per diamond

Columns and data types:
carat       float64
cut        category
color      category
clarity    category
depth       float64
table       float64
price         int64
x           float64
y           float64
z           float64
dtype: object

First 3 diamonds:
   carat      cut color clarity  depth  table  price     x     y     z
0   0.23    Ideal     E     SI2   61.5   55.0    326  3.95  3.98  2.43
1   0.21  Premium     E     SI1   59.8   61.0    326  3.89  3.84  2.31
2   0.23     Good     E     VS1   56.9   65.0    327  4.05  4.07  2.31

======================================================================
📊 DATASET COMPARISON
======================================================================
 Dataset  Rows  Columns  Memory (KB)           Type
 Titanic   891       15       278.87 Classification
    Tips   244        7         7.80     Regression
Diamonds 53940       10      3109.77     Regression

✅ All 3 datasets loaded successfully!
======================================================================


🍽️ DATASET 2: Restaurant Tips
----------------------------------------------------------------------
Purpose: Analyze tipping patterns at restaurants
Shape: (244, 7)
  → 244 restaurant bills
  → 7 features per bill

Columns and data types:
total_bill     float64
tip            float64
sex           category
smoker        category
day           category
time          category
size             int64
dtype: object

First 3 bills:
   total_bill   tip     sex smoker  day    time  size
0       16.99  1.01  Female     No  Sun  Dinner     2
1       10.34  1.66    Male     No  Sun  Dinner     3
2       21.01  3.50    Male     No  Sun  Dinner     3


💎 DATASET 3: Diamond Characteristics & Prices
----------------------------------------------------------------------
Purpose: Analyze diamond pricing based on quality features
Shape: (53940, 10)
  → 53,940 diamonds
  → 10 features per diamond

Columns and data types:
carat       float64
cut        category
color      category
clarity    category
depth       float64
table       float64
price         int64
x           float64
y           float64
z           float64
dtype: object

First 3 diamonds:
   carat      cut color clarity  depth  table  price     x     y     z
0   0.23    Ideal     E     SI2   61.5   55.0    326  3.95  3.98  2.43
1   0.21  Premium     E     SI1   59.8   61.0    326  3.89  3.84  2.31
2   0.23     Good     E     VS1   56.9   65.0    327  4.05  4.07  2.31

======================================================================
📊 DATASET COMPARISON
======================================================================
 Dataset  Rows  Columns  Memory (KB)           Type
 Titanic   891       15       278.87 Classification
    Tips   244        7         7.80     Regression
Diamonds 53940       10      3109.77     Regression

✅ All 3 datasets loaded successfully!
======================================================================

```

---
<a id="basics"></a>
## 3️⃣ Series & DataFrame Basics

### 📖 Concept: pandas Data Structures

pandas has two main data structures:

#### 1. **Series** (1-dimensional)
- Like a single column in Excel
- Has an index (row labels) and values
- All values must be the same data type
- Think: A list with labels

#### 2. **DataFrame** (2-dimensional)
- Like an Excel spreadsheet or SQL table
- Has rows (index) and columns
- Each column is a Series
- Columns can have different data types
- Think: A table with named columns

### 🔑 Key DataFrame Attributes:
- **`.shape`**: (rows, columns) tuple
- **`.columns`**: Column names
- **`.index`**: Row labels
- **`.dtypes`**: Data type of each column
- **`.values`**: Underlying NumPy array
- **`.info()`**: Summary of DataFrame
- **`.describe()`**: Statistical summary

### Why This Matters:
Understanding DataFrame structure is CRITICAL. Every pandas operation works with rows, columns, or both!
```python
print("=" * 70)
print("SERIES & DATAFRAME BASICS - PRACTICE ON 3 DATASETS")
print("=" * 70)

# ============================================================================
# PART 1: Understanding Series (1D labeled array)
# ============================================================================
print("\n1️⃣ SERIES (1-Dimensional)")
print("-" * 70)

# Extract a single column from each dataset
# In pandas, a single column is a Series object
titanic_ages = titanic['age']  # Get 'age' column from titanic DataFrame
tips_total_bill = tips['total_bill']  # Get 'total_bill' column
diamonds_price = diamonds['price']  # Get 'price' column

print("🚢 TITANIC - Age Series:")
print(f"Type: {type(titanic_ages)}")  # Shows this is pandas.Series
print(f"Shape: {titanic_ages.shape}")  # (891,) = 891 elements, 1D
print(f"First 5 values:\n{titanic_ages.head()}")  # head() shows first 5 by default

print("\n🍽️ TIPS - Total Bill Series:")
print(f"Type: {type(tips_total_bill)}")
print(f"Shape: {tips_total_bill.shape}")
print(f"First 5 values:\n{tips_total_bill.head()}")

print("\n💎 DIAMONDS - Price Series:")
print(f"Type: {type(diamonds_price)}")
print(f"Shape: {diamonds_price.shape}")
print(f"First 5 values:\n{diamonds_price.head()}")

# ============================================================================
# PART 2: DataFrame Attributes
# ============================================================================
print("\n\n2️⃣ DATAFRAME ATTRIBUTES")
print("-" * 70)

# Explore Titanic DataFrame structure
print("🚢 TITANIC DataFrame:")
print(f"Shape (rows, cols): {titanic.shape}")  # Tuple: (891 rows, 15 columns)
print(f"Number of rows: {titanic.shape[0]}")  # First element of tuple
print(f"Number of columns: {titanic.shape[1]}")  # Second element
print(f"Total elements: {titanic.size}")  # rows × columns = total cells
print(f"\nColumn names: {list(titanic.columns)}")  # List of all column names
print(f"\nIndex (first 5): {list(titanic.index[:5])}")  # Row labels (0, 1, 2...)

# Explore Tips DataFrame
print("\n🍽️ TIPS DataFrame:")
print(f"Shape: {tips.shape}")
print(f"Column names: {list(tips.columns)}")
print(f"Data types:\n{tips.dtypes}")  # Shows int64, float64, object for each column

# Explore Diamonds DataFrame  
print("\n💎 DIAMONDS DataFrame:")
print(f"Shape: {diamonds.shape}")
print(f"Column names: {list(diamonds.columns)}")
print(f"Memory usage: {diamonds.memory_usage(deep=True).sum() / 1024:.2f} KB")

# ============================================================================
# PART 3: .info() - Quick DataFrame Summary
# ============================================================================
print("\n\n3️⃣ .info() METHOD - Quick Summary")
print("-" * 70)

# .info() shows: column names, non-null counts, data types, memory usage
# This is usually the FIRST method you call on a new dataset
print("🚢 TITANIC .info():")
titanic.info()  # Automatically prints summary

print("\n🍽️ TIPS .info():")
tips.info()

# For large datasets, we can show just a preview
print("\n💎 DIAMONDS .info() (first 1000 rows):")
diamonds.head(1000).info()  # Show info for first 1000 rows only

# ============================================================================
# PART 4: .describe() - Statistical Summary
# ============================================================================
print("\n\n4️⃣ .describe() METHOD - Statistical Summary")
print("-" * 70)

# .describe() calculates statistics for numerical columns only
# Shows: count, mean, std, min, 25%, 50%, 75%, max
print("🚢 TITANIC Numerical Statistics:")
print(titanic.describe())
# Note: Only shows 'age', 'fare', etc. (numerical columns)

print("\n🍽️ TIPS Numerical Statistics:")
print(tips.describe())

# For categorical columns, use include='object'
print("\n🚢 TITANIC Categorical Statistics:")
print(titanic.describe(include='object'))  # Shows: count, unique, top, freq

# ============================================================================
# PART 5: Accessing DataFrame Values
# ============================================================================
print("\n\n5️⃣ ACCESSING VALUES")
print("-" * 70)

# Method 1: Square brackets (like dictionary)
print("Method 1: Square brackets df['column']")
print(f"Titanic ages (first 3): {titanic['age'].head(3).tolist()}")

# Method 2: Dot notation (only works if column name has no spaces)
print("\nMethod 2: Dot notation df.column")
print(f"Tips total_bill (first 3): {tips.total_bill.head(3).tolist()}")

# Method 3: .loc[] for label-based indexing
print("\nMethod 3: .loc[row, column]")
print(f"Titanic passenger 0, age: {titanic.loc[0, 'age']}")
print(f"Tips bill 0, tip amount: {tips.loc[0, 'tip']}")

# Method 4: .iloc[] for integer position-based indexing
print("\nMethod 4: .iloc[row_num, col_num]")
print(f"Titanic row 0, column 0: {titanic.iloc[0, 0]}")  # First row, first column
print(f"Diamonds row 100, column 6: {diamonds.iloc[100, 6]}")  # Row 100, col 6

print("\n✅ Series and DataFrame basics mastered on all 3 datasets!")
print("=" * 70)
```

**Output:**

```
======================================================================
SERIES & DATAFRAME BASICS - PRACTICE ON 3 DATASETS
======================================================================

1️⃣ SERIES (1-Dimensional)
----------------------------------------------------------------------
🚢 TITANIC - Age Series:
Type: <class 'pandas.core.series.Series'>
Shape: (891,)
First 5 values:
0    22.0
1    38.0
2    26.0
3    35.0
4    35.0
Name: age, dtype: float64

🍽️ TIPS - Total Bill Series:
Type: <class 'pandas.core.series.Series'>
Shape: (244,)
First 5 values:
0    16.99
1    10.34
2    21.01
3    23.68
4    24.59
Name: total_bill, dtype: float64

💎 DIAMONDS - Price Series:
Type: <class 'pandas.core.series.Series'>
Shape: (53940,)
First 5 values:
0    326
1    326
2    327
3    334
4    335
Name: price, dtype: int64


2️⃣ DATAFRAME ATTRIBUTES
----------------------------------------------------------------------
🚢 TITANIC DataFrame:
Shape (rows, cols): (891, 15)
Number of rows: 891
Number of columns: 15
Total elements: 13365

Column names: ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'class', 'who', 'adult_male', 'deck', 'embark_town', 'alive', 'alone']

Index (first 5): [0, 1, 2, 3, 4]

🍽️ TIPS DataFrame:
Shape: (244, 7)
Column names: ['total_bill', 'tip', 'sex', 'smoker', 'day', 'time', 'size']
Data types:
total_bill     float64
tip            float64
sex           category
smoker        category
day           category
time          category
size             int64
dtype: object

💎 DIAMONDS DataFrame:
Shape: (53940, 10)
Column names: ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'price', 'x', 'y', 'z']
Memory usage: 3109.77 KB


3️⃣ .info() METHOD - Quick Summary
----------------------------------------------------------------------
🚢 TITANIC .info():
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 15 columns):
 #   Column       Non-Null Count  Dtype   
---  ------       --------------  -----   
 0   survived     891 non-null    int64   
 1   pclass       891 non-null    int64   
 2   sex          891 non-null    object  
 3   age          714 non-null    float64 
 4   sibsp        891 non-null    int64   
 5   parch        891 non-null    int64   
 6   fare         891 non-null    float64 
 7   embarked     889 non-null    object  
 8   class        891 non-null    category
 9   who          891 non-null    object  
 10  adult_male   891 non-null    bool    
 11  deck         203 non-null    category
 12  embark_town  889 non-null    object  
 13  alive        891 non-null    object  
 14  alone        891 non-null    bool    
dtypes: bool(2), category(2), float64(2), int64(4), object(5)
memory usage: 80.7+ KB

🍽️ TIPS .info():
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 244 entries, 0 to 243
Data columns (total 7 columns):
 #   Column      Non-Null Count  Dtype   
---  ------      --------------  -----   
 0   total_bill  244 non-null    float64 
 1   tip         244 non-null    float64 
 2   sex         244 non-null    category
 3   smoker      244 non-null    category
 4   day         244 non-null    category
 5   time        244 non-null    category
 6   size        244 non-null    int64   
dtypes: category(4), float64(2), int64(1)
memory usage: 7.4 KB

💎 DIAMONDS .info() (first 1000 rows):
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1000 entries, 0 to 999
Data columns (total 10 columns):
 #   Column   Non-Null Count  Dtype   
---  ------   --------------  -----   
 0   carat    1000 non-null   float64 
 1   cut      1000 non-null   category
 2   color    1000 non-null   category
 3   clarity  1000 non-null   category
 4   depth    1000 non-null   float64 
 5   table    1000 non-null   float64 
 6   price    1000 non-null   int64   
 7   x        1000 non-null   float64 
 8   y        1000 non-null   float64 
 9   z        1000 non-null   float64 
dtypes: category(3), float64(6), int64(1)
memory usage: 58.7 KB


4️⃣ .describe() METHOD - Statistical Summary
----------------------------------------------------------------------
🚢 TITANIC Numerical Statistics:
         survived      pclass         age       sibsp       parch        fare
count  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000
mean     0.383838    2.308642   29.699118    0.523008    0.381594   32.204208
std      0.486592    0.836071   14.526497    1.102743    0.806057   49.693429
min      0.000000    1.000000    0.420000    0.000000    0.000000    0.000000
25%      0.000000    2.000000   20.125000    0.000000    0.000000    7.910400
50%      0.000000    3.000000   28.000000    0.000000    0.000000   14.454200
75%      1.000000    3.000000   38.000000    1.000000    0.000000   31.000000
max      1.000000    3.000000   80.000000    8.000000    6.000000  512.329200

🍽️ TIPS Numerical Statistics:
       total_bill         tip        size
count  244.000000  244.000000  244.000000
mean    19.785943    2.998279    2.569672
std      8.902412    1.383638    0.951100
min      3.070000    1.000000    1.000000
25%     13.347500    2.000000    2.000000
50%     17.795000    2.900000    2.000000
75%     24.127500    3.562500    3.000000
max     50.810000   10.000000    6.000000

🚢 TITANIC Categorical Statistics:
         sex embarked  who  embark_town alive
count    891      889  891          889   891
unique     2        3    3            3     2
top     male        S  man  Southampton    no
freq     577      644  537          644   549


5️⃣ ACCESSING VALUES
----------------------------------------------------------------------
Method 1: Square brackets df['column']
Titanic ages (first 3): [22.0, 38.0, 26.0]

Method 2: Dot notation df.column
Tips total_bill (first 3): [16.99, 10.34, 21.01]

Method 3: .loc[row, column]
Titanic passenger 0, age: 22.0
Tips bill 0, tip amount: 1.01

Method 4: .iloc[row_num, col_num]
Titanic row 0, column 0: 0
Diamonds row 100, column 6: 2760

✅ Series and DataFrame basics mastered on all 3 datasets!
======================================================================

```

---
<a id="exploration"></a>
## 4️⃣ Data Exploration & Inspection

### 📖 Concept: Understanding Your Data

Before analyzing data, you MUST understand it first. Key questions:
- How much data do I have?
- What columns exist?
- Are there missing values?
- What are the data types?
- What's the distribution of values?

### 🔑 Essential Methods:
1. **`.head(n)`** - First n rows (default=5)
2. **`.tail(n)`** - Last n rows  
3. **`.sample(n)`** - Random n rows
4. **`.info()`** - Summary: columns, types, non-null counts
5. **`.describe()`** - Statistics: mean, std, min, max, etc.
6. **`.value_counts()`** - Count unique values in a column
7. **`.unique()`** - Array of unique values
8. **`.nunique()`** - Number of unique values
9. **`.isnull()`** - Check for missing values
10. **`.dtypes`** - Data types of each column

These methods help you spot issues BEFORE analysis!
```python
print("=" * 70)
print("DATA EXPLORATION - PRACTICE ON 3 DATASETS")
print("=" * 70)

# ============================================================================
# PART 1: Viewing Data - head(), tail(), sample()
# ============================================================================
print("\n1️⃣ VIEWING DATA")
print("-" * 70)

# head() shows first n rows (default=5)
# Use this to quickly see what your data looks like
print("🚢 TITANIC - First 3 passengers:")
print(titanic.head(3))  # Shows first 3 rows with all columns

# tail() shows last n rows
# Useful to check if data was loaded completely
print("\n🍽️ TIPS - Last 3 bills:")
print(tips.tail(3))

# sample() shows random n rows
# Good for getting unbiased view of data
print("\n💎 DIAMONDS - Random 3 diamonds:")
print(diamonds.sample(3, random_state=42))  # random_state=42 for reproducibility

# ============================================================================
# PART 2: Checking Missing Values
# ============================================================================
print("\n\n2️⃣ MISSING VALUES CHECK")
print("-" * 70)

# .isnull() returns True/False for each cell (True = missing)
# .sum() counts True values (because True=1, False=0 in Python)
print("🚢 TITANIC - Missing values per column:")
missing_titanic = titanic.isnull().sum()  # Count missing values per column
print(missing_titanic[missing_titanic > 0])  # Show only columns with missing data

# Calculate percentage of missing data
# This helps decide whether to drop or fill missing values
print("\nMissing data percentage:")
total_cells = titanic.shape[0]  # Total number of rows
missing_pct = (missing_titanic / total_cells * 100).round(2)  # Convert to percentage
print(missing_pct[missing_pct > 0])

print("\n🍽️ TIPS - Missing values:")
tips_missing = tips.isnull().sum()
print(f"Total missing: {tips_missing.sum()}")  # Sum across all columns
if tips_missing.sum() == 0:
    print("✅ No missing values!")

print("\n💎 DIAMONDS - Missing values:")
diamonds_missing = diamonds.isnull().sum()
print(f"Total missing: {diamonds_missing.sum()}")
if diamonds_missing.sum() == 0:
    print("✅ No missing values!")

# ============================================================================
# PART 3: Unique Values & Value Counts
# ============================================================================
print("\n\n3️⃣ UNIQUE VALUES & COUNTS")
print("-" * 70)

# .unique() returns array of unique values in a column
# Useful for categorical data (sex, class, color, etc.)
print("🚢 TITANIC - Unique passenger classes:")
unique_classes = titanic['pclass'].unique()  # Get unique values
print(f"Classes: {unique_classes}")
print(f"Number of unique classes: {titanic['pclass'].nunique()}")  # Count unique

# .value_counts() counts how many times each value appears
# Shows distribution of categorical data
print("\n🚢 TITANIC - Passenger class distribution:")
class_counts = titanic['pclass'].value_counts()  # Count occurrences
print(class_counts)
print("\nAs percentages:")
print(titanic['pclass'].value_counts(normalize=True) * 100)  # normalize=True gives proportions

print("\n🍽️ TIPS - Days distribution:")
print(tips['day'].value_counts())  # Which days have most data?

print("\n🍽️ TIPS - Smoker vs Non-smoker:")
smoker_counts = tips['smoker'].value_counts()
print(smoker_counts)
print(f"Smokers: {smoker_counts['Yes']}, Non-smokers: {smoker_counts['No']}")

print("\n💎 DIAMONDS - Cut quality distribution:")
print(diamonds['cut'].value_counts())  # How many of each cut quality?

print("\n💎 DIAMONDS - Color grades:")
print(diamonds['color'].value_counts().sort_index())  # sort_index() orders alphabetically

# ============================================================================
# PART 4: Statistical Summary with .describe()
# ============================================================================
print("\n\n4️⃣ STATISTICAL SUMMARY")
print("-" * 70)

# .describe() for numerical columns
# Shows: count, mean, std, min, 25%, 50%, 75%, max
print("🚢 TITANIC - Age statistics:")
age_stats = titanic['age'].describe()  # Describe single column
print(age_stats)
print(f"\nInterpretation:")
print(f"  Average age: {age_stats['mean']:.1f} years")
print(f"  Youngest: {age_stats['min']:.0f} years")
print(f"  Oldest: {age_stats['max']:.0f} years")
print(f"  50% of passengers were under {age_stats['50%']:.0f} years")

print("\n🍽️ TIPS - Bill and tip statistics:")
print(tips[['total_bill', 'tip']].describe())  # Describe multiple columns

print("\n💎 DIAMONDS - Price statistics:")
price_stats = diamonds['price'].describe()
print(price_stats)
print(f"\nCheapest diamond: ${price_stats['min']:.0f}")
print(f"Most expensive: ${price_stats['max']:.0f}")
print(f"Average price: ${price_stats['mean']:.0f}")

# .describe() for categorical columns (include='object')
print("\n🚢 TITANIC - Categorical summaries:")
print(titanic[['sex', 'embarked']].describe(include='object'))

# ============================================================================
# PART 5: Correlation Analysis
# ============================================================================
print("\n\n5️⃣ CORRELATION ANALYSIS")
print("-" * 70)

# .corr() calculates correlation between numerical columns
# Values range from -1 (negative correlation) to +1 (positive correlation)
# 0 means no correlation
print("🍽️ TIPS - Correlation matrix:")
tips_corr = tips[['total_bill', 'tip', 'size']].corr()  # Select numerical columns
print(tips_corr)
print(f"\nInterpretation:")
print(f"  total_bill vs tip: {tips_corr.loc['total_bill', 'tip']:.3f}")
print(f"  → Strong positive correlation (bigger bill = bigger tip)")

print("\n💎 DIAMONDS - Price correlations:")
diamonds_numeric = diamonds[['carat', 'depth', 'table', 'price']]
diamonds_corr = diamonds_numeric.corr()
print("Correlation with price:")
print(diamonds_corr['price'].sort_values(ascending=False))  # Sort by correlation strength
print(f"\nCarat has strongest correlation with price: {diamonds_corr.loc['carat', 'price']:.3f}")

print("\n✅ Data exploration completed on all 3 datasets!")
print("=" * 70)
```

**Output:**

```
======================================================================
DATA EXPLORATION - PRACTICE ON 3 DATASETS
======================================================================

1️⃣ VIEWING DATA
----------------------------------------------------------------------
🚢 TITANIC - First 3 passengers:
   survived  pclass     sex   age  sibsp  parch     fare embarked  class  \
0         0       3    male  22.0      1      0   7.2500        S  Third   
1         1       1  female  38.0      1      0  71.2833        C  First   
2         1       3  female  26.0      0      0   7.9250        S  Third   

     who  adult_male deck  embark_town alive  alone  
0    man        True  NaN  Southampton    no  False  
1  woman       False    C    Cherbourg   yes  False  
2  woman       False  NaN  Southampton   yes   True  

🍽️ TIPS - Last 3 bills:
     total_bill   tip     sex smoker   day    time  size
241       22.67  2.00    Male    Yes   Sat  Dinner     2
242       17.82  1.75    Male     No   Sat  Dinner     2
243       18.78  3.00  Female     No  Thur  Dinner     2

💎 DIAMONDS - Random 3 diamonds:
       carat        cut color clarity  depth  table  price     x     y     z
1388    0.24      Ideal     G    VVS1   62.1   56.0    559  3.97  4.00  2.47
50052   0.58  Very Good     F    VVS2   60.0   57.0   2201  5.44  5.42  3.26
41645   0.40      Ideal     E    VVS2   62.1   55.0   1238  4.76  4.74  2.95


2️⃣ MISSING VALUES CHECK
----------------------------------------------------------------------
🚢 TITANIC - Missing values per column:
age            177
embarked         2
deck           688
embark_town      2
dtype: int64

Missing data percentage:
age            19.87
embarked        0.22
deck           77.22
embark_town     0.22
dtype: float64

🍽️ TIPS - Missing values:
Total missing: 0
✅ No missing values!

💎 DIAMONDS - Missing values:
Total missing: 0
✅ No missing values!


3️⃣ UNIQUE VALUES & COUNTS
----------------------------------------------------------------------
🚢 TITANIC - Unique passenger classes:
Classes: [3 1 2]
Number of unique classes: 3

🚢 TITANIC - Passenger class distribution:
pclass
3    491
1    216
2    184
Name: count, dtype: int64

As percentages:
pclass
3    55.106622
1    24.242424
2    20.650954
Name: proportion, dtype: float64

🍽️ TIPS - Days distribution:
day
Sat     87
Sun     76
Thur    62
Fri     19
Name: count, dtype: int64

🍽️ TIPS - Smoker vs Non-smoker:
smoker
No     151
Yes     93
Name: count, dtype: int64
Smokers: 93, Non-smokers: 151

💎 DIAMONDS - Cut quality distribution:
cut
Ideal        21551
Premium      13791
Very Good    12082
Good          4906
Fair          1610
Name: count, dtype: int64

💎 DIAMONDS - Color grades:
color
D     6775
E     9797
F     9542
G    11292
H     8304
I     5422
J     2808
Name: count, dtype: int64


4️⃣ STATISTICAL SUMMARY
----------------------------------------------------------------------
🚢 TITANIC - Age statistics:
count    714.000000
mean      29.699118
std       14.526497
min        0.420000
25%       20.125000
50%       28.000000
75%       38.000000
max       80.000000
Name: age, dtype: float64

Interpretation:
  Average age: 29.7 years
  Youngest: 0 years
  Oldest: 80 years
  50% of passengers were under 28 years

🍽️ TIPS - Bill and tip statistics:
       total_bill         tip
count  244.000000  244.000000
mean    19.785943    2.998279
std      8.902412    1.383638
min      3.070000    1.000000
25%     13.347500    2.000000
50%     17.795000    2.900000
75%     24.127500    3.562500
max     50.810000   10.000000

💎 DIAMONDS - Price statistics:
count    53940.000000
mean      3932.799722
std       3989.439738
min        326.000000
25%        950.000000
50%       2401.000000
75%       5324.250000
max      18823.000000
Name: price, dtype: float64

Cheapest diamond: $326
Most expensive: $18823
Average price: $3933

🚢 TITANIC - Categorical summaries:
         sex embarked
count    891      889
unique     2        3
top     male        S
freq     577      644


5️⃣ CORRELATION ANALYSIS
----------------------------------------------------------------------
🍽️ TIPS - Correlation matrix:
            total_bill       tip      size
total_bill    1.000000  0.675734  0.598315
tip           0.675734  1.000000  0.489299
size          0.598315  0.489299  1.000000

Interpretation:
  total_bill vs tip: 0.676
  → Strong positive correlation (bigger bill = bigger tip)

💎 DIAMONDS - Price correlations:
Correlation with price:
price    1.000000
carat    0.921591
table    0.127134
depth   -0.010647
Name: price, dtype: float64

Carat has strongest correlation with price: 0.922

✅ Data exploration completed on all 3 datasets!
======================================================================

```

---
<a id="indexing"></a>
## 5️⃣ Indexing & Selection

### 📖 Concept: Accessing Data in DataFrames

pandas provides multiple ways to select data:

#### **Selection Methods:**
1. **`df['column']`** - Select single column (returns Series)
2. **`df[['col1', 'col2']]`** - Select multiple columns (returns DataFrame)
3. **`df.loc[row_label, col_label]`** - Label-based indexing
4. **`df.iloc[row_num, col_num]`** - Integer position indexing
5. **`df[df['col'] > value]`** - Boolean indexing (filter rows)

#### **Key Differences:**
- **`.loc`** uses labels: `df.loc[0:5, 'name']` (row labels 0-5, column 'name')
- **`.iloc`** uses positions: `df.iloc[0:5, 0]` (first 5 rows, first column)
- **`.loc`** includes endpoint: `df.loc[0:5]` includes row 5
- **`.iloc`** excludes endpoint: `df.iloc[0:5]` excludes row 5 (like Python slicing)

#### **When to Use:**
- Use **`.loc`** when you know column/row names
- Use **`.iloc`** when working with positions
- Use **`[]`** for simple column selection
```python
print("=" * 70)
print("INDEXING & SELECTION - PRACTICE ON 3 DATASETS")
print("=" * 70)

# ============================================================================
# PART 1: Column Selection
# ============================================================================
print("\n1️⃣ COLUMN SELECTION")
print("-" * 70)

# Method 1: Single column with square brackets (returns Series)
print("🚢 TITANIC - Select 'age' column:")
age_series = titanic['age']  # This is a Series
print(f"Type: {type(age_series)}")  # pandas.core.series.Series
print(f"First 5 ages: {age_series.head().tolist()}")

# Method 2: Multiple columns with double brackets (returns DataFrame)
print("\n🚢 TITANIC - Select multiple columns:")
titanic_subset = titanic[['sex', 'age', 'fare']]  # Note: double [[ ]]
print(f"Type: {type(titanic_subset)}")  # pandas.core.frame.DataFrame
print(f"Shape: {titanic_subset.shape}")
print(titanic_subset.head(3))

print("\n🍽️ TIPS - Select bill and tip columns:")
tips_money = tips[['total_bill', 'tip']]
print(tips_money.head(3))

print("\n💎 DIAMONDS - Select quality features:")
diamond_quality = diamonds[['cut', 'color', 'clarity', 'price']]
print(diamond_quality.head(3))

# ============================================================================
# PART 2: .loc[] - Label-based Indexing
# ============================================================================
print("\n\n2️⃣ .loc[] - LABEL-BASED INDEXING")
print("-" * 70)

# .loc[row_label, column_label]
# Use actual row/column names (not positions)

# Single value selection
print("🚢 TITANIC - Get specific value:")
passenger_0_age = titanic.loc[0, 'age']  # Row 0, column 'age'
print(f"Passenger 0's age: {passenger_0_age}")

# Select multiple rows, single column
print("\n🚢 TITANIC - First 3 passengers' sex:")
first_3_sex = titanic.loc[0:2, 'sex']  # Rows 0-2 (INCLUDES 2!)
print(first_3_sex)

# Select multiple rows, multiple columns
print("\n🚢 TITANIC - Passengers 5-7, selected columns:")
subset = titanic.loc[5:7, ['who', 'sex', 'age']]  # Rows 5-7, 3 columns
print(subset)

# Select all rows, specific columns (use : for all)
print("\n🍽️ TIPS - All rows, bill and tip:")
tips_subset = tips.loc[:, ['total_bill', 'tip']]  # : means "all rows"
print(tips_subset.head(3))

# Boolean condition with .loc
print("\n💎 DIAMONDS - Expensive diamonds (price > 10000):")
expensive = diamonds.loc[diamonds['price'] > 10000, ['carat', 'cut', 'price']]
print(f"Found {len(expensive)} expensive diamonds")
print(expensive.head(3))

# ============================================================================
# PART 3: .iloc[] - Integer Position Indexing
# ============================================================================
print("\n\n3️⃣ .iloc[] - POSITION-BASED INDEXING")
print("-" * 70)

# .iloc[row_position, column_position]
# Use integer positions (0-based, like Python lists)

# Single value by position
print("🚢 TITANIC - Value at row 0, column 0:")
first_cell = titanic.iloc[0, 0]  # First row, first column
print(f"Value: {first_cell}")

# Select rows by position
print("\n🚢 TITANIC - First 3 rows, first 4 columns:")
top_left = titanic.iloc[0:3, 0:4]  # Rows 0-2, Columns 0-3 (EXCLUDES 3 and 4!)
print(top_left)

# Select specific rows and columns by position
print("\n🍽️ TIPS - Rows [0, 5, 10], first 3 columns:")
specific_rows = tips.iloc[[0, 5, 10], 0:3]  # List of row positions
print(specific_rows)

# Get last 3 rows using negative indexing
print("\n💎 DIAMONDS - Last 3 rows, last 2 columns:")
bottom_right = diamonds.iloc[-3:, -2:]  # Last 3 rows, last 2 columns
print(bottom_right)

# Every 100th row (useful for large datasets)
print("\n💎 DIAMONDS - Every 1000th row:")
sampled = diamonds.iloc[::1000, :]  # Start:Stop:Step format
print(f"Sampled {len(sampled)} rows from {len(diamonds)} total")
print(sampled[['carat', 'cut', 'price']])

# ============================================================================
# PART 4: Boolean Indexing (Filtering)
# ============================================================================
print("\n\n4️⃣ BOOLEAN INDEXING - FILTERING ROWS")
print("-" * 70)

# Create boolean mask (True/False array)
# Then use mask to filter DataFrame

# Example 1: Simple condition
print("🚢 TITANIC - Passengers over 60 years old:")
old_mask = titanic['age'] > 60  # Creates True/False for each row
old_passengers = titanic[old_mask]  # Keep only True rows
print(f"Found {len(old_passengers)} passengers over 60")
print(old_passengers[['who', 'age', 'survived']].head(3))

# Example 2: Multiple conditions with & (AND)
print("\n🚢 TITANIC - First class female passengers:")
# & means AND (both conditions must be True)
# | means OR (at least one condition must be True)
# Use parentheses around each condition!
first_class_female = titanic[(titanic['pclass'] == 1) & (titanic['sex'] == 'female')]
print(f"Found {len(first_class_female)} first class females")
print(first_class_female[['who', 'pclass', 'sex', 'age']].head(3))

# Example 3: Multiple conditions with | (OR)
print("\n🍽️ TIPS - Weekend meals (Saturday OR Sunday):")
weekend = tips[(tips['day'] == 'Sat') | (tips['day'] == 'Sun')]
print(f"Found {len(weekend)} weekend meals")
print(f"Average tip on weekends: ${weekend['tip'].mean():.2f}")

# Example 4: .isin() for multiple values
print("\n💎 DIAMONDS - Premium or Ideal cut:")
good_cuts = diamonds[diamonds['cut'].isin(['Premium', 'Ideal'])]  # Multiple values
print(f"Found {len(good_cuts):,} diamonds with Premium or Ideal cut")
print(f"Average price: ${good_cuts['price'].mean():.2f}")

# Example 5: String methods
print("\n🚢 TITANIC - Female passengers:")
# .str allows string operations on text columns
female_passengers = titanic[titanic['who'] == 'woman']
print(f"Found {len(female_passengers)} women passengers")
print(female_passengers[['who', 'sex', 'age']].head(3))

# Example 6: NOT condition with ~
print("\n🍽️ TIPS - Non-smokers:")
# ~ means NOT (flips True↔False)
non_smokers = tips[~(tips['smoker'] == 'Yes')]  # NOT smokers
print(f"Found {len(non_smokers)} non-smokers")
print(f"Average tip (non-smokers): ${non_smokers['tip'].mean():.2f}")

print("\n✅ Indexing and selection mastered on all 3 datasets!")
print("=" * 70)
```

**Output:**

```
======================================================================
INDEXING & SELECTION - PRACTICE ON 3 DATASETS
======================================================================

1️⃣ COLUMN SELECTION
----------------------------------------------------------------------
🚢 TITANIC - Select 'age' column:
Type: <class 'pandas.core.series.Series'>
First 5 ages: [22.0, 38.0, 26.0, 35.0, 35.0]

🚢 TITANIC - Select multiple columns:
Type: <class 'pandas.core.frame.DataFrame'>
Shape: (891, 3)
      sex   age     fare
0    male  22.0   7.2500
1  female  38.0  71.2833
2  female  26.0   7.9250

🍽️ TIPS - Select bill and tip columns:
   total_bill   tip
0       16.99  1.01
1       10.34  1.66
2       21.01  3.50

💎 DIAMONDS - Select quality features:
       cut color clarity  price
0    Ideal     E     SI2    326
1  Premium     E     SI1    326
2     Good     E     VS1    327


2️⃣ .loc[] - LABEL-BASED INDEXING
----------------------------------------------------------------------
🚢 TITANIC - Get specific value:
Passenger 0's age: 22.0

🚢 TITANIC - First 3 passengers' sex:
0      male
1    female
2    female
Name: sex, dtype: object

🚢 TITANIC - Passengers 5-7, selected columns:
     who   sex   age
5    man  male   NaN
6    man  male  54.0
7  child  male   2.0

🍽️ TIPS - All rows, bill and tip:
   total_bill   tip
0       16.99  1.01
1       10.34  1.66
2       21.01  3.50

💎 DIAMONDS - Expensive diamonds (price > 10000):
Found 5222 expensive diamonds
       carat        cut  price
21928   1.70      Ideal  10002
21929   1.03      Ideal  10003
21930   1.23  Very Good  10004


3️⃣ .iloc[] - POSITION-BASED INDEXING
----------------------------------------------------------------------
🚢 TITANIC - Value at row 0, column 0:
Value: 0

🚢 TITANIC - First 3 rows, first 4 columns:
   survived  pclass     sex   age
0         0       3    male  22.0
1         1       1  female  38.0
2         1       3  female  26.0

🍽️ TIPS - Rows [0, 5, 10], first 3 columns:
    total_bill   tip     sex
0        16.99  1.01  Female
5        25.29  4.71    Male
10       10.27  1.71    Male

💎 DIAMONDS - Last 3 rows, last 2 columns:
          y     z
53937  5.68  3.56
53938  6.12  3.74
53939  5.87  3.64

💎 DIAMONDS - Every 1000th row:
Sampled 54 rows from 53940 total
       carat        cut  price
0       0.23      Ideal    326
1000    0.75      Ideal   2898
2000    0.94    Premium   3099
3000    0.70      Ideal   3303
4000    0.53      Ideal   3517
5000    0.87    Premium   3742
6000    1.01    Premium   3959
7000    1.00    Premium   4155
8000    1.01      Ideal   4327
9000    0.91  Very Good   4512
10000   1.00       Fair   4704
11000   1.17       Good   4914
12000   1.01       Good   5147
13000   1.00    Premium   5404
14000   1.19    Premium   5698
15000   1.30       Good   6042
16000   1.57    Premium   6401
17000   1.01      Ideal   6787
18000   1.01       Good   7279
19000   1.00    Premium   7822
20000   1.71    Premium   8540
21000   1.10      Ideal   9215
22000   1.34      Ideal  10070
23000   1.01      Ideal  11057
24000   1.90      Ideal  12165
25000   1.29      Ideal  13530
26000   1.54  Very Good  15225
27000   1.56      Ideal  17108
28000   0.30  Very Good    658
29000   0.34      Ideal    686
30000   0.43      Ideal    716
31000   0.35      Ideal    747
32000   0.28  Very Good    777
33000   0.44  Very Good    813
34000   0.42      Ideal    847
35000   0.43    Premium    882
36000   0.42      Ideal    921
37000   0.50       Good    965
38000   0.43    Premium   1008
39000   0.41      Ideal   1055
40000   0.41    Premium   1107
41000   0.39  Very Good   1183
42000   0.59    Premium   1265
43000   0.50      Ideal   1368
44000   0.30       Good    394
45000   0.57      Ideal   1637
46000   0.55  Very Good   1725
47000   0.53      Ideal   1818
48000   0.52      Ideal   1919
49000   0.73       Fair   2050
50000   0.57      Ideal   2193
51000   0.73  Very Good   2326
52000   0.81  Very Good   2444
53000   0.70      Ideal   2596


4️⃣ BOOLEAN INDEXING - FILTERING ROWS
----------------------------------------------------------------------
🚢 TITANIC - Passengers over 60 years old:
Found 22 passengers over 60
    who   age  survived
33  man  66.0         0
54  man  65.0         0
96  man  71.0         0

🚢 TITANIC - First class female passengers:
Found 94 first class females
      who  pclass     sex   age
1   woman       1  female  38.0
3   woman       1  female  35.0
11  woman       1  female  58.0

🍽️ TIPS - Weekend meals (Saturday OR Sunday):
Found 163 weekend meals
Average tip on weekends: $3.12

💎 DIAMONDS - Premium or Ideal cut:
Found 35,342 diamonds with Premium or Ideal cut
Average price: $3897.20

🚢 TITANIC - Female passengers:
Found 271 women passengers
     who     sex   age
1  woman  female  38.0
2  woman  female  26.0
3  woman  female  35.0

🍽️ TIPS - Non-smokers:
Found 151 non-smokers
Average tip (non-smokers): $2.99

✅ Indexing and selection mastered on all 3 datasets!
======================================================================

```

---

## **4. Filtering & Sorting Data**

**Why Filter and Sort?**
- **Filtering:** Find specific rows that meet criteria (e.g., high-value transactions, failed tests)
- **Sorting:** Arrange data by values to see patterns (e.g., top performers, worst cases)

**Key Methods:**
1. **Boolean indexing:** `df[df['column'] > value]`
2. **Query:** `df.query('column > value')` - more readable for complex conditions
3. **Sort by values:** `df.sort_values(by='column')` - ascending/descending order
4. **Sort by index:** `df.sort_index()` - reorganize by row labels
5. **Largest/Smallest:** `df.nlargest(n, 'column')` and `df.nsmallest(n, 'column')` - quick top/bottom

**Practical Use Cases:**
- Find outliers (e.g., passengers who paid the highest fare)
- Identify patterns (e.g., which day has the best tips)
- Data quality checks (e.g., sort by missing values)
```python
print("=" * 70)
print("FILTERING & SORTING - ADVANCED DATA SELECTION")
print("=" * 70)

# ============================================================================
# PART 1: Advanced Filtering
# ============================================================================
print("\n1️⃣ ADVANCED FILTERING")
print("-" * 70)

# Filter 1: Top fare payers on Titanic
print("🚢 TITANIC - Passengers who paid > $100 fare:")
expensive_tickets = titanic[titanic['fare'] > 100]  # Simple filter
print(f"Found {len(expensive_tickets)} passengers")
# Calculate survival rate for expensive ticket holders
survival_rate = expensive_tickets['survived'].mean() * 100  # 1=survived, 0=died
print(f"Survival rate: {survival_rate:.1f}%")
print(expensive_tickets[['who', 'fare', 'pclass', 'survived']].head(3))

# Filter 2: Using .query() method (more readable)
print("\n🚢 TITANIC - Children (age < 12) in first class:")
# .query() lets you write conditions as strings (easier to read)
first_class_kids = titanic.query('age < 12 and pclass == 1')
print(f"Found {len(first_class_kids)} first class children")
print(first_class_kids[['who', 'age', 'pclass']].head(3))

# Filter 3: Multiple conditions
print("\n🍽️ TIPS - Large parties (>4 people) on weekends with high tips:")
# Combine multiple filters
large_weekend_tips = tips[
    (tips['size'] > 4) &  # More than 4 people
    (tips['day'].isin(['Sat', 'Sun'])) &  # Weekend
    (tips['tip'] > 5)  # Tip over $5
]
print(f"Found {len(large_weekend_tips)} matching meals")
print(f"Average bill: ${large_weekend_tips['total_bill'].mean():.2f}")
print(large_weekend_tips[['total_bill', 'tip', 'size', 'day']].head(3))

# Filter 4: Range filtering
print("\n💎 DIAMONDS - Mid-range carats (1.0 to 1.5):")
mid_carat = diamonds[(diamonds['carat'] >= 1.0) & (diamonds['carat'] <= 1.5)]
print(f"Found {len(mid_carat):,} diamonds in this range")
print(f"Price range: ${mid_carat['price'].min()} - ${mid_carat['price'].max()}")

# Filter 5: String filtering
print("\n🚢 TITANIC - Passengers from Southampton (embarked 'S'):")
southampton = titanic[titanic['embarked'] == 'S']
print(f"Found {len(southampton)} passengers from Southampton")
print(f"Survival rate: {southampton['survived'].mean() * 100:.1f}%")

# ============================================================================
# PART 2: Sorting Data
# ============================================================================
print("\n\n2️⃣ SORTING DATA")
print("-" * 70)

# Sort 1: Single column ascending
print("🚢 TITANIC - Youngest to oldest passengers:")
sorted_by_age = titanic.sort_values(by='age')  # Default: ascending=True
# .dropna() removes missing ages for cleaner display
print(sorted_by_age[['who', 'age']].dropna().head(3))
print(f"Youngest: {sorted_by_age['age'].min():.1f} years")

# Sort 2: Single column descending
print("\n🚢 TITANIC - Most expensive to cheapest fares:")
sorted_by_fare = titanic.sort_values(by='fare', ascending=False)  # Descending
print(sorted_by_fare[['who', 'fare', 'pclass']].head(3))
print(f"Highest fare paid: ${sorted_by_fare['fare'].max():.2f}")

# Sort 3: Multiple columns
print("\n🍽️ TIPS - Sort by day, then by total_bill:")
# First sort by day, then within each day sort by bill
sorted_tips = tips.sort_values(by=['day', 'total_bill'], ascending=[True, False])
print(sorted_tips[['day', 'total_bill', 'tip', 'size']].head(5))

# Sort 4: Using .nlargest() and .nsmallest()
print("\n💎 DIAMONDS - Top 5 most expensive:")
top_5_expensive = diamonds.nlargest(5, 'price')  # Fastest way to get top N
print(top_5_expensive[['carat', 'cut', 'color', 'clarity', 'price']])

print("\n💎 DIAMONDS - 5 cheapest diamonds:")
bottom_5_cheap = diamonds.nsmallest(5, 'price')  # Fastest way to get bottom N
print(bottom_5_cheap[['carat', 'cut', 'color', 'clarity', 'price']])

# Sort 5: Sorting by index
print("\n🍽️ TIPS - Original order (sorted by index):")
# Shuffle then restore original order
shuffled = tips.sample(frac=1)  # Shuffle 100% of rows
print(f"After shuffle: First index is {shuffled.index[0]}")
restored = shuffled.sort_index()  # Sort by index to restore order
print(f"After sort_index(): First index is {restored.index[0]}")

# ============================================================================
# PART 3: Combining Filter + Sort
# ============================================================================
print("\n\n3️⃣ COMBINING FILTERING AND SORTING")
print("-" * 70)

# Example 1: Filter then sort
print("🚢 TITANIC - Top 5 oldest survivors:")
survivors = titanic[titanic['survived'] == 1]  # Step 1: Filter survivors
oldest_survivors = survivors.nlargest(5, 'age')  # Step 2: Get 5 oldest
print(oldest_survivors[['who', 'age', 'sex', 'pclass']])

# Example 2: Complex pipeline
print("\n🍽️ TIPS - Best tippers on Friday:")
# Step 1: Filter Friday
friday_tips = tips[tips['day'] == 'Fri']
# Step 2: Calculate tip percentage
friday_tips = friday_tips.copy()  # Avoid SettingWithCopyWarning
friday_tips['tip_pct'] = (friday_tips['tip'] / friday_tips['total_bill']) * 100
# Step 3: Sort by tip percentage
best_tippers = friday_tips.sort_values('tip_pct', ascending=False).head(3)
print(best_tippers[['total_bill', 'tip', 'tip_pct', 'time']])

# Example 3: Finding outliers
print("\n💎 DIAMONDS - Unusually expensive small diamonds:")
# Find diamonds under 1 carat but over $10,000
small_expensive = diamonds[
    (diamonds['carat'] < 1.0) & 
    (diamonds['price'] > 10000)
].sort_values('price', ascending=False)
print(f"Found {len(small_expensive)} small but expensive diamonds")
if len(small_expensive) > 0:
    print(small_expensive[['carat', 'cut', 'color', 'clarity', 'price']].head(3))

print("\n✅ Filtering and sorting mastered on all 3 datasets!")
print("=" * 70)
```

**Output:**

```
======================================================================
FILTERING & SORTING - ADVANCED DATA SELECTION
======================================================================

1️⃣ ADVANCED FILTERING
----------------------------------------------------------------------
🚢 TITANIC - Passengers who paid > $100 fare:
Found 53 passengers
Survival rate: 73.6%
      who      fare  pclass  survived
27    man  263.0000       1         0
31  woman  146.5208       1         1
88  woman  263.0000       1         1

🚢 TITANIC - Children (age < 12) in first class:
Found 4 first class children
       who   age  pclass
297  child  2.00       1
305  child  0.92       1
445  child  4.00       1

🍽️ TIPS - Large parties (>4 people) on weekends with high tips:
Found 1 matching meals
Average bill: $29.85
     total_bill   tip  size  day
155       29.85  5.14     5  Sun

💎 DIAMONDS - Mid-range carats (1.0 to 1.5):
Found 13,618 diamonds in this range
Price range: $1262 - $18700

🚢 TITANIC - Passengers from Southampton (embarked 'S'):
Found 644 passengers from Southampton
Survival rate: 33.7%


2️⃣ SORTING DATA
----------------------------------------------------------------------
🚢 TITANIC - Youngest to oldest passengers:
       who   age
803  child  0.42
755  child  0.67
644  child  0.75
Youngest: 0.4 years

🚢 TITANIC - Most expensive to cheapest fares:
       who      fare  pclass
679    man  512.3292       1
258  woman  512.3292       1
737    man  512.3292       1
Highest fare paid: $512.33

🍽️ TIPS - Sort by day, then by total_bill:
      day  total_bill   tip  size
197  Thur       43.11  5.00     4
142  Thur       41.19  5.00     5
85   Thur       34.83  5.17     4
141  Thur       34.30  6.70     6
83   Thur       32.68  5.00     2

💎 DIAMONDS - Top 5 most expensive:
       carat        cut color clarity  price
27749   2.29    Premium     I     VS2  18823
27748   2.00  Very Good     G     SI1  18818
27747   1.51      Ideal     G      IF  18806
27746   2.07      Ideal     G     SI2  18804
27745   2.00  Very Good     H     SI1  18803

💎 DIAMONDS - 5 cheapest diamonds:
   carat      cut color clarity  price
0   0.23    Ideal     E     SI2    326
1   0.21  Premium     E     SI1    326
2   0.23     Good     E     VS1    327
3   0.29  Premium     I     VS2    334
4   0.31     Good     J     SI2    335

🍽️ TIPS - Original order (sorted by index):
After shuffle: First index is 37
After sort_index(): First index is 0


3️⃣ COMBINING FILTERING AND SORTING
----------------------------------------------------------------------
🚢 TITANIC - Top 5 oldest survivors:
       who   age     sex  pclass
630    man  80.0    male       1
275  woman  63.0  female       1
483  woman  63.0  female       3
570    man  62.0    male       2
829  woman  62.0  female       1

🍽️ TIPS - Best tippers on Friday:
     total_bill   tip    tip_pct    time
93        16.32  4.30  26.348039  Dinner
221       13.42  3.48  25.931446   Lunch
222        8.58  1.92  22.377622   Lunch

💎 DIAMONDS - Unusually expensive small diamonds:
Found 0 small but expensive diamonds

✅ Filtering and sorting mastered on all 3 datasets!
======================================================================

```

---

## **5. GroupBy Operations - Aggregate & Analyze**

**What is GroupBy?**
GroupBy is like **Excel's Pivot Table** - it splits data into groups and applies functions to each group.

**The Split-Apply-Combine Pattern:**
1. **Split:** Divide data into groups based on some criteria
2. **Apply:** Calculate something for each group (sum, mean, count, etc.)
3. **Combine:** Put results together in a new DataFrame

**Common Use Cases:**
- Average sales per region
- Count of customers per country
- Total revenue per product category
- Mean score per student
- Survival rate per passenger class

**Key Methods:**
- `.groupby('column')` - Group by one column
- `.groupby(['col1', 'col2'])` - Group by multiple columns
- `.agg()` - Apply multiple aggregation functions
- `.apply()` - Apply custom functions
- `.transform()` - Apply function and keep original shape
```python
print("=" * 70)
print("GROUPBY OPERATIONS - SPLIT-APPLY-COMBINE")
print("=" * 70)

# ============================================================================
# PART 1: Basic GroupBy
# ============================================================================
print("\n1️⃣ BASIC GROUPBY")
print("-" * 70)

# Example 1: Group by one column, single aggregation
print("🚢 TITANIC - Survival rate by passenger class:")
survival_by_class = titanic.groupby('pclass')['survived'].mean()  # Mean of 1s and 0s = percentage
print(survival_by_class)
print("\nInterpretation:")
print("- 1st class: {:.1f}% survived".format(survival_by_class[1] * 100))
print("- 2nd class: {:.1f}% survived".format(survival_by_class[2] * 100))
print("- 3rd class: {:.1f}% survived".format(survival_by_class[3] * 100))

# Example 2: Group by one column, count
print("\n🍽️ TIPS - Number of meals per day:")
meals_per_day = tips.groupby('day').size()  # .size() counts rows per group
print(meals_per_day)
print(f"\nBusiest day: {meals_per_day.idxmax()} with {meals_per_day.max()} meals")

# Example 3: Group by one column, sum
print("\n🍽️ TIPS - Total revenue per day:")
revenue_per_day = tips.groupby('day')['total_bill'].sum()  # Sum all bills per day
print(revenue_per_day)
print(f"Total revenue: ${revenue_per_day.sum():.2f}")

# Example 4: Group by categorical column
print("\n💎 DIAMONDS - Average price per cut quality:")
price_by_cut = diamonds.groupby('cut')['price'].mean()  # Mean price per cut
print(price_by_cut.sort_values(ascending=False))  # Sort by price
print(f"\nMost expensive cut (on average): {price_by_cut.idxmax()}")

# ============================================================================
# PART 2: GroupBy with Multiple Columns
# ============================================================================
print("\n\n2️⃣ GROUPBY WITH MULTIPLE COLUMNS")
print("-" * 70)

# Example 1: Two columns
print("🚢 TITANIC - Survival rate by class AND sex:")
survival_class_sex = titanic.groupby(['pclass', 'sex'])['survived'].mean()
print(survival_class_sex)
print("\nKey insight: Women in 1st class had {:.1f}% survival".format(
    survival_class_sex[1, 'female'] * 100
))

# Example 2: Count combinations
print("\n🍽️ TIPS - Meals by day AND time:")
meals_day_time = tips.groupby(['day', 'time']).size()  # Count per group
print(meals_day_time)

# Example 3: Multiple columns with aggregation
print("\n💎 DIAMONDS - Average price by cut AND color:")
price_cut_color = diamonds.groupby(['cut', 'color'])['price'].mean()
# Show top 5 combinations
print("Top 5 expensive combinations:")
print(price_cut_color.sort_values(ascending=False).head())

# ============================================================================
# PART 3: Multiple Aggregations with .agg()
# ============================================================================
print("\n\n3️⃣ MULTIPLE AGGREGATIONS")
print("-" * 70)

# Example 1: Multiple functions on one column
print("🚢 TITANIC - Fare statistics by class:")
fare_stats = titanic.groupby('pclass')['fare'].agg(['mean', 'median', 'min', 'max', 'std'])
# .agg() takes a list of function names
print(fare_stats.round(2))  # Round to 2 decimal places

# Example 2: Different functions for different columns
print("\n🍽️ TIPS - Summary by day:")
day_summary = tips.groupby('day').agg({
    'total_bill': ['sum', 'mean'],  # Two functions for total_bill
    'tip': ['sum', 'mean'],          # Two functions for tip
    'size': 'mean'                   # One function for size
})
print(day_summary.round(2))

# Example 3: Named aggregations (cleaner column names)
print("\n💎 DIAMONDS - Price summary by cut:")
cut_summary = diamonds.groupby('cut').agg(
    avg_price=('price', 'mean'),      # New name = (column, function)
    min_price=('price', 'min'),
    max_price=('price', 'max'),
    count=('price', 'count')          # Count how many in each group
)
print(cut_summary.round(0))  # No decimals for prices

# ============================================================================
# PART 4: Filtering Groups
# ============================================================================
print("\n\n4️⃣ FILTERING GROUPS")
print("-" * 70)

# Example 1: Keep groups that meet a condition
print("🚢 TITANIC - Embarkation ports with > 100 passengers:")
grouped = titanic.groupby('embarked')
# .filter() keeps entire groups that meet condition
large_ports = grouped.filter(lambda x: len(x) > 100)
print(large_ports['embarked'].value_counts())

# Example 2: Filter based on group statistics
print("\n🍽️ TIPS - Days where average tip > $3:")
day_groups = tips.groupby('day')
# Keep groups where mean tip exceeds threshold
good_tip_days = day_groups.filter(lambda x: x['tip'].mean() > 3)
print(good_tip_days.groupby('day')['tip'].mean().sort_values(ascending=False))

# ============================================================================
# PART 5: Transform - Keep Original Shape
# ============================================================================
print("\n\n5️⃣ TRANSFORM - ADD GROUP STATS TO ORIGINAL DATA")
print("-" * 70)

# Example 1: Add group mean to each row
print("🚢 TITANIC - Compare individual fare to class average:")
# Calculate mean fare per class
titanic_with_avg = titanic.copy()
titanic_with_avg['class_avg_fare'] = titanic.groupby('pclass')['fare'].transform('mean')
# Now each row has both individual fare AND class average
titanic_with_avg['fare_vs_avg'] = titanic_with_avg['fare'] - titanic_with_avg['class_avg_fare']
print(titanic_with_avg[['who', 'pclass', 'fare', 'class_avg_fare', 'fare_vs_avg']].head(5))

# Example 2: Standardize within groups
print("\n🍽️ TIPS - Tip as percentage of day's average:")
tips_with_pct = tips.copy()
# Transform keeps original DataFrame shape (244 rows)
tips_with_pct['day_avg_tip'] = tips.groupby('day')['tip'].transform('mean')
tips_with_pct['tip_vs_day_avg'] = (tips_with_pct['tip'] / tips_with_pct['day_avg_tip']) * 100
print(tips_with_pct[['day', 'tip', 'day_avg_tip', 'tip_vs_day_avg']].head())

# ============================================================================
# PART 6: Advanced GroupBy
# ============================================================================
print("\n\n6️⃣ ADVANCED GROUPBY")
print("-" * 70)

# Example 1: Custom aggregation function
print("🚢 TITANIC - Age range per class:")
def age_range(ages):
    """Calculate age range (max - min)"""
    return ages.max() - ages.min()

age_ranges = titanic.groupby('pclass')['age'].agg(age_range)  # Apply custom function
print(age_ranges)

# Example 2: Percentage of total
print("\n🍽️ TIPS - Each day's % of total revenue:")
day_revenue = tips.groupby('day')['total_bill'].sum()
total_revenue = tips['total_bill'].sum()
day_pct = (day_revenue / total_revenue * 100).round(1)
print(day_pct.sort_values(ascending=False))

# Example 3: Top N per group
print("\n💎 DIAMONDS - Most expensive diamond per cut:")
top_per_cut = diamonds.groupby('cut').apply(
    lambda x: x.nlargest(1, 'price')[['cut', 'carat', 'color', 'clarity', 'price']]
)
print(top_per_cut.reset_index(drop=True))

print("\n✅ GroupBy operations mastered on all 3 datasets!")
print("=" * 70)
```

**Output:**

```
======================================================================
GROUPBY OPERATIONS - SPLIT-APPLY-COMBINE
======================================================================

1️⃣ BASIC GROUPBY
----------------------------------------------------------------------
🚢 TITANIC - Survival rate by passenger class:
pclass
1    0.629630
2    0.472826
3    0.242363
Name: survived, dtype: float64

Interpretation:
- 1st class: 63.0% survived
- 2nd class: 47.3% survived
- 3rd class: 24.2% survived

🍽️ TIPS - Number of meals per day:
day
Thur    62
Fri     19
Sat     87
Sun     76
dtype: int64

Busiest day: Sat with 87 meals

🍽️ TIPS - Total revenue per day:
day
Thur    1096.33
Fri      325.88
Sat     1778.40
Sun     1627.16
Name: total_bill, dtype: float64
Total revenue: $4827.77

💎 DIAMONDS - Average price per cut quality:
cut
Premium      4584.257704
Fair         4358.757764
Very Good    3981.759891
Good         3928.864452
Ideal        3457.541970
Name: price, dtype: float64

Most expensive cut (on average): Premium


2️⃣ GROUPBY WITH MULTIPLE COLUMNS
----------------------------------------------------------------------
🚢 TITANIC - Survival rate by class AND sex:
pclass  sex   
1       female    0.968085
        male      0.368852
2       female    0.921053
        male      0.157407
3       female    0.500000
        male      0.135447
Name: survived, dtype: float64

Key insight: Women in 1st class had 96.8% survival

🍽️ TIPS - Meals by day AND time:
day   time  
Thur  Lunch     61
      Dinner     1
Fri   Lunch      7
      Dinner    12
Sat   Lunch      0
      Dinner    87
Sun   Lunch      0
      Dinner    76
dtype: int64

💎 DIAMONDS - Average price by cut AND color:
Top 5 expensive combinations:
cut        color
Premium    J        6294.591584
           I        5946.180672
Very Good  I        5255.879568
Premium    H        5216.706780
Fair       H        5135.683168
Name: price, dtype: float64


3️⃣ MULTIPLE AGGREGATIONS
----------------------------------------------------------------------
🚢 TITANIC - Fare statistics by class:
         mean  median  min     max    std
pclass                                   
1       84.15   60.29  0.0  512.33  78.38
2       20.66   14.25  0.0   73.50  13.42
3       13.68    8.05  0.0   69.55  11.78

🍽️ TIPS - Summary by day:
     total_bill            tip        size
            sum   mean     sum  mean  mean
day                                       
Thur    1096.33  17.68  171.83  2.77  2.45
Fri      325.88  17.15   51.96  2.73  2.11
Sat     1778.40  20.44  260.40  2.99  2.52
Sun     1627.16  21.41  247.39  3.26  2.84

💎 DIAMONDS - Price summary by cut:
           avg_price  min_price  max_price  count
cut                                              
Ideal         3458.0        326      18806  21551
Premium       4584.0        326      18823  13791
Very Good     3982.0        336      18818  12082
Good          3929.0        327      18788   4906
Fair          4359.0        337      18574   1610


4️⃣ FILTERING GROUPS
----------------------------------------------------------------------
🚢 TITANIC - Embarkation ports with > 100 passengers:
embarked
S    644
C    168
Name: count, dtype: int64

🍽️ TIPS - Days where average tip > $3:
day
Sun     3.255132
Thur         NaN
Fri          NaN
Sat          NaN
Name: tip, dtype: float64


5️⃣ TRANSFORM - ADD GROUP STATS TO ORIGINAL DATA
----------------------------------------------------------------------
🚢 TITANIC - Compare individual fare to class average:
     who  pclass     fare  class_avg_fare  fare_vs_avg
0    man       3   7.2500       13.675550    -6.425550
1  woman       1  71.2833       84.154687   -12.871387
2  woman       3   7.9250       13.675550    -5.750550
3  woman       1  53.1000       84.154687   -31.054687
4    man       3   8.0500       13.675550    -5.625550

🍽️ TIPS - Tip as percentage of day's average:
   day   tip  day_avg_tip  tip_vs_day_avg
0  Sun  1.01     3.255132       31.027932
1  Sun  1.66     3.255132       50.996402
2  Sun  3.50     3.255132      107.522535
3  Sun  3.31     3.255132      101.685598
4  Sun  3.61     3.255132      110.901815


6️⃣ ADVANCED GROUPBY
----------------------------------------------------------------------
🚢 TITANIC - Age range per class:
pclass
1    79.08
2    69.33
3    73.58
Name: age, dtype: float64

🍽️ TIPS - Each day's % of total revenue:
day
Sat     36.8
Sun     33.7
Thur    22.7
Fri      6.8
Name: total_bill, dtype: float64

💎 DIAMONDS - Most expensive diamond per cut:
         cut  carat color clarity  price
0      Ideal   1.51     G      IF  18806
1    Premium   2.29     I     VS2  18823
2  Very Good   2.00     G     SI1  18818
3       Good   2.80     G     SI2  18788
4       Fair   2.01     G     SI1  18574

✅ GroupBy operations mastered on all 3 datasets!
======================================================================

```

```
C:\Users\tparl\AppData\Local\Temp\ipykernel_18232\2232890795.py:22: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  meals_per_day = tips.groupby('day').size()  # .size() counts rows per group
C:\Users\tparl\AppData\Local\Temp\ipykernel_18232\2232890795.py:28: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  revenue_per_day = tips.groupby('day')['total_bill'].sum()  # Sum all bills per day
C:\Users\tparl\AppData\Local\Temp\ipykernel_18232\2232890795.py:34: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  price_by_cut = diamonds.groupby('cut')['price'].mean()  # Mean price per cut
C:\Users\tparl\AppData\Local\Temp\ipykernel_18232\2232890795.py:54: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  meals_day_time = tips.groupby(['day', 'time']).size()  # Count per group
C:\Users\tparl\AppData\Local\Temp\ipykernel_18232\2232890795.py:59: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  price_cut_color = diamonds.groupby(['cut', 'color'])['price'].mean()
C:\Users\tparl\AppData\Local\Temp\ipykernel_18232\2232890795.py:78: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  day_summary = tips.groupby('day').agg({
C:\Users\tparl\AppData\Local\Temp\ipykernel_18232\2232890795.py:87: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  cut_summary = diamonds.groupby('cut').agg(
C:\Users\tparl\AppData\Local\Temp\ipykernel_18232\2232890795.py:110: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  day_groups = tips.groupby('day')
C:\Users\tparl\AppData\Local\Temp\ipykernel_18232\2232890795.py:113: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  print(good_tip_days.groupby('day')['tip'].mean().sort_values(ascending=False))
C:\Users\tparl\AppData\Local\Temp\ipykernel_18232\2232890795.py:134: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  tips_with_pct['day_avg_tip'] = tips.groupby('day')['tip'].transform('mean')
C:\Users\tparl\AppData\Local\Temp\ipykernel_18232\2232890795.py:155: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  day_revenue = tips.groupby('day')['total_bill'].sum()
C:\Users\tparl\AppData\Local\Temp\ipykernel_18232\2232890795.py:162: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  top_per_cut = diamonds.groupby('cut').apply(
C:\Users\tparl\AppData\Local\Temp\ipykernel_18232\2232890795.py:162: FutureWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
  top_per_cut = diamonds.groupby('cut').apply(

```

---

## **6. Merging & Joining DataFrames**

**Why Merge Data?**
Real-world data is often split across multiple tables/files. Like SQL joins, pandas lets you combine DataFrames based on common columns.

**Types of Joins:**
1. **Inner Join:** Keep only rows that exist in BOTH DataFrames (intersection)
2. **Left Join:** Keep ALL rows from left DataFrame, match from right where possible
3. **Right Join:** Keep ALL rows from right DataFrame, match from left where possible
4. **Outer Join:** Keep ALL rows from BOTH DataFrames (union)

**Key Methods:**
- `pd.merge(df1, df2, on='key')` - Join on common column
- `pd.concat([df1, df2])` - Stack DataFrames vertically or horizontally
- `df1.join(df2)` - Join on index

**Real-World Examples:**
- Combine customer info + order history
- Match students with their exam scores
- Link products with supplier details
```python
print("=" * 70)
print("MERGING & JOINING DATAFRAMES")
print("=" * 70)

# ============================================================================
# PART 1: Creating Sample DataFrames for Merging
# ============================================================================
print("\n1️⃣ SAMPLE DATA PREPARATION")
print("-" * 70)

# Create two small DataFrames to demonstrate joins
passengers_info = pd.DataFrame({
    'passenger_id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, 30, 35, 40, 28]
})

ticket_info = pd.DataFrame({
    'passenger_id': [1, 2, 3, 6, 7],  # Note: 6 and 7 don't exist in passengers_info
    'ticket_class': ['First', 'Economy', 'Business', 'Economy', 'First'],
    'fare': [250, 80, 150, 90, 260]
})

print("PASSENGERS INFO:")
print(passengers_info)
print("\nTICKET INFO:")
print(ticket_info)
print("\nNotice: Some passenger_ids exist in one table but not the other!")

# ============================================================================
# PART 2: Inner Join - Keep Only Matching Rows
# ============================================================================
print("\n\n2️⃣ INNER JOIN - INTERSECTION")
print("-" * 70)

# Inner join: Keep only rows where passenger_id exists in BOTH tables
inner_merged = pd.merge(passengers_info, ticket_info, on='passenger_id', how='inner')
print("Inner join result (only passengers 1, 2, 3):")
print(inner_merged)
print(f"\nRows: {len(inner_merged)} (only matching IDs: 1, 2, 3)")

# ============================================================================
# PART 3: Left Join - Keep All Left Rows
# ============================================================================
print("\n\n3️⃣ LEFT JOIN - KEEP ALL LEFT ROWS")
print("-" * 70)

# Left join: Keep ALL passengers, add ticket info where available
left_merged = pd.merge(passengers_info, ticket_info, on='passenger_id', how='left')
print("Left join result (all passengers):")
print(left_merged)
print("\nNotice: David and Eve have NaN (missing) ticket info because IDs 4 and 5 don't exist in ticket_info")

# ============================================================================
# PART 4: Right Join - Keep All Right Rows
# ============================================================================
print("\n\n4️⃣ RIGHT JOIN - KEEP ALL RIGHT ROWS")
print("-" * 70)

# Right join: Keep ALL tickets, add passenger info where available
right_merged = pd.merge(passengers_info, ticket_info, on='passenger_id', how='right')
print("Right join result (all tickets):")
print(right_merged)
print("\nNotice: Tickets for IDs 6 and 7 have NaN passenger info")

# ============================================================================
# PART 5: Outer Join - Keep All Rows from Both
# ============================================================================
print("\n\n5️⃣ OUTER JOIN - UNION")
print("-" * 70)

# Outer join: Keep ALL rows from both tables
outer_merged = pd.merge(passengers_info, ticket_info, on='passenger_id', how='outer')
print("Outer join result (all passengers AND all tickets):")
print(outer_merged)
print(f"\nRows: {len(outer_merged)} (includes all unique IDs: 1, 2, 3, 4, 5, 6, 7)")

# ============================================================================
# PART 6: Merging on Multiple Columns
# ============================================================================
print("\n\n6️⃣ MERGING ON MULTIPLE COLUMNS")
print("-" * 70)

# Create DataFrames with multiple key columns
sales = pd.DataFrame({
    'date': ['2024-01-01', '2024-01-01', '2024-01-02'],
    'product': ['A', 'B', 'A'],
    'quantity': [10, 5, 8]
})

prices = pd.DataFrame({
    'date': ['2024-01-01', '2024-01-01', '2024-01-02'],
    'product': ['A', 'B', 'A'],
    'price': [100, 200, 110]
})

# Merge on BOTH date and product
merged_sales = pd.merge(sales, prices, on=['date', 'product'])
# Calculate total revenue
merged_sales['revenue'] = merged_sales['quantity'] * merged_sales['price']

print("SALES:")
print(sales)
print("\nPRICES:")
print(prices)
print("\nMERGED (with calculated revenue):")
print(merged_sales)

# ============================================================================
# PART 7: Merging with Different Column Names
# ============================================================================
print("\n\n7️⃣ MERGING WITH DIFFERENT COLUMN NAMES")
print("-" * 70)

# Sometimes key columns have different names
customers = pd.DataFrame({
    'customer_id': [101, 102, 103],
    'customer_name': ['Alice', 'Bob', 'Charlie']
})

orders = pd.DataFrame({
    'order_id': [1, 2, 3],
    'cust_id': [101, 102, 103],  # Different name!
    'amount': [500, 300, 700]
})

# Use left_on and right_on to specify different column names
merged_orders = pd.merge(
    customers, 
    orders, 
    left_on='customer_id',  # Column from left DataFrame
    right_on='cust_id'       # Column from right DataFrame
)
print("CUSTOMERS:")
print(customers)
print("\nORDERS:")
print(orders)
print("\nMERGED:")
print(merged_orders)
print("\nNotice: Both customer_id and cust_id are kept (they're duplicates)")

# ============================================================================
# PART 8: Concatenating DataFrames
# ============================================================================
print("\n\n8️⃣ CONCATENATING DATAFRAMES")
print("-" * 70)

# pd.concat() stacks DataFrames vertically or horizontally

# Example 1: Vertical stacking (adding more rows)
df1 = tips.head(3)  # First 3 rows
df2 = tips.tail(3)  # Last 3 rows

stacked_vertical = pd.concat([df1, df2], ignore_index=True)  # Stack vertically
print("Vertical concat (6 rows total):")
print(stacked_vertical[['total_bill', 'tip', 'day']])

# Example 2: Horizontal stacking (adding more columns)
df_left = tips[['total_bill', 'tip']].head(3)
df_right = tips[['sex', 'day', 'time']].head(3)

stacked_horizontal = pd.concat([df_left, df_right], axis=1)  # axis=1 means columns
print("\nHorizontal concat:")
print(stacked_horizontal)

# ============================================================================
# PART 9: Real-World Example with Titanic
# ============================================================================
print("\n\n9️⃣ REAL-WORLD EXAMPLE - TITANIC")
print("-" * 70)

# Split Titanic into two tables, then rejoin
# Table 1: Personal info (add passenger_id for merging)
personal = titanic[['sex', 'age']].head(5).reset_index()
personal = personal.rename(columns={'index': 'passenger_id'})

# Table 2: Travel info (same passengers)
travel = titanic[['pclass', 'fare', 'embarked']].head(5).reset_index()
travel = travel.rename(columns={'index': 'passenger_id'})

# Merge on 'passenger_id'
full_info = pd.merge(personal, travel, on='passenger_id')
print("PERSONAL INFO:")
print(personal)
print("\nTRAVEL INFO:")
print(travel)
print("\nMERGED FULL INFO:")
print(full_info)

# ============================================================================
# PART 10: Handling Duplicate Keys
# ============================================================================
print("\n\n🔟 HANDLING DUPLICATE KEYS")
print("-" * 70)

# What happens when the same key appears multiple times?
df_a = pd.DataFrame({
    'key': ['A', 'B', 'B'],  # B appears twice!
    'value_a': [1, 2, 3]
})

df_b = pd.DataFrame({
    'key': ['A', 'B'],
    'value_b': [10, 20]
})

merged_duplicates = pd.merge(df_a, df_b, on='key')
print("DF_A (B appears twice):")
print(df_a)
print("\nDF_B:")
print(df_b)
print("\nMERGED:")
print(merged_duplicates)
print("\nNotice: B from df_a (both rows) matched with B from df_b")
print("Result: 3 rows total (A once, B twice)")

print("\n✅ Merging and joining mastered!")
print("=" * 70)
```

**Output:**

```
======================================================================
MERGING & JOINING DATAFRAMES
======================================================================

1️⃣ SAMPLE DATA PREPARATION
----------------------------------------------------------------------
PASSENGERS INFO:
   passenger_id     name  age
0             1    Alice   25
1             2      Bob   30
2             3  Charlie   35
3             4    David   40
4             5      Eve   28

TICKET INFO:
   passenger_id ticket_class  fare
0             1        First   250
1             2      Economy    80
2             3     Business   150
3             6      Economy    90
4             7        First   260

Notice: Some passenger_ids exist in one table but not the other!


2️⃣ INNER JOIN - INTERSECTION
----------------------------------------------------------------------
Inner join result (only passengers 1, 2, 3):
   passenger_id     name  age ticket_class  fare
0             1    Alice   25        First   250
1             2      Bob   30      Economy    80
2             3  Charlie   35     Business   150

Rows: 3 (only matching IDs: 1, 2, 3)


3️⃣ LEFT JOIN - KEEP ALL LEFT ROWS
----------------------------------------------------------------------
Left join result (all passengers):
   passenger_id     name  age ticket_class   fare
0             1    Alice   25        First  250.0
1             2      Bob   30      Economy   80.0
2             3  Charlie   35     Business  150.0
3             4    David   40          NaN    NaN
4             5      Eve   28          NaN    NaN

Notice: David and Eve have NaN (missing) ticket info because IDs 4 and 5 don't exist in ticket_info


4️⃣ RIGHT JOIN - KEEP ALL RIGHT ROWS
----------------------------------------------------------------------
Right join result (all tickets):
   passenger_id     name   age ticket_class  fare
0             1    Alice  25.0        First   250
1             2      Bob  30.0      Economy    80
2             3  Charlie  35.0     Business   150
3             6      NaN   NaN      Economy    90
4             7      NaN   NaN        First   260

Notice: Tickets for IDs 6 and 7 have NaN passenger info


5️⃣ OUTER JOIN - UNION
----------------------------------------------------------------------
Outer join result (all passengers AND all tickets):
   passenger_id     name   age ticket_class   fare
0             1    Alice  25.0        First  250.0
1             2      Bob  30.0      Economy   80.0
2             3  Charlie  35.0     Business  150.0
3             4    David  40.0          NaN    NaN
4             5      Eve  28.0          NaN    NaN
5             6      NaN   NaN      Economy   90.0
6             7      NaN   NaN        First  260.0

Rows: 7 (includes all unique IDs: 1, 2, 3, 4, 5, 6, 7)


6️⃣ MERGING ON MULTIPLE COLUMNS
----------------------------------------------------------------------
SALES:
         date product  quantity
0  2024-01-01       A        10
1  2024-01-01       B         5
2  2024-01-02       A         8

PRICES:
         date product  price
0  2024-01-01       A    100
1  2024-01-01       B    200
2  2024-01-02       A    110

MERGED (with calculated revenue):
         date product  quantity  price  revenue
0  2024-01-01       A        10    100     1000
1  2024-01-01       B         5    200     1000
2  2024-01-02       A         8    110      880


7️⃣ MERGING WITH DIFFERENT COLUMN NAMES
----------------------------------------------------------------------
CUSTOMERS:
   customer_id customer_name
0          101         Alice
1          102           Bob
2          103       Charlie

ORDERS:
   order_id  cust_id  amount
0         1      101     500
1         2      102     300
2         3      103     700

MERGED:
   customer_id customer_name  order_id  cust_id  amount
0          101         Alice         1      101     500
1          102           Bob         2      102     300
2          103       Charlie         3      103     700

Notice: Both customer_id and cust_id are kept (they're duplicates)


8️⃣ CONCATENATING DATAFRAMES
----------------------------------------------------------------------
Vertical concat (6 rows total):
   total_bill   tip   day
0       16.99  1.01   Sun
1       10.34  1.66   Sun
2       21.01  3.50   Sun
3       22.67  2.00   Sat
4       17.82  1.75   Sat
5       18.78  3.00  Thur

Horizontal concat:
   total_bill   tip     sex  day    time
0       16.99  1.01  Female  Sun  Dinner
1       10.34  1.66    Male  Sun  Dinner
2       21.01  3.50    Male  Sun  Dinner


9️⃣ REAL-WORLD EXAMPLE - TITANIC
----------------------------------------------------------------------
PERSONAL INFO:
   passenger_id     sex   age
0             0    male  22.0
1             1  female  38.0
2             2  female  26.0
3             3  female  35.0
4             4    male  35.0

TRAVEL INFO:
   passenger_id  pclass     fare embarked
0             0       3   7.2500        S
1             1       1  71.2833        C
2             2       3   7.9250        S
3             3       1  53.1000        S
4             4       3   8.0500        S

MERGED FULL INFO:
   passenger_id     sex   age  pclass     fare embarked
0             0    male  22.0       3   7.2500        S
1             1  female  38.0       1  71.2833        C
2             2  female  26.0       3   7.9250        S
3             3  female  35.0       1  53.1000        S
4             4    male  35.0       3   8.0500        S


🔟 HANDLING DUPLICATE KEYS
----------------------------------------------------------------------
DF_A (B appears twice):
  key  value_a
0   A        1
1   B        2
2   B        3

DF_B:
  key  value_b
0   A       10
1   B       20

MERGED:
  key  value_a  value_b
0   A        1       10
1   B        2       20
2   B        3       20

Notice: B from df_a (both rows) matched with B from df_b
Result: 3 rows total (A once, B twice)

✅ Merging and joining mastered!
======================================================================

```

---

## **7. Handling Missing Data**

**Why Missing Data Matters:**
Real-world datasets are NEVER perfect. Missing values can:
- Break calculations (e.g., mean of [1, 2, NaN] fails)
- Cause model errors in machine learning
- Lead to incorrect insights

**Types of Missing Data:**
- `NaN` (Not a Number) - pandas default for missing values
- `None` - Python's null value
- Empty strings or placeholder values (e.g., -999, "N/A")

**Strategies:**
1. **Detect:** Find where missing values are
2. **Drop:** Remove rows/columns with missing values (simple but loses data)
3. **Fill:** Replace missing values with something sensible (mean, median, forward fill, etc.)
4. **Interpolate:** Estimate missing values from nearby values

**Key Methods:**
- `.isnull()` / `.isna()` - Detect missing values
- `.dropna()` - Remove missing values
- `.fillna()` - Replace missing values
- `.interpolate()` - Estimate missing values
```python
print("=" * 70)
print("HANDLING MISSING DATA")
print("=" * 70)

# ============================================================================
# PART 1: Detecting Missing Values
# ============================================================================
print("\n1️⃣ DETECTING MISSING VALUES")
print("-" * 70)

# Check Titanic for missing values
print("🚢 TITANIC - Missing value summary:")
missing_summary = titanic.isnull().sum()  # Count NaN per column
print(missing_summary[missing_summary > 0])  # Show only columns with missing data

# Calculate percentage missing
total_rows = len(titanic)
missing_pct = (missing_summary / total_rows * 100).round(1)
print("\nAs percentages:")
print(missing_pct[missing_pct > 0])

# Visualize missing data pattern
print("\n🚢 TITANIC - First 10 rows, showing which values are missing:")
print(titanic[['age', 'deck', 'embarked']].head(10).isnull())
# True = missing, False = present

# Check Tips dataset
print("\n🍽️ TIPS - Missing values:")
tips_missing = tips.isnull().sum()
if tips_missing.sum() == 0:
    print("✅ No missing values! This dataset is clean.")
else:
    print(tips_missing[tips_missing > 0])

# Check Diamonds dataset
print("\n💎 DIAMONDS - Missing values:")
diamonds_missing = diamonds.isnull().sum()
if diamonds_missing.sum() == 0:
    print("✅ No missing values! This dataset is clean.")
else:
    print(diamonds_missing[diamonds_missing > 0])

# ============================================================================
# PART 2: Dropping Missing Values
# ============================================================================
print("\n\n2️⃣ DROPPING MISSING VALUES")
print("-" * 70)

# Strategy 1: Drop rows with ANY missing value
print("🚢 TITANIC - Original shape:", titanic.shape)
titanic_dropany = titanic.dropna()  # Drop rows with any NaN
print(f"After dropna(): {titanic_dropany.shape}")
print(f"Lost {len(titanic) - len(titanic_dropany)} rows ({((len(titanic) - len(titanic_dropany))/len(titanic)*100):.1f}%)")

# Strategy 2: Drop rows only if ALL values are missing
print("\n🚢 TITANIC - Drop rows where ALL values are NaN:")
titanic_dropall = titanic.dropna(how='all')  # Only drop if entire row is NaN
print(f"After dropna(how='all'): {titanic_dropall.shape}")
print("Usually same as original because complete empty rows are rare")

# Strategy 3: Drop rows with missing values in specific columns
print("\n🚢 TITANIC - Drop rows where AGE is missing:")
titanic_no_missing_age = titanic.dropna(subset=['age'])  # Only check 'age' column
print(f"After dropna(subset=['age']): {titanic_no_missing_age.shape}")
print(f"Lost {len(titanic) - len(titanic_no_missing_age)} rows with missing age")

# Strategy 4: Drop columns with too many missing values
print("\n🚢 TITANIC - Drop columns with > 50% missing:")
threshold = len(titanic) * 0.5  # 50% of rows
titanic_drop_cols = titanic.dropna(axis=1, thresh=threshold)
# axis=1 means columns, thresh=minimum non-NaN values required
print(f"Dropped columns: {set(titanic.columns) - set(titanic_drop_cols.columns)}")

# ============================================================================
# PART 3: Filling Missing Values
# ============================================================================
print("\n\n3️⃣ FILLING MISSING VALUES")
print("-" * 70)

# Create a copy to avoid modifying original
titanic_filled = titanic.copy()

# Strategy 1: Fill with a constant value
print("🚢 TITANIC - Fill missing EMBARKED with 'Unknown':")
titanic_filled['embarked'] = titanic_filled['embarked'].fillna('Unknown')
print(f"Embarked missing count: {titanic_filled['embarked'].isnull().sum()}")
print(f"Value counts:\n{titanic_filled['embarked'].value_counts()}")

# Strategy 2: Fill numerical columns with mean
print("\n🚢 TITANIC - Fill missing AGE with mean:")
mean_age = titanic['age'].mean()  # Calculate mean age
titanic_filled['age'] = titanic_filled['age'].fillna(mean_age)
print(f"Mean age: {mean_age:.1f} years")
print(f"Age missing count after fill: {titanic_filled['age'].isnull().sum()}")

# Strategy 3: Fill with median (better for skewed data)
titanic_filled2 = titanic.copy()
median_age = titanic['age'].median()  # Middle value
titanic_filled2['age'] = titanic_filled2['age'].fillna(median_age)
print(f"\n🚢 TITANIC - Fill missing AGE with median:")
print(f"Median age: {median_age:.1f} years")

# Strategy 4: Fill with mode (most common value)
print("\n🚢 TITANIC - Fill missing EMBARKED with mode:")
mode_embarked = titanic['embarked'].mode()[0]  # Most common port
titanic_filled2['embarked'] = titanic_filled2['embarked'].fillna(mode_embarked)
print(f"Most common embarkation port: {mode_embarked}")

# Strategy 5: Forward fill (carry previous value forward)
print("\n🚢 TITANIC - Forward fill AGE:")
titanic_ffill = titanic.copy()
titanic_ffill['age'] = titanic_ffill['age'].fillna(method='ffill')
# ffill = forward fill (use previous row's value)
print(f"Age missing after forward fill: {titanic_ffill['age'].isnull().sum()}")

# Strategy 6: Backward fill (carry next value backward)
print("\n🚢 TITANIC - Backward fill AGE:")
titanic_bfill = titanic.copy()
titanic_bfill['age'] = titanic_bfill['age'].fillna(method='bfill')
# bfill = backward fill (use next row's value)
print(f"Age missing after backward fill: {titanic_bfill['age'].isnull().sum()}")

# ============================================================================
# PART 4: Interpolation (Estimating Missing Values)
# ============================================================================
print("\n\n4️⃣ INTERPOLATION")
print("-" * 70)

# Create sample data with missing values
time_series = pd.Series([10, None, None, 20, None, 30])
print("Original data with missing values:")
print(time_series)

# Linear interpolation
interpolated = time_series.interpolate()
# Estimates missing values on a line between known values
print("\nAfter linear interpolation:")
print(interpolated)
print("10 → [12.5, 15.0] → 20 → [25.0] → 30")

# Apply to Titanic
print("\n🚢 TITANIC - Interpolate AGE:")
titanic_interp = titanic.copy()
titanic_interp['age'] = titanic_interp['age'].interpolate()
print(f"Age missing after interpolation: {titanic_interp['age'].isnull().sum()}")
print(f"Age range: {titanic_interp['age'].min():.1f} - {titanic_interp['age'].max():.1f}")

# ============================================================================
# PART 5: Smart Filling Based on Groups
# ============================================================================
print("\n\n5️⃣ SMART FILLING WITH GROUPBY")
print("-" * 70)

# Fill missing ages with mean age of their passenger class
print("🚢 TITANIC - Fill missing AGE with class average:")
titanic_smart = titanic.copy()

# Calculate mean age per class
age_by_class = titanic.groupby('pclass')['age'].transform('mean')
# Fill missing ages with their class average
titanic_smart['age'] = titanic['age'].fillna(age_by_class)

print("Mean ages by class:")
print(titanic.groupby('pclass')['age'].mean())
print(f"\nAge missing after smart fill: {titanic_smart['age'].isnull().sum()}")

# Compare strategies
print("\n📊 COMPARISON OF FILLING STRATEGIES:")
print(f"Original mean age: {titanic['age'].mean():.2f}")
print(f"Mean fill mean age: {titanic_filled['age'].mean():.2f}")
print(f"Median fill mean age: {titanic_filled2['age'].mean():.2f}")
print(f"Smart fill mean age: {titanic_smart['age'].mean():.2f}")

# ============================================================================
# PART 6: Checking for Missing After Operations
# ============================================================================
print("\n\n6️⃣ FINAL VALIDATION")
print("-" * 70)

# Always verify after handling missing data
print("🚢 TITANIC - Final check on smartly filled data:")
final_missing = titanic_smart.isnull().sum()
print(final_missing[final_missing > 0])

if final_missing.sum() > 0:
    print(f"\n⚠️ Still have {final_missing.sum()} total missing values")
    print("Columns still with missing data:")
    print(final_missing[final_missing > 0].index.tolist())
else:
    print("\n✅ No missing values remaining!")

print("\n✅ Missing data handling mastered on all 3 datasets!")
print("=" * 70)
```

**Output:**

```
======================================================================
HANDLING MISSING DATA
======================================================================

1️⃣ DETECTING MISSING VALUES
----------------------------------------------------------------------
🚢 TITANIC - Missing value summary:
age            177
embarked         2
deck           688
embark_town      2
dtype: int64

As percentages:
age            19.9
embarked        0.2
deck           77.2
embark_town     0.2
dtype: float64

🚢 TITANIC - First 10 rows, showing which values are missing:
     age   deck  embarked
0  False   True     False
1  False  False     False
2  False   True     False
3  False  False     False
4  False   True     False
5   True   True     False
6  False  False     False
7  False   True     False
8  False   True     False
9  False   True     False

🍽️ TIPS - Missing values:
✅ No missing values! This dataset is clean.

💎 DIAMONDS - Missing values:
✅ No missing values! This dataset is clean.


2️⃣ DROPPING MISSING VALUES
----------------------------------------------------------------------
🚢 TITANIC - Original shape: (891, 15)
After dropna(): (182, 15)
Lost 709 rows (79.6%)

🚢 TITANIC - Drop rows where ALL values are NaN:
After dropna(how='all'): (891, 15)
Usually same as original because complete empty rows are rare

🚢 TITANIC - Drop rows where AGE is missing:
After dropna(subset=['age']): (714, 15)
Lost 177 rows with missing age

🚢 TITANIC - Drop columns with > 50% missing:
Dropped columns: {'deck'}


3️⃣ FILLING MISSING VALUES
----------------------------------------------------------------------
🚢 TITANIC - Fill missing EMBARKED with 'Unknown':
Embarked missing count: 0
Value counts:
embarked
S          644
C          168
Q           77
Unknown      2
Name: count, dtype: int64

🚢 TITANIC - Fill missing AGE with mean:
Mean age: 29.7 years
Age missing count after fill: 0

🚢 TITANIC - Fill missing AGE with median:
Median age: 28.0 years

🚢 TITANIC - Fill missing EMBARKED with mode:
Most common embarkation port: S

🚢 TITANIC - Forward fill AGE:
Age missing after forward fill: 0

🚢 TITANIC - Backward fill AGE:
Age missing after backward fill: 0


4️⃣ INTERPOLATION
----------------------------------------------------------------------
Original data with missing values:
0    10.0
1     NaN
2     NaN
3    20.0
4     NaN
5    30.0
dtype: float64

After linear interpolation:
0    10.000000
1    13.333333
2    16.666667
3    20.000000
4    25.000000
5    30.000000
dtype: float64
10 → [12.5, 15.0] → 20 → [25.0] → 30

🚢 TITANIC - Interpolate AGE:
Age missing after interpolation: 0
Age range: 0.4 - 80.0


5️⃣ SMART FILLING WITH GROUPBY
----------------------------------------------------------------------
🚢 TITANIC - Fill missing AGE with class average:
Mean ages by class:
pclass
1    38.233441
2    29.877630
3    25.140620
Name: age, dtype: float64

Age missing after smart fill: 0

📊 COMPARISON OF FILLING STRATEGIES:
Original mean age: 29.70
Mean fill mean age: 29.70
Median fill mean age: 29.36
Smart fill mean age: 29.29


6️⃣ FINAL VALIDATION
----------------------------------------------------------------------
🚢 TITANIC - Final check on smartly filled data:
embarked         2
deck           688
embark_town      2
dtype: int64

⚠️ Still have 692 total missing values
Columns still with missing data:
['embarked', 'deck', 'embark_town']

✅ Missing data handling mastered on all 3 datasets!
======================================================================

```

```
C:\Users\tparl\AppData\Local\Temp\ipykernel_12056\3528508196.py:112: FutureWarning: Series.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
  titanic_ffill['age'] = titanic_ffill['age'].fillna(method='ffill')
C:\Users\tparl\AppData\Local\Temp\ipykernel_12056\3528508196.py:119: FutureWarning: Series.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
  titanic_bfill['age'] = titanic_bfill['age'].fillna(method='bfill')

```

---
<a id="projects"></a>
## 🚀 Chapter Completed! Now Practice with Projects

### 🎉 Congratulations! You've Mastered pandas Basics!

### 📝 Recommended Projects:

#### 1. **Project 01: Data Cleaning & Exploration** ⭐ Beginner
**What you'll do:**
- Load Titanic and Housing datasets with pandas
- Inspect data with `.info()`, `.describe()`, `.head()`
- Handle missing values with various strategies
- Clean inconsistent data and fix data types
- Create derived features

**Skills practiced:**
- DataFrame operations
- Missing data handling  
- Data type conversions
- Feature engineering

**Time:** 2-3 hours  
**Link:** [Open Project 01](../projects/Project_01_DataCleaning.md)

---

#### 2. **Project 02: Visualization & EDA** ⭐⭐ Beginner-Intermediate
**What you'll do:**
- Perform Exploratory Data Analysis
- Use pandas with Matplotlib/Seaborn
- Analyze distributions and correlations
- Create multi-panel dashboards

**Skills practiced:**
- GroupBy aggregations
- Statistical analysis
- Data visualization
- Pattern recognition

**Time:** 3-4 hours  
**Link:** [Open Project 02](../projects/Project_02_Visualization.md)

---

## 📚 Continue Learning

### ➡️ **Chapter 03: Matplotlib - Data Visualization**
Learn to create professional visualizations for your data analysis.

**Link:** [Open Chapter 03](03_Matplotlib_Visualization.ipynb)

---

## 🔗 Navigation

- **Previous**: [Chapter 01: NumPy](01_NumPy_Foundations.ipynb)
- **Next**: [Chapter 03: Matplotlib](03_Matplotlib_Visualization.ipynb)
- **Home**: [START HERE](../START_HERE.md)
- **Index**: [Main Index](../index.md)
- **All Projects**: [Projects Overview](../projects/README.md)

---

**🎓 You're now ready to analyze real-world data with pandas!**

**Next action:** Complete **Project 01** to practice your pandas skills! 💻
