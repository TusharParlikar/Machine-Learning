# Project 02 — Visualization & Exploratory Data Analysis (EDA)
**Difficulty:** ⭐⭐ Beginner-Intermediate  
**Estimated Time:** 3-4 hours  
**Prerequisites:** Complete Chapter 01 (NumPy), Chapter 02 (pandas), Chapter 03 (Matplotlib), and Project 01

---

## 📋 Project Description

Learn to create meaningful visualizations that reveal patterns, relationships, and anomalies in your data. Visualization is how you UNDERSTAND your data and communicate findings!

**What you'll learn:**
- Creating different chart types (histograms, scatter, box, bar)
- Customizing plots (colors, labels, titles, legends)
- Building multi-panel figures for comparisons
- Finding patterns and outliers visually
- Saving publication-quality figures

---

## 🎯 Objectives

By the end of this project, you should be able to:
- [ ] Create histograms to show distributions
- [ ] Create scatter plots to show relationships
- [ ] Create box plots to compare groups
- [ ] Create bar charts for categorical comparisons
- [ ] Build multi-panel figures with subplots
- [ ] Annotate and customize plots professionally
- [ ] Save figures as PNG/PDF files

---

## 📊 Suggested Datasets (Pick ONE)

### Option 1: Titanic (RECOMMENDED - use your cleaned data from Project 01!)
- **Link:** Your `cleaned_data.csv` from Project 01
- **Why:** You already know the data, focus on visualization

### Option 2: Heart Disease UCI
- **Link:** https://www.kaggle.com/ronitf/heart-disease-uci
- **Why:** Medical data, clear relationships, categorical + numeric mix

### Option 3: Tips Dataset (Built-in with Seaborn)
- **Why:** Small, easy to understand, good for practice
- **Load with:** `import seaborn as sns; df = sns.load_dataset('tips')`

---

## 🛠️ Tools & Libraries You'll Use

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Makes Matplotlib prettier

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
```

---

## 📝 Step-by-Step Tasks

### Task 1: Load and Quick Check
**Goal:** Load your cleaned data and verify it's ready

```python
# TODO: Load your dataset
df = pd.read_csv('cleaned_data.csv')

# TODO: Quick check
print(df.head())
print(df.info())
print(df.describe())

# TODO: Check for any remaining missing values
print("Missing values:", df.isnull().sum().sum())
```

**Hints:**
- If you have missing values, go back to Project 01
- Make sure numeric columns are actually numeric (not strings)

---

### Task 2: Distribution Analysis (Histograms)
**Goal:** Understand the distribution of numeric variables

```python
# TODO: Create histograms for all numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()  # Flatten to 1D for easy iteration

for idx, col in enumerate(numeric_cols[:4]):  # First 4 numeric columns
    df[col].hist(bins=30, ax=axes[idx], edgecolor='black')
    axes[idx].set_title(f'Distribution of {col}')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('distributions.png', dpi=150)
plt.show()
```

**Hints:**
- Histograms show if data is normal, skewed, or has outliers
- Try different `bins` values (10, 20, 30, 50)
- Look for: symmetry, outliers, multiple peaks

**Questions to ask:**
- Is the data normally distributed (bell curve)?
- Are there extreme values (outliers)?
- Are there gaps in the distribution?

---

### Task 3: Relationships (Scatter Plots)
**Goal:** Find relationships between numeric variables

```python
# TODO: Create scatter plots for key relationships
# Example: Age vs Fare (Titanic) or Age vs Cholesterol (Heart Disease)

plt.figure(figsize=(10, 6))
plt.scatter(df['Column1'], df['Column2'], alpha=0.5, c=df['TargetColumn'], cmap='viridis')
plt.xlabel('Column1 Name')
plt.ylabel('Column2 Name')
plt.title('Relationship between Column1 and Column2')
plt.colorbar(label='Target Variable')
plt.grid(True, alpha=0.3)
plt.savefig('scatter_relationship.png', dpi=150)
plt.show()
```

**Hints:**
- Use `alpha=0.5` for transparency when points overlap
- Color points by a categorical variable to see patterns
- Add `s=df['Size']` to vary point size by a variable

**Look for:**
- Positive correlation (up-right trend)
- Negative correlation (down-right trend)
- Clusters or groups
- Outliers

**Titanic Example:**
```python
plt.scatter(df['Age'], df['Fare'], alpha=0.6, c=df['Survived'], cmap='RdYlGn')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('Age vs Fare (colored by Survival)')
plt.colorbar(label='Survived (0=No, 1=Yes)')
plt.show()
```

---

### Task 4: Group Comparisons (Box Plots)
**Goal:** Compare distributions across categories

```python
# TODO: Create box plots to compare groups
# Example: Fare by Passenger Class or Age by Survival

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Box plot 1
df.boxplot(column='NumericColumn', by='CategoryColumn', ax=axes[0])
axes[0].set_title('NumericColumn by CategoryColumn')
axes[0].set_xlabel('CategoryColumn')
axes[0].set_ylabel('NumericColumn')

# Box plot 2
df.boxplot(column='AnotherNumeric', by='AnotherCategory', ax=axes[1])
axes[1].set_title('AnotherNumeric by AnotherCategory')

plt.tight_layout()
plt.savefig('boxplots_comparison.png', dpi=150)
plt.show()
```

**Hints:**
- Box plots show median (line), quartiles (box), and outliers (dots)
- Use to compare distributions between groups
- Seaborn's `boxplot()` is often prettier: `sns.boxplot(data=df, x='Category', y='Numeric')`

**Titanic Example:**
```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
df.boxplot(column='Age', by='Pclass', ax=axes[0])
df.boxplot(column='Fare', by='Survived', ax=axes[1])
plt.tight_layout()
plt.show()
```

**Questions:**
- Which group has higher values?
- Are there more outliers in one group?
- Are the distributions similar or different?

---

### Task 5: Categorical Analysis (Bar Charts)
**Goal:** Show counts or averages for categories

```python
# TODO: Count plot (frequency of categories)
plt.figure(figsize=(10, 6))
category_counts = df['CategoryColumn'].value_counts()
category_counts.plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Frequency of CategoryColumn')
plt.xlabel('CategoryColumn')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('category_counts.png', dpi=150)
plt.show()

# TODO: Grouped bar chart (average by category)
plt.figure(figsize=(10, 6))
grouped = df.groupby('CategoryColumn')['NumericColumn'].mean()
grouped.plot(kind='bar', color='coral', edgecolor='black')
plt.title('Average NumericColumn by CategoryColumn')
plt.xlabel('CategoryColumn')
plt.ylabel('Average NumericColumn')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('grouped_averages.png', dpi=150)
plt.show()
```

**Hints:**
- Use `value_counts()` for frequency counts
- Use `groupby().mean()` for averages by category
- Rotate x-labels if they overlap: `plt.xticks(rotation=45)`

**Titanic Example:**
```python
# Survival rate by class
survival_by_class = df.groupby('Pclass')['Survived'].mean()
survival_by_class.plot(kind='bar', color=['red', 'yellow', 'green'])
plt.title('Survival Rate by Passenger Class')
plt.ylabel('Survival Rate')
plt.xticks(rotation=0)
plt.show()
```

---

### Task 6: Multi-Panel Figure (Dashboard)
**Goal:** Create a comprehensive visualization with multiple plots

```python
# TODO: Create a 2x2 dashboard
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Histogram
axes[0, 0].hist(df['NumericColumn'], bins=30, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Distribution of NumericColumn')
axes[0, 0].set_xlabel('NumericColumn')
axes[0, 0].set_ylabel('Frequency')

# Plot 2: Scatter
axes[0, 1].scatter(df['Col1'], df['Col2'], alpha=0.5, c=df['Target'], cmap='coolwarm')
axes[0, 1].set_title('Col1 vs Col2')
axes[0, 1].set_xlabel('Col1')
axes[0, 1].set_ylabel('Col2')

# Plot 3: Box plot
df.boxplot(column='Numeric', by='Category', ax=axes[1, 0])
axes[1, 0].set_title('Numeric by Category')

# Plot 4: Bar chart
category_means = df.groupby('Category')['Numeric'].mean()
axes[1, 1].bar(category_means.index, category_means.values, color='green', edgecolor='black')
axes[1, 1].set_title('Average Numeric by Category')
axes[1, 1].set_xlabel('Category')
axes[1, 1].set_ylabel('Average')

plt.suptitle('Comprehensive EDA Dashboard', fontsize=16, y=1.00)
plt.tight_layout()
plt.savefig('eda_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Hints:**
- `subplots(2, 2)` creates a 2x2 grid
- Use `axes[row, col]` to access each subplot
- `tight_layout()` prevents overlapping labels
- `suptitle()` adds an overall title

---

### Task 7: Annotate and Customize
**Goal:** Make your plots publication-ready

```python
# TODO: Create one highly customized plot
plt.figure(figsize=(12, 6))

# Example: Annotated scatter
plt.scatter(df['X'], df['Y'], alpha=0.6, s=100, c=df['Category'], cmap='Set2', edgecolor='black', linewidth=0.5)

# Add title and labels with larger fonts
plt.title('Professional Scatter Plot with Annotations', fontsize=16, fontweight='bold')
plt.xlabel('X Variable (units)', fontsize=14)
plt.ylabel('Y Variable (units)', fontsize=14)

# Add grid
plt.grid(True, alpha=0.3, linestyle='--')

# Add colorbar
cbar = plt.colorbar()
cbar.set_label('Category', fontsize=12)

# Annotate interesting points
# max_point = df.loc[df['Y'].idxmax()]
# plt.annotate('Highest Y value', 
#              xy=(max_point['X'], max_point['Y']),
#              xytext=(max_point['X']+1, max_point['Y']+1),
#              arrowprops=dict(arrowstyle='->', color='red', lw=2),
#              fontsize=12, color='red')

plt.tight_layout()
plt.savefig('professional_plot.png', dpi=300, bbox_inches='tight')
plt.show()
```

**Customization Options:**
- `fontsize`: Make text bigger/smaller
- `fontweight='bold'`: Bold text
- `color`, `alpha`: Control appearance
- `edgecolor`, `linewidth`: Add borders
- `cmap`: Color scheme ('viridis', 'plasma', 'coolwarm', etc.)
- `dpi=300`: High resolution for publication

---

## ✅ Success Criteria

You've completed this project successfully if:
- [ ] You created at least 1 histogram showing distributions
- [ ] You created at least 1 scatter plot showing relationships
- [ ] You created at least 1 box plot comparing groups
- [ ] You created at least 1 bar chart for categorical data
- [ ] You built a multi-panel figure (2x2 or larger)
- [ ] You customized plots with titles, labels, and colors
- [ ] You saved all figures as PNG files (300 dpi for best quality)
- [ ] You can explain what each plot reveals about the data

---

## 🎓 Bonus Challenges

1. Create a correlation heatmap: `sns.heatmap(df.corr(), annot=True, cmap='coolwarm')`
2. Create pair plots for multiple variables: `sns.pairplot(df, hue='TargetColumn')`
3. Add statistical annotations (mean lines, confidence intervals)
4. Create an animated plot showing changes over time
5. Try Seaborn's `FacetGrid` for advanced multi-panel plots

---

## 🐛 Common Errors & Solutions

### Error: "ValueError: could not convert string to float"
**Solution:** Make sure you're plotting numeric columns only.

```python
numeric_df = df.select_dtypes(include=[np.number])
```

### Plot looks cluttered or overlapping
**Solution:** Adjust figure size and use `tight_layout()`.

```python
plt.figure(figsize=(12, 8))
# ... your plot code ...
plt.tight_layout()
```

### Colors don't show up or legend missing
**Solution:** Add colorbar or legend explicitly.

```python
plt.colorbar(label='Variable Name')
# or
plt.legend(['Label1', 'Label2'])
```

### Saved figure is cut off
**Solution:** Use `bbox_inches='tight'` when saving.

```python
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
```

---

## 📚 Resources

- Matplotlib gallery: https://matplotlib.org/stable/gallery/index.html
- Seaborn gallery: https://seaborn.pydata.org/examples/index.html
- Chapter 03: Matplotlib Visualization
- Video: Matplotlib video in main README

---

## 🎯 Next Steps

After completing this project:
1. Move to **Project 03: Classification Baseline** to build ML models
2. Review Chapter 03 if you struggled with any visualization concepts
3. Create a visualization notebook for every dataset you work with

---

**Remember:** A good visualization is worth a thousand numbers. Master this skill to communicate insights effectively! 📊
