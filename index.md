# 3-Day Intensive Syllabus — NumPy, pandas, Matplotlib, scikit-learn
## Detailed Hierarchical Index

**Learning Path:** Start with NumPy → pandas → Matplotlib → scikit-learn. Each chapter builds on the previous one.

**🚀 NEW TO MACHINE LEARNING? START HERE:** [START_HERE.md](START_HERE.md)

**Chapter Navigation:**
- [01: NumPy Foundations](chapters/01_NumPy_Foundations.ipynb) ← Start here!
- [02: pandas Data Manipulation](chapters/02_Pandas_DataManipulation.ipynb)
- [03: Matplotlib Visualization](chapters/03_Matplotlib_Visualization.ipynb)
- [04: scikit-learn Machine Learning](chapters/04_ScikitLearn_MachineLearning.ipynb)
- [Projects Overview](projects/README.md)

---

## Day 1 — Foundation: NumPy and pandas Basics

### Chapter 1: NumPy — Numerical Computing Foundation
**Learn this FIRST** — NumPy is the foundation for all numeric work in Python.

#### 1.1 Introduction to NumPy
- 1.1.1 Why NumPy? (performance comparison with Python lists)
- 1.1.2 NumPy ecosystem and scientific Python stack
- 1.1.3 Installation and import conventions
- 1.1.4 Use cases: data science, machine learning, scientific computing

#### 1.2 Understanding ndarray (N-dimensional Array)
- 1.2.1 What is an ndarray? Memory layout and efficiency
- 1.2.2 Arrays vs Python lists: key differences
- 1.2.3 Homogeneous data types (dtype) importance
- 1.2.4 Array dimensions: 0D (scalar), 1D (vector), 2D (matrix), nD (tensor)

#### 1.3 Creating Arrays
- 1.3.1 From Python lists and tuples: `np.array()`
- 1.3.2 Range-based creation: `arange()`, `linspace()`, `logspace()`
- 1.3.3 Placeholder arrays: `zeros()`, `ones()`, `full()`, `empty()`
- 1.3.4 Identity and diagonal matrices: `eye()`, `identity()`, `diag()`
- 1.3.5 Random number generation: `random.rand()`, `random.randn()`, `random.randint()`, `random.choice()`
- 1.3.6 Array-like objects: `asarray()`, `copy()`

#### 1.4 Array Attributes and Properties
- 1.4.1 Shape: `shape`, `ndim`, `size`
- 1.4.2 Data type: `dtype`, `itemsize`, `nbytes`
- 1.4.3 Memory layout: C-contiguous vs Fortran-contiguous
- 1.4.4 Checking and modifying array properties

#### 1.5 Indexing and Slicing
- 1.5.1 Basic indexing: single element access (1D, 2D, nD)
- 1.5.2 Negative indexing: accessing from the end
- 1.5.3 Slicing: `start:stop:step` syntax
- 1.5.4 Multi-dimensional slicing
- 1.5.5 Views vs copies: understanding memory sharing
- 1.5.6 Ellipsis (`...`) for flexible indexing

#### 1.6 Advanced Indexing
- 1.6.1 Fancy indexing: integer array indexing
- 1.6.2 Boolean masking: conditional selection
- 1.6.3 `np.where()` for conditional operations
- 1.6.4 `np.take()` and `np.put()` for element access
- 1.6.5 Combining fancy indexing and boolean masks

#### 1.7 Universal Functions (ufuncs)
- 1.7.1 What are ufuncs? Element-wise operations
- 1.7.2 Arithmetic operations: `+`, `-`, `*`, `/`, `**`, `//`, `%`
- 1.7.3 Comparison operators: `<`, `>`, `==`, `!=`, `<=`, `>=`
- 1.7.4 Mathematical functions: `sqrt()`, `exp()`, `log()`, `sin()`, `cos()`, `abs()`
- 1.7.5 Custom ufuncs with `np.vectorize()`

#### 1.8 Broadcasting Rules
- 1.8.1 What is broadcasting? Implicit shape matching
- 1.8.2 Broadcasting rules (dimensions and alignment)
- 1.8.3 Common broadcasting patterns
- 1.8.4 When broadcasting fails and how to fix it
- 1.8.5 `np.newaxis` and `reshape()` for broadcasting control

#### 1.9 Aggregations and Statistics
- 1.9.1 Basic aggregations: `sum()`, `mean()`, `median()`, `std()`, `var()`
- 1.9.2 Min/max operations: `min()`, `max()`, `argmin()`, `argmax()`
- 1.9.3 Axis parameter: row-wise vs column-wise operations
- 1.9.4 Cumulative operations: `cumsum()`, `cumprod()`
- 1.9.5 Percentiles and quantiles: `percentile()`, `quantile()`

#### 1.10 Array Manipulation
- 1.10.1 Reshaping: `reshape()`, `ravel()`, `flatten()`
- 1.10.2 Transposing: `T`, `transpose()`, `swapaxes()`
- 1.10.3 Stacking: `vstack()`, `hstack()`, `dstack()`, `stack()`, `concatenate()`
- 1.10.4 Splitting: `vsplit()`, `hsplit()`, `dsplit()`, `split()`
- 1.10.5 Adding/removing dimensions: `expand_dims()`, `squeeze()`
- 1.10.6 Repeating elements: `repeat()`, `tile()`

#### 1.11 Linear Algebra Operations
- 1.11.1 Matrix multiplication: `dot()`, `@` operator, `matmul()`
- 1.11.2 Matrix decompositions: SVD, eigenvalues, eigenvectors
- 1.11.3 Matrix inversion and solving linear systems
- 1.11.4 Norms and determinants

#### 1.12 Performance Optimization
- 1.12.1 Vectorization vs loops: performance comparison
- 1.12.2 Memory views and in-place operations
- 1.12.3 Avoiding copies: when to use views
- 1.12.4 Choosing appropriate dtypes for memory efficiency
- 1.12.5 Common performance pitfalls

#### 1.13 Practice Exercises
- [See NumPy Chapter](chapters/numpy.md) for detailed code examples

---

### Chapter 2: pandas — Data Manipulation and Analysis
**Learn this SECOND** — pandas builds on NumPy for high-level data wrangling.

#### 2.1 Introduction to pandas
- 2.1.1 Why pandas? High-level data manipulation
- 2.1.2 pandas and NumPy relationship
- 2.1.3 Installation and import conventions
- 2.1.4 Use cases: data cleaning, exploration, transformation

#### 2.2 Core Data Structures
- 2.2.1 Series: 1D labeled array
  - Creating Series from lists, dicts, scalars
  - Index and values
  - Series operations and methods
- 2.2.2 DataFrame: 2D labeled data structure
  - Creating DataFrames from dicts, lists, arrays, files
  - Columns and index
  - Viewing data: `head()`, `tail()`, `sample()`

#### 2.3 Data I/O — Reading and Writing
- 2.3.1 CSV files: `read_csv()`, `to_csv()`
  - Handling delimiters, headers, encoding
  - Reading subsets: `usecols`, `nrows`, `skiprows`
- 2.3.2 Excel files: `read_excel()`, `to_excel()`
- 2.3.3 JSON: `read_json()`, `to_json()`
- 2.3.4 SQL databases: `read_sql()`, `to_sql()`
- 2.3.5 Parquet and HDF5 for large datasets
- 2.3.6 Clipboard and HTML tables

#### 2.4 Inspecting and Understanding Data
- 2.4.1 Quick overview: `info()`, `describe()`, `dtypes`
- 2.4.2 Shape and size: `shape`, `size`, `ndim`
- 2.4.3 Column and index access: `columns`, `index`
- 2.4.4 Memory usage: `memory_usage()`
- 2.4.5 Unique values and counts: `unique()`, `nunique()`, `value_counts()`

#### 2.5 Indexing and Selection
- 2.5.1 Column selection: `df['col']`, `df[['col1', 'col2']]`
- 2.5.2 Label-based indexing: `loc[]`
  - Single value, row slice, row+column selection
- 2.5.3 Position-based indexing: `iloc[]`
  - Integer-based slicing
- 2.5.4 Fast scalar access: `at[]`, `iat[]`
- 2.5.5 Boolean indexing and filtering
- 2.5.6 Query method: `query()` for SQL-like filtering
- 2.5.7 Multi-indexing basics: hierarchical indexes

#### 2.6 Data Cleaning and Preparation
- 2.6.1 Missing data detection: `isnull()`, `notnull()`, `isna()`
- 2.6.2 Handling missing data:
  - Dropping: `dropna()` with various options
  - Filling: `fillna()`, `ffill()`, `bfill()`
  - Interpolation: `interpolate()`
- 2.6.3 Duplicate detection: `duplicated()`, `drop_duplicates()`
- 2.6.4 Replacing values: `replace()`
- 2.6.5 Renaming: `rename()` for columns and index
- 2.6.6 Data type conversions: `astype()`, `to_numeric()`, `to_datetime()`
- 2.6.7 Categorical data: creating and using categories
- 2.6.8 String operations: `str` accessor methods

#### 2.7 Basic DataFrame Operations
- 2.7.1 Adding columns: direct assignment, `assign()`
- 2.7.2 Dropping columns/rows: `drop()`
- 2.7.3 Sorting: `sort_values()`, `sort_index()`
- 2.7.4 Ranking: `rank()`
- 2.7.5 Applying functions: `apply()`, `applymap()`, `map()`
- 2.7.6 Arithmetic operations and alignment
- 2.7.7 Method chaining for clean code

#### 2.8 Practice Exercises — Day 1
- [Project 1: Data Cleaning](projects/01-data-cleaning-exploration/README.md)

---

## Day 2 — Advanced Manipulation: pandas Wrangling and Matplotlib Visualization

### Chapter 3: Advanced pandas — GroupBy, Merge, Reshape

#### 3.1 GroupBy Operations
- 3.1.1 GroupBy mechanics: split-apply-combine
- 3.1.2 Grouping by single and multiple columns
- 3.1.3 Aggregation functions: `agg()`, `sum()`, `mean()`, `count()`
- 3.1.4 Multiple aggregations: dictionary and named aggregation
- 3.1.5 Transformation: `transform()` for group-wise operations
- 3.1.6 Filtering groups: `filter()`
- 3.1.7 Grouping with custom functions
- 3.1.8 Iteration over groups
- 3.1.9 GroupBy with datetime: resampling patterns

#### 3.2 Combining DataFrames
- 3.2.1 Concatenation: `concat()` along rows/columns
  - Handling different indexes
  - Keys and hierarchical indexes
- 3.2.2 Merging/Joining: `merge()` for SQL-like operations
  - Inner, outer, left, right joins
  - Merging on index vs columns
  - Handling suffixes for overlapping columns
- 3.2.3 DataFrame join method: `join()`
- 3.2.4 Combining with different indexes
- 3.2.5 Handling merge indicator: `indicator=True`

#### 3.3 Reshaping Data
- 3.3.1 Pivoting: `pivot()` for wide format
- 3.3.2 Pivot tables: `pivot_table()` with aggregation
- 3.3.3 Melting: `melt()` for long format
- 3.3.4 Stacking/Unstacking: `stack()`, `unstack()`
- 3.3.5 Wide-to-long and long-to-wide transformations
- 3.3.6 Cross-tabulation: `crosstab()`

#### 3.4 Time Series Basics
- 3.4.1 DateTime index: creating and converting
- 3.4.2 Date ranges: `date_range()`, `period_range()`
- 3.4.3 Resampling: `resample()` for frequency conversion
  - Upsampling and downsampling
  - Aggregation during resampling
- 3.4.4 Rolling windows: `rolling()` for moving statistics
- 3.4.5 Expanding windows: `expanding()`
- 3.4.6 Time shifts and lags: `shift()`, `tshift()`
- 3.4.7 Time zone handling: `tz_localize()`, `tz_convert()`

#### 3.5 Advanced Techniques
- 3.5.1 Method chaining for readable pipelines
- 3.5.2 `query()` and `eval()` for performance
- 3.5.3 Categorical dtype for memory optimization
- 3.5.4 Understanding `inplace` parameter: pros and cons
- 3.5.5 Working with large datasets: chunking and Dask preview
- 3.5.6 Sparse data structures

---

### Chapter 4: Matplotlib — Data Visualization
**Learn this THIRD** — Visualization is essential for understanding data and communicating results.

#### 4.1 Introduction to Data Visualization
- 4.1.1 Why visualize? The power of visual communication
- 4.1.2 Choosing the right chart type
  - Comparison: bar charts
  - Distribution: histograms, box plots
  - Relationship: scatter plots, line plots
  - Composition: pie charts, stacked bars
- 4.1.3 Visualization best practices and common mistakes
- 4.1.4 Matplotlib architecture: Figure, Axes, Artist hierarchy

#### 4.2 Getting Started with Matplotlib
- 4.2.1 Installation and import conventions
- 4.2.2 Two interfaces: pyplot (MATLAB-style) vs object-oriented
- 4.2.3 Creating figures and axes: `plt.subplots()`
- 4.2.4 First plot: line plot with `plot()`
- 4.2.5 Displaying plots: `show()`, inline in Jupyter

#### 4.3 Basic Plot Types
- 4.3.1 Line plots: `plot()`
  - Single and multiple lines
  - Line styles, colors, markers
- 4.3.2 Scatter plots: `scatter()`
  - Marker size, color mapping
- 4.3.3 Bar charts: `bar()`, `barh()`
  - Grouped and stacked bars
- 4.3.4 Histograms: `hist()`
  - Bins, density, cumulative
- 4.3.5 Box plots: `boxplot()`
- 4.3.6 Pie charts: `pie()`
- 4.3.7 Area plots: `fill_between()`

#### 4.4 Customizing Plots
- 4.4.1 Colors: named colors, hex codes, RGB tuples, colormaps
- 4.4.2 Line styles: solid, dashed, dotted
- 4.4.3 Markers: types, sizes, edge/face colors
- 4.4.4 Labels and titles: `xlabel()`, `ylabel()`, `title()`
- 4.4.5 Legends: `legend()`, positioning, styling
- 4.4.6 Grid: `grid()` with customization
- 4.4.7 Axis limits: `xlim()`, `ylim()`, `axis()`
- 4.4.8 Tick customization: `xticks()`, `yticks()`, formatters
- 4.4.9 Annotations: `annotate()`, `text()`
- 4.4.10 Arrows and shapes

#### 4.5 Subplots and Layouts
- 4.5.1 Creating subplots: `subplots()` with rows and columns
- 4.5.2 Individual subplot creation: `add_subplot()`
- 4.5.3 GridSpec for complex layouts
- 4.5.4 Sharing axes: `sharex`, `sharey`
- 4.5.5 Adjusting spacing: `tight_layout()`, `subplots_adjust()`
- 4.5.6 Inset axes for zoom-ins

#### 4.6 Advanced Visualization
- 4.6.1 Colormaps and color bars
- 4.6.2 Contour and filled contour plots
- 4.6.3 Heatmaps with `imshow()` and `pcolormesh()`
- 4.6.4 3D plotting basics
- 4.6.5 Error bars: `errorbar()`
- 4.6.6 Log scales: `semilogx()`, `semilogy()`, `loglog()`

#### 4.7 Styling and Themes
- 4.7.1 Built-in styles: `plt.style.use()`
- 4.7.2 Customizing with rcParams
- 4.7.3 Creating custom styles
- 4.7.4 Publication-quality figures

#### 4.8 Saving and Exporting
- 4.8.1 Saving figures: `savefig()`
- 4.8.2 File formats: PNG, SVG, PDF
- 4.8.3 DPI and resolution settings
- 4.8.4 Bbox and padding control

#### 4.9 Integration with pandas
- 4.9.1 DataFrame plot method: `df.plot()`
- 4.9.2 Plot types: `kind` parameter
- 4.9.3 Quick plotting for exploration
- 4.9.4 Customizing pandas plots with Matplotlib

#### 4.10 Seaborn Preview
- 4.10.1 When to use Seaborn vs Matplotlib
- 4.10.2 Statistical plots: `regplot()`, `boxplot()`, `violinplot()`
- 4.10.3 Faceting: `FacetGrid`
- 4.10.4 Themes and palettes

#### 4.11 Practice Exercises
- [Project 2: Visualization & EDA](projects/02-visualization-eda/README.md)

---

## Day 3 — Machine Learning: scikit-learn Foundations

### Chapter 5: scikit-learn — Supervised Learning
**Learn this FOURTH** — Apply NumPy, pandas, and Matplotlib skills to build ML models.

#### 5.1 Introduction to Machine Learning
- 5.1.1 What is machine learning? Types of learning
- 5.1.2 Supervised vs unsupervised vs reinforcement learning
- 5.1.3 Classification vs regression problems
- 5.1.4 ML workflow: data → model → evaluation → deployment

#### 5.2 scikit-learn Basics
- 5.2.1 Installation and import conventions
- 5.2.2 Estimator API: `fit()`, `predict()`, `transform()`
- 5.2.3 Consistent interface across algorithms
- 5.2.4 scikit-learn datasets: `load_iris()`, `load_boston()`, `make_classification()`

#### 5.3 Data Preparation for ML
- 5.3.1 Train-test split: `train_test_split()`
  - Stratification for classification
  - Random state for reproducibility
- 5.3.2 Feature scaling importance
  - StandardScaler: z-score normalization
  - MinMaxScaler: range normalization
  - RobustScaler: outlier-resistant scaling
- 5.3.3 Encoding categorical features
  - LabelEncoder for target labels
  - OneHotEncoder for nominal features
  - OrdinalEncoder for ordinal features
- 5.3.4 Handling missing values
  - SimpleImputer: mean, median, mode, constant
- 5.3.5 Feature engineering basics
  - Polynomial features
  - Binning continuous variables

#### 5.4 Classification Algorithms
- 5.4.1 Logistic Regression
  - Binary and multiclass classification
  - Regularization: L1 (Lasso), L2 (Ridge)
  - Interpreting coefficients
- 5.4.2 K-Nearest Neighbors (KNN)
  - Distance metrics
  - Choosing k
- 5.4.3 Decision Trees
  - Tree structure and splitting criteria
  - Visualization of trees
  - Overfitting and pruning
- 5.4.4 Random Forests
  - Ensemble learning concept
  - Feature importance
  - Out-of-bag error
- 5.4.5 Support Vector Machines (SVM) — brief intro
- 5.4.6 Naive Bayes — brief intro

#### 5.5 Regression Algorithms
- 5.5.1 Linear Regression
  - Ordinary least squares
  - Assumptions and diagnostics
  - Interpreting coefficients
- 5.5.2 Ridge Regression (L2 regularization)
- 5.5.3 Lasso Regression (L1 regularization)
- 5.5.4 ElasticNet (L1 + L2)
- 5.5.5 Decision Tree Regressor
- 5.5.6 Random Forest Regressor

#### 5.6 Model Evaluation
- 5.6.1 Classification metrics
  - Accuracy: when it's misleading
  - Confusion matrix: TP, TN, FP, FN
  - Precision, recall, F1-score
  - ROC curve and AUC
  - Classification report: `classification_report()`
- 5.6.2 Regression metrics
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R² (coefficient of determination)
  - Adjusted R²
- 5.6.3 Cross-validation
  - K-fold cross-validation: `cross_val_score()`
  - Stratified K-fold for classification
  - Leave-one-out cross-validation
  - Understanding bias-variance tradeoff

#### 5.7 Hyperparameter Tuning
- 5.7.1 What are hyperparameters?
- 5.7.2 Grid search: `GridSearchCV()`
  - Exhaustive search over parameter grid
  - Best parameters and best score
- 5.7.3 Randomized search: `RandomizedSearchCV()`
  - Sampling from distributions
  - Computational efficiency
- 5.7.4 Cross-validation during tuning
- 5.7.5 Avoiding data leakage

#### 5.8 Pipelines
- 5.8.1 Why use pipelines? Avoiding leakage and streamlining code
- 5.8.2 Creating pipelines: `Pipeline` and `make_pipeline()`
- 5.8.3 ColumnTransformer for different preprocessing per column
- 5.8.4 Integrating preprocessing and modeling
- 5.8.5 Pipelines in cross-validation and grid search

#### 5.9 Practice Exercises
- [Project 3: Classification Baseline](projects/03-classification-baseline/README.md)
- [Project 4: Regression Baseline](projects/04-regression-baseline/README.md)

---

### Chapter 6: scikit-learn — Unsupervised Learning & Advanced Topics

#### 6.1 Dimensionality Reduction
- 6.1.1 Curse of dimensionality
- 6.1.2 Principal Component Analysis (PCA)
  - How PCA works: eigenvectors and eigenvalues
  - Explained variance
  - Choosing number of components
  - Visualization in 2D/3D
  - PCA in pipelines
- 6.1.3 t-SNE for visualization (brief)
- 6.1.4 Feature selection methods
  - SelectKBest, RFE (Recursive Feature Elimination)

#### 6.2 Clustering
- 6.2.1 What is clustering? Unsupervised learning
- 6.2.2 K-Means clustering
  - Algorithm: centroids and assignments
  - Choosing k: elbow method
  - Limitations and when to use
- 6.2.3 Hierarchical clustering (brief)
- 6.2.4 DBSCAN (brief)
- 6.2.5 Evaluating clusters
  - Silhouette score
  - Davies-Bouldin index
  - Visual inspection

#### 6.3 Model Persistence and Deployment
- 6.3.1 Saving models: `joblib.dump()`, `pickle`
- 6.3.2 Loading models: `joblib.load()`
- 6.3.3 Versioning models
- 6.3.4 Reproducibility: `random_state` everywhere

#### 6.4 Advanced Topics (Brief Overview)
- 6.4.1 Ensemble methods: bagging, boosting, stacking
- 6.4.2 Gradient Boosting: XGBoost, LightGBM preview
- 6.4.3 Neural networks: MLPClassifier/Regressor basics
- 6.4.4 Anomaly detection
- 6.4.5 Imbalanced datasets: SMOTE, class weights

#### 6.5 End-to-End ML Project Workflow
- 6.5.1 Problem definition and success metrics
- 6.5.2 Data collection and exploration (pandas + Matplotlib)
- 6.5.3 Data cleaning and feature engineering (pandas + NumPy)
- 6.5.4 Model selection and training (scikit-learn)
- 6.5.5 Evaluation and interpretation (metrics + visualization)
- 6.5.6 Iteration and improvement
- 6.5.7 Documentation and presentation

#### 6.6 Practice Exercises
- [Project 5: Clustering & PCA](projects/05-clustering-pca/README.md)
- [Project 6: End-to-End Mini-Project](projects/06-end-to-end-mini-project/README.md)

---

## Appendices

### Appendix A: Recommended Datasets
- **Beginner-friendly Kaggle datasets:**
  - Titanic: https://www.kaggle.com/c/titanic
  - House Prices: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
  - Iris Dataset (built-in scikit-learn)
  - MNIST Digits (built-in scikit-learn)
  - Mall Customers: https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial
  - Heart Disease UCI: https://www.kaggle.com/ronitf/heart-disease-uci

### Appendix B: Video Resources
- NumPy: https://youtu.be/9DhZ-JCWvDw?si=eg8b2PRTGfy2_XYu
- pandas: https://youtu.be/VXtjG_GzO7Q?si=xkKkRSO6Ya--dsDl
- Matplotlib: https://youtu.be/c9vhHUGdav0?si=N1LUnRPywgEMssOk
- scikit-learn: https://youtu.be/pqNCD_5r0IU?si=ZYTtXrRBOhPB-SDm

### Appendix C: Quick Reference Cheatsheets
- NumPy cheatsheet: https://numpy.org/doc/stable/user/quickstart.html
- pandas cheatsheet: https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf
- Matplotlib cheatsheet: https://matplotlib.org/cheatsheets/
- scikit-learn cheatsheet: https://scikit-learn.org/stable/tutorial/machine_learning_map/

### Appendix D: Common Errors and Solutions
- **Shape mismatches:** Use `reshape()` or check broadcasting rules
- **dtype issues:** Use `astype()` for conversion, check for mixed types
- **Missing values:** Always check with `isnull()` before modeling
- **Data leakage:** Always split data BEFORE any preprocessing
- **Overfitting:** Use cross-validation, regularization, and more data

---

**Next Steps:** Navigate to each chapter for detailed content with code examples and expected outputs.
