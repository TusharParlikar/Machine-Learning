# Project 05 — Clustering & PCA (Unsupervised Learning)
**Difficulty:** ⭐⭐ Intermediate  
**Estimated Time:** 3 hours  
**Prerequisites:** Chapters 01-04

---

## 📋 Description
Explore unsupervised learning! Use PCA to reduce dimensions and KMeans to find natural groups in data without labels.

## 🎯 Objectives
- [ ] Apply PCA for dimensionality reduction
- [ ] Visualize explained variance
- [ ] Perform KMeans clustering
- [ ] Evaluate cluster quality
- [ ] Visualize clusters in 2D/3D

## 📊 Datasets
- **Mall Customers:** https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial (RECOMMENDED)
- **Iris:** Built-in scikit-learn

## 🛠️ Libraries
```python
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
```

## 📝 Tasks

### 1. Standardize and Apply PCA
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Plot explained variance
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.grid()
plt.show()
```

**Hint:** Choose number of components that explain ~90-95% variance.

### 2. Reduce to 2D
```python
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)
print(f"Explained variance: {pca_2d.explained_variance_ratio_.sum():.2%}")
```

### 3. Find Optimal K (Elbow Method)
```python
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot elbow curve
plt.plot(K_range, inertias, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# Plot silhouette scores
plt.plot(K_range, silhouette_scores, marker='o', color='orange')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score by K')
plt.show()
```

**Hint:** Look for "elbow" in inertia plot; higher silhouette score is better.

### 4. Cluster with Best K
```python
best_k = 3  # Choose based on elbow/silhouette
kmeans = KMeans(n_clusters=best_k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

print(f"Cluster sizes: {np.bincount(clusters)}")
print(f"Silhouette Score: {silhouette_score(X_scaled, clusters):.4f}")
```

### 5. Visualize Clusters
```python
plt.figure(figsize=(10, 6))
for i in range(best_k):
    plt.scatter(X_2d[clusters == i, 0], X_2d[clusters == i, 1], 
                label=f'Cluster {i}', alpha=0.6, s=50)

# Plot centroids
centroids_2d = pca_2d.transform(kmeans.cluster_centers_)
plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], 
            marker='X', s=300, c='red', edgecolor='black', label='Centroids')

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'K-Means Clustering (K={best_k})')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

### 6. Analyze Cluster Characteristics
```python
# Add cluster labels to original data
df['Cluster'] = clusters

# Compute cluster means
cluster_summary = df.groupby('Cluster').mean()
print("Cluster Characteristics:")
print(cluster_summary)
```

## ✅ Success Criteria
- [ ] PCA reduces dimensions meaningfully
- [ ] Optimal K chosen using elbow/silhouette
- [ ] Clusters visualized clearly
- [ ] Cluster characteristics interpreted
- [ ] Silhouette score > 0.3

## 🎓 Bonus
- Try hierarchical clustering (dendrogram)
- Use t-SNE for visualization
- Try DBSCAN (density-based clustering)
- 3D visualization with 3 PCs

**Next:** Project 06 - End-to-End Mini-Project 🚀
