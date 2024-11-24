# Customer Analytics Segmentation Project

## Overview
This project aims to perform customer segmentation using hierarchical and K-means clustering techniques. The goal is to identify distinct customer segments based on their behavior and characteristics, which can help in targeted marketing and personalized customer experiences.

## Table of Contents
* Libraries
* Data
* Exploratory Data Analysis
* Data Preprocessing
* Hierarchical Clustering
* K-means Clustering
* Model Interpretation and Results
* Visualizations

## Libraries
The following libraries are used in this project:

```python
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
```

## Data
The dataset used in this project is customer_segmentation_data.csv. It contains the following key features: minutes watched by each customer, Customer Lifetime Value (CLV), geographic region, and acquisition channel.

## Exploratory Data Analysis

### Descriptive Statistics
Descriptive statistics of the data are calculated to understand the distribution and central tendencies of the features:

```python
df.describe()
```

### Correlation Matrix
A correlation matrix visualization helps understand relationships between variables:

```python
plt.figure(figsize=(12, 9))
s = sns.heatmap(df_segmentation.corr(), annot=True, cmap='RdBu', vmin=-1, vmax=1)
s.set_yticklabels(s.get_yticklabels(), rotation=0, fontsize=12)
s.set_xticklabels(s.get_xticklabels(), rotation=90, fontsize=12)
plt.title('Correlation Heatmap')
plt.savefig('corr.png')
plt.show()
```

### Scatter Plot
Visualizing the relationship between minutes watched and CLV:

```python
plt.figure(figsize=(12, 9))
plt.scatter(df_segmentation.iloc[:, 0], df_segmentation.iloc[:, 1])
plt.xlabel('Minutes watched')
plt.ylabel('CLV')
plt.title('Visualization of raw data')
plt.savefig("scatter.png")
plt.show()
```

## Data Preprocessing

### Handling Missing Values

```python
df_segmentation = df_segmentation.fillna(0)
```

### Creating Dummy Variables
Converting categorical features into dummy variables:

```python
segment_dummies = pd.get_dummies(df_heard_from, prefix='channel', prefix_sep='_')
df_segmentation = pd.concat([df_segmentation, segment_dummies], axis=1)
segment_dummies_2 = pd.get_dummies(df_countries, prefix='country_region', prefix_sep='_')
df_segmentation = pd.concat([df_segmentation, segment_dummies_2], axis=1)
```

### Standardization

```python
scaler = StandardScaler()
segmentation_std = scaler.fit_transform(df_segmentation)
```

## Hierarchical Clustering
Implementing hierarchical clustering with the ward method:

```python
hier_clust = linkage(segmentation_std, method='ward')
plt.figure(figsize=(12, 9))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Observations')
plt.ylabel('Distance')
dendrogram(hier_clust, truncate_mode='level', p=5, show_leaf_counts=False, no_labels=True)
plt.savefig('hierarchical.png')
plt.show()
```

## K-means Clustering

### Elbow Method
Determining optimal number of clusters:

```python
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(segmentation_std)
    wcss.append(kmeans.inertia_)
```

### K-means Implementation

```python
kmeans = KMeans(n_clusters=8, init='k-means++', random_state=42)
kmeans.fit(segmentation_std)
```

## Model Interpretation and Results

### Cluster Analysis

```python
df_segm_kmeans = df_segmentation.copy()
df_segm_kmeans['Segment'] = kmeans.labels_
df_segm_analysis = df_segm_kmeans.groupby(['Segment']).mean()
```

### Segment Naming
The segments are given descriptive names such as Instagram Explorers, LinkedIn Networkers, and Friends' Influence, based on their characteristics.

## Visualizations

### Final Segment Visualization

```python
x_axis = df_segm_kmeans['CLV']
y_axis = df_segm_kmeans['minutes_watched']
plt.figure(figsize=(10, 8))
sns.scatterplot(x=x_axis, y=y_axis, hue=df_segm_kmeans['Labels'])
plt.title('Segmentation K-means')
plt.show()
```

## Conclusion
This project successfully identified distinct customer segments using clustering techniques. The resulting segments provide actionable insights for targeted marketing strategies and personalized customer experiences.