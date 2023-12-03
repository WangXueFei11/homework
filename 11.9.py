# -*- coding: utf-8 -*-
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 加载数据集
iris = load_iris()
X = iris.data
y = iris.target

# 设置K-means算法参数
kmeans = KMeans(n_clusters=3, random_state=0)

# 训练模型
kmeans.fit(X)

# 预测数据点的类别标签
y_kmeans = kmeans.predict(X)

# 输出结果
print("K-means Clustering Results:")
print("Cluster 0: ", kmeans.cluster_centers_[0])
print("Cluster 1: ", kmeans.cluster_centers_[1])
print("Cluster 2: ", kmeans.cluster_centers_[2])

# 可视化结果
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[0][0], kmeans.cluster_centers_[0][1], c='red', s=200, alpha=0.5)
plt.scatter(kmeans.cluster_centers_[1][0], kmeans.cluster_centers_[1][1], c='blue', s=200, alpha=0.5)
plt.scatter(kmeans.cluster_centers_[2][0], kmeans.cluster_centers_[2][1], c='green', s=200, alpha=0.5)
plt.show()
