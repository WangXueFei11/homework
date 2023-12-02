import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

dt = load_iris()
X = pd.DataFrame(dt.data, columns=dt.feature_names)
X = X[['petal length (cm)', 'petal width (cm)']]

kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto')
kmeans.fit(X)

labels = kmeans.labels_
plt.scatter(X['petal length (cm)'], X['petal width (cm)'], c=labels)
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=100,
            c='red',
            label='Centroids')
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.legend()
plt.show()
