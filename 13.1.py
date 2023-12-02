import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris

data = load_iris()
X = data.data
y = data.target
feature_names = data.feature_names

df = pd.DataFrame(data=np.c_[X, y], columns=feature_names + ['target'])
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

sns.set(style="ticks")
sns.pairplot(df, hue='species', markers=["o", "s", "D"])
plt.show()
