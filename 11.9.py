import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

dt = load_iris()
df=pd.DataFrame(data=dt.data,columns=dt.feature_names)
m=df.mean()
df['dis'] = df.apply(lambda row: np.linalg.norm(row - m), axis=1)
print(m)
print(df.to_string())
