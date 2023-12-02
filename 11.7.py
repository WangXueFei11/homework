import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

dt = load_iris()
X_train, X_test, y_train, y_test = train_test_split(dt['data'],
                                                    dt['target'],
                                                    test_size=0.3,
                                                    random_state=42)
lr = LogisticRegression()
lr.fit(X_train, y_train)

print(lr.coef_)
print(lr.score(X_train, y_train))
print(lr.score(X_test, y_test))
