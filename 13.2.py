import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

dt = load_iris()
X_train,X_test,y_train,y_test=train_test_split(dt['data'],dt['target'],test_size=0.2, random_state=42)
