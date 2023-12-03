# -*- coding: utf-8 -*-
from sklearn.datasets import load_iris  
from sklearn.model_selection import train_test_split  
  
# 导入iris数据集  
iris = load_iris()  
X = iris.data  
y = iris.target  
  
# 随机切分数据集为训练集和测试集，其中测试集占比为0.2，随机种子为42  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
  
# 输出训练集和测试集的大小  
print("size_for_train", len(X_train))  
print("size_for_test", len(X_test))
