# -*- coding: utf-8 -*-
from sklearn.datasets import load_iris  
from sklearn.model_selection import train_test_split  
from sklearn.naive_bayes import GaussianNB  
from sklearn.metrics import accuracy_score  
  
# 导入iris数据集  
iris = load_iris()  
X = iris.data  
y = iris.target  
  
# 随机切分数据集为训练集和测试集，其中测试集占比为0.2，随机种子为42  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
  
# 使用KUN分类器进行分类  
gnb = GaussianNB()  
gnb.fit(X_train, y_train)  
  
# 预测训练集和测试集的标签  
y_pred_train = gnb.predict(X_train)  
y_pred_test = gnb.predict(X_test)  
  
# 计算训练集和测试集的准确度  
accuracy_train = accuracy_score(y_train, y_pred_train)  
accuracy_test = accuracy_score(y_test, y_pred_test)  
  
# 输出训练集和测试集的准确度  
print("accuracy_train", accuracy_train)  
print("accuracy_test", accuracy_test)
