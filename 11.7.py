# -*- coding: utf-8 -*-
from sklearn.datasets import load_iris  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score  
  
# 加载数据集  
iris = load_iris()  
X = iris.data  
y = iris.target  
  
# 划分数据集为训练集和测试集  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  
  
# 创建 Logistic Regression 模型  
model = LogisticRegression()  
  
# 训练模型  
model.fit(X_train, y_train)  
  
# 使用模型进行预测  
y_pred = model.predict(X_test)  
  
# 计算预测准确率  
accuracy = accuracy_score(y_test, y_pred)  
print("Accuracy: ", accuracy)
