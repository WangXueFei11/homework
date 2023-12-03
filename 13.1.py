# -*- coding: utf-8 -*-
from sklearn import datasets  
import matplotlib.pyplot as plt  
  
# 导入iris数据集  
iris = datasets.load_iris()  
  
# 获取特征数据和标签数据  
X = iris.data  
y = iris.target  
  
# 可视化花的长度（单位：厘米）与花的宽度（单位：厘米）的关系  
plt.figure(figsize=(10, 6))  
plt.scatter(X[:, 2], X[:, 3])  # 第0列为花的类别，第2列和第3列分别是花的长度和宽度  
plt.xlabel('length')  
plt.ylabel('width')  
plt.title('relationship')  
plt.show()  
  
# 可视化花的形状（单位：厘米）与花的颜色（单位：厘米）的关系  
plt.figure(figsize=(10, 6))  
plt.scatter(X[:, 1], X[:, 0])  # 第1列和第0列分别是花的形状和颜色  
plt.xlabel('length')  
plt.ylabel('color')  
plt.title('relationship')  
plt.show()
