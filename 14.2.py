# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer  
  
# 定义要向量化的文本列表
texts = ['This is a boy,who likes playing football.',
    'This is a girl, who like playing tennis.']  
  
# 创建CountVectorizer对象
vectorizer = CountVectorizer()
  
# 将文本向量化为词频矩阵
X = vectorizer.fit_transform(texts)

print(X)

# 输出词频矩阵
print(X.toarray())  
  
# 输出特征名称（即词汇表)
print(vectorizer.get_feature_names_out())
