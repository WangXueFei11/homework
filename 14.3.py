# -*- coding: utf-8 -*-
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
# 加载数据集
newsgroups_train = fetch_20newsgroups(subset='train')
# 创建TfidfVectorizer对象
vectorizer = TfidfVectorizer()
# 对文本进行向量化 
vectors = vectorizer.fit_transform(newsgroups_train.data)
# 获取特征名（即词汇表）
feature_names = vectorizer.get_feature_names_out()
# 输出第一个文本的结果向量
first_document_vector = vectors[0]
print("first_document_vector:")
print(first_document_vector)
