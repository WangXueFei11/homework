from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

dt=fetch_20newsgroups()
text=[dt.data[0]]
tv=TfidfVectorizer()
v=tv.fit_transform(text)
print(v)
print(tv.get_feature_names_out())
print(v.toarray())
