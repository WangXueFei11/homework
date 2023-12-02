from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer()
text=['This is a boy,who likes playing football.',
    'This is a girl, who like playing tennis.']

v=cv.fit_transform(text)
print(v)
print(cv.get_feature_names_out())
print(v.toarray())
