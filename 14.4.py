from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

dt=fetch_20newsgroups()
tv=TfidfVectorizer()

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(dt['data'])

X_train, X_test, y_train, y_test = train_test_split(X, dt['target'], test_size=0.2, random_state=42)

classifier = MultinomialNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(report)
