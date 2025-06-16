
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import pickle

# Load dataset
data = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
data.columns = ['label', 'text']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Vectorizer dan training
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['label']

model = MultinomialNB()
model.fit(X, y)

# Save model dan vectorizer
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
