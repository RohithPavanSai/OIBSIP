import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Generate local text review records
reviews = [
    ("Love this product! Super fast shipping.", "positive"),
    ("Fantastic customer service and high quality.", "positive"),
    ("Terrible quality. Complete waste of my money.", "negative"),
    ("The item arrived completely broken and damaged.", "negative"),
    ("It is okay, nothing special but works fine.", "neutral"),
    ("Average shipping times, standard experience.", "neutral")
]
mock_data = [reviews[np.random.randint(0, len(reviews))] for _ in range(150)]
df = pd.DataFrame(mock_data, columns=['Review_Text', 'Sentiment'])

# 2. Text Preprocessing Pipeline
df['Cleaned_Review'] = df['Review_Text'].apply(lambda x: re.sub(r'[^\w\s]', '', x.lower()))

# 3. Feature Extraction (TF-IDF Vectorizer Matrix)
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['Cleaned_Review'])
y = df['Sentiment']

# 4. Stratified Train/Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Model Execution (Train Naive Bayes Classifier)
model = MultinomialNB()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# 6. Evaluation Matrix Print Out
print("--- Sentiment Model Performance Matrix ---")
print(f"Model Base Accuracy: {accuracy_score(y_test, predictions):.2f}\n")
print(classification_report(y_test, predictions))
print("Sentiment Analysis execution complete successfully!")