import pandas as pd
import re
import string
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib
import nltk
from nltk.corpus import stopwords
import emoji
import contractions

# Ensure the model directory exists
os.makedirs("model", exist_ok=True)

# Download necessary NLTK data
nltk.download("stopwords")

# Load dataset
df = pd.read_csv("twitter_training.csv", header=None)
df.columns = ["id", "entity", "sentiment", "text"]

# Drop rows with missing text or sentiment
df.dropna(subset=['text', 'sentiment'], inplace=True)

# Text cleaning function
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = contractions.fix(text)
    text = emoji.demojize(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text.strip()

df["clean_text"] = df["text"].apply(clean_text)

# Drop rows where clean_text is empty
df = df[df['clean_text'] != '']

# Use all sentiments for training
X = df["clean_text"]
y = df["sentiment"]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
X_tfidf = vectorizer.fit_transform(X)

# Train SVM model
model = LinearSVC(C=1.0, random_state=42, max_iter=2000) # Increased max_iter for convergence
model.fit(X_tfidf, y)

# Save the model and vectorizer
joblib.dump(model, "model/svm_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("Model and vectorizer trained on all sentiments and saved successfully.")