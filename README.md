

# 🐦 Twitter Sentiment Analysis

## 📌 Project Overview

This project aims to classify tweets into four categories: **Positive, Negative, Neutral, and Irrelevant**.
We used a dataset of **74K+ tweets**, applied advanced **text preprocessing**, and built a machine learning pipeline to achieve high accuracy in sentiment classification.

---

## 🔹 Problem Statement

Social media platforms like Twitter generate massive amounts of text data daily.
Understanding public sentiment is crucial for:

* **Brand monitoring**
* **Customer feedback analysis**
* **Political or social sentiment tracking**

The challenge was to design a **robust NLP pipeline** that can handle noisy tweet data (emojis, hashtags, contractions, slang, etc.) and accurately classify sentiment.

---

## 🔹 Approach

### 1️⃣ Data Preprocessing

* Expanded contractions (*don’t → do not*)
* Converted emojis to text (*😊 → smiling\_face*)
* Removed URLs, mentions, hashtags, and special characters
* Removed stopwords for cleaner text

### 2️⃣ Feature Engineering

* Applied TF-IDF vectorization with unigrams + bigrams
* Extracted 20,000 most important features

### 3️⃣ Model Training

* Logistic Regression** (baseline)
* Linear SVM (Support Vector Machine) → achieved best performance
* Tuned hyperparameters using GridSearchCV

### 4️⃣ Evaluation

* Accuracy: \~81% on test data
* Balanced precision & recall across classes

---

## 🔹 Results

✅ Cleaned and processed 74K+ tweets
✅ Built sentiment classifier achieving 81% accuracy
✅ Improved interpretability with n-grams capturing context (e.g., “not good”, “very happy”)


---

## 🔹 Tech Stack

* **Python**
* **Pandas, NumPy**
* **scikit-learn, NLTK**
* **Matplotlib, Seaborn** (for visualization)

---

## 🔹 Future Improvements

* Deploy as a **Flask/FastAPI web app for real-time sentiment analysis
* Use **Hugging Face Transformers (BERT/RoBERTa)** for state-of-the-art performance
* Incorporate **topic modeling** for deeper insights

---

