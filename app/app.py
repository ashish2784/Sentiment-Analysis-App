from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
import emoji
import contractions
import os

# Define the path to the frontend directory, which is one level up from the app's directory
frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend')
app = Flask(__name__, static_folder=frontend_dir)
CORS(app)

# Load the trained model and vectorizer
# The paths are relative to this script's location, so they will work correctly
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'svm_model.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'vectorizer.pkl')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Download necessary NLTK data
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = contractions.fix(text)
    text = emoji.demojize(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text.strip()

# API endpoint for text prediction
@app.route("/predict", methods=["POST"])
def predict():
    if not request.json or 'text' not in request.json:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    text = request.json["text"]
    if not text.strip():
        return jsonify({"error": "Input text cannot be empty"}), 400

    cleaned_text = clean_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)

    return jsonify({"sentiment": prediction[0]})

# API endpoint for file prediction
@app.route("/predict_file", methods=["POST"])
def predict_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    if file and file.filename.endswith('.txt'):
        try:
            text = file.read().decode('utf-8')
            if not text.strip():
                return jsonify({"error": "The uploaded file is empty"}), 400

            cleaned_text = clean_text(text)
            vectorized_text = vectorizer.transform([cleaned_text])
            prediction = model.predict(vectorized_text)

            return jsonify({"sentiment": prediction[0]})
        except Exception as e:
            return jsonify({"error": f"Error processing file: {e}"}), 500
    else:
        return jsonify({"error": "Invalid file type. Please upload a .txt file"}), 400

# Serve the frontend
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == "__main__":
    app.run(debug=True, port=5000)