from flask import Flask, request, jsonify
import pickle
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import nltk
import logging

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Text Cleaning Function
def clean_text(text):
    try:
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)

        # Remove stop words
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(filtered_tokens)
    except Exception as e:
        logging.error(f"Error cleaning text: {e}")
        return ""

# Load trained model and vectorizer
try:
    with open("naive_bayes_model.pkl", "rb") as model_file:
        nb_classifier = pickle.load(model_file)

    with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
except FileNotFoundError as e:
    raise Exception(f"Model or vectorizer file not found: {e}")

@app.route('/classify_email', methods=['POST'])
def classify_email():
    try:
        # Get input JSON
        data = request.get_json()
        if not data or 'subject' not in data or 'body' not in data:
            return jsonify({"error": "Invalid input. Please provide both subject and body."}), 400

        subject = data['subject']
        body = data['body']

        # Combine subject and body
        email_content = f"{subject} {body}"

        # Clean and vectorize email content
        cleaned_content = clean_text(email_content)
        logging.info(f"Cleaned Content: {cleaned_content}")
        text_vectorized = vectorizer.transform([cleaned_content])

        # Predict sentiment
        prediction = nb_classifier.predict(text_vectorized)
        logging.info(f"Prediction: {prediction[0]}")
        result = "Inappropriate" if prediction[0] == 1 else "Appropriate"

        return jsonify({"subject": subject, "body": body, "classification": result})
    except Exception as e:
        logging.error(f"Error in classify_email: {e}")
        return jsonify({"error": "An error occurred during classification."}), 500

@app.route('/classify_bulk', methods=['POST'])
def classify_bulk_emails():
    try:
        # Handle bulk email classification
        data = request.get_json()
        if not data or 'emails' not in data or not isinstance(data['emails'], list):
            return jsonify({"error": "Invalid input. Please provide a list of emails with 'subject' and 'body'."}), 400

        results = []
        for email in data['emails']:
            subject = email.get('subject', "")
            body = email.get('body', "")
            email_content = f"{subject} {body}"
            cleaned_content = clean_text(email_content)
            text_vectorized = vectorizer.transform([cleaned_content])
            prediction = nb_classifier.predict(text_vectorized)
            result = "Inappropriate" if prediction[0] == 1 else "Appropriate"
            results.append({"subject": subject, "body": body, "classification": result})

        return jsonify({"results": results})
    except Exception as e:
        logging.error(f"Error in classify_bulk_emails: {e}")
        return jsonify({"error": "An error occurred during bulk classification."}), 500

@app.route('/test_connection', methods=['GET'])
def test_connection():
    # Health check for the API
    return jsonify({"status": "API is running."})

if __name__ == '__main__':
    # Run the app
    app.run(debug=True, host="0.0.0.0", port=5001)
