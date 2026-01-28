from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# Initialize app
app = Flask(__name__)
CORS(app)  # allow browser access

# Load trained model
model = joblib.load("fake_news_model.pkl")


@app.route("/", methods=["GET"])
def home():
    return "Fake News Backend Running", 200


@app.route("/health", methods=["GET"])
def health():
    return "OK", 200


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    # Length check (VERY IMPORTANT)
    if len(text.split()) < 1:
        return jsonify({
            "result": "Input too short for reliable prediction",
            "confidence": 0
        })

    prediction = model.predict([text])[0]
    confidence = model.predict_proba([text])[0].max()

    return jsonify({
        "result": "Fake" if prediction == 0 else "Real",
        "confidence": round(confidence * 100, 2)
    })


if __name__ == "__main__":
    app.run()
