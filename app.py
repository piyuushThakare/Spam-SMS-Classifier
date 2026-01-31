from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Load model & vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

with open(os.path.join(MODEL_DIR, "spam_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    is_spam = None

    if request.method == "POST":
        message = request.form.get("message")

        message_vec = vectorizer.transform([message])
        proba = model.predict_proba(message_vec)[0]

          # 🔥 Correct class mapping
        class_labels = model.classes_

        prob_map = dict(zip(class_labels, proba))

        spam_prob = proba[1] * 100
        ham_prob = proba[0] * 100

        if spam_prob > ham_prob:
            is_spam = True
            prediction = "🚫 Spam Message"
            confidence = f"{spam_prob:.2f}%"
        else:
            is_spam = False
            prediction = "✅ Not Spam"
            confidence = f"{ham_prob:.2f}%"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        is_spam=is_spam
    )
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)

