import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    X_new = scaler.transform([data["features"]])
    prediction = model.predict(X_new)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
