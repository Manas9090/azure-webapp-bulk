import numpy as np
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load(open("model.pkl", "rb"))

@app.route("/predict_api", methods=["POST"])
def predict_api():
    try:
        # Get JSON input
        data = request.get_json(force=True)

        # Expecting: {"input": [age, bmi, children, smoker_flag, region_code]}
        features = np.array(data["input"]).reshape(1, -1)

        # Model prediction
        prediction = model.predict(features)
        output = round(float(prediction[0]), 2)

        return jsonify({"prediction": output})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
