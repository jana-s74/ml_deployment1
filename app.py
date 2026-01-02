from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os

app = Flask(__name__)

# ================================
# Load model, scaler, and columns
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(BASE_DIR, "columns.pkl"), "rb") as f:
    columns = pickle.load(f)


# ================================
# Prediction API
# ================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        input_df = pd.DataFrame([data])

        # Ensure correct column order
        input_df = input_df[columns]

        if input_df.isnull().any().any():
            return jsonify({"error": "Missing input values"}), 400

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)

        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================================
# Render entry point
# ================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


