from flask import Flask, request, jsonify
import pandas as pd
import pickle

# Initialize Flask app
app = Flask(__name__)

# ================================
# Load model, scaler, and columns
# ================================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("columns.pkl", "rb") as f:
    columns = pickle.load(f)   # this should be a list of feature names


# ================================
# Prediction API
# ================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.get_json()

        if data is None:
            return jsonify({"error": "No JSON data received"}), 400

        # Convert JSON to DataFrame with correct column order
        input_df = pd.DataFrame([data], columns=columns)

        # Check for missing values
        if input_df.isnull().values.any():
            return jsonify({"error": "Missing or invalid input values"}), 400

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)

        # Convert numpy type to normal int
        result = int(prediction[0])

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================================
# Run Flask App
# ================================
if __name__ == "__main__":
    app.run(debug=True)

