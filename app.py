from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Load column names used during training
X = pd.read_csv("X_columns.csv")  # only column names required

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json(force=True)

        # Convert JSON to DataFrame with correct column order
        input_df = pd.DataFrame([data])
        input_df = input_df[X.columns]

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)

        # Return result
        return jsonify({
            'prediction': int(prediction[0])
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        })

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
