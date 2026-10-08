from flask import Flask, request, jsonify
import joblib
import numpy as np
import joblib
app = Flask(__name__)
# Define export filenames
model_filename = 'logistic_regression_model.joblib'
scaler_filename = 'scaler.joblib'

# Save the model and scaler
joblib.dump(model, model_filename)
joblib.dump(scaler, scaler_filename)

print(f"Successfully exported model to: {model_filename}")
print(f"Successfully exported scaler to: {scaler_filename}")


# Load the exported scaler and model
scaler = joblib.load('scaler.joblib')
model = joblib.load('logistic_regression_model.joblib')

# Feature names in the exact order they were trained on
FEATURES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        
        # Extract features from the JSON payload in correct order
        features_list = [float(data[feat]) for feat in FEATURES]
        features_array = np.array([features_list])
        
        # Standardize the features using the loaded scaler
        scaled_features = scaler.transform(features_array)
        
        # Make prediction and obtain probability
        prediction = int(model.predict(scaled_features)[0])
        probability = float(model.predict_proba(scaled_features)[0][1])
        
        return jsonify({
            'prediction': prediction,
            'prediction_label': 'Diabetes' if prediction == 1 else 'No Diabetes',
            'probability': probability
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    app.run(host='0.0.0.0', port=5000)
