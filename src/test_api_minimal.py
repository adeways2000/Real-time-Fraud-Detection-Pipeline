"""
Minimal test API to verify basic functionality.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(app)

# Load model and feature engineer
base_dir = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(base_dir, 'models', 'fraud_model.pkl')
model = joblib.load(model_path)

# Simple feature transformation (without the complex feature engineering)
def simple_transform(data):
    """Simple feature transformation for testing."""
    features = [
        data['amount'],
        1 if data['merchant_category'] == 'online' else 0,  # Simple encoding
        data['hour'],
        data['day_of_week'],
        data['user_age'],
        data['account_age_days'],
        # Add some derived features
        1 if data['amount'] > 1000 else 0,  # high_amount
        1 if data['hour'] < 6 or data['hour'] > 22 else 0,  # unusual_hour
        1 if data['account_age_days'] < 30 else 0,  # new_account
        data['amount'] / (data['account_age_days'] + 1),  # amount_per_account_age
        0  # placeholder for last feature
    ]
    return [features]

@app.route('/predict', methods=['POST'])
def predict():
    """Make fraud prediction."""
    try:
        data = request.get_json()
        
        # Simple validation
        required_fields = ['amount', 'merchant_category', 'hour', 'day_of_week', 'user_age', 'account_age_days']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Transform features
        X = simple_transform(data)
        
        # Make prediction
        fraud_probability = float(model.predict_proba(X)[0][1])  # Convert to Python float
        prediction = "fraud" if fraud_probability > 0.5 else "legitimate"
        
        return jsonify({
            'fraud_probability': round(fraud_probability, 4),
            'prediction': prediction,
            'confidence': round(max(fraud_probability, 1 - fraud_probability), 4)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check."""
    return jsonify({'status': 'healthy', 'model_loaded': True})

if __name__ == '__main__':
    print("Starting minimal test API...")
    app.run(host='0.0.0.0', port=5000, debug=False)

