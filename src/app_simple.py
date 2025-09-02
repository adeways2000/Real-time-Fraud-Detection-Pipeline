"""
Simplified Flask application for real-time fraud detection API.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import json
import yaml
import logging
from datetime import datetime
import time
import os
from feature_engineering import FeatureEngineer

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load configuration
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelService:
    """Service for loading and managing the fraud detection model."""
    
    def __init__(self):
        self.model = None
        self.feature_engineer = None
        self.model_metadata = {}
        self.is_loaded = False
        self.load_model()
    
    def load_model(self):
        """Load the trained model and feature engineer."""
        try:
            base_dir = os.path.dirname(os.path.dirname(__file__))
            
            # Load model
            model_path = os.path.join(base_dir, 'models', 'fraud_model.pkl')
            self.model = joblib.load(model_path)
            
            # Load feature engineer
            self.feature_engineer = FeatureEngineer()
            fe_path = os.path.join(base_dir, 'models', 'feature_engineer.pkl')
            self.feature_engineer.load(fe_path)
            
            # Load metadata
            metadata_path = os.path.join(base_dir, 'models', 'model_metadata.json')
            with open(metadata_path, 'r') as f:
                self.model_metadata = json.load(f)
            
            self.is_loaded = True
            logger.info("Model and feature engineer loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            self.is_loaded = False
    
    def predict(self, transaction_data):
        """Make fraud prediction for a transaction."""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        start_time = time.time()
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame([transaction_data])
            
            # Transform features
            X = self.feature_engineer.transform(df)
            
            # Make prediction
            fraud_probability = self.model.predict_proba(X)[0][1]
            prediction = "fraud" if fraud_probability > config['model']['threshold'] else "legitimate"
            confidence = max(fraud_probability, 1 - fraud_probability)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # ms
            
            result = {
                'fraud_probability': round(fraud_probability, 4),
                'prediction': prediction,
                'confidence': round(confidence, 4),
                'processing_time_ms': round(processing_time, 2),
                'timestamp': datetime.now().isoformat(),
                'model_version': self.model_metadata.get('model_type', 'unknown')
            }
            
            logger.info(f"Prediction made: {prediction} (prob: {fraud_probability:.4f})")
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

# Initialize model service
model_service = ModelService()

@app.route('/')
def index():
    """Root endpoint with API information."""
    return jsonify({
        'name': config['app']['name'],
        'version': config['app']['version'],
        'description': 'Real-time fraud detection API',
        'model_loaded': model_service.is_loaded
    })

@app.route('/health', methods=['GET'])
def health():
    """Check system health and model status."""
    return jsonify({
        'status': 'healthy' if model_service.is_loaded else 'unhealthy',
        'model_loaded': model_service.is_loaded,
        'timestamp': datetime.now().isoformat(),
        'version': config['app']['version']
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make fraud prediction for a transaction."""
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate required fields
        required_fields = ['amount', 'merchant_category', 'hour', 'day_of_week', 'user_age', 'account_age_days']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {missing_fields}'}), 400
        
        # Validate data types and ranges
        try:
            data['amount'] = float(data['amount'])
            data['hour'] = int(data['hour'])
            data['day_of_week'] = int(data['day_of_week'])
            data['user_age'] = int(data['user_age'])
            data['account_age_days'] = int(data['account_age_days'])
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid data types in request'}), 400
        
        # Validate ranges
        if not (0 <= data['hour'] <= 23):
            return jsonify({'error': 'Hour must be between 0 and 23'}), 400
        
        if not (0 <= data['day_of_week'] <= 6):
            return jsonify({'error': 'Day of week must be between 0 and 6'}), 400
        
        if data['amount'] <= 0:
            return jsonify({'error': 'Amount must be positive'}), 400
        
        # Make prediction
        result = model_service.predict(data)
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Prediction endpoint error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info(f"Starting {config['app']['name']} v{config['app']['version']}")
    logger.info(f"Model loaded: {model_service.is_loaded}")
    
    app.run(
        host=config['app']['host'],
        port=config['app']['port'],
        debug=config['app']['debug']
    )

