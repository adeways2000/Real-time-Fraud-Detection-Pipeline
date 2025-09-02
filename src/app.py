"""
Main Flask application for real-time fraud detection API.
Provides RESTful endpoints for fraud prediction and system monitoring.
"""

from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import json
import yaml
import logging
from datetime import datetime
import time
import threading
from queue import Queue
import os
from feature_engineering import FeatureEngineer
from monitoring import SystemMonitor

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
api = Api(app)

# Load configuration
import os
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Setup logging
logging.basicConfig(
    level=getattr(logging, config['monitoring']['log_level']),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelService:
    """
    Service for loading and managing the fraud detection model.
    """
    
    def __init__(self):
        self.model = None
        self.feature_engineer = None
        self.model_metadata = {}
        self.is_loaded = False
        self.load_model()
    
    def load_model(self):
        """Load the trained model and feature engineer."""
        try:
            # Get base directory
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
        """
        Make fraud prediction for a transaction.
        
        Args:
            transaction_data (dict): Transaction features
            
        Returns:
            dict: Prediction results
        """
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
            
            logger.info(f"Prediction made: {prediction} (prob: {fraud_probability:.4f}, time: {processing_time:.2f}ms)")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

# Initialize model service
model_service = ModelService()

# Initialize system monitor
system_monitor = SystemMonitor()

class PredictResource(Resource):
    """
    Resource for fraud prediction endpoint.
    """
    
    def post(self):
        """
        Make fraud prediction for a transaction.
        
        Expected JSON payload:
        {
            "amount": 150.75,
            "merchant_category": "online",
            "hour": 23,
            "day_of_week": 6,
            "user_age": 35,
            "account_age_days": 365
        }
        """
        try:
            # Get request data
            data = request.get_json()
            
            if not data:
                return {'error': 'No JSON data provided'}, 400
            
            # Validate required fields
            required_fields = ['amount', 'merchant_category', 'hour', 'day_of_week', 'user_age', 'account_age_days']
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                return {'error': f'Missing required fields: {missing_fields}'}, 400
            
            # Validate data types and ranges
            try:
                data['amount'] = float(data['amount'])
                data['hour'] = int(data['hour'])
                data['day_of_week'] = int(data['day_of_week'])
                data['user_age'] = int(data['user_age'])
                data['account_age_days'] = int(data['account_age_days'])
            except (ValueError, TypeError):
                return {'error': 'Invalid data types in request'}, 400
            
            # Validate ranges
            if not (0 <= data['hour'] <= 23):
                return {'error': 'Hour must be between 0 and 23'}, 400
            
            if not (0 <= data['day_of_week'] <= 6):
                return {'error': 'Day of week must be between 0 and 6'}, 400
            
            if data['amount'] <= 0:
                return {'error': 'Amount must be positive'}, 400
            
            # Make prediction
            result = model_service.predict(data)
            
            # Update monitoring
            system_monitor.record_prediction(result)
            
            return result, 200
            
        except Exception as e:
            logger.error(f"Prediction endpoint error: {str(e)}")
            return {'error': 'Internal server error'}, 500

class HealthResource(Resource):
    """
    Resource for health check endpoint.
    """
    
    def get(self):
        """
        Check system health and model status.
        """
        try:
            health_status = {
                'status': 'healthy' if model_service.is_loaded else 'unhealthy',
                'model_loaded': model_service.is_loaded,
                'model_metadata': model_service.model_metadata,
                'timestamp': datetime.now().isoformat(),
                'uptime_seconds': system_monitor.get_uptime(),
                'version': config['app']['version']
            }
            
            status_code = 200 if model_service.is_loaded else 503
            return health_status, status_code
            
        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            return {'status': 'error', 'message': str(e)}, 500

class MetricsResource(Resource):
    """
    Resource for system metrics endpoint.
    """
    
    def get(self):
        """
        Get system metrics and model performance statistics.
        """
        try:
            metrics = system_monitor.get_metrics()
            return metrics, 200
            
        except Exception as e:
            logger.error(f"Metrics endpoint error: {str(e)}")
            return {'error': 'Failed to retrieve metrics'}, 500

class BatchPredictResource(Resource):
    """
    Resource for batch prediction endpoint.
    """
    
    def post(self):
        """
        Make fraud predictions for multiple transactions.
        
        Expected JSON payload:
        {
            "transactions": [
                {
                    "amount": 150.75,
                    "merchant_category": "online",
                    "hour": 23,
                    "day_of_week": 6,
                    "user_age": 35,
                    "account_age_days": 365
                },
                ...
            ]
        }
        """
        try:
            data = request.get_json()
            
            if not data or 'transactions' not in data:
                return {'error': 'No transactions provided'}, 400
            
            transactions = data['transactions']
            
            if len(transactions) > config['data']['max_batch_size']:
                return {'error': f'Batch size exceeds maximum of {config["data"]["max_batch_size"]}'}, 400
            
            results = []
            
            for i, transaction in enumerate(transactions):
                try:
                    result = model_service.predict(transaction)
                    result['transaction_index'] = i
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing transaction {i}: {str(e)}")
                    results.append({
                        'transaction_index': i,
                        'error': str(e)
                    })
            
            return {
                'predictions': results,
                'total_processed': len(results),
                'timestamp': datetime.now().isoformat()
            }, 200
            
        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            return {'error': 'Internal server error'}, 500

# Add resources to API
api.add_resource(PredictResource, '/predict')
api.add_resource(HealthResource, '/health')
api.add_resource(MetricsResource, '/metrics')
api.add_resource(BatchPredictResource, '/batch_predict')

@app.route('/')
def index():
    """
    Root endpoint with API information.
    """
    return jsonify({
        'name': config['app']['name'],
        'version': config['app']['version'],
        'description': 'Real-time fraud detection API',
        'endpoints': {
            'POST /predict': 'Make fraud prediction for a single transaction',
            'POST /batch_predict': 'Make fraud predictions for multiple transactions',
            'GET /health': 'Check system health and model status',
            'GET /metrics': 'Get system metrics and performance statistics'
        },
        'model_info': model_service.model_metadata if model_service.is_loaded else None
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Start system monitor in background
    monitor_thread = threading.Thread(target=system_monitor.start_monitoring, daemon=True)
    monitor_thread.start()
    
    # Start Flask app
    logger.info(f"Starting {config['app']['name']} v{config['app']['version']}")
    logger.info(f"Model loaded: {model_service.is_loaded}")
    
    app.run(
        host=config['app']['host'],
        port=config['app']['port'],
        debug=config['app']['debug']
    )

