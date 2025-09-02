"""
Direct test of the model and feature engineering pipeline.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from feature_engineering import FeatureEngineer

def test_model_directly():
    """Test the model directly without Flask."""
    print("Testing model directly...")
    
    try:
        # Get base directory
        base_dir = os.path.dirname(os.path.dirname(__file__))
        print(f"Base directory: {base_dir}")
        
        # Load model
        model_path = os.path.join(base_dir, 'models', 'fraud_model.pkl')
        print(f"Loading model from: {model_path}")
        model = joblib.load(model_path)
        print("Model loaded successfully")
        
        # Load feature engineer
        print("Loading feature engineer...")
        feature_engineer = FeatureEngineer()
        fe_path = os.path.join(base_dir, 'models', 'feature_engineer.pkl')
        print(f"Loading feature engineer from: {fe_path}")
        feature_engineer.load(fe_path)
        print("Feature engineer loaded successfully")
        
        # Test transaction
        transaction_data = {
            "amount": 100.50,
            "merchant_category": "grocery",
            "hour": 14,
            "day_of_week": 2,
            "user_age": 35,
            "account_age_days": 365
        }
        
        print(f"Test transaction: {transaction_data}")
        
        # Convert to DataFrame
        df = pd.DataFrame([transaction_data])
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {df.columns.tolist()}")
        
        # Transform features
        print("Transforming features...")
        X = feature_engineer.transform(df)
        print(f"Feature matrix shape: {X.shape}")
        print(f"Feature matrix: {X}")
        
        # Make prediction
        print("Making prediction...")
        fraud_probability = model.predict_proba(X)[0][1]
        prediction = "fraud" if fraud_probability > 0.5 else "legitimate"
        
        print(f"Fraud probability: {fraud_probability}")
        print(f"Prediction: {prediction}")
        
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_directly()
    print(f"Test {'PASSED' if success else 'FAILED'}")

