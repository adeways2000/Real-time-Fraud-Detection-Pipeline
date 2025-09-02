"""
Feature engineering module for fraud detection.
Handles data preprocessing, feature transformation, and validation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import yaml
import logging
from typing import Dict, List, Tuple, Any

class FeatureEngineer:
    """
    Feature engineering pipeline for fraud detection.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize feature engineer with configuration.
        
        Args:
            config_path (str): Path to configuration file
        """
        if config_path is None:
            # Use default path relative to this file
            import os
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.numerical_features = self.config['features']['numerical']
        self.categorical_features = self.config['features']['categorical']
        
        # Initialize preprocessors
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.is_fitted = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features from raw data.
        
        Args:
            df (pd.DataFrame): Raw transaction data
            
        Returns:
            pd.DataFrame: Data with engineered features
        """
        df = df.copy()
        
        # Time-based features
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Ensure is_weekend is created
        if 'is_weekend' not in df.columns:
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Amount-based features
        if 'amount' in df.columns:
            df['amount_log'] = np.log1p(df['amount'])
            df['amount_sqrt'] = np.sqrt(df['amount'])
            
            # Amount bins
            amount_bins = self.config['feature_engineering']['amount_bins']
            df['amount_bin'] = pd.cut(df['amount'], bins=amount_bins, 
                                    labels=[f'bin_{i}' for i in range(len(amount_bins)-1)])
        
        # Hour bins
        df['hour_bin'] = pd.cut(df['hour'], bins=[0, 6, 12, 18, 24], 
                               labels=['night', 'morning', 'afternoon', 'evening'])
        
        # Risk indicators
        df['high_amount'] = (df['amount'] > 1000).astype(int)
        df['unusual_hour'] = ((df['hour'] < 6) | (df['hour'] > 22)).astype(int)
        df['new_account'] = (df['account_age_days'] < 30).astype(int)
        
        # Interaction features
        df['amount_per_account_age'] = df['amount'] / (df['account_age_days'] + 1)
        df['weekend_high_amount'] = df['is_weekend'] * df['high_amount']
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate input data quality and completeness.
        
        Args:
            df (pd.DataFrame): Input data to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required columns
        required_cols = self.numerical_features + self.categorical_features
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Check for null values
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            issues.append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
        
        # Check data types and ranges
        if 'amount' in df.columns:
            if (df['amount'] < 0).any():
                issues.append("Negative amounts found")
            if (df['amount'] > 100000).any():
                issues.append("Extremely high amounts found (>100k)")
        
        if 'hour' in df.columns:
            if not df['hour'].between(0, 23).all():
                issues.append("Invalid hour values found")
        
        if 'day_of_week' in df.columns:
            if not df['day_of_week'].between(0, 6).all():
                issues.append("Invalid day_of_week values found")
        
        return len(issues) == 0, issues
    
    def fit(self, df: pd.DataFrame) -> 'FeatureEngineer':
        """
        Fit the feature engineering pipeline on training data.
        
        Args:
            df (pd.DataFrame): Training data
            
        Returns:
            FeatureEngineer: Fitted feature engineer
        """
        self.logger.info("Fitting feature engineering pipeline...")
        
        # Create features
        df_features = self.create_features(df)
        
        # Validate data
        is_valid, issues = self.validate_data(df_features)
        if not is_valid:
            raise ValueError(f"Data validation failed: {issues}")
        
        # Prepare features for training
        X = self.prepare_features(df_features, fit=True)
        
        self.is_fitted = True
        self.logger.info(f"Feature engineering pipeline fitted with {len(self.feature_names)} features")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted pipeline.
        
        Args:
            df (pd.DataFrame): Data to transform
            
        Returns:
            np.ndarray: Transformed features
        """
        if not self.is_fitted:
            raise ValueError("Feature engineer must be fitted before transform")
        
        # Create features
        df_features = self.create_features(df)
        
        # Validate data
        is_valid, issues = self.validate_data(df_features)
        if not is_valid:
            self.logger.warning(f"Data validation issues: {issues}")
        
        # Prepare features
        X = self.prepare_features(df_features, fit=False)
        
        return X
    
    def prepare_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """
        Prepare features for model training/prediction.
        
        Args:
            df (pd.DataFrame): Data with engineered features
            fit (bool): Whether to fit preprocessors
            
        Returns:
            np.ndarray: Processed feature matrix
        """
        feature_list = []
        
        # Process numerical features
        numerical_data = df[self.numerical_features].values
        if fit:
            numerical_data = self.scaler.fit_transform(numerical_data)
        else:
            numerical_data = self.scaler.transform(numerical_data)
        feature_list.append(numerical_data)
        
        # Process categorical features
        for col in self.categorical_features:
            if fit:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                encoded = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # Handle unseen categories
                encoded = []
                for val in df[col].astype(str):
                    try:
                        encoded.append(self.label_encoders[col].transform([val])[0])
                    except ValueError:
                        # Assign to most common class for unseen categories
                        encoded.append(0)
                encoded = np.array(encoded)
            
            feature_list.append(encoded.reshape(-1, 1))
        
        # Add engineered features
        if 'amount_log' in df.columns:
            feature_list.append(df[['amount_log']].values)
        if 'high_amount' in df.columns:
            feature_list.append(df[['high_amount', 'unusual_hour', 'new_account']].values)
        if 'amount_per_account_age' in df.columns:
            feature_list.append(df[['amount_per_account_age']].values)
        
        # Combine all features
        X = np.hstack(feature_list)
        
        if fit:
            # Store feature names for reference
            self.feature_names = (self.numerical_features + 
                                self.categorical_features + 
                                ['amount_log', 'high_amount', 'unusual_hour', 
                                 'new_account', 'amount_per_account_age'])
        
        return X
    
    def get_feature_importance_names(self) -> List[str]:
        """
        Get feature names for importance analysis.
        
        Returns:
            List[str]: Feature names
        """
        return self.feature_names
    
    def save(self, path: str) -> None:
        """
        Save fitted feature engineer.
        
        Args:
            path (str): Path to save the feature engineer
        """
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'config': self.config,
            'is_fitted': self.is_fitted
        }, path)
        self.logger.info(f"Feature engineer saved to {path}")
    
    def load(self, path: str) -> 'FeatureEngineer':
        """
        Load fitted feature engineer.
        
        Args:
            path (str): Path to load the feature engineer from
            
        Returns:
            FeatureEngineer: Loaded feature engineer
        """
        data = joblib.load(path)
        self.scaler = data['scaler']
        self.label_encoders = data['label_encoders']
        self.feature_names = data['feature_names']
        self.config = data['config']
        self.is_fitted = data['is_fitted']
        
        self.numerical_features = self.config['features']['numerical']
        self.categorical_features = self.config['features']['categorical']
        
        self.logger.info(f"Feature engineer loaded from {path}")
        return self

def prepare_training_data(data_path: str = '../data/train_data.csv') -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare training data for model training.
    
    Args:
        data_path (str): Path to training data
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (X, y) training data
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    # Fit and transform
    fe.fit(df)
    X = fe.transform(df)
    y = df['is_fraud'].values
    
    # Save fitted feature engineer
    fe.save('../models/feature_engineer.pkl')
    
    return X, y, fe

if __name__ == "__main__":
    # Test feature engineering pipeline
    print("Testing feature engineering pipeline...")
    
    X, y, fe = prepare_training_data()
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    print(f"Feature names: {fe.get_feature_importance_names()}")
    
    # Test on sample data
    test_df = pd.read_csv('../data/test_data.csv')
    X_test = fe.transform(test_df)
    print(f"Test feature matrix shape: {X_test.shape}")
    
    print("Feature engineering pipeline test completed successfully!")

