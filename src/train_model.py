"""
Model training script for fraud detection.
Trains and evaluates machine learning models for fraud detection.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import joblib
import json
import yaml
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from feature_engineering import FeatureEngineer, prepare_training_data

class FraudDetectionModel:
    """
    Fraud detection model trainer and evaluator.
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Initialize model trainer.
        
        Args:
            config_path (str): Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.feature_engineer = None
        self.model_metadata = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train_model(self, X_train, y_train, X_val, y_val, model_type='xgboost'):
        """
        Train fraud detection model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            model_type: Type of model to train ('xgboost', 'random_forest', 'logistic')
        """
        self.logger.info(f"Training {model_type} model...")
        
        if model_type == 'xgboost':
            # XGBoost with class balancing
            scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
            
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric='auc'
            )
            
            # Train with early stopping
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=20,
                verbose=False
            )
            
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
            
        elif model_type == 'logistic':
            self.model = LogisticRegression(
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            )
            self.model.fit(X_train, y_train)
        
        # Store model metadata
        self.model_metadata = {
            'model_type': model_type,
            'training_date': datetime.now().isoformat(),
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'fraud_rate_train': y_train.mean(),
            'fraud_rate_val': y_val.mean(),
            'feature_count': X_train.shape[1]
        }
        
        self.logger.info(f"Model training completed. Training samples: {len(X_train)}")
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'auc_score': auc_score,
            'accuracy': report['accuracy'],
            'precision_fraud': report['1']['precision'],
            'recall_fraud': report['1']['recall'],
            'f1_fraud': report['1']['f1-score'],
            'precision_legitimate': report['0']['precision'],
            'recall_legitimate': report['0']['recall'],
            'f1_legitimate': report['0']['f1-score'],
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        # Update metadata
        self.model_metadata.update(metrics)
        
        self.logger.info(f"Model evaluation completed. AUC: {auc_score:.4f}")
        
        return metrics
    
    def get_feature_importance(self, feature_names):
        """
        Get feature importance from trained model.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            dict: Feature importance scores
        """
        if self.model is None:
            raise ValueError("Model must be trained before getting feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
        else:
            return {}
        
        # Create feature importance dictionary
        feature_importance = dict(zip(feature_names, importance))
        
        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True))
        
        return feature_importance
    
    def hyperparameter_tuning(self, X_train, y_train, model_type='xgboost'):
        """
        Perform hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_type: Type of model to tune
        """
        self.logger.info(f"Starting hyperparameter tuning for {model_type}...")
        
        if model_type == 'xgboost':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.8, 0.9]
            }
            
            base_model = xgb.XGBClassifier(
                scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
                random_state=42
            )
            
        elif model_type == 'random_forest':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [8, 10, 12],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            
            base_model = RandomForestClassifier(
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        self.model_metadata['best_params'] = grid_search.best_params_
        self.model_metadata['best_cv_score'] = grid_search.best_score_
        
        self.logger.info(f"Best parameters: {grid_search.best_params_}")
        self.logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
    
    def save_model(self, model_path='../models/fraud_model.pkl', 
                   metadata_path='../models/model_metadata.json'):
        """
        Save trained model and metadata.
        
        Args:
            model_path: Path to save the model
            metadata_path: Path to save metadata
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save model
        joblib.dump(self.model, model_path)
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(self.model_metadata, f, indent=2)
        
        self.logger.info(f"Model saved to {model_path}")
        self.logger.info(f"Metadata saved to {metadata_path}")
    
    def load_model(self, model_path='../models/fraud_model.pkl',
                   metadata_path='../models/model_metadata.json'):
        """
        Load trained model and metadata.
        
        Args:
            model_path: Path to load the model from
            metadata_path: Path to load metadata from
        """
        # Load model
        self.model = joblib.load(model_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.model_metadata = json.load(f)
        
        self.logger.info(f"Model loaded from {model_path}")

def train_fraud_detection_model():
    """
    Main training pipeline for fraud detection model.
    """
    print("Starting fraud detection model training pipeline...")
    
    # Prepare training data
    print("Preparing training data...")
    X, y, feature_engineer = prepare_training_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Initialize model trainer
    model_trainer = FraudDetectionModel()
    
    # Train model (try XGBoost first)
    print("Training XGBoost model...")
    model_trainer.train_model(X_train, y_train, X_val, y_val, model_type='xgboost')
    
    # Evaluate model
    print("Evaluating model...")
    metrics = model_trainer.evaluate_model(X_test, y_test)
    
    print(f"Model Performance:")
    print(f"AUC Score: {metrics['auc_score']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Fraud Precision: {metrics['precision_fraud']:.4f}")
    print(f"Fraud Recall: {metrics['recall_fraud']:.4f}")
    print(f"Fraud F1-Score: {metrics['f1_fraud']:.4f}")
    
    # Feature importance
    feature_names = feature_engineer.get_feature_importance_names()
    feature_importance = model_trainer.get_feature_importance(feature_names)
    
    print("\\nTop 10 Most Important Features:")
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:10]):
        print(f"{i+1:2d}. {feature}: {importance:.4f}")
    
    # Save model
    print("Saving model...")
    model_trainer.save_model()
    
    print("Model training completed successfully!")
    
    return model_trainer, feature_engineer, metrics

if __name__ == "__main__":
    model_trainer, feature_engineer, metrics = train_fraud_detection_model()

