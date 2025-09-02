"""
Generate sample transaction data for fraud detection model training and testing.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_data(n_samples=10000, fraud_rate=0.05):
    """
    Generate synthetic transaction data with realistic patterns.
    
    Args:
        n_samples (int): Number of transactions to generate
        fraud_rate (float): Proportion of fraudulent transactions
    
    Returns:
        pd.DataFrame: Generated transaction data
    """
    np.random.seed(42)
    random.seed(42)
    
    # Generate base features
    data = []
    
    for i in range(n_samples):
        # Determine if transaction is fraudulent
        is_fraud = np.random.random() < fraud_rate
        
        # Generate features with different patterns for fraud vs legitimate
        if is_fraud:
            # Fraudulent transactions tend to be:
            # - Higher amounts
            # - At unusual hours
            # - From newer accounts
            # - In certain categories
            amount = np.random.lognormal(mean=5.5, sigma=1.5)  # Higher amounts
            hour = np.random.choice([0, 1, 2, 3, 22, 23], p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])  # Unusual hours
            merchant_category = np.random.choice(['online', 'atm', 'gas_station'], p=[0.6, 0.3, 0.1])
            user_age = np.random.randint(18, 40)  # Younger users
            account_age_days = np.random.randint(1, 180)  # Newer accounts
        else:
            # Legitimate transactions
            amount = np.random.lognormal(mean=3.5, sigma=1.0)  # Normal amounts
            hour = np.random.randint(6, 22)  # Normal business hours
            merchant_category = np.random.choice(['grocery', 'restaurant', 'retail', 'online', 'gas_station'], 
                                               p=[0.3, 0.25, 0.2, 0.15, 0.1])
            user_age = np.random.randint(18, 80)
            account_age_days = np.random.randint(30, 3650)  # Established accounts
        
        # Common features
        day_of_week = np.random.randint(0, 7)
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Add some noise and edge cases
        amount = max(0.01, amount)  # Minimum transaction amount
        if amount > 10000:  # Cap very high amounts
            amount = np.random.uniform(1000, 10000)
        
        transaction = {
            'transaction_id': f'txn_{i:06d}',
            'amount': round(amount, 2),
            'merchant_category': merchant_category,
            'hour': hour,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'user_age': user_age,
            'account_age_days': account_age_days,
            'is_fraud': int(is_fraud),
            'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 365))
        }
        
        data.append(transaction)
    
    df = pd.DataFrame(data)
    
    # Add derived features
    df['amount_log'] = np.log1p(df['amount'])
    df['amount_bin'] = pd.cut(df['amount'], bins=[0, 10, 50, 100, 500, 1000, float('inf')], 
                             labels=['very_low', 'low', 'medium', 'high', 'very_high', 'extreme'])
    df['hour_bin'] = pd.cut(df['hour'], bins=[0, 6, 12, 18, 24], 
                           labels=['night', 'morning', 'afternoon', 'evening'])
    
    return df

def save_sample_data():
    """Generate and save sample data to CSV file."""
    print("Generating sample transaction data...")
    
    # Generate training data
    train_data = generate_sample_data(n_samples=8000, fraud_rate=0.05)
    
    # Generate test data
    test_data = generate_sample_data(n_samples=2000, fraud_rate=0.05)
	
	# Create directories if they don't exist
    os.makedirs('../data', exist_ok=True)
    os.makedirs('../models', exist_ok=True)
	
    
    # Save to CSV files
    train_data.to_csv('../data/train_data.csv', index=False)
    test_data.to_csv('../data/test_data.csv', index=False)
    
    # Create a smaller sample for quick testing
    sample_data = train_data.head(100)
    sample_data.to_csv('../data/sample_data.csv', index=False)
    
    print(f"Generated {len(train_data)} training samples")
    print(f"Generated {len(test_data)} test samples")
    print(f"Fraud rate in training data: {train_data['is_fraud'].mean():.3f}")
    print(f"Fraud rate in test data: {test_data['is_fraud'].mean():.3f}")
    
    return train_data, test_data

if __name__ == "__main__":
    train_data, test_data = save_sample_data()
    print("\nSample of generated data:")
    print(train_data.head())
    print("\nData types:")
    print(train_data.dtypes)
    print("\nFraud distribution:")
    print(train_data['is_fraud'].value_counts())

