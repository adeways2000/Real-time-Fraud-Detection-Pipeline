"""
Test script for fraud detection API endpoints.
"""

import requests
import json
import time
import random

# API base URL
BASE_URL = "http://localhost:5000"

def test_health_endpoint():
    """Test the health endpoint."""
    print("Testing health endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing health endpoint: {e}")
        return False

def test_predict_endpoint():
    """Test the prediction endpoint."""
    print("\\nTesting prediction endpoint...")
    
    # Test data - legitimate transaction
    legitimate_transaction = {
        "amount": 45.50,
        "merchant_category": "grocery",
        "hour": 14,
        "day_of_week": 2,
        "user_age": 35,
        "account_age_days": 365
    }
    
    # Test data - suspicious transaction
    suspicious_transaction = {
        "amount": 2500.00,
        "merchant_category": "online",
        "hour": 2,
        "day_of_week": 6,
        "user_age": 22,
        "account_age_days": 15
    }
    
    test_cases = [
        ("Legitimate Transaction", legitimate_transaction),
        ("Suspicious Transaction", suspicious_transaction)
    ]
    
    for test_name, transaction in test_cases:
        print(f"\\n{test_name}:")
        print(f"Input: {json.dumps(transaction, indent=2)}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                json=transaction,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"Status Code: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            
        except Exception as e:
            print(f"Error: {e}")

def test_batch_predict_endpoint():
    """Test the batch prediction endpoint."""
    print("\\nTesting batch prediction endpoint...")
    
    transactions = [
        {
            "amount": 25.00,
            "merchant_category": "restaurant",
            "hour": 12,
            "day_of_week": 1,
            "user_age": 28,
            "account_age_days": 500
        },
        {
            "amount": 1500.00,
            "merchant_category": "atm",
            "hour": 23,
            "day_of_week": 6,
            "user_age": 19,
            "account_age_days": 5
        },
        {
            "amount": 75.25,
            "merchant_category": "gas_station",
            "hour": 8,
            "day_of_week": 3,
            "user_age": 45,
            "account_age_days": 1200
        }
    ]
    
    batch_data = {"transactions": transactions}
    
    try:
        response = requests.post(
            f"{BASE_URL}/batch_predict",
            json=batch_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
    except Exception as e:
        print(f"Error: {e}")

def test_metrics_endpoint():
    """Test the metrics endpoint."""
    print("\\nTesting metrics endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/metrics")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
    except Exception as e:
        print(f"Error: {e}")

def test_error_handling():
    """Test error handling."""
    print("\\nTesting error handling...")
    
    # Test missing fields
    print("Testing missing fields:")
    incomplete_data = {"amount": 100.0}
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=incomplete_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Test invalid data types
    print("\\nTesting invalid data types:")
    invalid_data = {
        "amount": "not_a_number",
        "merchant_category": "grocery",
        "hour": 14,
        "day_of_week": 2,
        "user_age": 35,
        "account_age_days": 365
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=invalid_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
    except Exception as e:
        print(f"Error: {e}")

def performance_test():
    """Test API performance."""
    print("\\nRunning performance test...")
    
    test_transaction = {
        "amount": 100.0,
        "merchant_category": "retail",
        "hour": 15,
        "day_of_week": 3,
        "user_age": 30,
        "account_age_days": 200
    }
    
    num_requests = 50
    response_times = []
    
    print(f"Making {num_requests} requests...")
    
    for i in range(num_requests):
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                json=test_transaction,
                headers={"Content-Type": "application/json"}
            )
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # ms
            response_times.append(response_time)
            
            if i % 10 == 0:
                print(f"Completed {i+1}/{num_requests} requests")
                
        except Exception as e:
            print(f"Request {i+1} failed: {e}")
    
    if response_times:
        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        
        print(f"\\nPerformance Results:")
        print(f"Average response time: {avg_time:.2f} ms")
        print(f"Min response time: {min_time:.2f} ms")
        print(f"Max response time: {max_time:.2f} ms")
        print(f"Successful requests: {len(response_times)}/{num_requests}")

def main():
    """Run all tests."""
    print("Starting API tests...")
    print("=" * 50)
    
    # Wait for API to be ready
    print("Waiting for API to be ready...")
    for i in range(10):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                print("API is ready!")
                break
        except:
            pass
        
        time.sleep(2)
        print(f"Attempt {i+1}/10...")
    else:
        print("API is not responding. Please start the API first.")
        return
    
    # Run tests
    test_health_endpoint()
    test_predict_endpoint()
    test_batch_predict_endpoint()
    test_metrics_endpoint()
    test_error_handling()
    performance_test()
    
    print("\\n" + "=" * 50)
    print("API tests completed!")

if __name__ == "__main__":
    main()

