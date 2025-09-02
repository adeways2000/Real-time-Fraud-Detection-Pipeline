"""
System monitoring module for fraud detection API.
Tracks performance metrics, model accuracy, and system health.
"""

import time
import threading
from collections import deque, defaultdict
from datetime import datetime, timedelta
import logging
import psutil
import json

class SystemMonitor:
    """
    System monitoring class for tracking API performance and model metrics.
    """
    
    def __init__(self, max_history=1000):
        """
        Initialize system monitor.
        
        Args:
            max_history (int): Maximum number of records to keep in memory
        """
        self.max_history = max_history
        self.start_time = time.time()
        
        # Metrics storage
        self.prediction_history = deque(maxlen=max_history)
        self.performance_metrics = defaultdict(list)
        self.error_count = 0
        self.total_predictions = 0
        
        # Threading lock for thread-safe operations
        self.lock = threading.Lock()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # System metrics
        self.system_metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'disk_usage': []
        }
    
    def record_prediction(self, prediction_result):
        """
        Record a prediction result for monitoring.
        
        Args:
            prediction_result (dict): Result from model prediction
        """
        with self.lock:
            timestamp = datetime.now()
            
            # Store prediction record
            record = {
                'timestamp': timestamp.isoformat(),
                'fraud_probability': prediction_result.get('fraud_probability', 0),
                'prediction': prediction_result.get('prediction', 'unknown'),
                'confidence': prediction_result.get('confidence', 0),
                'processing_time_ms': prediction_result.get('processing_time_ms', 0)
            }
            
            self.prediction_history.append(record)
            self.total_predictions += 1
            
            # Update performance metrics
            self.performance_metrics['processing_times'].append(
                prediction_result.get('processing_time_ms', 0)
            )
            
            # Keep only recent performance metrics
            if len(self.performance_metrics['processing_times']) > self.max_history:
                self.performance_metrics['processing_times'] = \
                    self.performance_metrics['processing_times'][-self.max_history:]
    
    def record_error(self, error_type, error_message):
        """
        Record an error for monitoring.
        
        Args:
            error_type (str): Type of error
            error_message (str): Error message
        """
        with self.lock:
            self.error_count += 1
            
            error_record = {
                'timestamp': datetime.now().isoformat(),
                'error_type': error_type,
                'message': error_message
            }
            
            # Store in performance metrics
            if 'errors' not in self.performance_metrics:
                self.performance_metrics['errors'] = []
            
            self.performance_metrics['errors'].append(error_record)
            
            # Keep only recent errors
            if len(self.performance_metrics['errors']) > 100:
                self.performance_metrics['errors'] = \
                    self.performance_metrics['errors'][-100:]
    
    def get_uptime(self):
        """
        Get system uptime in seconds.
        
        Returns:
            float: Uptime in seconds
        """
        return time.time() - self.start_time
    
    def get_prediction_stats(self):
        """
        Get prediction statistics.
        
        Returns:
            dict: Prediction statistics
        """
        with self.lock:
            if not self.prediction_history:
                return {
                    'total_predictions': 0,
                    'fraud_rate': 0,
                    'avg_confidence': 0,
                    'avg_processing_time_ms': 0
                }
            
            # Calculate statistics
            fraud_predictions = sum(1 for p in self.prediction_history 
                                  if p['prediction'] == 'fraud')
            
            total_confidence = sum(p['confidence'] for p in self.prediction_history)
            total_processing_time = sum(p['processing_time_ms'] for p in self.prediction_history)
            
            return {
                'total_predictions': len(self.prediction_history),
                'fraud_rate': fraud_predictions / len(self.prediction_history),
                'avg_confidence': total_confidence / len(self.prediction_history),
                'avg_processing_time_ms': total_processing_time / len(self.prediction_history),
                'min_processing_time_ms': min(p['processing_time_ms'] for p in self.prediction_history),
                'max_processing_time_ms': max(p['processing_time_ms'] for p in self.prediction_history)
            }
    
    def get_performance_metrics(self):
        """
        Get performance metrics.
        
        Returns:
            dict: Performance metrics
        """
        with self.lock:
            processing_times = self.performance_metrics.get('processing_times', [])
            
            if not processing_times:
                return {
                    'avg_latency_ms': 0,
                    'p95_latency_ms': 0,
                    'p99_latency_ms': 0,
                    'throughput_per_second': 0
                }
            
            # Calculate percentiles
            sorted_times = sorted(processing_times)
            n = len(sorted_times)
            
            p95_index = int(0.95 * n)
            p99_index = int(0.99 * n)
            
            # Calculate throughput (predictions per second)
            if len(self.prediction_history) > 1:
                time_span = (datetime.fromisoformat(self.prediction_history[-1]['timestamp']) - 
                           datetime.fromisoformat(self.prediction_history[0]['timestamp'])).total_seconds()
                throughput = len(self.prediction_history) / max(time_span, 1)
            else:
                throughput = 0
            
            return {
                'avg_latency_ms': sum(processing_times) / len(processing_times),
                'p95_latency_ms': sorted_times[p95_index] if p95_index < n else sorted_times[-1],
                'p99_latency_ms': sorted_times[p99_index] if p99_index < n else sorted_times[-1],
                'throughput_per_second': throughput,
                'total_requests': len(processing_times)
            }
    
    def get_system_metrics(self):
        """
        Get current system metrics.
        
        Returns:
            dict: System metrics
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_usage_percent': cpu_percent,
                'memory_usage_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_usage_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3)
            }
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {str(e)}")
            return {
                'cpu_usage_percent': 0,
                'memory_usage_percent': 0,
                'memory_available_gb': 0,
                'disk_usage_percent': 0,
                'disk_free_gb': 0
            }
    
    def get_model_metrics(self):
        """
        Get model-specific metrics.
        
        Returns:
            dict: Model metrics
        """
        with self.lock:
            if not self.prediction_history:
                return {
                    'predictions_last_hour': 0,
                    'fraud_rate_last_hour': 0,
                    'avg_confidence_last_hour': 0
                }
            
            # Filter predictions from last hour
            one_hour_ago = datetime.now() - timedelta(hours=1)
            recent_predictions = [
                p for p in self.prediction_history 
                if datetime.fromisoformat(p['timestamp']) > one_hour_ago
            ]
            
            if not recent_predictions:
                return {
                    'predictions_last_hour': 0,
                    'fraud_rate_last_hour': 0,
                    'avg_confidence_last_hour': 0
                }
            
            fraud_count = sum(1 for p in recent_predictions if p['prediction'] == 'fraud')
            total_confidence = sum(p['confidence'] for p in recent_predictions)
            
            return {
                'predictions_last_hour': len(recent_predictions),
                'fraud_rate_last_hour': fraud_count / len(recent_predictions),
                'avg_confidence_last_hour': total_confidence / len(recent_predictions)
            }
    
    def get_metrics(self):
        """
        Get comprehensive metrics.
        
        Returns:
            dict: All metrics
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': self.get_uptime(),
            'prediction_stats': self.get_prediction_stats(),
            'performance_metrics': self.get_performance_metrics(),
            'system_metrics': self.get_system_metrics(),
            'model_metrics': self.get_model_metrics(),
            'error_count': self.error_count,
            'total_predictions_lifetime': self.total_predictions
        }
    
    def start_monitoring(self):
        """
        Start background monitoring thread.
        """
        self.logger.info("Starting system monitoring...")
        
        while True:
            try:
                # Collect system metrics periodically
                system_metrics = self.get_system_metrics()
                
                with self.lock:
                    # Store system metrics
                    for key, value in system_metrics.items():
                        if key not in self.system_metrics:
                            self.system_metrics[key] = []
                        
                        self.system_metrics[key].append({
                            'timestamp': datetime.now().isoformat(),
                            'value': value
                        })
                        
                        # Keep only recent metrics
                        if len(self.system_metrics[key]) > 100:
                            self.system_metrics[key] = self.system_metrics[key][-100:]
                
                # Sleep for 60 seconds before next collection
                time.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring thread: {str(e)}")
                time.sleep(60)
    
    def export_metrics(self, filepath):
        """
        Export metrics to JSON file.
        
        Args:
            filepath (str): Path to save metrics
        """
        try:
            metrics = self.get_metrics()
            
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            self.logger.info(f"Metrics exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {str(e)}")
    
    def get_health_status(self):
        """
        Get overall health status.
        
        Returns:
            dict: Health status
        """
        system_metrics = self.get_system_metrics()
        performance_metrics = self.get_performance_metrics()
        
        # Determine health based on thresholds
        is_healthy = True
        issues = []
        
        # Check CPU usage
        if system_metrics['cpu_usage_percent'] > 80:
            is_healthy = False
            issues.append("High CPU usage")
        
        # Check memory usage
        if system_metrics['memory_usage_percent'] > 85:
            is_healthy = False
            issues.append("High memory usage")
        
        # Check disk usage
        if system_metrics['disk_usage_percent'] > 90:
            is_healthy = False
            issues.append("High disk usage")
        
        # Check latency
        if performance_metrics['avg_latency_ms'] > 200:
            is_healthy = False
            issues.append("High latency")
        
        return {
            'healthy': is_healthy,
            'issues': issues,
            'timestamp': datetime.now().isoformat()
        }

