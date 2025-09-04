"""
Performance monitoring and analytics for computer vision system.
Demonstrates system optimization and performance analysis.
"""

import time
import psutil
import cv2
import numpy as np
from typing import Dict, List, Any, Optional
from collections import deque
import threading
import json
from datetime import datetime

class PerformanceMetrics:
    """Container for performance metrics."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.fps_history = deque(maxlen=max_history)
        self.memory_usage = deque(maxlen=max_history)
        self.cpu_usage = deque(maxlen=max_history)
        self.detection_times = deque(maxlen=max_history)
        self.frame_processing_times = deque(maxlen=max_history)
        self.timestamps = deque(maxlen=max_history)
        
    def add_measurement(self, fps: float, memory_mb: float, cpu_percent: float,
                       detection_time: float, frame_time: float):
        """Add a new performance measurement."""
        self.fps_history.append(fps)
        self.memory_usage.append(memory_mb)
        self.cpu_usage.append(cpu_percent)
        self.detection_times.append(detection_time)
        self.frame_processing_times.append(frame_time)
        self.timestamps.append(time.time())
    
    def get_average_fps(self) -> float:
        """Get average FPS over recent history."""
        if not self.fps_history:
            return 0.0
        return sum(self.fps_history) / len(self.fps_history)
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage."""
        return max(self.memory_usage) if self.memory_usage else 0.0
    
    def get_average_cpu(self) -> float:
        """Get average CPU usage."""
        if not self.cpu_usage:
            return 0.0
        return sum(self.cpu_usage) / len(self.cpu_usage)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            'average_fps': self.get_average_fps(),
            'peak_memory_mb': self.get_peak_memory(),
            'average_cpu_percent': self.get_average_cpu(),
            'average_detection_time_ms': np.mean(self.detection_times) * 1000 if self.detection_times else 0,
            'average_frame_time_ms': np.mean(self.frame_processing_times) * 1000 if self.frame_processing_times else 0,
            'total_measurements': len(self.fps_history)
        }

class PerformanceMonitor:
    """Real-time performance monitoring for computer vision system."""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.metrics = PerformanceMetrics()
        self.is_monitoring = False
        self.monitor_thread = None
        self.start_time = None
        self.frame_count = 0
        self.last_fps_time = time.time()
        
    def start_monitoring(self):
        """Start performance monitoring."""
        self.is_monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Get system metrics
                memory_info = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent()
                memory_mb = memory_info.used / (1024 * 1024)
                
                # Calculate FPS
                current_time = time.time()
                fps = self.frame_count / max(current_time - self.last_fps_time, 0.001)
                
                # Add measurement (we'll update detection times separately)
                self.metrics.add_measurement(
                    fps=fps,
                    memory_mb=memory_mb,
                    cpu_percent=cpu_percent,
                    detection_time=0.0,  # Will be updated by system
                    frame_time=0.0       # Will be updated by system
                )
                
                # Reset frame count
                self.frame_count = 0
                self.last_fps_time = current_time
                
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"Performance monitoring error: {e}")
                time.sleep(self.update_interval)
    
    def record_frame_processed(self, detection_time: float, frame_time: float):
        """Record that a frame was processed."""
        self.frame_count += 1
        
        # Update the most recent measurement with actual times
        if self.metrics.detection_times:
            self.metrics.detection_times[-1] = detection_time
        if self.metrics.frame_processing_times:
            self.metrics.frame_processing_times[-1] = frame_time
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.metrics.get_performance_summary()
    
    def save_metrics_to_file(self, filename: str):
        """Save metrics to JSON file."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'performance_summary': self.get_current_metrics(),
            'detailed_metrics': {
                'fps_history': list(self.metrics.fps_history),
                'memory_usage': list(self.metrics.memory_usage),
                'cpu_usage': list(self.metrics.cpu_usage),
                'detection_times': list(self.metrics.detection_times),
                'frame_processing_times': list(self.metrics.frame_processing_times)
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Performance metrics saved to {filename}")

class SystemOptimizer:
    """System optimization recommendations based on performance metrics."""
    
    def __init__(self, performance_monitor: PerformanceMonitor):
        self.monitor = performance_monitor
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance and provide recommendations."""
        metrics = self.monitor.get_current_metrics()
        recommendations = []
        
        # FPS analysis
        if metrics['average_fps'] < 15:
            recommendations.append({
                'type': 'performance',
                'priority': 'high',
                'issue': 'Low FPS detected',
                'recommendation': 'Consider reducing input resolution or using faster detection models',
                'current_value': f"{metrics['average_fps']:.1f} FPS",
                'target_value': '> 20 FPS'
            })
        
        # Memory analysis
        if metrics['peak_memory_mb'] > 2000:
            recommendations.append({
                'type': 'memory',
                'priority': 'medium',
                'issue': 'High memory usage',
                'recommendation': 'Consider reducing batch size or model complexity',
                'current_value': f"{metrics['peak_memory_mb']:.1f} MB",
                'target_value': '< 1500 MB'
            })
        
        # CPU analysis
        if metrics['average_cpu_percent'] > 80:
            recommendations.append({
                'type': 'cpu',
                'priority': 'medium',
                'issue': 'High CPU usage',
                'recommendation': 'Consider using GPU acceleration or optimizing algorithms',
                'current_value': f"{metrics['average_cpu_percent']:.1f}%",
                'target_value': '< 70%'
            })
        
        # Detection time analysis
        if metrics['average_detection_time_ms'] > 100:
            recommendations.append({
                'type': 'detection',
                'priority': 'high',
                'issue': 'Slow detection processing',
                'recommendation': 'Consider using faster detection models or reducing input size',
                'current_value': f"{metrics['average_detection_time_ms']:.1f} ms",
                'target_value': '< 50 ms'
            })
        
        return {
            'metrics': metrics,
            'recommendations': recommendations,
            'overall_health': 'good' if len(recommendations) == 0 else 'needs_attention'
        }
    
    def get_optimization_suggestions(self) -> List[str]:
        """Get specific optimization suggestions."""
        suggestions = []
        metrics = self.monitor.get_current_metrics()
        
        if metrics['average_fps'] < 20:
            suggestions.extend([
                "Reduce input resolution from 640x480 to 320x240",
                "Use MobileNet instead of YOLO for person detection",
                "Implement frame skipping (process every 2nd frame)",
                "Use hardware acceleration (GPU)"
            ])
        
        if metrics['peak_memory_mb'] > 1500:
            suggestions.extend([
                "Reduce model batch size",
                "Clear detection history more frequently",
                "Use model quantization",
                "Implement memory pooling"
            ])
        
        return suggestions
