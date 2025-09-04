"""
Data visualization and analytics dashboard for computer vision system.
Demonstrates data analysis and visualization skills.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import tkinter as tk
from tkinter import ttk
import seaborn as sns
from collections import deque
import time
from datetime import datetime, timedelta

class RealTimeVisualizer:
    """Real-time data visualization for computer vision metrics."""
    
    def __init__(self, max_points: int = 100):
        self.max_points = max_points
        self.fps_data = deque(maxlen=max_points)
        self.memory_data = deque(maxlen=max_points)
        self.cpu_data = deque(maxlen=max_points)
        self.detection_data = deque(maxlen=max_points)
        self.timestamps = deque(maxlen=max_points)
        
        # Setup matplotlib style
        plt.style.use('seaborn-v0_8')
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Computer Vision System Performance Dashboard', fontsize=16)
        
        # Initialize plots
        self._setup_plots()
        
    def _setup_plots(self):
        """Setup the subplots."""
        # FPS plot
        self.axes[0, 0].set_title('Frames Per Second')
        self.axes[0, 0].set_ylabel('FPS')
        self.axes[0, 0].grid(True, alpha=0.3)
        self.fps_line, = self.axes[0, 0].plot([], [], 'b-', linewidth=2)
        
        # Memory usage plot
        self.axes[0, 1].set_title('Memory Usage')
        self.axes[0, 1].set_ylabel('Memory (MB)')
        self.axes[0, 1].grid(True, alpha=0.3)
        self.memory_line, = self.axes[0, 1].plot([], [], 'r-', linewidth=2)
        
        # CPU usage plot
        self.axes[1, 0].set_title('CPU Usage')
        self.axes[1, 0].set_ylabel('CPU (%)')
        self.axes[1, 0].set_xlabel('Time')
        self.axes[1, 0].grid(True, alpha=0.3)
        self.cpu_line, = self.axes[1, 0].plot([], [], 'g-', linewidth=2)
        
        # Detection count plot
        self.axes[1, 1].set_title('Detections per Second')
        self.axes[1, 1].set_ylabel('Detections/sec')
        self.axes[1, 1].set_xlabel('Time')
        self.axes[1, 1].grid(True, alpha=0.3)
        self.detection_line, = self.axes[1, 1].plot([], [], 'm-', linewidth=2)
        
        plt.tight_layout()
    
    def update_data(self, fps: float, memory_mb: float, cpu_percent: float, 
                   detections_per_sec: float):
        """Update the data with new measurements."""
        current_time = time.time()
        
        self.fps_data.append(fps)
        self.memory_data.append(memory_mb)
        self.cpu_data.append(cpu_percent)
        self.detection_data.append(detections_per_sec)
        self.timestamps.append(current_time)
        
        self._update_plots()
    
    def _update_plots(self):
        """Update all plots with current data."""
        if not self.timestamps:
            return
        
        # Convert timestamps to relative time
        start_time = self.timestamps[0]
        relative_times = [(t - start_time) for t in self.timestamps]
        
        # Update FPS plot
        self.fps_line.set_data(relative_times, list(self.fps_data))
        self.axes[0, 0].relim()
        self.axes[0, 0].autoscale_view()
        
        # Update memory plot
        self.memory_line.set_data(relative_times, list(self.memory_data))
        self.axes[0, 1].relim()
        self.axes[0, 1].autoscale_view()
        
        # Update CPU plot
        self.cpu_line.set_data(relative_times, list(self.cpu_data))
        self.axes[1, 0].relim()
        self.axes[1, 0].autoscale_view()
        
        # Update detection plot
        self.detection_line.set_data(relative_times, list(self.detection_data))
        self.axes[1, 1].relim()
        self.axes[1, 1].autoscale_view()
    
    def show(self):
        """Display the visualization."""
        plt.show()

class DetectionAnalytics:
    """Analytics for detection patterns and performance."""
    
    def __init__(self):
        self.detection_history = []
        self.performance_history = []
        
    def add_detection(self, detection_type: str, confidence: float, 
                     bbox: tuple, timestamp: float, method: str):
        """Add a detection event to history."""
        self.detection_history.append({
            'type': detection_type,
            'confidence': confidence,
            'bbox': bbox,
            'timestamp': timestamp,
            'method': method,
            'datetime': datetime.fromtimestamp(timestamp)
        })
    
    def add_performance_metrics(self, fps: float, memory_mb: float, 
                               cpu_percent: float, detection_time: float):
        """Add performance metrics to history."""
        self.performance_history.append({
            'fps': fps,
            'memory_mb': memory_mb,
            'cpu_percent': cpu_percent,
            'detection_time': detection_time,
            'timestamp': time.time(),
            'datetime': datetime.now()
        })
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """Get summary statistics for detections."""
        if not self.detection_history:
            return {}
        
        df = pd.DataFrame(self.detection_history)
        
        summary = {
            'total_detections': len(df),
            'detection_types': df['type'].value_counts().to_dict(),
            'detection_methods': df['method'].value_counts().to_dict(),
            'average_confidence': df['confidence'].mean(),
            'confidence_std': df['confidence'].std(),
            'detections_per_hour': len(df) / max((df['timestamp'].max() - df['timestamp'].min()) / 3600, 1),
            'time_range': {
                'start': df['datetime'].min().isoformat(),
                'end': df['datetime'].max().isoformat()
            }
        }
        
        return summary
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary statistics for performance."""
        if not self.performance_history:
            return {}
        
        df = pd.DataFrame(self.performance_history)
        
        summary = {
            'average_fps': df['fps'].mean(),
            'min_fps': df['fps'].min(),
            'max_fps': df['fps'].max(),
            'average_memory_mb': df['memory_mb'].mean(),
            'peak_memory_mb': df['memory_mb'].max(),
            'average_cpu_percent': df['cpu_percent'].mean(),
            'peak_cpu_percent': df['cpu_percent'].max(),
            'average_detection_time': df['detection_time'].mean(),
            'total_runtime_hours': (df['timestamp'].max() - df['timestamp'].min()) / 3600
        }
        
        return summary
    
    def create_detection_heatmap(self, save_path: Optional[str] = None):
        """Create a heatmap of detection patterns over time."""
        if not self.detection_history:
            return None
        
        df = pd.DataFrame(self.detection_history)
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.day_name()
        
        # Create pivot table for heatmap
        heatmap_data = df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd')
        plt.title('Detection Patterns by Day and Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Day of Week')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def create_performance_trends(self, save_path: Optional[str] = None):
        """Create performance trend analysis."""
        if not self.performance_history:
            return None
        
        df = pd.DataFrame(self.performance_history)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Performance Trends Analysis', fontsize=16)
        
        # FPS over time
        axes[0, 0].plot(df['datetime'], df['fps'], alpha=0.7)
        axes[0, 0].set_title('FPS Over Time')
        axes[0, 0].set_ylabel('FPS')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Memory usage over time
        axes[0, 1].plot(df['datetime'], df['memory_mb'], color='red', alpha=0.7)
        axes[0, 1].set_title('Memory Usage Over Time')
        axes[0, 1].set_ylabel('Memory (MB)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # CPU usage over time
        axes[1, 0].plot(df['datetime'], df['cpu_percent'], color='green', alpha=0.7)
        axes[1, 0].set_title('CPU Usage Over Time')
        axes[1, 0].set_ylabel('CPU (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Detection time over time
        axes[1, 1].plot(df['datetime'], df['detection_time'] * 1000, color='purple', alpha=0.7)
        axes[1, 1].set_title('Detection Time Over Time')
        axes[1, 1].set_ylabel('Detection Time (ms)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

class TkinterDashboard:
    """Tkinter-based real-time dashboard."""
    
    def __init__(self, analytics: DetectionAnalytics):
        self.analytics = analytics
        self.root = tk.Tk()
        self.root.title("Computer Vision System Dashboard")
        self.root.geometry("800x600")
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface."""
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True)
        
        # Performance tab
        self.perf_frame = ttk.Frame(notebook)
        notebook.add(self.perf_frame, text="Performance")
        self.setup_performance_tab()
        
        # Detections tab
        self.det_frame = ttk.Frame(notebook)
        notebook.add(self.det_frame, text="Detections")
        self.setup_detections_tab()
        
        # Analytics tab
        self.anal_frame = ttk.Frame(notebook)
        notebook.add(self.anal_frame, text="Analytics")
        self.setup_analytics_tab()
    
    def setup_performance_tab(self):
        """Setup performance monitoring tab."""
        # Performance metrics labels
        ttk.Label(self.perf_frame, text="Performance Metrics", font=('Arial', 14, 'bold')).pack(pady=10)
        
        self.fps_label = ttk.Label(self.perf_frame, text="FPS: --")
        self.fps_label.pack()
        
        self.memory_label = ttk.Label(self.perf_frame, text="Memory: -- MB")
        self.memory_label.pack()
        
        self.cpu_label = ttk.Label(self.perf_frame, text="CPU: -- %")
        self.cpu_label.pack()
        
        self.detection_time_label = ttk.Label(self.perf_frame, text="Detection Time: -- ms")
        self.detection_time_label.pack()
        
        # Update button
        ttk.Button(self.perf_frame, text="Update Metrics", 
                  command=self.update_performance_display).pack(pady=10)
    
    def setup_detections_tab(self):
        """Setup detections monitoring tab."""
        ttk.Label(self.det_frame, text="Detection Summary", font=('Arial', 14, 'bold')).pack(pady=10)
        
        self.detection_text = tk.Text(self.det_frame, height=20, width=80)
        self.detection_text.pack(pady=10)
        
        ttk.Button(self.det_frame, text="Update Detections", 
                  command=self.update_detections_display).pack()
    
    def setup_analytics_tab(self):
        """Setup analytics tab."""
        ttk.Label(self.anal_frame, text="Analytics", font=('Arial', 14, 'bold')).pack(pady=10)
        
        ttk.Button(self.anal_frame, text="Generate Detection Heatmap", 
                  command=self.generate_heatmap).pack(pady=5)
        
        ttk.Button(self.anal_frame, text="Generate Performance Trends", 
                  command=self.generate_trends).pack(pady=5)
        
        ttk.Button(self.anal_frame, text="Export Analytics Report", 
                  command=self.export_report).pack(pady=5)
    
    def update_performance_display(self):
        """Update performance metrics display."""
        summary = self.analytics.get_performance_summary()
        
        if summary:
            self.fps_label.config(text=f"FPS: {summary.get('average_fps', 0):.1f}")
            self.memory_label.config(text=f"Memory: {summary.get('average_memory_mb', 0):.1f} MB")
            self.cpu_label.config(text=f"CPU: {summary.get('average_cpu_percent', 0):.1f} %")
            self.detection_time_label.config(text=f"Detection Time: {summary.get('average_detection_time', 0)*1000:.1f} ms")
    
    def update_detections_display(self):
        """Update detections display."""
        summary = self.analytics.get_detection_summary()
        
        self.detection_text.delete(1.0, tk.END)
        if summary:
            self.detection_text.insert(tk.END, f"Total Detections: {summary.get('total_detections', 0)}\n")
            self.detection_text.insert(tk.END, f"Average Confidence: {summary.get('average_confidence', 0):.2f}\n")
            self.detection_text.insert(tk.END, f"Detections per Hour: {summary.get('detections_per_hour', 0):.1f}\n\n")
            
            self.detection_text.insert(tk.END, "Detection Types:\n")
            for det_type, count in summary.get('detection_types', {}).items():
                self.detection_text.insert(tk.END, f"  {det_type}: {count}\n")
            
            self.detection_text.insert(tk.END, "\nDetection Methods:\n")
            for method, count in summary.get('detection_methods', {}).items():
                self.detection_text.insert(tk.END, f"  {method}: {count}\n")
    
    def generate_heatmap(self):
        """Generate detection heatmap."""
        fig = self.analytics.create_detection_heatmap()
        if fig:
            fig.show()
    
    def generate_trends(self):
        """Generate performance trends."""
        fig = self.analytics.create_performance_trends()
        if fig:
            fig.show()
    
    def export_report(self):
        """Export analytics report."""
        # This would generate a comprehensive report
        print("Exporting analytics report...")
    
    def run(self):
        """Run the dashboard."""
        self.root.mainloop()
