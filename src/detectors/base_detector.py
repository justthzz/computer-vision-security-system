"""
Base detector class for all computer vision detection methods.
This demonstrates proper OOP design and abstraction in computer vision.
"""

from abc import ABC, abstractmethod
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
import time

class DetectionResult:
    """Data class to hold detection results with metadata."""
    
    def __init__(self, bbox: Tuple[int, int, int, int], confidence: float, 
                 class_name: str, timestamp: float, method: str):
        self.bbox = bbox  # (x, y, w, h)
        self.confidence = confidence
        self.class_name = class_name
        self.timestamp = timestamp
        self.method = method
        self.id = None  # For tracking
    
    def __repr__(self):
        return f"DetectionResult({self.class_name}, conf={self.confidence:.2f}, method={self.method})"

class BaseDetector(ABC):
    """Abstract base class for all detection methods."""
    
    def __init__(self, name: str, confidence_threshold: float = 0.5):
        self.name = name
        self.confidence_threshold = confidence_threshold
        self.is_initialized = False
        self.detection_count = 0
        self.total_processing_time = 0.0
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the detector. Returns True if successful."""
        pass
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """Detect objects in the frame. Returns list of DetectionResult objects."""
        pass
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for detection. Override if needed."""
        return frame
    
    def postprocess_detections(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """Postprocess detection results. Override if needed."""
        return [d for d in detections if d.confidence >= self.confidence_threshold]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this detector."""
        avg_time = self.total_processing_time / max(self.detection_count, 1)
        return {
            'name': self.name,
            'detection_count': self.detection_count,
            'total_processing_time': self.total_processing_time,
            'average_processing_time': avg_time,
            'fps': 1.0 / avg_time if avg_time > 0 else 0
        }
    
    def reset_metrics(self):
        """Reset performance metrics."""
        self.detection_count = 0
        self.total_processing_time = 0.0
