"""
Advanced motion detection using multiple algorithms.
Demonstrates computer vision fundamentals and algorithm comparison.
"""

import cv2
import numpy as np
import time
from typing import List, Tuple
from .base_detector import BaseDetector, DetectionResult

class MotionDetector(BaseDetector):
    """Advanced motion detection with multiple algorithms."""
    
    def __init__(self, confidence_threshold: float = 0.5, 
                 min_contour_area: int = 1000,
                 algorithm: str = 'MOG2'):
        super().__init__("MotionDetector", confidence_threshold)
        self.min_contour_area = min_contour_area
        self.algorithm = algorithm
        self.bg_subtractor = None
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
    def initialize(self) -> bool:
        """Initialize the background subtractor."""
        try:
            if self.algorithm == 'MOG2':
                self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                    detectShadows=True, varThreshold=50, history=500
                )
            elif self.algorithm == 'KNN':
                self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
                    detectShadows=True, dist2Threshold=400, history=500
                )
            elif self.algorithm == 'GMG':
                self.bg_subtractor = cv2.bgsegm.createBackgroundSubtractorGMG()
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")
            
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"Failed to initialize MotionDetector: {e}")
            return False
    
    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """Detect motion in the frame."""
        if not self.is_initialized:
            return []
        
        start_time = time.time()
        
        # Preprocess frame
        frame = self.preprocess_frame(frame)
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Apply morphological operations
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                confidence = min(area / (self.min_contour_area * 2), 1.0)
                
                detection = DetectionResult(
                    bbox=(x, y, w, h),
                    confidence=confidence,
                    class_name="motion",
                    timestamp=time.time(),
                    method=f"motion_{self.algorithm}"
                )
                detections.append(detection)
        
        # Update metrics
        self.detection_count += 1
        self.total_processing_time += time.time() - start_time
        
        return self.postprocess_detections(detections)
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for better motion detection."""
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        
        return frame
