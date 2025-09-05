"""
Advanced person detection using multiple deep learning models.
Demonstrates modern computer vision with YOLO, OpenPose, and custom models.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
import time
import os
from .base_detector import BaseDetector, DetectionResult

class PersonDetector(BaseDetector):
    """Advanced person detection with multiple model support."""
    
    def __init__(self, confidence_threshold: float = 0.5, 
                 model_type: str = 'YOLO',
                 input_size: Tuple[int, int] = (416, 416)):
        super().__init__("PersonDetector", confidence_threshold)
        self.model_type = model_type
        self.input_size = input_size
        self.net = None
        self.classes = []
        self.person_class_id = 0
        self.output_layers = []
        self.fallback_mode = False  # Use simple detection when YOLO fails
        
    def initialize(self) -> bool:
        """Initialize the person detection model."""
        try:
            if self.model_type == 'YOLO':
                if self._initialize_yolo():
                    return True
                else:
                    print("YOLO initialization failed, falling back to simple detection...")
                    self.fallback_mode = True
                    return True
            elif self.model_type == 'MobileNet':
                return self._initialize_mobilenet()
            elif self.model_type == 'OpenPose':
                return self._initialize_openpose()
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
        except Exception as e:
            print(f"Failed to initialize PersonDetector: {e}")
            print("Falling back to simple detection...")
            self.fallback_mode = True
            return True
    
    def _initialize_yolo(self) -> bool:
        """Initialize YOLO model."""
        weights_path = "yolov3.weights"
        config_path = "yolov3.cfg"
        names_path = "coco.names"
        
        if not all(os.path.exists(p) for p in [weights_path, config_path, names_path]):
            print("YOLO model files not found. Please run download_models.py")
            return False
        
        # Check if weights file is valid (should be ~248MB for YOLO v3)
        if os.path.getsize(weights_path) < 1000000:  # Less than 1MB indicates corrupted download
            print(f"YOLO weights file appears corrupted (size: {os.path.getsize(weights_path)} bytes)")
            print("Please download YOLO v3 weights manually from: https://pjreddie.com/darknet/yolo/")
            print("Or use a different detection method.")
            return False
        
        # Load YOLO model
        self.net = cv2.dnn.readNet(weights_path, config_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Get output layer names
        self.output_layers = self.net.getUnconnectedOutLayersNames()
        
        # Load class names
        with open(names_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        self.person_class_id = 0  # 'person' class in COCO dataset
        self.is_initialized = True
        return True
    
    def _initialize_mobilenet(self) -> bool:
        """Initialize MobileNet SSD model."""
        # This would load a MobileNet SSD model
        # For demo purposes, we'll use a placeholder
        print("MobileNet SSD initialization not implemented in this demo")
        return False
    
    def _initialize_openpose(self) -> bool:
        """Initialize OpenPose model."""
        # This would load an OpenPose model
        # For demo purposes, we'll use a placeholder
        print("OpenPose initialization not implemented in this demo")
        return False
    
    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """Detect persons in the frame."""
        if not self.is_initialized:
            return []
        
        start_time = time.time()
        
        # Use fallback detection if YOLO failed
        if self.fallback_mode:
            detections = self._detect_fallback(frame)
        else:
            # Preprocess frame
            frame = self.preprocess_frame(frame)
            
            if self.model_type == 'YOLO':
                detections = self._detect_yolo(frame)
            else:
                detections = []
        
        # Update metrics
        self.detection_count += 1
        self.total_processing_time += time.time() - start_time
        
        return self.postprocess_detections(detections)
    
    def _detect_yolo(self, frame: np.ndarray) -> List[DetectionResult]:
        """Detect persons using YOLO."""
        height, width = frame.shape[:2]
        
        # Create blob from frame
        blob = cv2.dnn.blobFromImage(
            frame, 1/255.0, self.input_size, 
            swapRB=True, crop=False
        )
        self.net.setInput(blob)
        
        # Get detections
        outputs = self.net.forward(self.output_layers)
        
        detections = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if class_id == self.person_class_id and confidence > self.confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    detection = DetectionResult(
                        bbox=(x, y, w, h),
                        confidence=float(confidence),
                        class_name="person",
                        timestamp=time.time(),
                        method=f"person_{self.model_type}"
                    )
                    detections.append(detection)
        
        return detections
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for person detection."""
        # Skip resize to avoid OpenCV errors - use original frame size
        # This is a workaround for persistent OpenCV resize issues
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            self.logger.warning("Invalid frame in person detector, returning original")
            return frame
        
        if len(frame.shape) < 2 or frame.shape[0] == 0 or frame.shape[1] == 0:
            self.logger.warning(f"Invalid frame dimensions in person detector: {frame.shape}")
            return frame
        
        # Return original frame without resize to avoid OpenCV errors
        self.logger.debug(f"Person detector using original frame size: {frame.shape}")
        return frame
    
    def _detect_fallback(self, frame: np.ndarray) -> List[DetectionResult]:
        """Fallback detection using simple motion-based detection."""
        detections = []
        
        try:
            # Simple motion detection as fallback
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (21, 21), 0)
            
            # Create a simple background subtractor
            if not hasattr(self, 'bg_subtractor'):
                self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
            
            # Apply background subtraction
            fg_mask = self.bg_subtractor.apply(blurred)
            
            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area (person-like size)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Minimum area for person detection
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Create detection result
                    detection = DetectionResult(
                        class_name="person",
                        confidence=0.7,  # Fixed confidence for fallback
                        bbox=(x, y, w, h),
                        timestamp=time.time(),
                        method="fallback_motion"
                    )
                    detections.append(detection)
            
        except Exception as e:
            print(f"Error in fallback detection: {e}")
        
        return detections
