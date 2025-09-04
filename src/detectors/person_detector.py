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
        
    def initialize(self) -> bool:
        """Initialize the person detection model."""
        try:
            if self.model_type == 'YOLO':
                return self._initialize_yolo()
            elif self.model_type == 'MobileNet':
                return self._initialize_mobilenet()
            elif self.model_type == 'OpenPose':
                return self._initialize_openpose()
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
        except Exception as e:
            print(f"Failed to initialize PersonDetector: {e}")
            return False
    
    def _initialize_yolo(self) -> bool:
        """Initialize YOLO model."""
        weights_path = "models/yolov3.weights"
        config_path = "models/yolov3.cfg"
        names_path = "models/coco.names"
        
        if not all(os.path.exists(p) for p in [weights_path, config_path, names_path]):
            print("YOLO model files not found. Please run download_models.py")
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
        # Resize frame to input size
        frame = cv2.resize(frame, self.input_size)
        return frame
