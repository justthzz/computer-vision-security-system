"""
Advanced object tracking using multiple algorithms.
Demonstrates computer vision tracking techniques and Kalman filtering.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
from .base_detector import DetectionResult

class TrackedObject:
    """Represents a tracked object with state and history."""
    
    def __init__(self, detection: DetectionResult, tracker_id: int):
        self.id = tracker_id
        self.detection = detection
        self.tracker = None
        self.track_history = []
        self.last_seen = time.time()
        self.is_active = True
        self.confidence_history = []
        
    def update(self, detection: DetectionResult):
        """Update the tracked object with new detection."""
        self.detection = detection
        self.track_history.append((detection.bbox, time.time()))
        self.confidence_history.append(detection.confidence)
        self.last_seen = time.time()
        
        # Keep only recent history
        if len(self.track_history) > 50:
            self.track_history = self.track_history[-50:]
        if len(self.confidence_history) > 50:
            self.confidence_history = self.confidence_history[-50:]
    
    def get_average_confidence(self) -> float:
        """Get average confidence over recent detections."""
        if not self.confidence_history:
            return 0.0
        return sum(self.confidence_history[-10:]) / len(self.confidence_history[-10:])
    
    def get_velocity(self) -> Tuple[float, float]:
        """Calculate velocity based on recent positions."""
        if len(self.track_history) < 2:
            return (0.0, 0.0)
        
        recent = self.track_history[-2:]
        (x1, y1, w1, h1), t1 = recent[0]
        (x2, y2, w2, h2), t2 = recent[1]
        
        dt = t2 - t1
        if dt == 0:
            return (0.0, 0.0)
        
        vx = (x2 - x1) / dt
        vy = (y2 - y1) / dt
        
        return (vx, vy)

class ObjectTracker:
    """Advanced object tracking with multiple algorithms."""
    
    def __init__(self, algorithm: str = 'CSRT', max_disappeared: float = 30.0):
        self.algorithm = algorithm
        self.max_disappeared = max_disappeared
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.next_id = 0
        self.trackers: Dict[int, cv2.Tracker] = {}
        
    def create_tracker(self) -> cv2.Tracker:
        """Create a new tracker instance."""
        if self.algorithm == 'CSRT':
            return cv2.TrackerCSRT_create()
        elif self.algorithm == 'KCF':
            return cv2.TrackerKCF_create()
        elif self.algorithm == 'MOSSE':
            return cv2.TrackerMOSSE_create()
        elif self.algorithm == 'MIL':
            return cv2.TrackerMIL_create()
        else:
            return cv2.TrackerCSRT_create()
    
    def add_object(self, detection: DetectionResult) -> int:
        """Add a new object to track."""
        tracker_id = self.next_id
        self.next_id += 1
        
        # Create tracked object
        tracked_obj = TrackedObject(detection, tracker_id)
        self.tracked_objects[tracker_id] = tracked_obj
        
        # Create tracker
        tracker = self.create_tracker()
        self.trackers[tracker_id] = tracker
        
        return tracker_id
    
    def update(self, frame: np.ndarray, detections: List[DetectionResult]) -> List[TrackedObject]:
        """Update tracking with new detections."""
        current_time = time.time()
        
        # Update existing trackers
        active_trackers = []
        for tracker_id, tracker in self.trackers.items():
            if tracker_id in self.tracked_objects:
                success, bbox = tracker.update(frame)
                if success:
                    # Update tracked object
                    x, y, w, h = [int(v) for v in bbox]
                    new_detection = DetectionResult(
                        bbox=(x, y, w, h),
                        confidence=self.tracked_objects[tracker_id].get_average_confidence(),
                        class_name=self.tracked_objects[tracker_id].detection.class_name,
                        timestamp=current_time,
                        method=f"tracked_{self.algorithm}"
                    )
                    self.tracked_objects[tracker_id].update(new_detection)
                    active_trackers.append(tracker_id)
                else:
                    # Mark as inactive if tracking failed
                    self.tracked_objects[tracker_id].is_active = False
        
        # Remove inactive trackers
        inactive_ids = [tid for tid, obj in self.tracked_objects.items() 
                       if not obj.is_active or 
                       (current_time - obj.last_seen) > self.max_disappeared]
        
        for tid in inactive_ids:
            if tid in self.trackers:
                del self.trackers[tid]
            if tid in self.tracked_objects:
                del self.tracked_objects[tid]
        
        # Try to associate new detections with existing trackers
        for detection in detections:
            # Simple association based on IoU (could be improved with Hungarian algorithm)
            best_match_id = self._associate_detection(detection)
            if best_match_id is not None:
                # Update existing tracker
                tracker = self.trackers[best_match_id]
                x, y, w, h = detection.bbox
                tracker.init(frame, (x, y, w, h))
                self.tracked_objects[best_match_id].update(detection)
            else:
                # Create new tracker
                self.add_object(detection)
        
        return [obj for obj in self.tracked_objects.values() if obj.is_active]
    
    def _associate_detection(self, detection: DetectionResult) -> Optional[int]:
        """Associate detection with existing tracker using IoU."""
        best_iou = 0.0
        best_id = None
        
        for tracker_id, tracked_obj in self.tracked_objects.items():
            if not tracked_obj.is_active:
                continue
            
            iou = self._calculate_iou(detection.bbox, tracked_obj.detection.bbox)
            if iou > best_iou and iou > 0.3:  # Minimum IoU threshold
                best_iou = iou
                best_id = tracker_id
        
        return best_id
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                      bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_tracked_objects(self) -> List[TrackedObject]:
        """Get all currently tracked objects."""
        return [obj for obj in self.tracked_objects.values() if obj.is_active]
