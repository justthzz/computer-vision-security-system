#!/usr/bin/env python3
"""
Demo Computer Vision Security System
A simplified version that works without OpenCV resize issues
"""

import cv2
import numpy as np
import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple

class DemoSecuritySystem:
    """Simplified security system for demonstration purposes."""
    
    def __init__(self):
        self.setup_logging()
        self.setup_config()
        self.setup_detectors()
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_config(self):
        """Setup system configuration."""
        self.config = {
            "output_resolution": [640, 480],
            "detection_threshold": 0.5,
            "fps_target": 30
        }
        
    def setup_detectors(self):
        """Setup detection algorithms."""
        # Motion detection using MOG2
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, 
            varThreshold=50
        )
        
        # Simple contour detection
        self.min_contour_area = 500
        
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process frame with motion detection."""
        if frame is None or frame.size == 0:
            return self._get_empty_result()
            
        # Convert to grayscale for motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply motion detection
        fg_mask = self.mog2.apply(gray)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                detections.append({
                    'bbox': (x, y, w, h),
                    'confidence': min(area / 1000, 1.0),
                    'class_name': 'Motion'
                })
        
        return {
            'detections': detections,
            'performance': {
                'fps': 30,  # Simulated FPS
                'frame_time': 0.033,
                'detection_times': {'motion': 0.01}
            }
        }
    
    def draw_annotations(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """Draw detection annotations on frame."""
        if frame is None or frame.size == 0:
            return np.zeros((480, 640, 3), dtype=np.uint8)
            
        annotated_frame = frame.copy()
        
        # Draw detections
        for detection in results['detections']:
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label
            label = f"Motion: {confidence:.2f}"
            cv2.putText(annotated_frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw performance info
        perf = results['performance']
        info_text = [
            f"FPS: {perf['fps']:.1f}",
            f"Detections: {len(results['detections'])}",
            f"Status: {'ALERT' if results['detections'] else 'CLEAR'}"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(annotated_frame, text, (10, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame
    
    def _get_empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'detections': [],
            'performance': {
                'fps': 0,
                'frame_time': 0,
                'detection_times': {}
            }
        }
    
    def run_demo(self, video_source: str = None):
        """Run the demo system."""
        print("🚀 Demo Computer Vision Security System")
        print("=" * 50)
        
        # Initialize video capture
        if video_source is None:
            print("📹 Using camera feed...")
            cap = cv2.VideoCapture(0)
        else:
            print(f"📹 Using video file: {video_source}")
            cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print("❌ Failed to open video source")
            return
        
        print("✅ Video capture initialized")
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Save current frame")
        print("  'r' - Reset detection")
        print("=" * 50)
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("📹 End of video or camera disconnected")
                    break
                
                # Process frame
                results = self.process_frame(frame)
                
                # Draw annotations
                annotated_frame = self.draw_annotations(frame, results)
                
                # Display frame
                cv2.imshow('Demo Security System', annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"demo_capture_{timestamp}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"💾 Frame saved: {filename}")
                elif key == ord('r'):
                    self.mog2 = cv2.createBackgroundSubtractorMOG2(
                        detectShadows=True, 
                        varThreshold=50
                    )
                    print("🔄 Detection reset")
                
                # Calculate FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"📊 FPS: {fps:.1f}, Detections: {len(results['detections'])}")
                
        except KeyboardInterrupt:
            print("\n⏹️  Demo stopped by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("🧹 Demo cleanup completed")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Demo Computer Vision Security System')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    
    args = parser.parse_args()
    
    # Create and run demo system
    system = DemoSecuritySystem()
    
    if args.video:
        system.run_demo(args.video)
    else:
        system.run_demo()

if __name__ == "__main__":
    main()
