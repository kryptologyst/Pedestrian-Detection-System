"""
Pedestrian Detection Module

This module provides a modern, type-safe implementation of pedestrian detection
using YOLOv8 (latest version) with comprehensive error handling and logging.
"""

import logging
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class Detection:
    """Represents a single pedestrian detection."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str = "person"


@dataclass
class DetectionResult:
    """Container for detection results."""
    detections: List[Detection]
    image_shape: Tuple[int, int, int]  # (height, width, channels)
    processing_time: float


class PedestrianDetector:
    """
    Modern pedestrian detection using YOLOv8.
    
    This class provides a clean interface for detecting pedestrians in images
    with comprehensive error handling and logging.
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        """
        Initialize the pedestrian detector.
        
        Args:
            model_path: Path to YOLO model file
            confidence_threshold: Minimum confidence for detections
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
        
        try:
            self.model = YOLO(model_path)
            self.logger.info(f"Successfully loaded YOLO model: {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def detect_pedestrians(self, image_path: str) -> DetectionResult:
        """
        Detect pedestrians in an image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            DetectionResult containing all detections and metadata
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be processed
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            # Load and validate image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run inference
            import time
            start_time = time.time()
            results = self.model(rgb_image)[0]
            processing_time = time.time() - start_time
            
            # Extract pedestrian detections
            detections = self._extract_detections(results)
            
            self.logger.info(f"Detected {len(detections)} pedestrians in {processing_time:.3f}s")
            
            return DetectionResult(
                detections=detections,
                image_shape=rgb_image.shape,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Error during detection: {e}")
            raise
    
    def _extract_detections(self, results) -> List[Detection]:
        """Extract pedestrian detections from YOLO results."""
        detections = []
        
        if results.boxes is not None:
            for box in results.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Only process 'person' class (class ID 0 in COCO)
                if class_id == 0 and confidence >= self.confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detection = Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=confidence,
                        class_id=class_id
                    )
                    detections.append(detection)
        
        return detections
    
    def visualize_detections(self, image_path: str, result: DetectionResult, 
                           save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize detections on the image.
        
        Args:
            image_path: Path to original image
            result: Detection results
            save_path: Optional path to save the visualization
            
        Returns:
            Image array with drawn detections
        """
        # Load original image
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Draw bounding boxes
        for detection in result.detections:
            x1, y1, x2, y2 = detection.bbox
            confidence = detection.confidence
            
            # Draw rectangle
            cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"Person {confidence:.2f}"
            cv2.putText(rgb_image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Save if requested
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
            self.logger.info(f"Visualization saved to: {save_path}")
        
        return rgb_image
    
    def get_detection_stats(self, result: DetectionResult) -> Dict[str, Any]:
        """Get statistics about the detection results."""
        if not result.detections:
            return {
                "count": 0,
                "avg_confidence": 0.0,
                "processing_time": result.processing_time
            }
        
        confidences = [d.confidence for d in result.detections]
        return {
            "count": len(result.detections),
            "avg_confidence": np.mean(confidences),
            "max_confidence": np.max(confidences),
            "min_confidence": np.min(confidences),
            "processing_time": result.processing_time
        }


def create_sample_image(output_path: str, width: int = 800, height: int = 600) -> None:
    """
    Create a synthetic sample image with simple shapes for testing.
    
    Args:
        output_path: Path where to save the sample image
        width: Image width
        height: Image height
    """
    # Create a simple synthetic image
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some background elements
    cv2.rectangle(image, (100, 100), (200, 300), (128, 128, 128), -1)  # Building
    cv2.rectangle(image, (300, 200), (400, 350), (128, 128, 128), -1)  # Building
    
    # Add some simple "pedestrian" shapes (rectangles)
    cv2.rectangle(image, (150, 250), (170, 400), (255, 255, 255), -1)  # Person 1
    cv2.rectangle(image, (180, 200), (200, 350), (255, 255, 255), -1)  # Person 2
    cv2.rectangle(image, (350, 300), (370, 450), (255, 255, 255), -1)  # Person 3
    
    # Add some noise for realism
    noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
    image = cv2.add(image, noise)
    
    cv2.imwrite(output_path, image)
    logging.info(f"Sample image created: {output_path}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data directory
    os.makedirs("data", exist_ok=True)
    
    # Create a sample image
    sample_path = "data/sample_pedestrians.jpg"
    create_sample_image(sample_path)
    
    # Initialize detector
    detector = PedestrianDetector()
    
    # Run detection
    try:
        result = detector.detect_pedestrians(sample_path)
        
        # Print statistics
        stats = detector.get_detection_stats(result)
        print(f"Detection Statistics: {stats}")
        
        # Visualize results
        vis_image = detector.visualize_detections(sample_path, result, 
                                                "data/detection_result.jpg")
        
        # Display results
        plt.figure(figsize=(12, 8))
        plt.imshow(vis_image)
        plt.title(f"Pedestrian Detection - Found {stats['count']} pedestrians")
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        logging.error(f"Detection failed: {e}")
