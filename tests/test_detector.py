"""
Test suite for pedestrian detection system.

This module contains unit tests for the core functionality.
"""

import pytest
import os
import tempfile
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from detector import PedestrianDetector, Detection, DetectionResult, create_sample_image
from config import ConfigManager, ModelConfig, UIConfig, LoggingConfig, AppConfig


class TestPedestrianDetector:
    """Test cases for PedestrianDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create a detector instance for testing."""
        return PedestrianDetector(confidence_threshold=0.3)
    
    @pytest.fixture
    def sample_image_path(self):
        """Create a temporary sample image for testing."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            create_sample_image(tmp.name, 400, 300)
            yield tmp.name
        os.unlink(tmp.name)
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        detector = PedestrianDetector()
        assert detector.confidence_threshold == 0.5
        assert detector.model is not None
    
    def test_detector_custom_params(self):
        """Test detector with custom parameters."""
        detector = PedestrianDetector(confidence_threshold=0.7)
        assert detector.confidence_threshold == 0.7
    
    def test_detect_pedestrians_file_not_found(self, detector):
        """Test detection with non-existent file."""
        with pytest.raises(FileNotFoundError):
            detector.detect_pedestrians("non_existent.jpg")
    
    def test_detect_pedestrians_success(self, detector, sample_image_path):
        """Test successful pedestrian detection."""
        result = detector.detect_pedestrians(sample_image_path)
        
        assert isinstance(result, DetectionResult)
        assert isinstance(result.detections, list)
        assert len(result.image_shape) == 3  # height, width, channels
        assert result.processing_time > 0
    
    def test_extract_detections(self, detector, sample_image_path):
        """Test detection extraction."""
        result = detector.detect_pedestrians(sample_image_path)
        
        for detection in result.detections:
            assert isinstance(detection, Detection)
            assert len(detection.bbox) == 4
            assert 0 <= detection.confidence <= 1
            assert detection.class_id == 0  # person class
            assert detection.class_name == "person"
    
    def test_visualize_detections(self, detector, sample_image_path):
        """Test visualization generation."""
        result = detector.detect_pedestrians(sample_image_path)
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            vis_image = detector.visualize_detections(sample_image_path, result, tmp.name)
            
            assert isinstance(vis_image, np.ndarray)
            assert os.path.exists(tmp.name)
            
            # Clean up
            os.unlink(tmp.name)
    
    def test_get_detection_stats(self, detector, sample_image_path):
        """Test statistics generation."""
        result = detector.detect_pedestrians(sample_image_path)
        stats = detector.get_detection_stats(result)
        
        assert isinstance(stats, dict)
        assert 'count' in stats
        assert 'avg_confidence' in stats
        assert 'processing_time' in stats
        assert stats['count'] >= 0
        assert stats['processing_time'] > 0


class TestDetectionClasses:
    """Test cases for detection data classes."""
    
    def test_detection_creation(self):
        """Test Detection class creation."""
        detection = Detection(
            bbox=(10, 20, 30, 40),
            confidence=0.85,
            class_id=0
        )
        
        assert detection.bbox == (10, 20, 30, 40)
        assert detection.confidence == 0.85
        assert detection.class_id == 0
        assert detection.class_name == "person"
    
    def test_detection_result_creation(self):
        """Test DetectionResult class creation."""
        detections = [
            Detection(bbox=(10, 20, 30, 40), confidence=0.8, class_id=0),
            Detection(bbox=(50, 60, 70, 80), confidence=0.9, class_id=0)
        ]
        
        result = DetectionResult(
            detections=detections,
            image_shape=(480, 640, 3),
            processing_time=0.123
        )
        
        assert len(result.detections) == 2
        assert result.image_shape == (480, 640, 3)
        assert result.processing_time == 0.123


class TestConfigManager:
    """Test cases for configuration management."""
    
    def test_config_manager_initialization(self):
        """Test ConfigManager initialization."""
        config_manager = ConfigManager()
        assert config_manager.config is not None
    
    def test_model_config(self):
        """Test ModelConfig creation."""
        config = ModelConfig()
        assert config.model_path == "yolov8n.pt"
        assert config.confidence_threshold == 0.5
        assert config.device == "cpu"
        assert config.input_size == 640
    
    def test_ui_config(self):
        """Test UIConfig creation."""
        config = UIConfig()
        assert config.title == "Pedestrian Detection System"
        assert config.theme == "light"
        assert config.max_file_size_mb == 10
        assert ".jpg" in config.supported_formats
    
    def test_logging_config(self):
        """Test LoggingConfig creation."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert "asctime" in config.format
        assert config.file_path == "logs/detection.log"
    
    def test_app_config(self):
        """Test AppConfig creation."""
        config = AppConfig(
            model=ModelConfig(),
            ui=UIConfig(),
            logging=LoggingConfig()
        )
        
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.ui, UIConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert config.data_dir == "data"


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_create_sample_image(self):
        """Test sample image creation."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            create_sample_image(tmp.name, 400, 300)
            
            assert os.path.exists(tmp.name)
            
            # Check if it's a valid image
            import cv2
            img = cv2.imread(tmp.name)
            assert img is not None
            assert img.shape == (300, 400, 3)
            
            # Clean up
            os.unlink(tmp.name)
    
    def test_create_sample_image_custom_size(self):
        """Test sample image creation with custom size."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            create_sample_image(tmp.name, 800, 600)
            
            import cv2
            img = cv2.imread(tmp.name)
            assert img.shape == (600, 800, 3)
            
            os.unlink(tmp.name)


# Integration tests
class TestIntegration:
    """Integration tests for the complete workflow."""
    
    def test_end_to_end_detection(self):
        """Test complete detection workflow."""
        # Create sample image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            create_sample_image(tmp.name, 400, 300)
            
            try:
                # Initialize detector
                detector = PedestrianDetector(confidence_threshold=0.3)
                
                # Run detection
                result = detector.detect_pedestrians(tmp.name)
                
                # Verify results
                assert isinstance(result, DetectionResult)
                assert result.processing_time > 0
                
                # Get statistics
                stats = detector.get_detection_stats(result)
                assert stats['count'] >= 0
                
                # Create visualization
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as vis_tmp:
                    vis_image = detector.visualize_detections(tmp.name, result, vis_tmp.name)
                    assert isinstance(vis_image, np.ndarray)
                    assert os.path.exists(vis_tmp.name)
                    os.unlink(vis_tmp.name)
                
            finally:
                os.unlink(tmp.name)


if __name__ == "__main__":
    pytest.main([__file__])
