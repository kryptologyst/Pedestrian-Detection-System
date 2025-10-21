"""
Streamlit web interface for pedestrian detection.

This module provides a user-friendly web interface for uploading images
and visualizing pedestrian detection results.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import logging
from pathlib import Path
import time

# Import our custom modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detector import PedestrianDetector, DetectionResult
from src.config import ConfigManager, setup_logging


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'detector' not in st.session_state:
        st.session_state.detector = None
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None


def load_detector():
    """Load the pedestrian detector."""
    if st.session_state.detector is None:
        try:
            with st.spinner("Loading YOLO model..."):
                st.session_state.detector = PedestrianDetector()
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return False
    return True


def process_uploaded_image(uploaded_file) -> DetectionResult:
    """Process uploaded image and return detection results."""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Run detection
        detector = st.session_state.detector
        result = detector.detect_pedestrians(tmp_path)
        
        # Visualize results
        vis_image = detector.visualize_detections(tmp_path, result)
        
        return result, vis_image, tmp_path
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def display_detection_stats(result: DetectionResult):
    """Display detection statistics in a nice format."""
    stats = st.session_state.detector.get_detection_stats(result)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Pedestrians Detected", stats['count'])
    
    with col2:
        st.metric("Avg Confidence", f"{stats['avg_confidence']:.2f}")
    
    with col3:
        st.metric("Processing Time", f"{stats['processing_time']:.3f}s")
    
    with col4:
        if stats['count'] > 0:
            st.metric("Max Confidence", f"{stats['max_confidence']:.2f}")
        else:
            st.metric("Max Confidence", "N/A")


def create_sample_images():
    """Create sample images for demo purposes."""
    from src.detector import create_sample_image
    
    sample_dir = Path("data/samples")
    sample_dir.mkdir(exist_ok=True)
    
    # Create different types of sample images
    samples = [
        ("Simple Scene", "simple_scene.jpg", 600, 400),
        ("Crowded Scene", "crowded_scene.jpg", 800, 600),
        ("Street View", "street_view.jpg", 1000, 600)
    ]
    
    for name, filename, width, height in samples:
        sample_path = sample_dir / filename
        if not sample_path.exists():
            create_sample_image(str(sample_path), width, height)
    
    return samples


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Pedestrian Detection System",
        page_icon="🚶",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Load configuration
    config_manager = ConfigManager()
    ui_config = config_manager.get_ui_config()
    setup_logging(config_manager.get_logging_config())
    
    # Title and description
    st.title("🚶 " + ui_config.title)
    st.markdown("""
    Upload an image to detect pedestrians using state-of-the-art YOLOv8 object detection.
    The system will identify and highlight all detected pedestrians with confidence scores.
    """)
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model settings
        st.subheader("Detection Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.5, 
            step=0.05,
            help="Minimum confidence score for detections"
        )
        
        # Update detector if threshold changed
        if st.session_state.detector and st.session_state.detector.confidence_threshold != confidence_threshold:
            st.session_state.detector.confidence_threshold = confidence_threshold
        
        st.subheader("📊 Sample Images")
        if st.button("Generate Sample Images"):
            with st.spinner("Creating sample images..."):
                samples = create_sample_images()
                st.success(f"Created {len(samples)} sample images!")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload an image containing pedestrians"
        )
        
        # Sample images section
        st.subheader("🎯 Try Sample Images")
        samples = create_sample_images()
        
        for name, filename, _, _ in samples:
            if st.button(f"Load {name}", key=f"sample_{filename}"):
                sample_path = f"data/samples/{filename}"
                if os.path.exists(sample_path):
                    with open(sample_path, "rb") as f:
                        st.session_state.uploaded_file = f.read()
                    st.success(f"Loaded {name} sample image!")
                else:
                    st.error(f"Sample image not found: {sample_path}")
    
    with col2:
        st.header("🔍 Detection Results")
        
        # Load detector
        if not load_detector():
            st.stop()
        
        # Process uploaded file
        if uploaded_file is not None or st.session_state.uploaded_file is not None:
            file_to_process = uploaded_file if uploaded_file is not None else st.session_state.uploaded_file
            
            if file_to_process is not None:
                # Convert bytes to file-like object if needed
                if isinstance(file_to_process, bytes):
                    import io
                    file_to_process = io.BytesIO(file_to_process)
                
                # Process the image
                with st.spinner("Detecting pedestrians..."):
                    try:
                        result, vis_image, _ = process_uploaded_image(file_to_process)
                        st.session_state.last_result = result
                        
                        # Display results
                        st.image(vis_image, caption="Detection Results", use_column_width=True)
                        
                        # Display statistics
                        display_detection_stats(result)
                        
                        # Detailed results
                        if result.detections:
                            st.subheader("📋 Detailed Results")
                            for i, detection in enumerate(result.detections):
                                with st.expander(f"Detection {i+1}"):
                                    st.write(f"**Confidence:** {detection.confidence:.3f}")
                                    st.write(f"**Bounding Box:** {detection.bbox}")
                                    st.write(f"**Class:** {detection.class_name}")
                        
                    except Exception as e:
                        st.error(f"Error processing image: {e}")
                        logging.error(f"Processing error: {e}")
        else:
            st.info("👆 Upload an image or select a sample image to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with Streamlit, YOLOv8, and OpenCV</p>
        <p>For educational and research purposes</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
