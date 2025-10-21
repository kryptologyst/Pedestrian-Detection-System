# Pedestrian Detection System

A production-ready pedestrian detection system built with YOLOv8, featuring both web interface and command-line tools.

## Features

- **State-of-the-art Detection**: Uses YOLOv8 for accurate pedestrian detection
- **Multiple Interfaces**: Web UI (Streamlit) and CLI for different use cases
- **Modern Architecture**: Type hints, comprehensive logging, and configuration management
- **Batch Processing**: Process multiple images efficiently
- **Visualization**: Automatic bounding box drawing and confidence scoring
- **Extensible**: Easy to customize and extend for specific use cases

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Pedestrian-Detection-System.git
cd Pedestrian-Detection-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the web interface:
```bash
streamlit run web_app/app.py
```

4. Or use the command line:
```bash
python src/cli.py detect path/to/image.jpg
```

## Usage

### Web Interface

The Streamlit web interface provides an intuitive way to:

- Upload images for detection
- Adjust confidence thresholds
- View detection results with visualizations
- Access sample images for testing

```bash
streamlit run web_app/app.py
```

### Command Line Interface

#### Single Image Detection
```bash
python src/cli.py detect image.jpg --confidence 0.7 --output results/
```

#### Batch Processing
```bash
python src/cli.py batch input_folder/ --output output_folder/ --pattern "*.jpg"
```

#### Create Sample Dataset
```bash
python src/cli.py create-samples data/samples --num-images 20
```

### Python API

```python
from src.detector import PedestrianDetector

# Initialize detector
detector = PedestrianDetector(confidence_threshold=0.5)

# Detect pedestrians
result = detector.detect_pedestrians("image.jpg")

# Get statistics
stats = detector.get_detection_stats(result)
print(f"Found {stats['count']} pedestrians")

# Visualize results
vis_image = detector.visualize_detections("image.jpg", result, "output.jpg")
```

## Project Structure

```
pedestrian-detection/
├── src/                    # Core source code
│   ├── detector.py         # Main detection logic
│   ├── config.py           # Configuration management
│   └── cli.py              # Command-line interface
├── web_app/                # Web interface
│   └── app.py              # Streamlit application
├── data/                   # Data directory
├── models/                 # Model files
├── config/                 # Configuration files
├── tests/                  # Test files
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## Configuration

The system uses YAML configuration files for easy customization:

```yaml
model:
  model_path: "yolov8n.pt"
  confidence_threshold: 0.5
  device: "cpu"
  input_size: 640

ui:
  title: "Pedestrian Detection System"
  theme: "light"
  max_file_size_mb: 10
  supported_formats: [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file_path: "logs/detection.log"
```

## 🔧 Advanced Usage

### Custom Model

Use a different YOLO model:

```python
detector = PedestrianDetector(model_path="yolov8s.pt")
```

### GPU Acceleration

For GPU support, install PyTorch with CUDA:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Then set device in configuration:

```yaml
model:
  device: "cuda"
```

### Batch Processing with Custom Patterns

```bash
python src/cli.py batch input/ --pattern "*.png" --confidence 0.6
```

## Testing

Run the test suite:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=src tests/
```

## Performance

The system provides detailed performance metrics:

- **Processing Time**: Time taken for detection
- **Confidence Scores**: Individual and average confidence
- **Detection Count**: Number of pedestrians found
- **Model Statistics**: Detailed performance breakdown

## Detection Classes

The system detects the following classes from COCO dataset:
- **Person** (class ID: 0) - Primary focus for pedestrian detection

## 🛠️ Development

### Code Style

The project follows PEP 8 standards. Format code with:

```bash
black src/ web_app/
```

### Type Checking

Run type checking:

```bash
mypy src/
```

### Linting

Check code quality:

```bash
flake8 src/ web_app/
```

## API Reference

### PedestrianDetector

Main detection class with methods:

- `detect_pedestrians(image_path)`: Detect pedestrians in an image
- `visualize_detections(image_path, result, save_path)`: Create visualization
- `get_detection_stats(result)`: Get performance statistics

### DetectionResult

Result container with:

- `detections`: List of Detection objects
- `image_shape`: Original image dimensions
- `processing_time`: Time taken for detection

### Detection

Individual detection with:

- `bbox`: Bounding box coordinates (x1, y1, x2, y2)
- `confidence`: Detection confidence score
- `class_id`: COCO class ID
- `class_name`: Human-readable class name

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [OpenCV](https://opencv.org/) for image processing
- [Streamlit](https://streamlit.io/) for web interface
- [COCO Dataset](https://cocodataset.org/) for training data

## Support

For questions and support:

- Create an issue on GitHub
- Check the documentation
- Review the examples in the `data/samples/` directory


# Pedestrian-Detection-System
