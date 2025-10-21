"""
Command-line interface for pedestrian detection.

This module provides a CLI for batch processing images and
running detection from the command line.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional
import json
import time

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from detector import PedestrianDetector, DetectionResult
from config import ConfigManager, setup_logging


class PedestrianDetectionCLI:
    """Command-line interface for pedestrian detection."""
    
    def __init__(self):
        """Initialize CLI."""
        self.config_manager = ConfigManager()
        self.detector = None
    
    def setup_logging(self, verbose: bool = False):
        """Setup logging configuration."""
        log_config = self.config_manager.get_logging_config()
        if verbose:
            log_config.level = "DEBUG"
        setup_logging(log_config)
    
    def load_detector(self, model_path: Optional[str] = None, 
                     confidence: Optional[float] = None):
        """Load the detection model."""
        model_config = self.config_manager.get_model_config()
        
        if model_path:
            model_config.model_path = model_path
        if confidence:
            model_config.confidence_threshold = confidence
        
        self.detector = PedestrianDetector(
            model_path=model_config.model_path,
            confidence_threshold=model_config.confidence_threshold
        )
    
    def process_single_image(self, image_path: str, output_dir: Optional[str] = None,
                           save_visualization: bool = True, save_results: bool = True) -> DetectionResult:
        """Process a single image."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        logging.info(f"Processing image: {image_path}")
        
        # Run detection
        result = self.detector.detect_pedestrians(image_path)
        
        # Save visualization if requested
        if save_visualization:
            vis_path = self._get_output_path(image_path, output_dir, "_detected.jpg")
            self.detector.visualize_detections(image_path, result, vis_path)
            logging.info(f"Visualization saved: {vis_path}")
        
        # Save results if requested
        if save_results:
            results_path = self._get_output_path(image_path, output_dir, "_results.json")
            self._save_results_json(result, results_path)
            logging.info(f"Results saved: {results_path}")
        
        return result
    
    def process_batch(self, input_dir: str, output_dir: Optional[str] = None,
                     pattern: str = "*.jpg", save_visualization: bool = True,
                     save_results: bool = True) -> List[DetectionResult]:
        """Process multiple images in batch."""
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Find all matching images
        image_files = list(input_path.glob(pattern))
        if not image_files:
            logging.warning(f"No images found matching pattern: {pattern}")
            return []
        
        logging.info(f"Found {len(image_files)} images to process")
        
        results = []
        for i, image_file in enumerate(image_files, 1):
            try:
                logging.info(f"Processing {i}/{len(image_files)}: {image_file.name}")
                result = self.process_single_image(
                    str(image_file), output_dir, save_visualization, save_results
                )
                results.append(result)
                
                # Print progress
                stats = self.detector.get_detection_stats(result)
                print(f"  ✓ Found {stats['count']} pedestrians (confidence: {stats['avg_confidence']:.2f})")
                
            except Exception as e:
                logging.error(f"Error processing {image_file}: {e}")
                print(f"  ✗ Error processing {image_file.name}: {e}")
        
        return results
    
    def _get_output_path(self, input_path: str, output_dir: Optional[str], suffix: str) -> str:
        """Generate output path for processed files."""
        input_file = Path(input_path)
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            return str(output_path / f"{input_file.stem}{suffix}")
        else:
            return str(input_file.parent / f"{input_file.stem}{suffix}")
    
    def _save_results_json(self, result: DetectionResult, output_path: str):
        """Save detection results as JSON."""
        data = {
            "detections": [
                {
                    "bbox": detection.bbox,
                    "confidence": detection.confidence,
                    "class_id": detection.class_id,
                    "class_name": detection.class_name
                }
                for detection in result.detections
            ],
            "image_shape": result.image_shape,
            "processing_time": result.processing_time,
            "statistics": self.detector.get_detection_stats(result)
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_sample_dataset(self, output_dir: str, num_images: int = 10):
        """Create a sample dataset for testing."""
        from detector import create_sample_image
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Creating {num_images} sample images in {output_dir}")
        
        for i in range(num_images):
            # Vary image sizes and content
            width = 600 + (i % 3) * 200
            height = 400 + (i % 2) * 200
            
            image_path = output_path / f"sample_{i+1:03d}.jpg"
            create_sample_image(str(image_path), width, height)
        
        logging.info(f"Sample dataset created: {output_dir}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Pedestrian Detection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image
  python cli.py detect image.jpg
  
  # Process batch of images
  python cli.py batch input_folder/ --output output_folder/
  
  # Create sample dataset
  python cli.py create-samples data/samples --num-images 20
  
  # Process with custom settings
  python cli.py detect image.jpg --confidence 0.7 --model yolov8s.pt
        """
    )
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Detect pedestrians in a single image')
    detect_parser.add_argument('image', help='Path to input image')
    detect_parser.add_argument('--output', '-o', help='Output directory')
    detect_parser.add_argument('--confidence', '-c', type=float, default=0.5,
                              help='Confidence threshold (default: 0.5)')
    detect_parser.add_argument('--model', '-m', help='Model path')
    detect_parser.add_argument('--no-viz', action='store_true',
                              help='Skip saving visualization')
    detect_parser.add_argument('--no-results', action='store_true',
                              help='Skip saving results JSON')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Process multiple images')
    batch_parser.add_argument('input_dir', help='Input directory')
    batch_parser.add_argument('--output', '-o', help='Output directory')
    batch_parser.add_argument('--pattern', '-p', default='*.jpg',
                             help='File pattern (default: *.jpg)')
    batch_parser.add_argument('--confidence', '-c', type=float, default=0.5,
                             help='Confidence threshold (default: 0.5)')
    batch_parser.add_argument('--model', '-m', help='Model path')
    batch_parser.add_argument('--no-viz', action='store_true',
                             help='Skip saving visualizations')
    batch_parser.add_argument('--no-results', action='store_true',
                             help='Skip saving results JSON')
    
    # Create samples command
    samples_parser = subparsers.add_parser('create-samples', help='Create sample dataset')
    samples_parser.add_argument('output_dir', help='Output directory')
    samples_parser.add_argument('--num-images', '-n', type=int, default=10,
                               help='Number of sample images (default: 10)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize CLI
    cli = PedestrianDetectionCLI()
    cli.setup_logging(args.verbose)
    
    try:
        if args.command == 'detect':
            # Load detector
            cli.load_detector(args.model, args.confidence)
            
            # Process image
            result = cli.process_single_image(
                args.image,
                args.output,
                not args.no_viz,
                not args.no_results
            )
            
            # Print summary
            stats = cli.detector.get_detection_stats(result)
            print(f"\n✓ Detection complete!")
            print(f"  Pedestrians found: {stats['count']}")
            print(f"  Average confidence: {stats['avg_confidence']:.3f}")
            print(f"  Processing time: {stats['processing_time']:.3f}s")
        
        elif args.command == 'batch':
            # Load detector
            cli.load_detector(args.model, args.confidence)
            
            # Process batch
            results = cli.process_batch(
                args.input_dir,
                args.output,
                args.pattern,
                not args.no_viz,
                not args.no_results
            )
            
            # Print summary
            total_detections = sum(len(r.detections) for r in results)
            avg_time = sum(r.processing_time for r in results) / len(results) if results else 0
            
            print(f"\n✓ Batch processing complete!")
            print(f"  Images processed: {len(results)}")
            print(f"  Total pedestrians found: {total_detections}")
            print(f"  Average processing time: {avg_time:.3f}s")
        
        elif args.command == 'create-samples':
            cli.create_sample_dataset(args.output_dir, args.num_images)
            print(f"\n✓ Created {args.num_images} sample images in {args.output_dir}")
    
    except Exception as e:
        logging.error(f"CLI error: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
