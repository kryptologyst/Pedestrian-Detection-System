#!/usr/bin/env python3
"""
Main entry point for the pedestrian detection system.

This script provides easy access to all functionality including
CLI, web interface, and utility functions.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def main():
    """Main entry point with subcommands."""
    parser = argparse.ArgumentParser(
        description="Pedestrian Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available commands:
  detect     Detect pedestrians in images (CLI)
  batch      Process multiple images
  web        Launch web interface
  samples    Create sample dataset
  test       Run test suite
  report     Generate detection report

Examples:
  python main.py detect image.jpg
  python main.py web
  python main.py samples --num-images 20
        """
    )
    
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
    
    # Web command
    web_parser = subparsers.add_parser('web', help='Launch web interface')
    web_parser.add_argument('--port', '-p', type=int, default=8501,
                           help='Port number (default: 8501)')
    web_parser.add_argument('--host', default='localhost',
                           help='Host address (default: localhost)')
    
    # Samples command
    samples_parser = subparsers.add_parser('samples', help='Create sample dataset')
    samples_parser.add_argument('output_dir', help='Output directory')
    samples_parser.add_argument('--num-images', '-n', type=int, default=10,
                               help='Number of sample images (default: 10)')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run test suite')
    test_parser.add_argument('--verbose', '-v', action='store_true',
                            help='Verbose output')
    test_parser.add_argument('--coverage', action='store_true',
                            help='Run with coverage')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate detection report')
    report_parser.add_argument('input_dir', help='Directory containing detection results')
    report_parser.add_argument('--output', '-o', default='reports',
                              help='Output directory for report (default: reports)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'detect':
            from src.cli import PedestrianDetectionCLI
            cli = PedestrianDetectionCLI()
            cli.setup_logging()
            cli.load_detector(args.model, args.confidence)
            
            result = cli.process_single_image(
                args.image,
                args.output,
                not args.no_viz,
                not args.no_results
            )
            
            stats = cli.detector.get_detection_stats(result)
            print(f"\n✓ Detection complete!")
            print(f"  Pedestrians found: {stats['count']}")
            print(f"  Average confidence: {stats['avg_confidence']:.3f}")
            print(f"  Processing time: {stats['processing_time']:.3f}s")
        
        elif args.command == 'batch':
            from src.cli import PedestrianDetectionCLI
            cli = PedestrianDetectionCLI()
            cli.setup_logging()
            cli.load_detector(args.model, args.confidence)
            
            results = cli.process_batch(
                args.input_dir,
                args.output,
                args.pattern,
                not args.no_viz,
                not args.no_results
            )
            
            total_detections = sum(len(r.detections) for r in results)
            avg_time = sum(r.processing_time for r in results) / len(results) if results else 0
            
            print(f"\n✓ Batch processing complete!")
            print(f"  Images processed: {len(results)}")
            print(f"  Total pedestrians found: {total_detections}")
            print(f"  Average processing time: {avg_time:.3f}s")
        
        elif args.command == 'web':
            import subprocess
            import os
            
            web_app_path = os.path.join(os.path.dirname(__file__), 'web_app', 'app.py')
            if not os.path.exists(web_app_path):
                print(f"Web app not found at: {web_app_path}")
                return
            
            print(f"Launching web interface on http://{args.host}:{args.port}")
            subprocess.run([
                'streamlit', 'run', web_app_path,
                '--server.port', str(args.port),
                '--server.address', args.host
            ])
        
        elif args.command == 'samples':
            from src.detector import create_sample_image
            
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            print(f"Creating {args.num_images} sample images in {args.output_dir}")
            for i in range(args.num_images):
                width = 600 + (i % 3) * 200
                height = 400 + (i % 2) * 200
                image_path = output_path / f"sample_{i+1:03d}.jpg"
                create_sample_image(str(image_path), width, height)
            
            print(f"✓ Created {args.num_images} sample images")
        
        elif args.command == 'test':
            import subprocess
            
            cmd = ['pytest', 'tests/']
            if args.verbose:
                cmd.append('-v')
            if args.coverage:
                cmd.extend(['--cov=src', '--cov-report=html'])
            
            subprocess.run(cmd)
        
        elif args.command == 'report':
            from src.visualization import create_detection_report
            from src.detector import PedestrianDetector
            
            # Load detection results from JSON files
            input_path = Path(args.input_dir)
            json_files = list(input_path.glob("*_results.json"))
            
            if not json_files:
                print(f"No detection result files found in {args.input_dir}")
                return
            
            print(f"Found {len(json_files)} result files")
            
            # For now, create a simple report
            # In a real implementation, you'd load the JSON results
            print(f"Report generation not fully implemented yet.")
            print(f"Would process {len(json_files)} result files from {args.input_dir}")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
