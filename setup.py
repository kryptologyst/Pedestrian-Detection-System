#!/usr/bin/env python3
"""
Setup script for pedestrian detection system.

This script handles initial setup, dependency installation,
and configuration generation.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle errors."""
    print(f"Running: {description or cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description or cmd} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running {description or cmd}: {e}")
        print(f"Error output: {e.stderr}")
        return False


def create_directories():
    """Create necessary directories."""
    directories = [
        "data",
        "data/samples", 
        "outputs",
        "temp",
        "logs",
        "reports",
        "models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def install_dependencies():
    """Install Python dependencies."""
    print("Installing Python dependencies...")
    return run_command("pip install -r requirements.txt", "Installing dependencies")


def create_sample_data():
    """Create sample data for testing."""
    print("Creating sample data...")
    
    # Import here to avoid issues if dependencies aren't installed yet
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        from detector import create_sample_image
        
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
                print(f"✓ Created sample image: {name}")
        
        return True
    except Exception as e:
        print(f"✗ Error creating sample data: {e}")
        return False


def verify_installation():
    """Verify that the installation works."""
    print("Verifying installation...")
    
    try:
        # Test imports
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        from detector import PedestrianDetector
        from config import ConfigManager
        
        print("✓ Core modules import successfully")
        
        # Test configuration
        config_manager = ConfigManager()
        print("✓ Configuration system works")
        
        # Test detector initialization (this will download the model)
        print("Testing detector initialization (this may take a moment to download the model)...")
        detector = PedestrianDetector()
        print("✓ Detector initialized successfully")
        
        return True
    except Exception as e:
        print(f"✗ Installation verification failed: {e}")
        return False


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup pedestrian detection system")
    parser.add_argument('--skip-deps', action='store_true',
                       help='Skip dependency installation')
    parser.add_argument('--skip-samples', action='store_true',
                       help='Skip sample data creation')
    parser.add_argument('--skip-verify', action='store_true',
                       help='Skip installation verification')
    
    args = parser.parse_args()
    
    print("🚶 Setting up Pedestrian Detection System")
    print("=" * 50)
    
    # Create directories
    print("\n1. Creating directories...")
    create_directories()
    
    # Install dependencies
    if not args.skip_deps:
        print("\n2. Installing dependencies...")
        if not install_dependencies():
            print("✗ Dependency installation failed. Please check your Python environment.")
            return False
    else:
        print("\n2. Skipping dependency installation")
    
    # Create sample data
    if not args.skip_samples:
        print("\n3. Creating sample data...")
        if not create_sample_data():
            print("⚠ Sample data creation failed, but continuing...")
    else:
        print("\n3. Skipping sample data creation")
    
    # Verify installation
    if not args.skip_verify:
        print("\n4. Verifying installation...")
        if not verify_installation():
            print("✗ Installation verification failed.")
            return False
    else:
        print("\n4. Skipping installation verification")
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Launch web interface: python main.py web")
    print("2. Run CLI detection: python main.py detect data/samples/simple_scene.jpg")
    print("3. Process batch images: python main.py batch data/samples/")
    print("4. Run tests: python main.py test")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
