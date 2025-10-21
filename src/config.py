"""
Configuration module for pedestrian detection project.

This module handles all configuration settings using YAML files
for easy customization and deployment.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    """Configuration for the detection model."""
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.5
    device: str = "cpu"  # "cpu" or "cuda"
    input_size: int = 640


@dataclass
class UIConfig:
    """Configuration for the user interface."""
    title: str = "Pedestrian Detection System"
    theme: str = "light"
    max_file_size_mb: int = 10
    supported_formats: list = None
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = "logs/detection.log"


@dataclass
class AppConfig:
    """Main application configuration."""
    model: ModelConfig
    ui: UIConfig
    logging: LoggingConfig
    data_dir: str = "data"
    output_dir: str = "outputs"
    temp_dir: str = "temp"


class ConfigManager:
    """Manages application configuration."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "config/config.yaml"
        self.config = self._load_config()
    
    def _load_config(self) -> AppConfig:
        """Load configuration from file or create default."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
                return self._dict_to_config(config_dict)
            except Exception as e:
                print(f"Error loading config file: {e}. Using defaults.")
        
        return self._create_default_config()
    
    def _create_default_config(self) -> AppConfig:
        """Create default configuration."""
        return AppConfig(
            model=ModelConfig(),
            ui=UIConfig(),
            logging=LoggingConfig()
        )
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> AppConfig:
        """Convert dictionary to configuration object."""
        return AppConfig(
            model=ModelConfig(**config_dict.get("model", {})),
            ui=UIConfig(**config_dict.get("ui", {})),
            logging=LoggingConfig(**config_dict.get("logging", {})),
            data_dir=config_dict.get("data_dir", "data"),
            output_dir=config_dict.get("output_dir", "outputs"),
            temp_dir=config_dict.get("temp_dir", "temp")
        )
    
    def save_config(self, config_path: Optional[str] = None) -> None:
        """Save current configuration to file."""
        save_path = config_path or self.config_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        config_dict = {
            "model": asdict(self.config.model),
            "ui": asdict(self.config.ui),
            "logging": asdict(self.config.logging),
            "data_dir": self.config.data_dir,
            "output_dir": self.config.output_dir,
            "temp_dir": self.config.temp_dir
        }
        
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration."""
        return self.config.model
    
    def get_ui_config(self) -> UIConfig:
        """Get UI configuration."""
        return self.config.ui
    
    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration."""
        return self.config.logging


def setup_logging(config: LoggingConfig) -> None:
    """Setup logging configuration."""
    import logging
    
    # Create logs directory if it doesn't exist
    if config.file_path:
        os.makedirs(os.path.dirname(config.file_path), exist_ok=True)
    
    # Configure logging
    handlers = [logging.StreamHandler()]
    if config.file_path:
        handlers.append(logging.FileHandler(config.file_path))
    
    logging.basicConfig(
        level=getattr(logging, config.level.upper()),
        format=config.format,
        handlers=handlers
    )


if __name__ == "__main__":
    # Create default configuration file
    config_manager = ConfigManager()
    config_manager.save_config()
    print("Default configuration created at config/config.yaml")
