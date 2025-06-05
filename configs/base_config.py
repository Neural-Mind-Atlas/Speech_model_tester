"""
Base Configuration for TTS/STT Testing Framework
====

This module provides the base configuration class for the framework.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import yaml


@dataclass
class BaseConfig:
    """Base configuration class for the TTS/STT testing framework."""
    
    # Framework metadata
    framework_name: str = "TTS-STT Testing Framework"
    framework_version: str = "1.0.0"
    
    # Logging configuration
    log_level: str = "INFO"
    debug_mode: bool = False
    
    # Directory paths
    test_data_dir: str = "data/test_inputs"
    output_data_dir: str = "data/test_outputs"
    reference_data_dir: str = "data/reference"
    results_dir: str = "results"
    
    # Execution configuration
    max_workers: int = 4
    timeout_seconds: int = 300
    enable_parallel: bool = True
    
    # Report generation
    enable_html_reports: bool = True
    enable_json_reports: bool = True
    enable_yaml_reports: bool = True
    
    # Provider settings
    enabled_providers: List[str] = field(default_factory=list)
    enabled_models: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Ensure directories exist
        for dir_path in [self.test_data_dir, self.output_data_dir, 
                        self.reference_data_dir, self.results_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_file(cls, config_path: str) -> 'BaseConfig':
        """Load configuration from file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config_data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported configuration format: {config_path.suffix}")
        
        return cls(**config_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'framework_name': self.framework_name,
            'framework_version': self.framework_version,
            'log_level': self.log_level,
            'debug_mode': self.debug_mode,
            'test_data_dir': self.test_data_dir,
            'output_data_dir': self.output_data_dir,
            'reference_data_dir': self.reference_data_dir,
            'results_dir': self.results_dir,
            'max_workers': self.max_workers,
            'timeout_seconds': self.timeout_seconds,
            'enable_parallel': self.enable_parallel,
            'enable_html_reports': self.enable_html_reports,
            'enable_json_reports': self.enable_json_reports,
            'enable_yaml_reports': self.enable_yaml_reports,
            'enabled_providers': self.enabled_providers,
            'enabled_models': self.enabled_models
        }
    
    def dict(self) -> Dict[str, Any]:
        """Alias for to_dict() method for compatibility."""
        return self.to_dict()
    
    def update(self, **kwargs) -> None:
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default."""
        return getattr(self, key, default)
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        try:
            # Validate numeric parameters
            if self.max_workers <= 0:
                raise ValueError("max_workers must be positive")
            if self.timeout_seconds <= 0:
                raise ValueError("timeout_seconds must be positive")
            
            # Validate log level
            valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if self.log_level.upper() not in valid_log_levels:
                raise ValueError(f"Invalid log_level: {self.log_level}")
            
            return True
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False