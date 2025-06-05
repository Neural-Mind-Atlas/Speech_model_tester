"""
Base Configuration for TTS/STT Testing Framework
===============================================

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