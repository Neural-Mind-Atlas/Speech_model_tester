"""
TTS/STT Testing Framework - Utils Module
===========================================

This module provides utility functions and classes for the TTS/STT testing framework.
It includes logging, file operations, audio processing, and metrics calculation utilities.

Author: TTS/STT Testing Framework Team
Version: 1.0.0
Created: 2024-06-04
"""

import os
import sys
from typing import Dict, Any, Optional
from pathlib import Path

# Module metadata
__version__ = "1.0.0"
__author__ = "TTS/STT Testing Framework Team"
__email__ = "support@tts-stt-framework.com"
__status__ = "Production"

# Import core utilities
try:
    from .logger import FrameworkLogger, get_logger, setup_logging
    from .file_utils import FileManager, validate_file_path, ensure_directory
    from .audio_utils import AudioProcessor, validate_audio_file, get_audio_info
    from .metrics_utils import MetricsCalculator, calculate_wer, calculate_cer
except ImportError as e:
    # Fallback logging if logger module fails to import
    import logging
    logging.error(f"Failed to import utils modules: {e}")
    raise ImportError(f"Critical utils module import failed: {e}")

# Export main classes and functions
__all__ = [
    # Logger utilities
    'FrameworkLogger',
    'get_logger',
    'setup_logging',
    
    # File utilities
    'FileManager',
    'validate_file_path',
    'ensure_directory',
    
    # Audio utilities
    'AudioProcessor',
    'validate_audio_file',
    'get_audio_info',
    
    # Metrics utilities
    'MetricsCalculator',
    'calculate_wer',
    'calculate_cer',
    
    # Module metadata
    '__version__',
    '__author__',
    '__email__',
    '__status__'
]

def get_module_info() -> Dict[str, Any]:
    """
    Get comprehensive module information.
    
    Returns:
        Dict[str, Any]: Module metadata and status information
    """
    return {
        'name': 'TTS/STT Utils Module',
        'version': __version__,
        'author': __author__,
        'email': __email__,
        'status': __status__,
        'python_version': sys.version,
        'module_path': str(Path(__file__).parent.absolute()),
        'available_utilities': {
            'logging': 'FrameworkLogger, get_logger, setup_logging',
            'file_operations': 'FileManager, validate_file_path, ensure_directory',
            'audio_processing': 'AudioProcessor, validate_audio_file, get_audio_info',
            'metrics_calculation': 'MetricsCalculator, calculate_wer, calculate_cer'
        }
    }

def validate_environment() -> bool:
    """
    Validate the environment for utils module functionality.
    
    Returns:
        bool: True if environment is valid, False otherwise
    """
    try:
        # Check Python version
        if sys.version_info < (3, 8):
            raise RuntimeError("Python 3.8 or higher is required")
        
        # Check required directories
        utils_dir = Path(__file__).parent
        if not utils_dir.exists():
            raise FileNotFoundError(f"Utils directory not found: {utils_dir}")
        
        # Validate critical imports
        required_modules = ['logger', 'file_utils', 'audio_utils', 'metrics_utils']
        for module in required_modules:
            module_path = utils_dir / f"{module}.py"
            if not module_path.exists():
                raise FileNotFoundError(f"Required module not found: {module_path}")
        
        return True
    except Exception as e:
        import logging
        logging.error(f"Environment validation failed: {e}")
        return False

# Initialize logging on module import
try:
    setup_logging()
    logger = get_logger(__name__)
    logger.info(f"Utils module initialized successfully - Version {__version__}")
except Exception as e:
    import logging
    logging.error(f"Failed to initialize utils module logging: {e}")

# Validate environment on import
if not validate_environment():
    raise RuntimeError("Utils module environment validation failed")