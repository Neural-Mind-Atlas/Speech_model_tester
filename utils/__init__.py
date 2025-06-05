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
import logging

# Module metadata
__version__ = "1.0.0"
__author__ = "TTS/STT Testing Framework Team"
__email__ = "support@tts-stt-framework.com"
__status__ = "Production"

# Configure module logger
logger = logging.getLogger(__name__)

# Track import errors
_import_errors = []
_available_modules = []

# Import core utilities with error handling
try:
    from .logger import FrameworkLogger, get_logger, setup_logging
    _available_modules.extend(['FrameworkLogger', 'get_logger', 'setup_logging'])
    logger.info("Successfully imported logger utilities")
except ImportError as e:
    logger.error(f"Failed to import logger utilities: {e}")
    _import_errors.append(('logger', str(e)))
    raise ImportError(f"Critical logger module import failed: {e}")

try:
    from .file_utils import FileManager, validate_file_path, ensure_directory
    _available_modules.extend(['FileManager', 'validate_file_path', 'ensure_directory'])
    logger.info("Successfully imported file utilities")
except ImportError as e:
    logger.error(f"Failed to import file utilities: {e}")
    _import_errors.append(('file_utils', str(e)))

try:
    from .audio_utils import AudioProcessor, AudioValidator, validate_audio_file, get_audio_info
    _available_modules.extend(['AudioProcessor', 'AudioValidator', 'validate_audio_file', 'get_audio_info'])
    logger.info("Successfully imported audio utilities")
except ImportError as e:
    logger.warning(f"Failed to import audio utilities: {e}")
    _import_errors.append(('audio_utils', str(e)))

try:
    from .metrics_utils import MetricsCalculator, calculate_wer, calculate_cer
    _available_modules.extend(['MetricsCalculator', 'calculate_wer', 'calculate_cer'])
    logger.info("Successfully imported metrics utilities")
except ImportError as e:
    logger.warning(f"Failed to import metrics utilities: {e}")
    _import_errors.append(('metrics_utils', str(e)))

try:
    from .report_generator import ReportGenerator
    _available_modules.extend(['ReportGenerator'])
    logger.info("Successfully imported report generator")
except ImportError as e:
    logger.warning(f"Failed to import report generator: {e}")
    _import_errors.append(('report_generator', str(e)))

# Export only available classes and functions
__all__ = _available_modules + [
    '__version__',
    '__author__',
    '__email__',
    '__status__',
    'get_module_info',
    'get_import_errors'
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
        'available_modules': _available_modules,
        'import_errors': _import_errors,
        'total_available': len(_available_modules),
        'total_errors': len(_import_errors)
    }

def get_import_errors():
    """Get list of import errors."""
    return _import_errors.copy()

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
        
        # At least logger must be available
        if 'get_logger' not in _available_modules:
            raise ImportError("Critical logger module not available")
        
        return True
    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        return False

# Initialize logging on module import (only if logger is available)
if 'setup_logging' in _available_modules:
    try:
        setup_logging()
        module_logger = get_logger(__name__)
        module_logger.info(f"Utils module initialized successfully - Version {__version__}")
        if _import_errors:
            module_logger.warning(f"Utils module initialized with {len(_import_errors)} import errors")
    except Exception as e:
        print(f"Failed to initialize utils module logging: {e}")

# Validate environment on import
if not validate_environment():
    logger.warning("Utils module environment validation failed, but continuing with available modules")

logger.info(f"Utils module loaded with {len(_available_modules)} available utilities")