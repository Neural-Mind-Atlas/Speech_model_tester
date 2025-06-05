"""
TTS/STT Testing Framework - Core Module
=====================================

This module contains the core evaluation functionality for Text-to-Speech (TTS) 
and Speech-to-Text (STT) models. It provides comprehensive testing capabilities
with robust error handling, detailed logging, and configurable evaluation metrics.

Author: AI Testing Team
Version: 1.0.0
Created: 2024-01-01
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

# Configure module-level logging
logger = logging.getLogger(__name__)

# Version information
__version__ = "1.0.0"
__author__ = "AI Testing Team"
__email__ = "testing@company.com"

# Module metadata
MODULE_INFO = {
    "name": "TTS-STT Core Evaluation Module",
    "version": __version__,
    "description": "Core evaluation functionality for TTS and STT models",
    "components": [
        "tts_evaluator",
        "stt_evaluator", 
        "evaluator_factory"
    ],
    "supported_formats": ["wav", "mp3", "flac", "ogg"],
    "supported_metrics": [
        "accuracy", "latency", "quality", "naturalness", 
        "word_error_rate", "character_error_rate"
    ]
}

# Import core components with graceful error handling
_import_errors = []

try:
    from .tts_evaluator import TTSEvaluator
    logger.info("Successfully imported TTSEvaluator")
except ImportError as e:
    logger.warning(f"Failed to import TTSEvaluator: {e}")
    _import_errors.append(('TTSEvaluator', str(e)))
    TTSEvaluator = None

try:
    from .stt_evaluator import STTEvaluator
    logger.info("Successfully imported STTEvaluator")
except ImportError as e:
    logger.warning(f"Failed to import STTEvaluator: {e}")
    _import_errors.append(('STTEvaluator', str(e)))
    STTEvaluator = None

try:
    from .evaluator_factory import EvaluatorFactory
    logger.info("Successfully imported EvaluatorFactory")
except ImportError as e:
    logger.warning(f"Failed to import EvaluatorFactory: {e}")
    _import_errors.append(('EvaluatorFactory', str(e)))
    EvaluatorFactory = None

# Log successful imports
if not _import_errors:
    logger.info("Successfully imported all core evaluation components")
else:
    logger.warning(f"Some core components failed to import: {[err[0] for err in _import_errors]}")

# Export available components only
__all__ = []
if TTSEvaluator is not None:
    __all__.append('TTSEvaluator')
if STTEvaluator is not None:
    __all__.append('STTEvaluator')
if EvaluatorFactory is not None:
    __all__.append('EvaluatorFactory')

__all__.extend([
    "MODULE_INFO",
    "get_module_info",
    "validate_audio_file",
    "get_supported_formats",
    "get_import_errors"
])

def get_module_info() -> Dict[str, Any]:
    """
    Get comprehensive information about the core module.
    
    Returns:
        Dict[str, Any]: Module metadata and capabilities
    """
    info = MODULE_INFO.copy()
    info['available_components'] = __all__
    info['import_errors'] = _import_errors
    return info

def get_import_errors() -> List[Tuple[str, str]]:
    """
    Get list of import errors that occurred during module initialization.
    
    Returns:
        List[Tuple[str, str]]: List of (component_name, error_message) tuples
    """
    return _import_errors.copy()

def validate_audio_file(file_path: str) -> bool:
    """
    Validate if an audio file exists and has a supported format.
    
    Args:
        file_path (str): Path to the audio file
        
    Returns:
        bool: True if file is valid, False otherwise
    """
    try:
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            logger.warning(f"Audio file not found: {file_path}")
            return False
            
        # Check file extension
        extension = path.suffix.lower().lstrip('.')
        if extension not in MODULE_INFO["supported_formats"]:
            logger.warning(f"Unsupported audio format: {extension}")
            return False
            
        # Check file size (must be > 0)
        if path.stat().st_size == 0:
            logger.warning(f"Audio file is empty: {file_path}")
            return False
            
        logger.debug(f"Audio file validation successful: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error validating audio file {file_path}: {e}")
        return False

def get_supported_formats() -> List[str]:
    """
    Get list of supported audio formats.
    
    Returns:
        List[str]: List of supported audio file extensions
    """
    return MODULE_INFO["supported_formats"].copy()

# Module initialization logging
logger.info(f"TTS/STT Core Module v{__version__} initialized")
logger.debug(f"Available components: {[comp for comp in __all__ if not comp.startswith('_')]}")
logger.debug(f"Supported formats: {MODULE_INFO['supported_formats']}")

if _import_errors:
    logger.warning(f"Module initialized with {len(_import_errors)} import errors")