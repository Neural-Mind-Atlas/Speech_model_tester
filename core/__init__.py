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

# Import core components
try:
    from .tts_evaluator import TTSEvaluator
    from .stt_evaluator import STTEvaluator
    from .evaluator_factory import EvaluatorFactory
    
    logger.info("Successfully imported all core evaluation components")
    
except ImportError as e:
    logger.error(f"Failed to import core components: {e}")
    raise ImportError(f"Core module initialization failed: {e}")

# Export public interface
__all__ = [
    "TTSEvaluator",
    "STTEvaluator", 
    "EvaluatorFactory",
    "MODULE_INFO",
    "get_module_info",
    "validate_audio_file",
    "get_supported_formats"
]

def get_module_info() -> Dict[str, Any]:
    """
    Get comprehensive information about the core module.
    
    Returns:
        Dict[str, Any]: Module metadata and capabilities
    """
    return MODULE_INFO.copy()

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
logger.info(f"TTS/STT Core Module v{__version__} initialized successfully")
logger.debug(f"Available components: {MODULE_INFO['components']}")
logger.debug(f"Supported formats: {MODULE_INFO['supported_formats']}")