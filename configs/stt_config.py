"""
STT Configuration Module
========================

Configuration specific to Speech-to-Text evaluation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from .base_config import BaseConfig

@dataclass
class STTConfig(BaseConfig):
    """STT-specific configuration."""
    
    # STT specific settings
    default_language: str = "en"
    max_audio_duration: int = 300  # seconds
    
    # Accuracy thresholds
    min_word_accuracy: float = 0.85
    max_word_error_rate: float = 0.15
    min_confidence_score: float = 0.8
    
    # Supported languages
    supported_languages: List[str] = field(default_factory=lambda: [
        "en", "es", "fr", "de", "hi", "zh"
    ])
    
    # STT evaluation metrics
    stt_metrics: List[str] = field(default_factory=lambda: [
        "word_error_rate", "character_error_rate", "accuracy",
        "latency", "confidence_score", "language_detection"
    ])