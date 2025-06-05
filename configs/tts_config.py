"""
TTS Configuration Module
========================

Configuration specific to Text-to-Speech evaluation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from .base_config import BaseConfig

@dataclass
class TTSConfig(BaseConfig):
    """TTS-specific configuration."""
    
    # TTS specific settings
    default_voice: str = "default"
    default_sample_rate: int = 22050
    max_text_length: int = 1000
    
    # Quality thresholds
    min_audio_quality: float = 0.7
    max_latency_ms: float = 5000.0
    min_naturalness_score: float = 0.6
    
    # Supported voices
    supported_voices: List[str] = field(default_factory=lambda: [
        "default", "male", "female", "neutral"
    ])
    
    # TTS evaluation metrics
    tts_metrics: List[str] = field(default_factory=lambda: [
        "audio_quality", "naturalness", "intelligibility", 
        "prosody", "voice_consistency", "latency"
    ])