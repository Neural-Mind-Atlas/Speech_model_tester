"""
TTS/STT Testing Framework - Metrics Module (Compatibility Layer)
==============================================================

This module provides backward compatibility for metrics imports.
It re-exports everything from metrics_utils for compatibility.

Author: TTS/STT Testing Framework Team
Version: 1.0.0
"""

# Re-export everything from metrics_utils for compatibility
from .metrics_utils import *

# Ensure all expected classes are available
from .metrics_utils import MetricsCalculator, MetricResult, calculate_wer, calculate_cer, calculate_bleu, calculate_similarity

__all__ = [
    'MetricsCalculator',
    'MetricResult', 
    'calculate_wer',
    'calculate_cer',
    'calculate_bleu',
    'calculate_similarity'
]