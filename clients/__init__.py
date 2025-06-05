"""
TTS/STT Testing Framework - Clients Module
Provides unified interface for various TTS/STT service providers
"""

from .base_client import BaseTTSSTTClient
from .sarvam_client import SarvamClient
from .chatterbox_client import ChatterboxClient
from .openai_client import OpenAIClient
from .azure_client import AzureClient
from .google_client import GoogleClient
from .client_factory import ClientFactory

__all__ = [
    'BaseTTSSTTClient',
    'SarvamClient',
    'ChatterboxClient',
    'OpenAIClient',
    'AzureClient',
    'GoogleClient',
    'ClientFactory'
]

__version__ = "1.0.0"