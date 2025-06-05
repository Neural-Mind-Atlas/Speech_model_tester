"""
Abstract base client interface for TTS/STT services
Defines the contract that all client implementations must follow
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass
import logging

# Configure logger
logger = logging.getLogger(__name__)

@dataclass
class TTSResponse:
    """Standardized TTS response format"""
    success: bool
    audio_data: Optional[bytes] = None
    audio_format: str = "wav"
    sample_rate: int = 22050
    duration: float = 0.0
    latency: float = 0.0
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass 
class STTResponse:
    """Standardized STT response format"""
    success: bool
    transcript: Optional[str] = None
    confidence: float = 0.0
    processing_time: float = 0.0
    rtf: float = 0.0  # Real-time factor
    error_message: Optional[str] = None
    word_timestamps: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None

class BaseTTSSTTClient(ABC):
    """
    Abstract base class for TTS/STT service clients
    All provider implementations must inherit from this class
    """
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        Initialize the client with model configuration
        
        Args:
            model_name: Name of the model to use
            config: Configuration dictionary containing model settings
        """
        self.model_name = model_name
        self.config = config
        self.provider = config.get('provider', 'unknown')
        self.api_key = self._get_api_key()
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1.0)
        
        # Service capabilities
        self.supports_tts = config.get('supports_tts', False)
        self.supports_stt = config.get('supports_stt', False)
        
        logger.info(f"Initialized {self.provider} client for model: {model_name}")
        logger.debug(f"TTS Support: {self.supports_tts}, STT Support: {self.supports_stt}")
        
    def _get_api_key(self) -> str:
        """
        Retrieve API key from configuration or environment variables
        
        Returns:
            API key string
            
        Raises:
            ValueError: If API key is not found
        """
        # Check config first
        if 'api_key' in self.config:
            return self.config['api_key']
            
        # Check environment variables
        api_key_env = self.config.get('api_key_env')
        if api_key_env:
            api_key = os.getenv(api_key_env)
            if api_key:
                return api_key
                
        # Try provider-specific environment variable
        provider_env = f"{self.provider.upper()}_API_KEY"
        api_key = os.getenv(provider_env)
        if api_key:
            return api_key
            
        raise ValueError(f"API key not found for {self.provider}. "
                        f"Set {api_key_env or provider_env} environment variable or provide in config")
    
    @abstractmethod
    def text_to_speech(self, 
                      text: str, 
                      voice: str = "default",
                      language: str = "en",
                      **kwargs) -> TTSResponse:
        """
        Convert text to speech
        
        Args:
            text: Text to convert to speech
            voice: Voice identifier/name
            language: Language code (e.g., 'en', 'hi', 'ta')
            **kwargs: Additional provider-specific parameters
            
        Returns:
            TTSResponse object containing audio data and metadata
        """
        pass
    
    @abstractmethod
    def speech_to_text(self, 
                      audio_data: bytes,
                      audio_format: str = "wav",
                      language: str = "en",
                      **kwargs) -> STTResponse:
        """
        Convert speech to text
        
        Args:
            audio_data: Audio data in bytes
            audio_format: Audio format (wav, mp3, etc.)
            language: Language code (e.g., 'en', 'hi', 'ta')
            **kwargs: Additional provider-specific parameters
            
        Returns:
            STTResponse object containing transcript and metadata
        """
        pass
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """
        Execute function with exponential backoff retry logic
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retries are exhausted
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed for {self.provider}: {str(e)}. "
                                 f"Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed for {self.provider}")
                    break
        
        raise last_exception
    
    def _validate_text_input(self, text: str) -> None:
        """
        Validate text input for TTS
        
        Args:
            text: Input text
            
        Raises:
            ValueError: If text is invalid
        """
        if not text or not text.strip():
            raise ValueError("Text input cannot be empty")
            
        max_length = self.config.get('max_text_length', 5000)
        if len(text) > max_length:
            raise ValueError(f"Text length ({len(text)}) exceeds maximum allowed ({max_length})")
    
    def _validate_audio_input(self, audio_data: bytes) -> None:
        """
        Validate audio input for STT
        
        Args:
            audio_data: Audio data in bytes
            
        Raises:
            ValueError: If audio data is invalid
        """
        if not audio_data:
            raise ValueError("Audio data cannot be empty")
            
        max_size = self.config.get('max_audio_size', 25 * 1024 * 1024)  # 25MB default
        if len(audio_data) > max_size:
            raise ValueError(f"Audio size ({len(audio_data)} bytes) exceeds maximum allowed ({max_size} bytes)")
    
    def get_supported_voices(self) -> List[str]:
        """
        Get list of supported voices for TTS
        
        Returns:
            List of supported voice identifiers
        """
        return self.config.get('supported_voices', ['default'])
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages
        
        Returns:
            List of supported language codes
        """
        return self.config.get('supported_languages', ['en'])
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get client capabilities and limitations
        
        Returns:
            Dictionary containing capability information
        """
        return {
            'provider': self.provider,
            'model': self.model_name,
            'supports_tts': self.supports_tts,
            'supports_stt': self.supports_stt,
            'supported_voices': self.get_supported_voices(),
            'supported_languages': self.get_supported_languages(),
            'max_text_length': self.config.get('max_text_length', 5000),
            'max_audio_size': self.config.get('max_audio_size', 25 * 1024 * 1024),
            'supported_audio_formats': self.config.get('supported_audio_formats', ['wav', 'mp3'])
        }
    
    def health_check(self) -> bool:
        """
        Perform health check to verify service availability
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            # Basic connectivity test - can be overridden by implementations
            return True
        except Exception as e:
            logger.error(f"Health check failed for {self.provider}: {str(e)}")
            return False