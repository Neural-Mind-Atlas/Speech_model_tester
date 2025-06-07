"""
OpenAI TTS/STT client implementation
Provides integration with OpenAI's TTS and Whisper STT services
"""

import openai
import time
import tempfile
import aiofiles
import asyncio
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path

from .base_client import BaseTTSSTTClient, TTSResponse, STTResponse

logger = logging.getLogger(__name__)

class OpenAIClient(BaseTTSSTTClient):
    """Client for OpenAI TTS and Whisper STT services"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OpenAI client
        
        Args:
            config: Configuration dictionary containing:
                - api_key: OpenAI API key
                - organization: Optional organization ID
                - tts_model: TTS model to use (default: tts-1)
                - stt_model: STT model to use (default: whisper-1)
                - default_voice: Default voice for TTS
        """
        # Extract model name from config or use default
        model_name = config.get('model_name', 'openai-speech')
        
        super().__init__(model_name, config)
        
        # Initialize OpenAI client
        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            organization=config.get('organization')
        )
        
        # Model configuration
        self.tts_model = config.get('tts_model', 'tts-1')
        self.stt_model = config.get('stt_model', 'whisper-1')
        
        # Default settings
        self.default_voice = config.get('default_voice', 'alloy')
        self.default_response_format = config.get('default_response_format', 'mp3')
        
        logger.info(f"OpenAI client initialized - TTS: {self.tts_model}, STT: {self.stt_model}")
    
    async def text_to_speech(self, 
                           text: str, 
                           voice: str = "alloy",
                           language: str = "en",
                           **kwargs) -> TTSResponse:
        """
        Convert text to speech using OpenAI TTS
        
        Args:
            text: Text to convert
            voice: Voice name (alloy, echo, fable, onyx, nova, shimmer)
            language: Language code (OpenAI TTS automatically detects)
            **kwargs: Additional parameters
        
        Returns:
            TTSResponse with audio data
        """
        if not self.supports_tts:
            return TTSResponse(
                success=False,
                error_message="TTS not supported by this model configuration"
            )
        
        try:
            # Validate input
            self._validate_text_input(text)
            
            start_time = time.time()
            
            # Prepare parameters
            voice = voice or self.default_voice
            model = kwargs.get('model', self.tts_model)
            response_format = kwargs.get('response_format', self.default_response_format)
            speed = kwargs.get('speed', 1.0)
            
            # Validate voice
            supported_voices = self.get_supported_voices()
            if voice not in supported_voices:
                logger.warning(f"Voice '{voice}' not supported, using default: {self.default_voice}")
                voice = self.default_voice
            
            # Validate speed
            if not (0.25 <= speed <= 4.0):
                logger.warning(f"Speed {speed} out of range [0.25, 4.0], clamping")
                speed = max(0.25, min(4.0, speed))
            
            logger.debug(f"OpenAI TTS request: model={model}, voice={voice}, "
                        f"format={response_format}, speed={speed}")
            
            # Make API call
            response = await self.client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                response_format=response_format,
                speed=speed
            )
            
            latency = time.time() - start_time
            
            # Read audio data
            audio_data = response.content
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=f".{response_format}",
                                           delete=False) as temp_file:
                temp_file.write(audio_data)
                audio_file_path = temp_file.name
            
            # Estimate duration (rough calculation)
            word_count = len(text.split())
            estimated_duration = word_count * 0.6  # ~0.6 seconds per word
            
            return TTSResponse(
                success=True,
                audio_url=audio_file_path,
                latency=latency,
                metadata={
                    'model': model,
                    'voice': voice,
                    'response_format': response_format,
                    'speed': speed,
                    'text_length': len(text),
                    'estimated_duration': estimated_duration,
                    'audio_size_bytes': len(audio_data)
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI TTS failed: {str(e)}")
            return TTSResponse(
                success=False,
                error_message=f"OpenAI TTS error: {str(e)}",
                latency=time.time() - start_time if 'start_time' in locals() else 0
            )
    
    async def speech_to_text(self, 
                           audio_path: str, 
                           language: str = "en",
                           **kwargs) -> STTResponse:
        """
        Convert speech to text using OpenAI Whisper
        
        Args:
            audio_path: Path to audio file
            language: Language code (optional, Whisper auto-detects)
            **kwargs: Additional parameters
        
        Returns:
            STTResponse with transcribed text
        """
        if not self.supports_stt:
            return STTResponse(
                success=False,
                error_message="STT not supported by this model configuration"
            )
        
        try:
            # Validate audio file
            audio_file = Path(audio_path)
            if not audio_file.exists():
                return STTResponse(
                    success=False,
                    error_message=f"Audio file not found: {audio_path}"
                )
            
            start_time = time.time()
            
            # Prepare parameters
            model = kwargs.get('model', self.stt_model)
            response_format = kwargs.get('response_format', 'text')
            temperature = kwargs.get('temperature', 0)
            
            logger.debug(f"OpenAI STT request: model={model}, language={language}, "
                        f"format={response_format}")
            
            # Open and transcribe audio file
            async with aiofiles.open(audio_path, 'rb') as audio_file:
                response = await self.client.audio.transcriptions.create(
                    model=model,
                    file=audio_file,
                    language=language if language != "auto" else None,
                    response_format=response_format,
                    temperature=temperature
                )
            
            latency = time.time() - start_time
            
            # Extract text from response
            if hasattr(response, 'text'):
                transcribed_text = response.text
            else:
                transcribed_text = str(response)
            
            return STTResponse(
                success=True,
                text=transcribed_text,
                latency=latency,
                metadata={
                    'model': model,
                    'language': language,
                    'response_format': response_format,
                    'temperature': temperature,
                    'audio_file': audio_path,
                    'text_length': len(transcribed_text)
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI STT failed: {str(e)}")
            return STTResponse(
                success=False,
                error_message=f"OpenAI STT error: {str(e)}",
                latency=time.time() - start_time if 'start_time' in locals() else 0
            )
    
    def get_supported_voices(self) -> List[str]:
        """
        Get list of supported voices for TTS
        
        Returns:
            List of voice names
        """
        return ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages
        
        Returns:
            List of language codes
        """
        # OpenAI supports many languages, here are the most common ones
        return [
            'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh',
            'ar', 'hi', 'tr', 'pl', 'nl', 'sv', 'da', 'no', 'fi'
        ]
    
    async def health_check(self) -> bool:
        """
        Perform health check for OpenAI services
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            # Test TTS with a simple request
            test_response = await self.text_to_speech(
                text="Health check test",
                voice=self.default_voice
            )
            
            return test_response.success
            
        except Exception as e:
            logger.error(f"OpenAI health check failed: {str(e)}")
            return False