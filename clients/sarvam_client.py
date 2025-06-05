"""
Sarvam AI TTS/STT client implementation
Provides integration with Sarvam's multilingual speech services
"""

import requests
import time
from typing import Dict, Any, List
import logging
from io import BytesIO

from .base_client import BaseTTSSTTClient, TTSResponse, STTResponse

logger = logging.getLogger(__name__)

class SarvamClient(BaseTTSSTTClient):
    """Client for Sarvam AI TTS/STT services"""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        
        # Sarvam API endpoints
        self.tts_endpoint = config.get('tts_endpoint', 'https://api.sarvam.ai/text-to-speech')
        self.stt_endpoint = config.get('stt_endpoint', 'https://api.sarvam.ai/speech-to-text')
        
        # Default parameters
        self.default_voice = config.get('default_voice', 'meera')
        self.default_sample_rate = config.get('default_sample_rate', 22050)
        
        logger.info(f"Sarvam client initialized with model: {model_name}")
    
    def text_to_speech(self, 
                      text: str, 
                      voice: str = "meera",
                      language: str = "hi",
                      **kwargs) -> TTSResponse:
        """
        Convert text to speech using Sarvam TTS
        
        Args:
            text: Text to convert
            voice: Voice name (meera, arjun, etc.)
            language: Language code (hi, en, ta, etc.)
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
            
            # Prepare request
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'text': text,
                'voice': voice or self.default_voice,
                'language': language,
                'sample_rate': kwargs.get('sample_rate', self.default_sample_rate),
                'format': kwargs.get('format', 'wav')
            }
            
            # Add model-specific parameters
            if 'speed' in kwargs:
                payload['speed'] = kwargs['speed']
            if 'pitch' in kwargs:
                payload['pitch'] = kwargs['pitch']
                
            logger.debug(f"Sarvam TTS request: {payload}")
            
            # Make API call with retry logic
            response = self._retry_with_backoff(
                self._make_tts_request, 
                headers, 
                payload
            )
            
            latency = time.time() - start_time
            
            if response.status_code == 200:
                audio_data = response.content
                
                # Extract metadata from response headers
                metadata = {
                    'provider': 'sarvam',
                    'model': self.model_name,
                    'voice': voice,
                    'language': language,
                    'request_id': response.headers.get('X-Request-ID'),
                    'audio_size': len(audio_data)
                }
                
                # Estimate duration (approximate)
                estimated_duration = len(text) * 0.1  # Rough estimate: 0.1s per character
                
                return TTSResponse(
                    success=True,
                    audio_data=audio_data,
                    audio_format=payload['format'],
                    sample_rate=payload['sample_rate'],
                    duration=estimated_duration,
                    latency=latency,
                    metadata=metadata
                )
            else:
                error_msg = f"Sarvam TTS API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return TTSResponse(
                    success=False,
                    error_message=error_msg,
                    latency=latency
                )
                
        except Exception as e:
            logger.error(f"Sarvam TTS error: {str(e)}")
            return TTSResponse(
                success=False,
                error_message=str(e)
            )
    
    def speech_to_text(self, 
                      audio_data: bytes,
                      audio_format: str = "wav",
                      language: str = "hi",
                      **kwargs) -> STTResponse:
        """
        Convert speech to text using Sarvam STT
        
        Args:
            audio_data: Audio data in bytes
            audio_format: Audio format
            language: Language code
            **kwargs: Additional parameters
            
        Returns:
            STTResponse with transcript
        """
        if not self.supports_stt:
            return STTResponse(
                success=False,
                error_message="STT not supported by this model configuration"
            )
        
        try:
            # Validate input
            self._validate_audio_input(audio_data)
            
            start_time = time.time()
            
            # Prepare request
            headers = {
                'Authorization': f'Bearer {self.api_key}'
            }
            
            files = {
                'audio': (f'audio.{audio_format}', BytesIO(audio_data), f'audio/{audio_format}')
            }
            
            data = {
                'language': language,
                'format': audio_format,
                'enable_timestamps': kwargs.get('enable_timestamps', False),
                'model': self.model_name
            }
            
            logger.debug(f"Sarvam STT request for {len(audio_data)} bytes of audio")
            
            # Make API call with retry logic
            response = self._retry_with_backoff(
                self._make_stt_request,
                headers,
                files,
                data
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                transcript = result.get('transcript', '')
                confidence = result.get('confidence', 0.0)
                
                # Calculate RTF (Real-time Factor)
                audio_duration = result.get('audio_duration', processing_time)
                rtf = processing_time / audio_duration if audio_duration > 0 else 0
                
                metadata = {
                    'provider': 'sarvam',
                    'model': self.model_name,
                    'language': language,
                    'audio_duration': audio_duration,
                    'request_id': response.headers.get('X-Request-ID')
                }
                
                # Extract word timestamps if available
                word_timestamps = result.get('word_timestamps', [])
                
                return STTResponse(
                    success=True,
                    transcript=transcript,
                    confidence=confidence,
                    processing_time=processing_time,
                    rtf=rtf,
                    word_timestamps=word_timestamps,
                    metadata=metadata
                )
            else:
                error_msg = f"Sarvam STT API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return STTResponse(
                    success=False,
                    error_message=error_msg,
                    processing_time=processing_time
                )
                
        except Exception as e:
            logger.error(f"Sarvam STT error: {str(e)}")
            return STTResponse(
                success=False,
                error_message=str(e)
            )
    
    def _make_tts_request(self, headers: Dict[str, str], payload: Dict[str, Any]) -> requests.Response:
        """Make TTS API request"""
        response = requests.post(
            self.tts_endpoint,
            headers=headers,
            json=payload,
            timeout=30
        )
        return response
    
    def _make_stt_request(self, headers: Dict[str, str], files: Dict, data: Dict[str, Any]) -> requests.Response:
        """Make STT API request"""
        response = requests.post(
            self.stt_endpoint,
            headers=headers,
            files=files,
            data=data,
            timeout=60
        )
        return response
    
    def get_supported_voices(self) -> List[str]:
        """Get Sarvam supported voices"""
        return self.config.get('supported_voices', [
            'meera',    # Female Hindi
            'arjun',    # Male Hindi  
            'kavya',    # Female Tamil
            'arun',     # Male Tamil
            'sarah',    # Female English
            'david'     # Male English
        ])
    
    def get_supported_languages(self) -> List[str]:
        """Get Sarvam supported languages"""
        return self.config.get('supported_languages', [
            'hi',  # Hindi
            'en',  # English
            'ta',  # Tamil
            'te',  # Telugu
            'kn',  # Kannada
            'ml',  # Malayalam
            'gu',  # Gujarati
            'bn',  # Bengali
            'or',  # Odia
            'pa'   # Punjabi
        ])
    
    def health_check(self) -> bool:
        """Check Sarvam service health"""
        try:
            # Simple TTS test with minimal text
            if self.supports_tts:
                test_response = self.text_to_speech("test", voice="meera", language="hi")
                return test_response.success
            elif self.supports_stt:
                # For STT, we'd need test audio - return True if we have valid credentials
                return bool(self.api_key)
            return True
        except Exception as e:
            logger.error(f"Sarvam health check failed: {str(e)}")
            return False