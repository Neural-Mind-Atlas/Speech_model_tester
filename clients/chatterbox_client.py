"""
Chatterbox TTS/STT client implementation  
Provides integration with Chatterbox speech services
"""

import requests
import time
import json
from typing import Dict, Any, List
import logging
from io import BytesIO

from .base_client import BaseTTSSTTClient, TTSResponse, STTResponse

logger = logging.getLogger(__name__)

class ChatterboxClient(BaseTTSSTTClient):
    """Client for Chatterbox TTS/STT services"""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        
        # Chatterbox API configuration
        self.base_url = config.get('base_url', 'https://api.chatterbox.ai')
        self.tts_endpoint = f"{self.base_url}/v1/text-to-speech"
        self.stt_endpoint = f"{self.base_url}/v1/speech-to-text"
        
        # Default settings
        self.default_voice = config.get('default_voice', 'natural')
        self.default_quality = config.get('default_quality', 'standard')
        
        logger.info(f"Chatterbox client initialized with model: {model_name}")
    
    def text_to_speech(self, 
                      text: str, 
                      voice: str = "natural",
                      language: str = "en",
                      **kwargs) -> TTSResponse:
        """
        Convert text to speech using Chatterbox TTS
        
        Args:
            text: Text to convert
            voice: Voice identifier
            language: Language code
            **kwargs: Additional parameters (quality, speed, emotion, etc.)
            
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
            
            # Prepare request headers
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                'User-Agent': 'TTS-STT-Framework/1.0'
            }
            
            # Prepare payload
            payload = {
                'text': text,
                'voice': voice or self.default_voice,
                'language': language,
                'model': self.model_name,
                'quality': kwargs.get('quality', self.default_quality),
                'format': kwargs.get('format', 'wav'),
                'sample_rate': kwargs.get('sample_rate', 22050)
            }
            
            # Add optional parameters
            optional_params = ['speed', 'pitch', 'emotion', 'style', 'stability']
            for param in optional_params:
                if param in kwargs:
                    payload[param] = kwargs[param]
            
            logger.debug(f"Chatterbox TTS request: voice={voice}, language={language}, "
                        f"text_length={len(text)}")
            
            # Make API call with retry logic
            response = self._retry_with_backoff(
                self._make_tts_request,
                headers,
                payload
            )
            
            latency = time.time() - start_time
            
            if response.status_code == 200:
                # Check if response is JSON (error) or binary (audio)
                content_type = response.headers.get('Content-Type', '')
                
                if 'application/json' in content_type:
                    # Error response
                    error_data = response.json()
                    error_msg = error_data.get('error', 'Unknown error')
                    logger.error(f"Chatterbox TTS error: {error_msg}")
                    return TTSResponse(
                        success=False,
                        error_message=error_msg,
                        latency=latency
                    )
                else:
                    # Audio response
                    audio_data = response.content
                    
                    # Extract metadata from headers
                    metadata = {
                        'provider': 'chatterbox',
                        'model': self.model_name,
                        'voice': voice,
                        'language': language,
                        'quality': payload['quality'],
                        'request_id': response.headers.get('X-Request-ID'),
                        'audio_size': len(audio_data),
                        'generation_time': response.headers.get('X-Generation-Time')
                    }
                    
                    # Extract duration from headers if available
                    duration = float(response.headers.get('X-Audio-Duration', 0))
                    if duration == 0:
                        # Estimate duration based on text length
                        duration = len(text.split()) * 0.6  # ~0.6 seconds per word
                    
                    return TTSResponse(
                        success=True,
                        audio_data=audio_data,
                        audio_format=payload['format'],
                        sample_rate=payload['sample_rate'],
                        duration=duration,
                        latency=latency,
                        metadata=metadata
                    )
            else:
                error_msg = f"Chatterbox TTS API error: {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg += f" - {error_data.get('error', response.text)}"
                except:
                    error_msg += f" - {response.text}"
                
                logger.error(error_msg)
                return TTSResponse(
                    success=False,
                    error_message=error_msg,
                    latency=latency
                )
                
        except Exception as e:
            logger.error(f"Chatterbox TTS error: {str(e)}")
            return TTSResponse(
                success=False,
                error_message=str(e)
            )
    
    def speech_to_text(self, 
                      audio_data: bytes,
                      audio_format: str = "wav",
                      language: str = "en",
                      **kwargs) -> STTResponse:
        """
        Convert speech to text using Chatterbox STT
        
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
            
            # Prepare request headers
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'User-Agent': 'TTS-STT-Framework/1.0'
            }
            
            # Prepare multipart form data
            files = {
                'audio': (f'audio.{audio_format}', BytesIO(audio_data), f'audio/{audio_format}')
            }
            
            data = {
                'language': language,
                'model': self.model_name,
                'format': audio_format,
                'enable_word_timestamps': kwargs.get('enable_word_timestamps', True),
                'enable_confidence_scores': kwargs.get('enable_confidence_scores', True),
                'noise_reduction': kwargs.get('noise_reduction', True)
            }
            
            # Add optional parameters
            if 'vocabulary' in kwargs:
                data['vocabulary'] = json.dumps(kwargs['vocabulary'])
            if 'speaker_diarization' in kwargs:
                data['speaker_diarization'] = kwargs['speaker_diarization']
            
            logger.debug(f"Chatterbox STT request: language={language}, "
                        f"audio_size={len(audio_data)} bytes")
            
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
                audio_duration = result.get('audio_duration', processing_time)
                
                # Calculate RTF
                rtf = processing_time / audio_duration if audio_duration > 0 else 0
                
                # Extract metadata
                metadata = {
                    'provider': 'chatterbox',
                    'model': self.model_name,
                    'language': language,
                    'audio_duration': audio_duration,
                    'request_id': response.headers.get('X-Request-ID'),
                    'processing_info': result.get('processing_info', {})
                }
                
                # Extract word-level timestamps
                word_timestamps = result.get('word_timestamps', [])
                
                # Handle speaker diarization if present
                if 'speakers' in result:
                    metadata['speaker_info'] = result['speakers']
                
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
                error_msg = f"Chatterbox STT API error: {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg += f" - {error_data.get('error', response.text)}"
                except:
                    error_msg += f" - {response.text}"
                
                logger.error(error_msg)
                return STTResponse(
                    success=False,
                    error_message=error_msg,
                    processing_time=processing_time
                )
                
        except Exception as e:
            logger.error(f"Chatterbox STT error: {str(e)}")
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
            timeout=45,
            stream=True
        )
        return response
    
    def _make_stt_request(self, headers: Dict[str, str], files: Dict, data: Dict[str, Any]) -> requests.Response:
        """Make STT API request"""
        response = requests.post(
            self.stt_endpoint,
            headers=headers,
            files=files,
            data=data,
            timeout=90
        )
        return response
    
    def get_supported_voices(self) -> List[str]:
        """Get Chatterbox supported voices"""
        return self.config.get('supported_voices', [
            'natural',     # Natural voice
            'professional', # Professional tone
            'friendly',    # Friendly tone
            'authoritative', # Authoritative tone
            'calm',        # Calm voice
            'energetic',   # Energetic voice
            'narrative'    # Narrative style
        ])
    
    def get_supported_languages(self) -> List[str]:
        """Get Chatterbox supported languages"""
        return self.config.get('supported_languages', [
            'en',    # English
            'es',    # Spanish
            'fr',    # French
            'de',    # German
            'it',    # Italian
            'pt',    # Portuguese
            'nl',    # Dutch
            'ja',    # Japanese
            'ko',    # Korean
            'zh',    # Chinese
            'hi',    # Hindi
            'ar'     # Arabic
        ])
    
    def health_check(self) -> bool:
        """Check Chatterbox service health"""
        try:
            # Check service status endpoint
            health_url = f"{self.base_url}/v1/health"
            headers = {'Authorization': f'Bearer {self.api_key}'}
            
            response = requests.get(health_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                return health_data.get('status') == 'healthy'
            
            return False
            
        except Exception as e:
            logger.error(f"Chatterbox health check failed: {str(e)}")
            return False