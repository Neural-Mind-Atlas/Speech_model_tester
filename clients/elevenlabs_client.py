"""
ElevenLabs TTS client implementation
Provides integration with ElevenLabs Text-to-Speech API
"""

import io
import json
import time
import wave
from typing import Dict, Any, List, Optional
import logging
import requests

from .base_client import BaseTTSSTTClient, TTSResponse, STTResponse

logger = logging.getLogger(__name__)

class ElevenLabsClient(BaseTTSSTTClient):
    """Client for ElevenLabs Text-to-Speech API"""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        
        # ElevenLabs specific configuration
        self.base_url = config.get('api_base_url', 'https://api.elevenlabs.io/v1')
        self.default_voice_id = config.get('default_voice_id', 'pNInz6obpgDQGcFmaJgB')  # Adam voice
        self.default_model_id = config.get('default_model_id', 'eleven_monolingual_v1')
        
        # API headers
        self.headers = {
            'Accept': 'audio/mpeg',
            'Content-Type': 'application/json',
            'xi-api-key': self.api_key
        }
        
        # Voice settings
        self.default_voice_settings = {
            'stability': 0.5,
            'similarity_boost': 0.5,
            'style': 0.0,
            'use_speaker_boost': True
        }
        
        # ElevenLabs only supports TTS, not STT
        self.supports_tts = True
        self.supports_stt = False
        
        logger.info(f"ElevenLabs client initialized - Model: {model_name}, Base URL: {self.base_url}")
    
    def text_to_speech(self, 
                      text: str, 
                      voice_id: str = None,
                      model_id: str = None,
                      voice_settings: Dict[str, Any] = None,
                      **kwargs) -> TTSResponse:
        """
        Convert text to speech using ElevenLabs TTS
        
        Args:
            text: Text to convert
            voice_id: ElevenLabs voice ID
            model_id: ElevenLabs model ID
            voice_settings: Voice configuration settings
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
            
            # Use defaults if not provided
            voice_id = voice_id or self.default_voice_id
            model_id = model_id or self.default_model_id
            
            # Merge voice settings
            final_voice_settings = self.default_voice_settings.copy()
            if voice_settings:
                final_voice_settings.update(voice_settings)
            
            # Prepare request payload
            payload = {
                'text': text,
                'model_id': model_id,
                'voice_settings': final_voice_settings
            }
            
            # Optional parameters
            if 'previous_text' in kwargs:
                payload['previous_text'] = kwargs['previous_text']
            if 'next_text' in kwargs:
                payload['next_text'] = kwargs['next_text']
            
            # API endpoint
            url = f"{self.base_url}/text-to-speech/{voice_id}"
            
            logger.debug(f"ElevenLabs TTS request: voice_id={voice_id}, model_id={model_id}, text_length={len(text)}")
            
            # Make API request
            response = requests.post(
                url,
                json=payload,
                headers=self.headers,
                timeout=kwargs.get('timeout', 30)
            )
            
            latency = time.time() - start_time
            
            if response.status_code == 200:
                audio_data = response.content
                
                # ElevenLabs returns MP3 by default
                audio_format = 'mp3'
                sample_rate = 22050  # ElevenLabs default
                
                # Calculate duration (approximate for MP3)
                duration = self._estimate_mp3_duration(audio_data)
                
                metadata = {
                    'provider': 'elevenlabs',
                    'model': model_id,
                    'voice_id': voice_id,
                    'voice_settings': final_voice_settings,
                    'audio_size': len(audio_data),
                    'sample_rate': sample_rate,
                    'response_headers': dict(response.headers)
                }
                
                return TTSResponse(
                    success=True,
                    audio_data=audio_data,
                    audio_format=audio_format,
                    sample_rate=sample_rate,
                    duration=duration,
                    latency=latency,
                    metadata=metadata
                )
            else:
                error_msg = f"ElevenLabs API error: {response.status_code}"
                try:
                    error_details = response.json()
                    error_msg += f" - {error_details.get('detail', {}).get('message', 'Unknown error')}"
                except:
                    error_msg += f" - {response.text}"
                
                logger.error(error_msg)
                return TTSResponse(
                    success=False,
                    error_message=error_msg,
                    latency=latency
                )
        
        except Exception as e:
            logger.error(f"ElevenLabs TTS error: {str(e)}")
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
        ElevenLabs does not support Speech-to-Text
        
        Returns:
            STTResponse indicating STT is not supported
        """
        return STTResponse(
            success=False,
            error_message="Speech-to-Text is not supported by ElevenLabs"
        )
    
    def get_available_models(self, model_type: str) -> List[str]:
        """
        Get available models for the specified type
        
        Args:
            model_type: 'tts' or 'stt'
        
        Returns:
            List of available model names
        """
        if model_type == 'tts':
            try:
                url = f"{self.base_url}/models"
                response = requests.get(url, headers={'xi-api-key': self.api_key})
                
                if response.status_code == 200:
                    models_data = response.json()
                    return [model['model_id'] for model in models_data]
                else:
                    logger.warning(f"Failed to fetch ElevenLabs models: {response.status_code}")
                    # Return default models
                    return [
                        'eleven_monolingual_v1',
                        'eleven_multilingual_v1',
                        'eleven_multilingual_v2',
                        'eleven_turbo_v2'
                    ]
            except Exception as e:
                logger.error(f"Error fetching ElevenLabs models: {str(e)}")
                return ['eleven_monolingual_v1']
        
        elif model_type == 'stt':
            return []  # ElevenLabs doesn't support STT
        
        return []
    
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """
        Get available voices from ElevenLabs
        
        Returns:
            List of voice information dictionaries
        """
        try:
            url = f"{self.base_url}/voices"
            response = requests.get(url, headers={'xi-api-key': self.api_key})
            
            if response.status_code == 200:
                voices_data = response.json()
                return voices_data.get('voices', [])
            else:
                logger.warning(f"Failed to fetch ElevenLabs voices: {response.status_code}")
                return []
        
        except Exception as e:
            logger.error(f"Error fetching ElevenLabs voices: {str(e)}")
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check for ElevenLabs API
        
        Returns:
            Health check results
        """
        try:
            start_time = time.time()
            
            # Check user info endpoint
            url = f"{self.base_url}/user"
            response = requests.get(
                url, 
                headers={'xi-api-key': self.api_key},
                timeout=10
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                user_data = response.json()
                return {
                    'healthy': True,
                    'response_time': response_time,
                    'api_status': 'operational',
                    'user_info': {
                        'character_count': user_data.get('subscription', {}).get('character_count', 0),
                        'character_limit': user_data.get('subscription', {}).get('character_limit', 0),
                        'tier': user_data.get('subscription', {}).get('tier', 'unknown')
                    }
                }
            else:
                return {
                    'healthy': False,
                    'response_time': response_time,
                    'error': f"API returned status {response.status_code}"
                }
        
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e)
            }
    
    def _estimate_mp3_duration(self, audio_data: bytes) -> float:
        """
        Estimate MP3 duration (rough calculation)
        
        Args:
            audio_data: MP3 audio data
        
        Returns:
            Estimated duration in seconds
        """
        try:
            # Very rough estimation: assume 128kbps MP3
            # This is not accurate but provides an estimate
            bitrate = 128000  # bits per second
            file_size_bits = len(audio_data) * 8
            duration = file_size_bits / bitrate
            return max(0.1, duration)  # Minimum 0.1 seconds
        except:
            return 1.0  # Default fallback
    
    def _validate_text_input(self, text: str) -> None:
        """
        Validate text input for ElevenLabs TTS
        
        Args:
            text: Input text to validate
        
        Raises:
            ValueError: If text is invalid
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        if len(text) > 5000:  # ElevenLabs character limit
            raise ValueError(f"Text too long: {len(text)} characters (max 5000)")
        
        # Check for unsupported characters (basic validation)
        if any(ord(char) > 65535 for char in text):
            raise ValueError("Text contains unsupported Unicode characters")