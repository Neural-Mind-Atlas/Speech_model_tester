# """
# OpenAI TTS/STT client implementation
# Provides integration with OpenAI's TTS and Whisper STT services
# """

# # import openai
# # import time
# # from typing import Dict, Any, List
# # import logging
# # from io import BytesIO

# # from .base_client import BaseTTSSTTClient, TTSResponse, STTResponse

# # logger = logging.getLogger(__name__)

# # class OpenAIClient(BaseTTSSTTClient):
# #     """Client for OpenAI TTS and Whisper STT services"""
    
# #     def __init__(self, model_name: str, config: Dict[str, Any]):
# #         super().__init__(model_name, config)
        
# #         # Initialize OpenAI client
# #         self.client = openai.OpenAI(api_key=self.api_key)
        
# #         # Model configuration
# #         self.tts_model = config.get('tts_model', 'tts-1')
# #         self.stt_model = config.get('stt_model', 'whisper-1')
        
# #         # Default settings
# #         self.default_voice = config.get('default_voice', 'alloy')
# #         self.default_response_format = config.get('default_response_format', 'mp3')
        
# #         logger.info(f"OpenAI client initialized - TTS: {self.tts_model}, STT: {self.stt_model}")

# """
# OpenAI TTS/STT client implementation
# Provides integration with OpenAI's speech services including Whisper and TTS
# """

# import openai
# import time
# import tempfile
# import os
# from typing import Dict, Any, List, Optional
# import logging
# import aiofiles
# import asyncio

# from .base_client import BaseTTSSTTClient, TTSResponse, STTResponse

# logger = logging.getLogger(__name__)

# class OpenAIClient(BaseTTSSTTClient):  # Make sure this is the correct class name
#     """Client for OpenAI Speech Services (TTS and Whisper STT)"""
    
#     def __init__(self, model_name: str, config: Dict[str, Any]):
#         super().__init__(model_name, config)
        
#         # Initialize OpenAI client
#         self.client = openai.AsyncOpenAI(
#             api_key=self.api_key,
#             organization=config.get('organization')
#         )
        
#         # Model configurations
#         self.tts_models = config.get('tts_models', ['tts-1', 'tts-1-hd'])
#         self.stt_models = config.get('stt_models', ['whisper-1'])
        
#         # Default settings
#         self.default_voice = config.get('default_voice', 'alloy')
#         self.default_tts_model = config.get('default_tts_model', 'tts-1')
#         self.default_stt_model = config.get('default_stt_model', 'whisper-1')
        
#         logger.info(f"OpenAI client initialized - Model: {model_name}")
    
#     def text_to_speech(self, 
#                       text: str, 
#                       voice: str = "alloy",
#                       language: str = "en",
#                       **kwargs) -> TTSResponse:
#         """
#         Convert text to speech using OpenAI TTS
        
#         Args:
#             text: Text to convert
#             voice: Voice name (alloy, echo, fable, onyx, nova, shimmer)
#             language: Language code (OpenAI TTS automatically detects)
#             **kwargs: Additional parameters
            
#         Returns:
#             TTSResponse with audio data
#         """
#         if not self.supports_tts:
#             return TTSResponse(
#                 success=False,
#                 error_message="TTS not supported by this model configuration"
#             )
        
#         try:
#             # Validate input
#             self._validate_text_input(text)
            
#             start_time = time.time()
            
#             # Prepare parameters
#             voice = voice or self.default_voice
#             response_format = kwargs.get('response_format', self.default_response_format)
#             speed = kwargs.get('speed', 1.0)
            
#             # Validate voice
#             if voice not in self.get_supported_voices():
#                 logger.warning(f"Voice '{voice}' not supported, using default: {self.default_voice}")
#                 voice = self.default_voice
            
#             # Validate speed
#             if not (0.25 <= speed <= 4.0):
#                 logger.warning(f"Speed {speed} out of range [0.25, 4.0], clamping")
#                 speed = max(0.25, min(4.0, speed))
            
#             logger.debug(f"OpenAI TTS request: model={self.tts_model}, voice={voice}, "
#                         f"format={response_format}, speed={speed}")
            
#             # Make API call with retry logic
#             response = self._retry_with_backoff(
#                 self._make_tts_request,
#                 text,
#                 voice,
#                 response_format,
#                 speed
#             )
            
#             latency = time.time() - start_time
            
#             # Read audio data
#             audio_data = response.content
            
#             # Estimate duration (OpenAI doesn't provide this directly)
#             # Rough estimate: ~150 words per minute
#             word_count = len(text.split())
#             estimated_duration = (word_count / 150) * 60  # Convert to seconds
            
#             metadata = {
#                 'provider': 'openai',
#                 'model': self.tts_model,
#                 'voice': voice,
#                 'response_format': response_format,
#                 'speed': speed,
#                 'audio_size': len(audio_data),
#                 'estimated_word_count': word_count
#             }
            
#             return TTSResponse(
#                 success=True,
#                 audio_data=audio_data,
#                 audio_format=response_format,
#                 sample_rate=22050 if response_format == 'wav' else 24000,  # OpenAI defaults
#                 duration=estimated_duration,
#                 latency=latency,
#                 metadata=metadata
#             )
            
#         except openai.APIError as e:
#             logger.error(f"OpenAI TTS API error: {str(e)}")
#             return TTSResponse(
#                 success=False,
#                 error_message=f"OpenAI API error: {str(e)}"
#             )
#         except Exception as e:
#             logger.error(f"OpenAI TTS error: {str(e)}")
#             return TTSResponse(
#                 success=False,
#                 error_message=str(e)
#             )
    
#     def speech_to_text(self, 
#                       audio_data: bytes,
#                       audio_format: str = "wav",
#                       language: str = "en",
#                       **kwargs) -> STTResponse:
#         """
#         Convert speech to text using OpenAI Whisper
        
#         Args:
#             audio_data: Audio data in bytes
#             audio_format: Audio format
#             language: Language code (optional, Whisper can auto-detect)
#             **kwargs: Additional parameters
            
#         Returns:
#             STTResponse with transcript
#         """
#         if not self.supports_stt:
#             return STTResponse(
#                 success=False,
#                 error_message="STT not supported by this model configuration"
#             )
        
#         try:
#             # Validate input
#             self._validate_audio_input(audio_data)
            
#             start_time = time.time()
            
#             # Prepare audio file object
#             audio_file = BytesIO(audio_data)
#             audio_file.name = f"audio.{audio_format}"
            
#             # Prepare parameters
#             response_format = kwargs.get('response_format', 'verbose_json')
#             temperature = kwargs.get('temperature', 0.0)
            
#             # Language handling
#             language_param = language if language != 'auto' else None
            
#             logger.debug(f"OpenAI STT request: model={self.stt_model}, "
#                         f"language={language_param}, format={response_format}")
            
#             # Make API call with retry logic
#             if response_format == 'verbose_json':
#                 transcript_response = self._retry_with_backoff(
#                     self._make_stt_request_verbose,
#                     audio_file,
#                     language_param,
#                     response_format,
#                     temperature
#                 )
#             else:
#                 transcript_response = self._retry_with_backoff(
#                     self._make_stt_request_simple,
#                     audio_file,
#                     language_param,
#                     response_format,
#                     temperature
#                 )
            
#             processing_time = time.time() - start_time
            
#             # Process response based on format
#             if response_format == 'verbose_json':
#                 transcript = transcript_response.text
#                 language_detected = getattr(transcript_response, 'language', language)
#                 duration = getattr(transcript_response, 'duration', processing_time)
                
#                 # Extract segments for word timestamps
#                 segments = getattr(transcript_response, 'segments', [])
#                 word_timestamps = []
                
#                 for segment in segments:
#                     words = getattr(segment, 'words', [])
#                     for word in words:
#                         word_timestamps.append({
#                             'word': getattr(word, 'word', ''),
#                             'start': getattr(word, 'start', 0),
#                             'end': getattr(word, 'end', 0),
#                             'confidence': getattr(word, 'probability', 0.0)
#                         })
                
#                 # Calculate overall confidence (average of word probabilities)
#                 if word_timestamps:
#                     confidence = sum(w['confidence'] for w in word_timestamps) / len(word_timestamps)
#                 else:
#                     confidence = 0.95  # Default high confidence for Whisper
                
#             else:
#                 # Simple text response
#                 transcript = transcript_response
#                 language_detected = language
#                 duration = processing_time  # Estimate
#                 confidence = 0.95  # Default
#                 word_timestamps = []
            
#             # Calculate RTF
#             rtf = processing_time / duration if duration > 0 else 0
            
#             metadata = {
#                 'provider': 'openai',
#                 'model': self.stt_model,
#                 'language_detected': language_detected,
#                 'response_format': response_format,
#                 'audio_duration': duration,
#                 'temperature': temperature
#             }
            
#             return STTResponse(
#                 success=True,
#                 transcript=transcript,
#                 confidence=confidence,
#                 processing_time=processing_time,
#                 rtf=rtf,
#                 word_timestamps=word_timestamps,
#                 metadata=metadata
#             )
            
#         except openai.APIError as e:
#             logger.error(f"OpenAI STT API error: {str(e)}")
#             return STTResponse(
#                 success=False,
#                 error_message=f"OpenAI API error: {str(e)}"
#             )
#         except Exception as e:
#             logger.error(f"OpenAI STT error: {str(e)}")
#             return STTResponse(
#                 success=False,
#                 error_message=str(e)
#             )
    
#     def _make_tts_request(self, text: str, voice: str, response_format: str, speed: float):
#         """Make TTS API request"""
#         response = self.client.audio.speech.create(
#             model=self.tts_model,
#             voice=voice,
#             input=text,
#             response_format=response_format,
#             speed=speed
#         )
#         return response
    
#     def _make_stt_request_verbose(self, audio_file, language: str, response_format: str, temperature: float):
#         """Make STT API request with verbose JSON response"""
#         kwargs = {
#             'model': self.stt_model,
#             'file': audio_file,
#             'response_format': response_format,
#             'timestamp_granularities': ['word'],
#             'temperature': temperature
#         }
        
#         if language:
#             kwargs['language'] = language
        
#         return self.client.audio.transcriptions.create(**kwargs)
    
#     def _make_stt_request_simple(self, audio_file, language: str, response_format: str, temperature: float):
#         """Make STT API request with simple text response"""
#         kwargs = {
#             'model': self.stt_model,
#             'file': audio_file,
#             'response_format': response_format,
#             'temperature': temperature
#         }
        
#         if language:
#             kwargs['language'] = language
        
#         return self.client.audio.transcriptions.create(**kwargs)
    
#     def get_supported_voices(self) -> List[str]:
#         """Get OpenAI TTS supported voices"""
#         return [
#             'alloy',    # Neutral, balanced voice
#             'echo',     # Male voice
#             'fable',    # British accent
#             'onyx',     # Deep male voice
#             'nova',     # Female voice
#             'shimmer'   # Soft female voice
#         ]
    
#     def get_supported_languages(self) -> List[str]:
#         """Get OpenAI supported languages (Whisper supports 100+ languages)"""
#         return [
#             'en', 'es', 'fr', 'de', 'it', 'pt', 'nl', 'pl', 'tr', 'ru',
#             'ja', 'ko', 'zh', 'ar', 'hi', 'th', 'vi', 'id', 'ms', 'he',
#             'uk', 'bg', 'hr', 'cs', 'da', 'et', 'fi', 'el', 'hu', 'is',
#             'lv', 'lt', 'mt', 'no', 'ro', 'sk', 'sl', 'sv', 'ca', 'eu',
#             'gl', 'cy', 'ga', 'mk', 'sq', 'az', 'be', 'bn', 'bs', 'ka',
#             'kk', 'ky', 'lv', 'lb', 'ml', 'mr', 'ne', 'pa', 'fa', 'ta',
#             'te', 'tl', 'ur', 'uz', 'auto'  # auto-detect
#         ]
    
#     def health_check(self) -> bool:
#         """Check OpenAI service health"""
#         try:
#             # Simple test with minimal TTS request
#             if self.supports_tts:
#                 test_response = self.text_to_speech("test", voice="alloy")
#                 return test_response.success
#             elif self.supports_stt:
#                 # For STT, just verify we have valid credentials
#                 return bool(self.api_key)
#             return True
#         except Exception as e:
#             logger.error(f"OpenAI health check failed: {str(e)}")
#             return False




"""
OpenAI TTS/STT client implementation
Provides integration with OpenAI's TTS and Whisper STT services
"""

import openai
import time
import tempfile
from typing import Dict, Any, List
import logging
from io import BytesIO
from pathlib import Path

from .base_client import BaseTTSSTTClient, TTSResponse, STTResponse

logger = logging.getLogger(__name__)

class OpenAIClient(BaseTTSSTTClient):
    """Client for OpenAI TTS and Whisper STT services"""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Model configuration
        self.tts_model = config.get('tts_model', 'tts-1')
        self.stt_model = config.get('stt_model', 'whisper-1')
        
        # Default settings
        self.default_voice = config.get('default_voice', 'alloy')
        self.default_response_format = config.get('default_response_format', 'mp3')
        
        logger.info(f"OpenAI client initialized - TTS: {self.tts_model}, STT: {self.stt_model}")
    
    def text_to_speech(self, 
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
            response_format = kwargs.get('response_format', self.default_response_format)
            speed = kwargs.get('speed', 1.0)
            
            # Validate voice
            if voice not in self.get_supported_voices():
                logger.warning(f"Voice '{voice}' not supported, using default: {self.default_voice}")
                voice = self.default_voice
            
            # Validate speed
            if not (0.25 <= speed <= 4.0):
                logger.warning(f"Speed {speed} out of range [0.25, 4.0], clamping")
                speed = max(0.25, min(4.0, speed))
            
            logger.debug(f"OpenAI TTS request: model={self.tts_model}, voice={voice}, "
                        f"format={response_format}, speed={speed}")
            
            # Make API call
            response = self.client.audio.speech.create(
                model=self.tts_model,
                voice=voice,
                input=text,
                response_format=response_format,
                speed=speed
            )
            
            latency = time.time() - start_time
            
            # Read audio data
            audio_data = response.content
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=f".{response_format}", delete=False) as temp_file:
                temp_file.write(audio_data)
                audio_url = temp_file.name
            
            # Estimate duration (OpenAI doesn't provide this directly)
            word_count = len(text.split())
            estimated_duration = (word_count / 150) * 60  # Convert to seconds
            
            metadata = {
                'provider': 'openai',
                'model': self.tts_model,
                'voice': voice,
                'response_format': response_format,
                'speed': speed,
                'audio_size': len(audio_data),
                'estimated_word_count': word_count
            }
            
            return TTSResponse(
                success=True,
                audio_data=audio_data,
                audio_url=audio_url,
                audio_format=response_format,
                sample_rate=22050 if response_format == 'wav' else 24000,
                duration=estimated_duration,
                latency=latency,
                metadata=metadata
            )
            
        except openai.APIError as e:
            logger.error(f"OpenAI TTS API error: {str(e)}")
            return TTSResponse(
                success=False,
                error_message=f"OpenAI API error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"OpenAI TTS error: {str(e)}")
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
        Convert speech to text using OpenAI Whisper
        
        Args:
            audio_data: Audio data in bytes
            audio_format: Audio format
            language: Language code (optional, Whisper can auto-detect)
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
            
            # Prepare audio file object
            audio_file = BytesIO(audio_data)
            audio_file.name = f"audio.{audio_format}"
            
            # Optional parameters
            model = kwargs.get('model', self.stt_model)
            prompt = kwargs.get('prompt')
            response_format = kwargs.get('response_format', 'text')
            temperature = kwargs.get('temperature', 0)
            
            logger.debug(f"OpenAI STT request: model={model}, language={language}, "
                        f"format={response_format}")
            
            # Make API call
            if response_format == 'verbose_json':
                response = self.client.audio.transcriptions.create(
                    model=model,
                    file=audio_file,
                    language=language if language != 'auto' else None,
                    prompt=prompt,
                    response_format=response_format,
                    temperature=temperature
                )
                transcript = response.text
                confidence = getattr(response, 'confidence', 0.0)
                word_timestamps = getattr(response, 'words', [])
            else:
                response = self.client.audio.transcriptions.create(
                    model=model,
                    file=audio_file,
                    language=language if language != 'auto' else None,
                    prompt=prompt,
                    response_format='text',
                    temperature=temperature
                )
                transcript = response
                confidence = 0.0
                word_timestamps = []
            
            processing_time = time.time() - start_time
            
            metadata = {
                'provider': 'openai',
                'model': model,
                'language': language,
                'response_format': response_format,
                'temperature': temperature,
                'audio_size': len(audio_data)
            }
            
            return STTResponse(
                success=True,
                transcript=transcript,
                confidence=confidence,
                processing_time=processing_time,
                word_timestamps=word_timestamps,
                metadata=metadata
            )
            
        except openai.APIError as e:
            logger.error(f"OpenAI STT API error: {str(e)}")
            return STTResponse(
                success=False,
                error_message=f"OpenAI API error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"OpenAI STT error: {str(e)}")
            return STTResponse(
                success=False,
                error_message=str(e)
            )
    
    def get_supported_voices(self) -> List[str]:
        """Get OpenAI supported voices"""
        return ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
    
    def get_supported_languages(self) -> List[str]:
        """Get OpenAI supported languages (Whisper supports many)"""
        return [
            'en', 'zh', 'de', 'es', 'ru', 'ko', 'fr', 'ja', 'pt', 'tr', 'pl', 'ca', 'nl',
            'ar', 'sv', 'it', 'id', 'hi', 'fi', 'vi', 'he', 'uk', 'el', 'ms', 'cs', 'ro',
            'da', 'hu', 'ta', 'no', 'th', 'ur', 'hr', 'bg', 'lt', 'la', 'mi', 'ml', 'cy',
            'sk', 'te', 'fa', 'lv', 'bn', 'sr', 'az', 'sl', 'kn', 'et', 'mk', 'br', 'eu',
            'is', 'hy', 'ne', 'mn', 'bs', 'kk', 'sq', 'sw', 'gl', 'mr', 'pa', 'si', 'km',
            'sn', 'yo', 'so', 'af', 'oc', 'ka', 'be', 'tg', 'sd', 'gu', 'am', 'yi', 'lo',
            'uz', 'fo', 'ht', 'ps', 'tk', 'nn', 'mt', 'sa', 'lb', 'my', 'bo', 'tl', 'mg',
            'as', 'tt', 'haw', 'ln', 'ha', 'ba', 'jw', 'su'
        ]
    
    def health_check(self) -> bool:
        """Check OpenAI service health"""
        try:
            # Simple test with minimal text
            test_response = self.text_to_speech("test", voice="alloy")
            return test_response.success
        except Exception as e:
            logger.error(f"OpenAI health check failed: {str(e)}")
            return False