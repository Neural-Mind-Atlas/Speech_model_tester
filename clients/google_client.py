"""
Google Cloud TTS/STT client implementation
Provides integration with Google Cloud Speech-to-Text and Text-to-Speech
"""

from google.cloud import texttospeech
from google.cloud import speech
import time
from typing import Dict, Any, List, Optional
import logging
import io
import os
import tempfile
import pathlib

from .base_client import BaseTTSSTTClient, TTSResponse, STTResponse

logger = logging.getLogger(__name__)

class GoogleClient(BaseTTSSTTClient):
    """Client for Google Cloud Speech services"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Google Cloud Speech client
        
        Args:
            config: Configuration dictionary containing:
                - credentials_path: Path to service account JSON file
                - project_id: Google Cloud project ID
                - default_voice_name: Default voice for TTS
                - default_language_code: Default language code
        """
        # Extract model name from config or use default
        model_name = config.get('model_name', 'google-cloud-speech')
        
        super().__init__(model_name, config)
        
        # Set up Google Cloud credentials
        credentials_path = config.get('credentials_path')
        if credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        
        # Project ID for Google Cloud
        self.project_id = config.get('project_id')
        
        # Initialize clients
        try:
            if self.supports_tts:
                self.tts_client = texttospeech.TextToSpeechClient()
            if self.supports_stt:
                self.stt_client = speech.SpeechClient()
        except Exception as e:
            logger.error(f"Failed to initialize Google clients: {str(e)}")
            raise
        
        # Default settings
        self.default_voice_name = config.get('default_voice_name', 'en-US-Wavenet-D')
        self.default_language_code = config.get('default_language_code', 'en-US')
        
        logger.info(f"Google Cloud Speech client initialized - Model: {model_name}")
    
    async def text_to_speech(self, 
                           text: str, 
                           voice: str = "en-US-Wavenet-D",
                           language: str = "en-US",
                           **kwargs) -> TTSResponse:
        """
        Convert text to speech using Google Cloud TTS
        
        Args:
            text: Text to convert
            voice: Google voice name (e.g., en-US-Wavenet-D)
            language: Language code (e.g., en-US)
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
            
            # Configure synthesis input
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Configure voice
            voice = voice or self.default_voice_name
            language_code = language or self.default_language_code
            
            # Parse voice parameters
            voice_parts = voice.split('-')
            if len(voice_parts) >= 3:
                voice_language = '-'.join(voice_parts[:2])
                voice_name = voice
            else:
                voice_language = language_code
                voice_name = voice
            
            # Determine voice gender and type
            ssml_gender = kwargs.get('ssml_gender', texttospeech.SsmlVoiceGender.NEUTRAL)
            if isinstance(ssml_gender, str):
                gender_map = {
                    'male': texttospeech.SsmlVoiceGender.MALE,
                    'female': texttospeech.SsmlVoiceGender.FEMALE,
                    'neutral': texttospeech.SsmlVoiceGender.NEUTRAL
                }
                ssml_gender = gender_map.get(ssml_gender.lower(), texttospeech.SsmlVoiceGender.NEUTRAL)
            
            voice_selection = texttospeech.VoiceSelectionParams(
                language_code=voice_language,
                name=voice_name,
                ssml_gender=ssml_gender
            )
            
            # Configure audio output
            audio_encoding = kwargs.get('audio_encoding', 'LINEAR16')
            sample_rate = kwargs.get('sample_rate', 22050)
            
            encoding_map = {
                'LINEAR16': texttospeech.AudioEncoding.LINEAR16,
                'MP3': texttospeech.AudioEncoding.MP3,
                'OGG_OPUS': texttospeech.AudioEncoding.OGG_OPUS,
                'MULAW': texttospeech.AudioEncoding.MULAW,
                'ALAW': texttospeech.AudioEncoding.ALAW
            }
            
            audio_encoding_enum = encoding_map.get(audio_encoding, texttospeech.AudioEncoding.LINEAR16)
            
            audio_config = texttospeech.AudioConfig(
                audio_encoding=audio_encoding_enum,
                sample_rate_hertz=sample_rate,
                speaking_rate=kwargs.get('speaking_rate', 1.0),
                pitch=kwargs.get('pitch', 0.0),
                volume_gain_db=kwargs.get('volume_gain_db', 0.0)
            )
            
            logger.debug(f"Google TTS request: voice={voice_name}, language={voice_language}, "
                        f"encoding={audio_encoding}")
            
            # Perform the text-to-speech request
            response = self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice_selection,
                audio_config=audio_config
            )
            
            latency = time.time() - start_time
            
            audio_data = response.audio_content
            
            # Estimate duration
            if audio_encoding == 'LINEAR16':
                # Calculate duration from audio data
                bytes_per_sample = 2  # 16-bit
                total_samples = len(audio_data) // bytes_per_sample
                duration = total_samples / sample_rate
            else:
                # Estimate for compressed formats
                word_count = len(text.split())
                duration = word_count * 0.6
            
            # Save audio to temporary file
            file_extension = 'wav' if audio_encoding == 'LINEAR16' else 'mp3'
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as temp_file:
                temp_file.write(audio_data)
                audio_url = temp_file.name
            
            # Update statistics
            self._update_stats(success=True, latency=latency)
            
            return TTSResponse(
                success=True,
                audio_url=audio_url,
                audio_data=audio_data,
                metadata={
                    'voice': voice_name,
                    'language': voice_language,
                    'encoding': audio_encoding,
                    'sample_rate': sample_rate,
                    'duration': duration,
                    'latency': latency,
                    'audio_size': len(audio_data)
                }
            )
            
        except Exception as e:
            error_msg = f"Google TTS failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Update statistics
            self._update_stats(success=False, latency=time.time() - start_time)
            
            return TTSResponse(
                success=False,
                error_message=error_msg,
                metadata={'voice': voice, 'language': language}
            )
    
    async def speech_to_text(self, 
                           audio_path: str, 
                           language: str = "en-US",
                           **kwargs) -> STTResponse:
        """
        Convert speech to text using Google Cloud STT
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., en-US)
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
            # Validate audio file exists
            audio_file = pathlib.Path(audio_path)
            if not audio_file.exists():
                return STTResponse(
                    success=False,
                    error_message=f"Audio file not found: {audio_path}"
                )
            
            start_time = time.time()
            
            # Read audio file
            with open(audio_path, 'rb') as audio_file:
                audio_content = audio_file.read()
            
            # Configure audio
            audio = speech.RecognitionAudio(content=audio_content)
            
            # Configure recognition
            language_code = language or self.default_language_code
            model = kwargs.get('model', 'latest_short')
            enable_automatic_punctuation = kwargs.get('enable_automatic_punctuation', True)
            
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=kwargs.get('sample_rate', 16000),
                language_code=language_code,
                model=model,
                enable_automatic_punctuation=enable_automatic_punctuation,
                enable_word_time_offsets=kwargs.get('enable_word_time_offsets', False),
                enable_word_confidence=kwargs.get('enable_word_confidence', True)
            )
            
            logger.debug(f"Google STT request: language={language_code}, model={model}")
            
            # Perform the speech-to-text request
            response = self.stt_client.recognize(config=config, audio=audio)
            
            latency = time.time() - start_time
            
            # Process results
            if not response.results:
                return STTResponse(
                    success=False,
                    error_message="No speech detected in audio",
                    metadata={'language': language_code, 'latency': latency}
                )
            
            # Get the best alternative
            result = response.results[0]
            alternative = result.alternatives[0]
            
            transcript = alternative.transcript
            confidence = alternative.confidence if hasattr(alternative, 'confidence') else 0.0
            
            # Extract word-level information if available
            words = []
            if hasattr(alternative, 'words'):
                for word_info in alternative.words:
                    words.append({
                        'word': word_info.word,
                        'start_time': word_info.start_time.total_seconds() if hasattr(word_info, 'start_time') else 0,
                        'end_time': word_info.end_time.total_seconds() if hasattr(word_info, 'end_time') else 0,
                        'confidence': word_info.confidence if hasattr(word_info, 'confidence') else 0.0
                    })
            
            # Update statistics
            self._update_stats(success=True, latency=latency)
            
            return STTResponse(
                success=True,
                text=transcript,
                metadata={
                    'language': language_code,
                    'confidence': confidence,
                    'model': model,
                    'latency': latency,
                    'words': words,
                    'audio_duration': len(audio_content) / (16000 * 2)  # Estimate duration
                }
            )
            
        except Exception as e:
            error_msg = f"Google STT failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Update statistics
            self._update_stats(success=False, latency=time.time() - start_time)
            
            return STTResponse(
                success=False,
                error_message=error_msg,
                metadata={'language': language, 'audio_path': audio_path}
            )
    
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """
        Get list of available voices for TTS
        
        Returns:
            List of voice dictionaries with name, language, and gender
        """
        try:
            if not self.supports_tts:
                return []
            
            # Get available voices from Google Cloud
            voices_response = self.tts_client.list_voices()
            
            voices = []
            for voice in voices_response.voices:
                for language_code in voice.language_codes:
                    voices.append({
                        'name': voice.name,
                        'language': language_code,
                        'gender': voice.ssml_gender.name.lower(),
                        'natural_sample_rate': voice.natural_sample_rate_hertz
                    })
            
            return voices
            
        except Exception as e:
            logger.error(f"Failed to get available voices: {str(e)}")
            # Return default voices if API call fails
            return [
                {'name': 'en-US-Wavenet-D', 'language': 'en-US', 'gender': 'male'},
                {'name': 'en-US-Wavenet-C', 'language': 'en-US', 'gender': 'female'},
                {'name': 'en-US-Wavenet-B', 'language': 'en-US', 'gender': 'male'},
                {'name': 'en-US-Wavenet-A', 'language': 'en-US', 'gender': 'female'},
                {'name': 'en-GB-Wavenet-A', 'language': 'en-GB', 'gender': 'female'},
                {'name': 'en-GB-Wavenet-B', 'language': 'en-GB', 'gender': 'male'},
            ]
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages
        
        Returns:
            List of language codes
        """
        return [
            'en-US', 'en-GB', 'en-AU', 'en-CA', 'en-IN',
            'es-ES', 'es-US', 'fr-FR', 'fr-CA', 'de-DE',
            'it-IT', 'pt-BR', 'pt-PT', 'ru-RU', 'ja-JP',
            'ko-KR', 'zh-CN', 'zh-TW', 'hi-IN', 'ar-XA'
        ]
    
    async def health_check(self) -> bool:
        """
        Perform health check for Google Cloud Speech services
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            # Test TTS if supported
            if self.supports_tts:
                test_response = await self.text_to_speech(
                    text="Health check test",
                    voice=self.default_voice_name
                )
                if not test_response.success:
                    return False
            
            # Test STT if supported (would need a test audio file)
            # For now, just check if clients are initialized
            if self.supports_stt and not hasattr(self, 'stt_client'):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Google Cloud health check failed: {str(e)}")
            return False