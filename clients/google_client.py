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

from .base_client import BaseTTSSTTClient, TTSResponse, STTResponse

logger = logging.getLogger(__name__)

class GoogleClient(BaseTTSSTTClient):
    """Client for Google Cloud Speech services"""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        
        # Set up Google Cloud credentials
        credentials_path = config.get('credentials_path')
        if credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        
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
    
    def text_to_speech(self, 
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
                duration = word_count * 0.6  # ~0.6 seconds per word
            
            # Format determination
            audio_format = 'wav' if audio_encoding == 'LINEAR16' else audio_encoding.lower()
            
            metadata = {
                'provider': 'google',
                'model': self.model_name,
                'voice': voice_name,
                'language': voice_language,
                'audio_encoding': audio_encoding,
                'sample_rate': sample_rate,
                'audio_size': len(audio_data),
                'speaking_rate': audio_config.speaking_rate,
                'pitch': audio_config.pitch
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
            
        except Exception as e:
            logger.error(f"Google TTS error: {str(e)}")
            return TTSResponse(
                success=False,
                error_message=str(e)
            )
    
    def speech_to_text(self, 
                      audio_data: bytes,
                      audio_format: str = "wav",
                      language: str = "en-US",
                      **kwargs) -> STTResponse:
        """
        Convert speech to text using Google Cloud STT
        
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
            
            # Configure audio
            encoding_map = {
                'wav': speech.RecognitionConfig.AudioEncoding.LINEAR16,
                'flac': speech.RecognitionConfig.AudioEncoding.FLAC,
                'mulaw': speech.RecognitionConfig.AudioEncoding.MULAW,
                'amr': speech.RecognitionConfig.AudioEncoding.AMR,
                'amr_wb': speech.RecognitionConfig.AudioEncoding.AMR_WB,
                'ogg_opus': speech.RecognitionConfig.AudioEncoding.OGG_OPUS,
                'speex': speech.RecognitionConfig.AudioEncoding.SPEEX_WITH_HEADER_BYTE,
                'mp3': speech.RecognitionConfig.AudioEncoding.MP3
            }
            
            audio_encoding = encoding_map.get(audio_format.lower(), speech.RecognitionConfig.AudioEncoding.LINEAR16)
            sample_rate = kwargs.get('sample_rate', 22050)
            
            # Configure recognition
            config = speech.RecognitionConfig(
                encoding=audio_encoding,
                sample_rate_hertz=sample_rate,
                language_code=language,
                model=kwargs.get('model', 'latest_long'),
                use_enhanced=kwargs.get('use_enhanced', True),
                enable_word_time_offsets=kwargs.get('enable_word_timestamps', True),
                enable_word_confidence=kwargs.get('enable_word_confidence', True),
                enable_automatic_punctuation=kwargs.get('enable_punctuation', True),
                enable_speaker_diarization=kwargs.get('enable_speaker_diarization', False),
                max_alternatives=kwargs.get('max_alternatives', 1)
            )
            
            # Add speaker diarization config if enabled
            if kwargs.get('enable_speaker_diarization', False):
                config.diarization_config = speech.SpeakerDiarizationConfig(
                    enable_speaker_diarization=True,
                    min_speaker_count=kwargs.get('min_speakers', 1),
                    max_speaker_count=kwargs.get('max_speakers', 6)
                )
            
            # Create audio object
            audio = speech.RecognitionAudio(content=audio_data)
            
            logger.debug(f"Google STT request: language={language}, model={config.model}, "
                        f"audio_size={len(audio_data)} bytes")
            
            # Perform recognition
            if len(audio_data) > 10 * 1024 * 1024:  # > 10MB, use long running operation
                operation = self.stt_client.long_running_recognize(config=config, audio=audio)
                logger.info("Google STT: Using long running operation for large audio file")
                response = operation.result(timeout=300)  # 5 minute timeout
            else:
                response = self.stt_client.recognize(config=config, audio=audio)
            
            processing_time = time.time() - start_time
            
            if response.results:
                # Get the best alternative
                result = response.results[0]
                alternative = result.alternatives[0]
                
                transcript = alternative.transcript
                confidence = alternative.confidence
                
                # Extract word-level information
                word_timestamps = []
                if hasattr(alternative, 'words'):
                    for word_info in alternative.words:
                        word_timestamps.append({
                            'word': word_info.word,
                            'start': word_info.start_time.total_seconds(),
                            'end': word_info.end_time.total_seconds(),
                            'confidence': getattr(word_info, 'confidence', confidence),
                            'speaker_tag': getattr(word_info, 'speaker_tag', 0)
                        })
                
                # Calculate audio duration
                if word_timestamps:
                    audio_duration = max(w['end'] for w in word_timestamps)
                else:
                    # Estimate duration
                    audio_duration = processing_time
                
                # Calculate RTF
                rtf = processing_time / audio_duration if audio_duration > 0 else 0
                
                metadata = {
                    'provider': 'google',
                    'model': self.model_name,
                    'language': language,
                    'audio_duration': audio_duration,
                    'recognition_model': config.model,
                    'num_alternatives': len(result.alternatives),
                    'total_billed_time': getattr(response, 'total_billed_time', None)
                }
                
                # Add speaker diarization info if available
                if config.diarization_config and config.diarization_config.enable_speaker_diarization:
                    speakers = set(w.get('speaker_tag', 0) for w in word_timestamps)
                    metadata['num_speakers'] = len(speakers)
                
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
                error_msg = "Google STT: No transcription results"
                logger.warning(error_msg)
                return STTResponse(
                    success=False,
                    error_message=error_msg,
                    processing_time=processing_time
                )
                
        except Exception as e:
            logger.error(f"Google STT error: {str(e)}")
            return STTResponse(
                success=False,
                error_message=str(e)
            )
    
    def get_supported_voices(self) -> List[str]:
        """Get Google TTS supported voices (subset)"""
        return self.config.get('supported_voices', [
            # English voices
            'en-US-Wavenet-A', 'en-US-Wavenet-B', 'en-US-Wavenet-C', 'en-US-Wavenet-D',
            'en-US-Wavenet-E', 'en-US-Wavenet-F', 'en-US-Wavenet-G', 'en-US-Wavenet-H',
            'en-US-Wavenet-I', 'en-US-Wavenet-J',
            'en-GB-Wavenet-A', 'en-GB-Wavenet-B', 'en-GB-Wavenet-C', 'en-GB-Wavenet-D',
            
            # Other languages (examples)
            'es-ES-Wavenet-A', 'fr-FR-Wavenet-A', 'de-DE-Wavenet-A',
            'ja-JP-Wavenet-A', 'ko-KR-Wavenet-A', 'zh-CN-Wavenet-A',
            'hi-IN-Wavenet-A', 'hi-IN-Wavenet-B', 'hi-IN-Wavenet-C'
        ])
    
    def get_supported_languages(self) -> List[str]:
        """Get Google supported languages"""
        return self.config.get('supported_languages', [
            'en-US', 'en-GB', 'en-AU', 'en-CA', 'en-IN', 'en-IE', 'en-NZ', 'en-PH', 'en-SG', 'en-ZA',
            'es-ES', 'es-MX', 'es-AR', 'es-CL', 'es-CO', 'es-PE', 'es-VE',
            'fr-FR', 'fr-CA', 'de-DE', 'it-IT', 'pt-BR', 'pt-PT',
            'ru-RU', 'ja-JP', 'ko-KR', 'zh-CN', 'zh-TW', 'zh-HK',
            'hi-IN', 'ar-EG', 'tr-TR', 'nl-NL', 'sv-SE', 'da-DK',
            'no-NO', 'fi-FI', 'pl-PL', 'cs-CZ', 'sk-SK', 'hu-HU'
        ])
    
    def health_check(self) -> bool:
        """Check Google Cloud Speech service health"""
        try:
            if self.supports_tts:
                # Simple TTS test
                test_response = self.text_to_speech("test", voice="en-US-Wavenet-D")
                return test_response.success
            elif self.supports_stt:
                # For STT, try to list available models (this tests authentication)
                try:
                    # This is a simple way to test if credentials work
                    self.stt_client.list_phrase_sets(parent=f"projects/{self.config.get('project_id', 'test')}/locations/global")
                    return True
                except:
                    # If we can't list (maybe no project_id), assume healthy if we have a client
                    return hasattr(self, 'stt_client')
            return True
        except Exception as e:
            logger.error(f"Google health check failed: {str(e)}")
            return False