"""
Microsoft Azure TTS/STT client implementation
Provides integration with Azure Cognitive Services Speech
"""

import azure.cognitiveservices.speech as speechsdk
import time
from typing import Dict, Any, List, Optional
import logging
import io
import wave

from .base_client import BaseTTSSTTClient, TTSResponse, STTResponse

logger = logging.getLogger(__name__)

class AzureClient(BaseTTSSTTClient):
    """Client for Microsoft Azure Speech Services"""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        
        # Azure Speech configuration
        self.region = config.get('region') or config.get('speech_region')
        if not self.region:
            raise ValueError("Azure region must be specified in config")
        
        # Initialize Azure Speech Config
        self.speech_config = speechsdk.SpeechConfig(
            subscription=self.api_key, 
            region=self.region
        )
        
        # Default settings
        self.default_voice = config.get('default_voice', 'en-US-AriaNeural')
        self.default_language = config.get('default_language', 'en-US')
        
        logger.info(f"Azure Speech client initialized - Region: {self.region}, Model: {model_name}")
    
    def text_to_speech(self, 
                      text: str, 
                      voice: str = "en-US-AriaNeural",
                      language: str = "en-US",
                      **kwargs) -> TTSResponse:
        """
        Convert text to speech using Azure TTS
        
        Args:
            text: Text to convert
            voice: Azure voice name (e.g., en-US-AriaNeural)
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
            
            # Configure voice
            voice = voice or self.default_voice
            self.speech_config.speech_synthesis_voice_name = voice
            
            # Set output format
            audio_format = kwargs.get('format', 'wav')
            sample_rate = kwargs.get('sample_rate', 22050)
            
            if audio_format == 'wav' and sample_rate == 22050:
                self.speech_config.set_speech_synthesis_output_format(
                    speechsdk.SpeechSynthesisOutputFormat.Riff22050Hz16BitMonoPcm
                )
            elif audio_format == 'wav' and sample_rate == 16000:
                self.speech_config.set_speech_synthesis_output_format(
                    speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
                )
            else:
                # Default to 22kHz WAV
                self.speech_config.set_speech_synthesis_output_format(
                    speechsdk.SpeechSynthesisOutputFormat.Riff22050Hz16BitMonoPcm
                )
                sample_rate = 22050
            
            # Create synthesizer
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self.speech_config, 
                audio_config=None
            )
            
            # Prepare SSML if advanced parameters are provided
            ssml_text = self._prepare_ssml(text, voice, **kwargs)
            
            logger.debug(f"Azure TTS request: voice={voice}, text_length={len(text)}")
            
            # Perform synthesis
            if ssml_text != text:
                # Use SSML
                result = synthesizer.speak_ssml_async(ssml_text).get()
            else:
                # Use plain text
                result = synthesizer.speak_text_async(text).get()
            
            latency = time.time() - start_time
            
            # Check result
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                audio_data = result.audio_data
                
                # Calculate duration from audio data
                duration = self._calculate_audio_duration(audio_data, sample_rate)
                
                metadata = {
                    'provider': 'azure',
                    'model': self.model_name,
                    'voice': voice,
                    'language': language,
                    'audio_size': len(audio_data),
                    'sample_rate': sample_rate,
                    'result_id': result.result_id
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
                error_msg = f"Azure TTS synthesis failed: {result.reason}"
                if result.reason == speechsdk.ResultReason.Canceled:
                    cancellation_details = speechsdk.CancellationDetails(result)
                    error_msg += f" - {cancellation_details.reason}: {cancellation_details.error_details}"
                
                logger.error(error_msg)
                return TTSResponse(
                    success=False,
                    error_message=error_msg,
                    latency=latency
                )
                
        except Exception as e:
            logger.error(f"Azure TTS error: {str(e)}")
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
        Convert speech to text using Azure STT
        
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
            
            # Configure language
            self.speech_config.speech_recognition_language = language
            
            # Enable detailed results if requested
            if kwargs.get('enable_word_timestamps', False):
                self.speech_config.request_word_level_timestamps()
            
            # Configure continuous recognition settings
            self.speech_config.enable_dictation()
            
            # Create audio config from bytes
            audio_format_info = self._get_azure_audio_format(audio_format)
            audio_stream = speechsdk.audio.PushAudioInputStream(audio_format_info)
            audio_config = speechsdk.audio.AudioConfig(stream=audio_stream)
            
            # Create recognizer
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )
            
            # Set up result collection
            results = []
            done = False
            
            def handle_result(evt):
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    results.append(evt.result)
                elif evt.result.reason == speechsdk.ResultReason.NoMatch:
                    logger.warning("Azure STT: No speech could be recognized")
                elif evt.result.reason == speechsdk.ResultReason.Canceled:
                    cancellation_details = speechsdk.CancellationDetails(evt.result)
                    logger.error(f"Azure STT canceled: {cancellation_details.reason}")
            
            def handle_session_stopped(evt):
                nonlocal done
                done = True
            
            # Connect callbacks
            recognizer.recognized.connect(handle_result)
            recognizer.session_stopped.connect(handle_session_stopped)
            
            # Start recognition
            recognizer.start_continuous_recognition()
            
            # Push audio data
            audio_stream.write(audio_data)
            audio_stream.close()
            
            # Wait for completion (with timeout)
            timeout = kwargs.get('timeout', 30)
            elapsed = 0
            while not done and elapsed < timeout:
                time.sleep(0.1)
                elapsed += 0.1
            
            # Stop recognition
            recognizer.stop_continuous_recognition()
            
            processing_time = time.time() - start_time
            
            if results:
                # Combine all results
                full_transcript = ' '.join([result.text for result in results])
                
                # Calculate confidence (Azure provides this in detailed results)
                confidences = []
                word_timestamps = []
                
                for result in results:
                    # Extract detailed information if available
                    if hasattr(result, 'json') and result.json:
                        import json
                        result_json = json.loads(result.json)
                        
                        if 'NBest' in result_json and result_json['NBest']:
                            best_result = result_json['NBest'][0]
                            confidences.append(best_result.get('Confidence', 0.0))
                            
                            # Extract word-level information
                            if 'Words' in best_result:
                                for word_info in best_result['Words']:
                                    word_timestamps.append({
                                        'word': word_info.get('Word', ''),
                                        'start': word_info.get('Offset', 0) / 10000000,  # Convert to seconds
                                        'end': (word_info.get('Offset', 0) + word_info.get('Duration', 0)) / 10000000,
                                        'confidence': word_info.get('Confidence', 0.0)
                                    })
                
                # Calculate overall confidence
                confidence = sum(confidences) / len(confidences) if confidences else 0.8
                
                # Estimate audio duration
                if word_timestamps:
                    audio_duration = max(w['end'] for w in word_timestamps)
                else:
                    audio_duration = processing_time  # Fallback estimate
                
                # Calculate RTF
                rtf = processing_time / audio_duration if audio_duration > 0 else 0
                
                metadata = {
                    'provider': 'azure',
                    'model': self.model_name,
                    'language': language,
                    'audio_duration': audio_duration,
                    'num_results': len(results),
                    'region': self.region
                }
                
                return STTResponse(
                    success=True,
                    transcript=full_transcript,
                    confidence=confidence,
                    processing_time=processing_time,
                    rtf=rtf,
                    word_timestamps=word_timestamps,
                    metadata=metadata
                )
            else:
                error_msg = "Azure STT: No speech was recognized"
                logger.warning(error_msg)
                return STTResponse(
                    success=False,
                    error_message=error_msg,
                    processing_time=processing_time
                )
                
        except Exception as e:
            logger.error(f"Azure STT error: {str(e)}")
            return STTResponse(
                success=False,
                error_message=str(e)
            )
    
    def _prepare_ssml(self, text: str, voice: str, **kwargs) -> str:
        """
        Prepare SSML markup for advanced speech synthesis
        
        Args:
            text: Original text
            voice: Voice name
            **kwargs: SSML parameters
            
        Returns:
            SSML string or original text if no SSML needed
        """
        # Check if any SSML parameters are provided
        ssml_params = ['rate', 'pitch', 'volume', 'emphasis', 'break_time']
        if not any(param in kwargs for param in ssml_params):
            return text
        
        # Build SSML
        ssml = f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
        ssml += f'<voice name="{voice}">'
        
        # Add prosody if specified
        prosody_attrs = []
        if 'rate' in kwargs:
            prosody_attrs.append(f'rate="{kwargs["rate"]}"')
        if 'pitch' in kwargs:
            prosody_attrs.append(f'pitch="{kwargs["pitch"]}"')
        if 'volume' in kwargs:
            prosody_attrs.append(f'volume="{kwargs["volume"]}"')
        
        if prosody_attrs:
            ssml += f'<prosody {" ".join(prosody_attrs)}>'
            ssml += text
            ssml += '</prosody>'
        else:
            ssml += text
        
        ssml += '</voice></speak>'
        
        return ssml
    
    def _get_azure_audio_format(self, audio_format: str) -> speechsdk.audio.AudioStreamFormat:
        """Get Azure audio format from string"""
        if audio_format.lower() == 'wav':
            return speechsdk.audio.AudioStreamFormat(samples_per_second=22050, bits_per_sample=16, channels=1)
        elif audio_format.lower() == 'mp3':
            # For MP3, we need to convert or use a different approach
            # For now, assume WAV format
            return speechsdk.audio.AudioStreamFormat(samples_per_second=22050, bits_per_sample=16, channels=1)
        else:
            # Default
            return speechsdk.audio.AudioStreamFormat(samples_per_second=22050, bits_per_sample=16, channels=1)
    
    def _calculate_audio_duration(self, audio_data: bytes, sample_rate: int) -> float:
        """Calculate audio duration from raw audio data"""
        try:
            # Assume 16-bit mono PCM
            bytes_per_sample = 2
            total_samples = len(audio_data) // bytes_per_sample
            duration = total_samples / sample_rate
            return duration
        except:
            # Fallback estimate
            return len(audio_data) / (sample_rate * 2)  # Rough estimate
    
    def get_supported_voices(self) -> List[str]:
        """Get Azure TTS supported voices (subset)"""
        return self.config.get('supported_voices', [
            # English voices
            'en-US-AriaNeural',
            'en-US-JennyNeural',
            'en-US-GuyNeural',
            'en-US-DavisNeural',
            'en-US-AmberNeural',
            'en-US-AnaNeural',
            'en-US-BrandonNeural',
            'en-US-ChristopherNeural',
            'en-US-CoraNeural',
            'en-US-ElizabethNeural',
            
            # Other languages (examples)
            'es-ES-ElviraNeural',
            'fr-FR-DeniseNeural',
            'de-DE-KatjaNeural',
            'hi-IN-SwaraNeural',
            'ja-JP-NanamiNeural',
            'ko-KR-SunHiNeural',
            'zh-CN-XiaoxiaoNeural'
        ])
    
    def get_supported_languages(self) -> List[str]:
        """Get Azure supported languages"""
        return self.config.get('supported_languages', [
            'en-US', 'en-GB', 'es-ES', 'es-MX', 'fr-FR', 'fr-CA',
            'de-DE', 'it-IT', 'pt-BR', 'pt-PT', 'ru-RU', 'ja-JP',
            'ko-KR', 'zh-CN', 'zh-TW', 'hi-IN', 'ar-EG', 'tr-TR',
            'nl-NL', 'sv-SE', 'da-DK', 'no-NO', 'fi-FI', 'pl-PL'
        ])
    
    def health_check(self) -> bool:
        """Check Azure Speech service health"""
        try:
            # Simple TTS test
            if self.supports_tts:
                test_response = self.text_to_speech("test", voice="en-US-AriaNeural")
                return test_response.success
            elif self.supports_stt:
                # For STT, verify credentials are valid
                return bool(self.api_key and self.region)
            return True
        except Exception as e:
            logger.error(f"Azure health check failed: {str(e)}")
            return False