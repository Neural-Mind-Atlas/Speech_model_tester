"""
TTS/STT Testing Framework - Audio Utilities
==========================================

This module provides comprehensive audio processing utilities for the TTS/STT testing framework.
It includes audio validation, format conversion, analysis, and quality assessment functionality.

Author: TTS/STT Testing Framework Team
Version: 1.0.0
Created: 2024-06-04
"""

import os
import wave
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import tempfile
import subprocess
import math

from .logger import get_logger, log_function_call
from .file_utils import FileManager, validate_file_path

class AudioProcessor:
    """
    Comprehensive audio processing class for the TTS/STT testing framework.
    
    Features:
    - Audio file validation and format detection
    - Audio quality analysis and metrics
    - Format conversion and normalization
    - Audio segmentation and manipulation
    - Spectral analysis and feature extraction
    """
    
    SUPPORTED_FORMATS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma']
    SUPPORTED_SAMPLE_RATES = [8000, 16000, 22050, 44100, 48000]
    
    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialize the audio processor.
        
        Args:
            temp_dir: Temporary directory for processing
        """
        self.logger = get_logger(__name__)
        self.file_manager = FileManager()
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir())
        
        # Ensure temp directory exists
        self.file_manager.ensure_directory(self.temp_dir)
        
        self.logger.info(f"AudioProcessor initialized with temp directory: {self.temp_dir}")
    
    @log_function_call
    def validate_audio_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate audio file and return comprehensive information.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dict[str, Any]: Validation results and audio information
        """
        validation_result = {
            'is_valid': False,
            'file_path': str(file_path),
            'errors': [],
            'warnings': [],
            'audio_info': None
        }
        
        try:
            path = Path(file_path)
            
            # Check if file exists
            if not validate_file_path(path, must_exist=True):
                validation_result['errors'].append("File does not exist or is not accessible")
                return validation_result
            
            # Check file extension
            if path.suffix.lower() not in self.SUPPORTED_FORMATS:
                validation_result['errors'].append(f"Unsupported format: {path.suffix}")
                return validation_result
            
            # Get audio information
            audio_info = self.get_audio_info(path)
            if not audio_info:
                validation_result['errors'].append("Failed to read audio file information")
                return validation_result
            
            validation_result['audio_info'] = audio_info
            
            # Validate audio parameters
            if audio_info['duration'] <= 0:
                validation_result['errors'].append("Invalid audio duration")
            
            if audio_info['sample_rate'] not in self.SUPPORTED_SAMPLE_RATES:
                validation_result['warnings'].append(f"Unusual sample rate: {audio_info['sample_rate']} Hz")
            
            if audio_info['channels'] > 2:
                validation_result['warnings'].append(f"Multi-channel audio: {audio_info['channels']} channels")
            
            # Check for audio quality issues
            quality_issues = self._check_audio_quality(path, audio_info)
            validation_result['warnings'].extend(quality_issues)
            
            # Mark as valid if no critical errors
            validation_result['is_valid'] = len(validation_result['errors']) == 0
            
            self.logger.debug(f"Audio validation completed: {path} - Valid: {validation_result['is_valid']}")
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Audio validation failed for {file_path}", e)
            validation_result['errors'].append(f"Validation error: {str(e)}")
            return validation_result
    
    @log_function_call
    def get_audio_info(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive audio file information.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Optional[Dict[str, Any]]: Audio information or None if failed
        """
        try:
            path = Path(file_path)
            
            # Get basic file info
            file_info = self.file_manager.get_file_info(path)
            if not file_info:
                return None
            
            # Load audio with librosa
            y, sr = librosa.load(str(path), sr=None)
            
            # Calculate audio metrics
            duration = len(y) / sr
            rms_energy = float(np.sqrt(np.mean(y**2)))
            zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(y)))
            spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
            
            # Detect silence
            silence_threshold = 0.01
            silence_frames = np.sum(np.abs(y) < silence_threshold)
            silence_percentage = (silence_frames / len(y)) * 100
            
            # Detect clipping
            clipping_threshold = 0.95
            clipped_samples = np.sum(np.abs(y) > clipping_threshold)
            clipping_percentage = (clipped_samples / len(y)) * 100
            
            audio_info = {
                'file_path': str(path),
                'file_size_bytes': file_info['size_bytes'],
                'file_size_human': file_info['size_human'],
                'format': path.suffix.lower(),
                'duration': duration,
                'sample_rate': sr,
                'channels': 1,  # librosa loads as mono by default
                'samples': len(y),
                'bit_depth': 'unknown',  # librosa doesn't provide this directly
                'bitrate': 'unknown',
                'audio_metrics': {
                    'rms_energy': rms_energy,
                    'max_amplitude': float(np.max(np.abs(y))),
                    'min_amplitude': float(np.min(np.abs(y))),
                    'zero_crossing_rate': zero_crossing_rate,
                    'spectral_centroid': spectral_centroid,
                    'silence_percentage': silence_percentage,
                    'clipping_percentage': clipping_percentage
                },
                'quality_assessment': {
                    'has_silence': silence_percentage > 10,
                    'has_clipping': clipping_percentage > 1,
                    'is_normalized': 0.7 <= np.max(np.abs(y)) <= 1.0,
                    'dynamic_range': float(np.max(y) - np.min(y))
                },
                'extraction_timestamp': datetime.now().isoformat()
            }
            
            # Try to get more detailed info with soundfile
            try:
                with sf.SoundFile(str(path)) as f:
                    audio_info.update({
                        'channels': f.channels,
                        'sample_rate': f.samplerate,
                        'samples': len(f),
                        'duration': len(f) / f.samplerate,
                        'subtype': f.subtype,
                        'format_info': f.format_info
                    })
            except Exception:
                pass  # Use librosa info as fallback
            
            self.logger.debug(f"Audio info extracted: {path}")
            return audio_info
            
        except Exception as e:
            self.logger.error(f"Failed to get audio info for {file_path}", e)
            return None
    
    @log_function_call
    def convert_audio_format(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        target_format: str = 'wav',
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        normalize: bool = False
    ) -> bool:
        """
        Convert audio file to different format.
        
        Args:
            input_path: Input audio file path
            output_path: Output audio file path
            target_format: Target format ('wav', 'mp3', 'flac')
            sample_rate: Target sample rate
            channels: Target number of channels
            normalize: Whether to normalize audio
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            input_path = Path(input_path)
            output_path = Path(output_path)
            
            # Validate input file
            if not validate_file_path(input_path, must_exist=True):
                self.logger.error(f"Invalid input file: {input_path}")
                return False
            
            # Ensure output directory exists
            if not self.file_manager.ensure_directory(output_path.parent):
                return False
            
            # Load audio
            y, sr = librosa.load(str(input_path), sr=sample_rate)
            
            # Convert to target channels
            if channels:
                if channels == 1 and len(y.shape) > 1:
                    # Convert to mono
                    y = librosa.to_mono(y)
                elif channels == 2 and len(y.shape) == 1:
                    # Convert to stereo
                    y = np.stack([y, y])
            
            # Normalize if requested
            if normalize:
                y = librosa.util.normalize(y)
            
            # Save in target format
            if target_format.lower() == 'wav':
                sf.write(str(output_path), y, sr, format='WAV')
            elif target_format.lower() == 'flac':
                sf.write(str(output_path), y, sr, format='FLAC')
            elif target_format.lower() == 'mp3':
                # For MP3, we need to use ffmpeg or similar
                temp_wav = self.temp_dir / f"temp_{datetime.now().timestamp()}.wav"
                sf.write(str(temp_wav), y, sr, format='WAV')
                
                # Use ffmpeg for MP3 conversion
                if self._convert_with_ffmpeg(temp_wav, output_path, 'mp3'):
                    temp_wav.unlink()  # Remove temp file
                else:
                    self.logger.error("Failed to convert to MP3")
                    return False
            else:
                self.logger.error(f"Unsupported target format: {target_format}")
                return False
            
            self.logger.info(f"Audio conversion successful: {input_path} -> {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Audio conversion failed: {input_path} -> {output_path}", e)
            return False
    
    @log_function_call
    def analyze_audio_quality(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Perform comprehensive audio quality analysis.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dict[str, Any]: Quality analysis results
        """
        try:
            path = Path(file_path)
            
            # Get basic audio info
            audio_info = self.get_audio_info(path)
            if not audio_info:
                return {'success': False, 'error': 'Failed to load audio'}
            
            # Load audio for analysis
            y, sr = librosa.load(str(path), sr=None)
            
            # Calculate advanced metrics
            quality_metrics = {
                'file_info': {
                    'path': str(path),
                    'duration': audio_info['duration'],
                    'sample_rate': sr,
                    'samples': len(y)
                },
                'amplitude_analysis': {
                    'max_amplitude': float(np.max(np.abs(y))),
                    'min_amplitude': float(np.min(np.abs(y))),
                    'mean_amplitude': float(np.mean(np.abs(y))),
                    'rms_energy': float(np.sqrt(np.mean(y**2))),
                    'dynamic_range_db': float(20 * np.log10(np.max(np.abs(y)) / (np.mean(np.abs(y)) + 1e-10)))
                },
                'frequency_analysis': {
                    'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
                    'spectral_rolloff': float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))),
                    'spectral_bandwidth': float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))),
                    'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(y)))
                },
                'quality_indicators': {
                    'signal_to_noise_ratio': self._calculate_snr(y),
                    'total_harmonic_distortion': self._calculate_thd(y, sr),
                    'silence_percentage': self._calculate_silence_percentage(y),
                    'clipping_percentage': self._calculate_clipping_percentage(y),
                    'dc_offset': float(np.mean(y))
                },
                'perceptual_metrics': {
                    'loudness_lufs': self._calculate_loudness(y, sr),
                    'pitch_stability': self._calculate_pitch_stability(y, sr),
                    'rhythm_regularity': self._calculate_rhythm_regularity(y, sr)
                },
                'quality_score': 0.0,  # Will be calculated
                'recommendations': [],
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Calculate overall quality score
            quality_metrics['quality_score'] = self._calculate_quality_score(quality_metrics)
            
            # Generate recommendations
            quality_metrics['recommendations'] = self._generate_quality_recommendations(quality_metrics)
            
            self.logger.debug(f"Audio quality analysis completed: {path}")
            return {
                'success': True,
                'file_path': str(path),
                'quality_metrics': quality_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Audio quality analysis failed for {file_path}", e)
            return {'success': False, 'error': str(e)}
    
    @log_function_call
    def extract_audio_features(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract comprehensive audio features for ML/AI analysis.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dict[str, Any]: Extracted features
        """
        try:
            path = Path(file_path)
            
            # Load audio
            y, sr = librosa.load(str(path), sr=None)
            
            # Extract various features
            features = {
                'basic_features': {
                    'duration': len(y) / sr,
                    'sample_rate': sr,
                    'samples': len(y),
                    'rms_energy': float(np.sqrt(np.mean(y**2)))
                },
                'spectral_features': {
                    'mfcc': librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).tolist(),
                    'spectral_centroid': librosa.feature.spectral_centroid(y=y, sr=sr).tolist(),
                    'spectral_rolloff': librosa.feature.spectral_rolloff(y=y, sr=sr).tolist(),
                    'spectral_bandwidth': librosa.feature.spectral_bandwidth(y=y, sr=sr).tolist(),
                    'zero_crossing_rate': librosa.feature.zero_crossing_rate(y).tolist(),
                    'chroma': librosa.feature.chroma_stft(y=y, sr=sr).tolist()
                },
                'temporal_features': {
                    'tempo': float(librosa.beat.tempo(y=y, sr=sr)[0]),
                    'onset_frames': librosa.onset.onset_detect(y=y, sr=sr).tolist(),
                    'beat_frames': librosa.beat.beat_track(y=y, sr=sr)[1].tolist()
                },
                'harmonic_features': {
                    'harmonics': librosa.effects.harmonic(y).tolist(),
                    'percussive': librosa.effects.percussive(y).tolist()
                },
                'extraction_timestamp': datetime.now().isoformat()
            }
            
            self.logger.debug(f"Audio features extracted: {path}")
            return {
                'success': True,
                'file_path': str(path),
                'features': features
            }
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed for {file_path}", e)
            return {'success': False, 'error': str(e)}
    
    def _check_audio_quality(self, file_path: Path, audio_info: Dict[str, Any]) -> List[str]:
        """Check for common audio quality issues."""
        warnings = []
        
        try:
            # Check duration
            if audio_info['duration'] < 0.1:
                warnings.append("Very short audio duration")
            elif audio_info['duration'] > 3600:  # 1 hour
                warnings.append("Very long audio duration")
            
            # Check for silence
            if audio_info['audio_metrics']['silence_percentage'] > 50:
                warnings.append("High percentage of silence detected")
            
            # Check for clipping
            if audio_info['audio_metrics']['clipping_percentage'] > 5:
                warnings.append("Audio clipping detected")
            
            # Check amplitude levels
            if audio_info['audio_metrics']['max_amplitude'] < 0.1:
                warnings.append("Very low audio levels")
            elif audio_info['audio_metrics']['max_amplitude'] > 0.99:
                warnings.append("Audio levels near maximum")
            
        except Exception as e:
            self.logger.error(f"Quality check failed for {file_path}", e)
            warnings.append("Quality check failed")
        
        return warnings
    
    def _convert_with_ffmpeg(self, input_path: Path, output_path: Path, format_type: str) -> bool:
        """Convert audio using ffmpeg."""
        try:
            cmd = [
                'ffmpeg',
                '-i', str(input_path),
                '-y',  # Overwrite output
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
            
        except Exception as e:
            self.logger.error(f"FFmpeg conversion failed", e)
            return False
    
    def _calculate_snr(self, y: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio."""
        try:
            # Simple SNR calculation
            signal_power = np.mean(y**2)
            noise_power = np.var(y - np.mean(y))
            if noise_power > 0:
                snr_db = 10 * np.log10(signal_power / noise_power)
            else:
                snr_db = float('inf')
            return float(snr_db)
        except:
            return 0.0
    
    def _calculate_thd(self, y: np.ndarray, sr: int) -> float:
        """Calculate Total Harmonic Distortion."""
        try:
            # Simplified THD calculation
            fft = np.fft.fft(y)
            freqs = np.fft.fftfreq(len(fft), 1/sr)
            
            # Find fundamental frequency
            magnitude = np.abs(fft)
            fundamental_idx = np.argmax(magnitude[1:len(magnitude)//2]) + 1
            fundamental_power = magnitude[fundamental_idx]**2
            
            # Calculate harmonic powers
            harmonic_power = 0
            for h in range(2, 6):  # 2nd to 5th harmonics
                harmonic_idx = fundamental_idx * h
                if harmonic_idx < len(magnitude)//2:
                    harmonic_power += magnitude[harmonic_idx]**2
            
            if fundamental_power > 0:
                thd = np.sqrt(harmonic_power / fundamental_power) * 100
            else:
                thd = 0
            
            return float(thd)
        except:
            return 0.0
    
    def _calculate_silence_percentage(self, y: np.ndarray, threshold: float = 0.01) -> float:
        """Calculate percentage of silence in audio."""
        silence_frames = np.sum(np.abs(y) < threshold)
        return float((silence_frames / len(y)) * 100)
    
    def _calculate_clipping_percentage(self, y: np.ndarray, threshold: float = 0.95) -> float:
        """Calculate percentage of clipped samples."""
        clipped_samples = np.sum(np.abs(y) > threshold)
        return float((clipped_samples / len(y)) * 100)
    
    def _calculate_loudness(self, y: np.ndarray, sr: int) -> float:
        """Calculate loudness in LUFS (simplified)."""
        try:
            # Simplified loudness calculation
            rms = np.sqrt(np.mean(y**2))
            loudness_lufs = 20 * np.log10(rms + 1e-10) - 23  # Rough LUFS approximation
            return float(loudness_lufs)
        except:
            return -70.0  # Default very quiet value
    
    def _calculate_pitch_stability(self, y: np.ndarray, sr: int) -> float:
        """Calculate pitch stability metric."""
        try:
            # Extract pitch using librosa
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            
            # Get dominant pitch over time
            pitch_track = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_track.append(pitch)
            
            if len(pitch_track) > 1:
                pitch_std = np.std(pitch_track)
                pitch_mean = np.mean(pitch_track)
                stability = 1.0 - (pitch_std / (pitch_mean + 1e-10))
                return float(max(0, min(1, stability)))
            else:
                return 0.0
        except:
            return 0.0
    
    def _calculate_rhythm_regularity(self, y: np.ndarray, sr: int) -> float:
        """Calculate rhythm regularity metric."""
        try:
            # Extract tempo and beats
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            
            if len(beats) > 2:
                # Calculate inter-beat intervals
                intervals = np.diff(beats) / sr
                interval_std = np.std(intervals)
                interval_mean = np.mean(intervals)
                regularity = 1.0 - (interval_std / (interval_mean + 1e-10))
                return float(max(0, min(1, regularity)))
            else:
                return 0.0
        except:
            return 0.0
    
    def _calculate_quality_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score from metrics."""
        try:
            score = 100.0  # Start with perfect score
            
            # Penalize for quality issues
            quality = metrics['quality_indicators']
            
            # SNR penalty
            if quality['signal_to_noise_ratio'] < 20:
                score -= 20
            elif quality['signal_to_noise_ratio'] < 30:
                score -= 10
            
            # Clipping penalty
            if quality['clipping_percentage'] > 5:
                score -= 30
            elif quality['clipping_percentage'] > 1:
                score -= 15
            
            # Silence penalty
            if quality['silence_percentage'] > 50:
                score -= 25
            elif quality['silence_percentage'] > 30:
                score -= 10
            
            # THD penalty
            if quality['total_harmonic_distortion'] > 10:
                score -= 20
            elif quality['total_harmonic_distortion'] > 5:
                score -= 10
            
            return float(max(0, min(100, score)))
        except:
            return 50.0  # Default moderate score
    
    def _generate_quality_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        try:
            quality = metrics['quality_indicators']
            
            if quality['clipping_percentage'] > 1:
                recommendations.append("Reduce input levels to prevent clipping")
            
            if quality['silence_percentage'] > 30:
                recommendations.append("Remove excessive silence from audio")
            
            if quality['signal_to_noise_ratio'] < 20:
                recommendations.append("Improve recording environment to reduce noise")
            
            if abs(quality['dc_offset']) > 0.01:
                recommendations.append("Remove DC offset from audio")
            
            if quality['total_harmonic_distortion'] > 5:
                recommendations.append("Check recording equipment for distortion")
            
            amplitude = metrics['amplitude_analysis']
            if amplitude['max_amplitude'] < 0.3:
                recommendations.append("Increase audio levels for better utilization")
            
        except Exception as e:
            self.logger.error("Failed to generate recommendations", e)
            recommendations.append("Unable to generate specific recommendations")
        
        return recommendations


class AudioValidator:
    """
    Audio file validation class for the TTS/STT testing framework.
    
    This is a simplified validator that wraps AudioProcessor functionality.
    """
    
    def __init__(self):
        """Initialize the audio validator."""
        self.logger = get_logger(__name__)
        self.processor = AudioProcessor()
        
    def validate_audio_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate audio file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dict[str, Any]: Validation results
        """
        return self.processor.validate_audio_file(file_path)
    
    def is_valid_audio_file(self, file_path: Union[str, Path]) -> bool:
        """
        Check if audio file is valid.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            bool: True if valid, False otherwise
        """
        result = self.validate_audio_file(file_path)
        return result.get('is_valid', False)


# Convenience functions
def validate_audio_file(file_path: Union[str, Path]) -> bool:
    """Validate audio file using default AudioProcessor."""
    processor = AudioProcessor()
    result = processor.validate_audio_file(file_path)
    return result['is_valid']

def get_audio_info(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Get audio info using default AudioProcessor."""
    processor = AudioProcessor()
    return processor.get_audio_info(file_path)

def convert_audio(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    target_format: str = 'wav'
) -> bool:
    """Convert audio format using default AudioProcessor."""
    processor = AudioProcessor()
    return processor.convert_audio_format(input_path, output_path, target_format)