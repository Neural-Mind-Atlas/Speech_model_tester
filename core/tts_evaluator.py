# """
# Text-to-Speech (TTS) Evaluator
# ==============================

# Comprehensive evaluation module for TTS models with support for multiple metrics,
# quality assessment, and performance benchmarking.

# Author: AI Testing Team
# Version: 1.0.0
# """

# import logging
# import time
# import hashlib
# import json
# import numpy as np
# from typing import Dict, Any, List, Optional, Tuple, Union
# from pathlib import Path
# from dataclasses import dataclass, asdict
# from datetime import datetime, timezone
# import asyncio
# import librosa
# import soundfile as sf
# from scipy import signal
# from scipy.stats import pearsonr

# # Configure logging
# logger = logging.getLogger(__name__)

# @dataclass
# class TTSEvaluationResult:
#     """Data class for TTS evaluation results."""
#     model_id: str
#     text_input: str
#     audio_output_path: Optional[str]
#     evaluation_metrics: Dict[str, float]
#     quality_scores: Dict[str, float]
#     performance_metrics: Dict[str, float]
#     metadata: Dict[str, Any]
#     timestamp: str
#     success: bool
#     error_message: Optional[str] = None

# @dataclass
# class TTSTestCase:
#     """Data class for TTS test case configuration."""
#     text: str
#     language: str
#     voice_id: Optional[str] = None
#     expected_duration_range: Optional[Tuple[float, float]] = None
#     quality_threshold: float = 0.7
#     test_id: Optional[str] = None

# class TTSEvaluator:
#     """
#     Comprehensive TTS model evaluator with advanced metrics and quality assessment.
    
#     This class provides functionality to evaluate TTS models across multiple dimensions:
#     - Audio quality metrics (SNR, THD, spectral analysis)
#     - Naturalness and intelligibility scoring
#     - Performance metrics (latency, throughput)
#     - Comparative analysis across models
#     """
    
#     def __init__(self, config: Dict[str, Any]):
#         """
#         Initialize TTS evaluator with configuration.
        
#         Args:
#             config (Dict[str, Any]): Evaluation configuration
#         """
#         self.config = config
#         self.evaluation_id = self._generate_evaluation_id()
#         self.results: List[TTSEvaluationResult] = []
        
#         # Initialize evaluation parameters
#         self.sample_rate = config.get('audio', {}).get('sample_rate', 16000)
#         self.quality_thresholds = config.get('testing', {}).get('tts', {}).get('quality_thresholds', {})
#         self.metrics_config = config.get('testing', {}).get('tts', {}).get('metrics', [])
        
#         # Audio analysis parameters
#         self.frame_length = 2048
#         self.hop_length = 512
#         self.n_mels = 128
        
#         logger.info(f"TTS Evaluator initialized with ID: {self.evaluation_id}")
#         logger.debug(f"Sample rate: {self.sample_rate}, Metrics: {self.metrics_config}")

#     def _generate_evaluation_id(self) -> str:
#         """Generate unique evaluation ID."""
#         timestamp = datetime.now(timezone.utc).isoformat()
#         content = f"tts_evaluation_{timestamp}_{id(self)}"
#         return hashlib.md5(content.encode()).hexdigest()[:12]

#     async def evaluate_model(
#         self, 
#         client, 
#         test_cases: List[TTSTestCase],
#         model_config: Dict[str, Any]
#     ) -> List[TTSEvaluationResult]:
#         """
#         Evaluate a TTS model with multiple test cases.
        
#         Args:
#             client: TTS client instance
#             test_cases (List[TTSTestCase]): Test cases to evaluate
#             model_config (Dict[str, Any]): Model configuration
            
#         Returns:
#             List[TTSEvaluationResult]: Evaluation results for all test cases
#         """
#         model_id = model_config.get('model_id', 'unknown')
#         logger.info(f"Starting TTS evaluation for model: {model_id}")
#         logger.info(f"Evaluating {len(test_cases)} test cases")
        
#         model_results = []
        
#         for i, test_case in enumerate(test_cases, 1):
#             logger.info(f"Evaluating test case {i}/{len(test_cases)}: {test_case.test_id or 'unnamed'}")
            
#             try:
#                 result = await self._evaluate_single_test_case(
#                     client, test_case, model_config
#                 )
#                 model_results.append(result)
#                 self.results.append(result)
                
#             except Exception as e:
#                 logger.error(f"Failed to evaluate test case {i}: {e}")
                
#                 # Create failed result
#                 failed_result = TTSEvaluationResult(
#                     model_id=model_id,
#                     text_input=test_case.text,
#                     audio_output_path=None,
#                     evaluation_metrics={},
#                     quality_scores={},
#                     performance_metrics={},
#                     metadata={
#                         'test_case_id': test_case.test_id,
#                         'language': test_case.language,
#                         'voice_id': test_case.voice_id
#                     },
#                     timestamp=datetime.now(timezone.utc).isoformat(),
#                     success=False,
#                     error_message=str(e)
#                 )
#                 model_results.append(failed_result)
#                 self.results.append(failed_result)
        
#         logger.info(f"Completed TTS evaluation for model: {model_id}")
#         return model_results

#     async def _evaluate_single_test_case(
#         self,
#         client,
#         test_case: TTSTestCase,
#         model_config: Dict[str, Any]
#     ) -> TTSEvaluationResult:
#         """
#         Evaluate a single TTS test case.
        
#         Args:
#             client: TTS client instance
#             test_case (TTSTestCase): Test case to evaluate
#             model_config (Dict[str, Any]): Model configuration
            
#         Returns:
#             TTSEvaluationResult: Evaluation result
#         """
#         start_time = time.time()
#         model_id = model_config.get('model_id', 'unknown')
        
#         try:
#             # Generate speech
#             logger.debug(f"Generating speech for text: '{test_case.text[:50]}...'")
#             generation_start = time.time()
            
#             audio_data, audio_path = await self._generate_speech(
#                 client, test_case, model_config
#             )
            
#             generation_time = time.time() - generation_start
#             logger.debug(f"Speech generation completed in {generation_time:.2f}s")
            
#             # Perform comprehensive evaluation
#             evaluation_metrics = await self._calculate_evaluation_metrics(
#                 audio_data, test_case, model_config
#             )
            
#             quality_scores = await self._calculate_quality_scores(
#                 audio_data, test_case
#             )
            
#             performance_metrics = {
#                 'generation_time_seconds': generation_time,
#                 'total_evaluation_time_seconds': time.time() - start_time,
#                 'audio_duration_seconds': len(audio_data) / self.sample_rate,
#                 'real_time_factor': generation_time / (len(audio_data) / self.sample_rate),
#                 'text_length_characters': len(test_case.text),
#                 'text_length_words': len(test_case.text.split()),
#                 'audio_file_size_bytes': Path(audio_path).stat().st_size if audio_path else 0
#             }
            
#             # Compile metadata
#             metadata = {
#                 'test_case_id': test_case.test_id,
#                 'language': test_case.language,
#                 'voice_id': test_case.voice_id,
#                 'sample_rate': self.sample_rate,
#                 'model_config': model_config,
#                 'evaluation_id': self.evaluation_id,
#                 'audio_analysis': await self._analyze_audio_properties(audio_data)
#             }
            
#             result = TTSEvaluationResult(
#                 model_id=model_id,
#                 text_input=test_case.text,
#                 audio_output_path=audio_path,
#                 evaluation_metrics=evaluation_metrics,
#                 quality_scores=quality_scores,
#                 performance_metrics=performance_metrics,
#                 metadata=metadata,
#                 timestamp=datetime.now(timezone.utc).isoformat(),
#                 success=True
#             )
            
#             logger.debug(f"Successfully evaluated test case for model: {model_id}")
#             return result
            
#         except Exception as e:
#             logger.error(f"Error evaluating test case for model {model_id}: {e}")
#             raise

#     async def _generate_speech(
#         self,
#         client,
#         test_case: TTSTestCase,
#         model_config: Dict[str, Any]
#     ) -> Tuple[np.ndarray, str]:
#         """
#         Generate speech using the TTS client.
        
#         Args:
#             client: TTS client instance
#             test_case (TTSTestCase): Test case configuration
#             model_config (Dict[str, Any]): Model configuration
            
#         Returns:
#             Tuple[np.ndarray, str]: Audio data and output file path
#         """
#         # Prepare generation parameters
#         params = {
#             'text': test_case.text,
#             'language': test_case.language,
#             'model_id': model_config.get('model_id'),
#             'voice_id': test_case.voice_id,
#             'sample_rate': self.sample_rate
#         }
        
#         # Add model-specific parameters
#         if 'default_params' in model_config:
#             params.update(model_config['default_params'])
        
#         # Generate speech
#         response = await client.text_to_speech(**params)
        
#         if not response.success:
#             raise RuntimeError(f"TTS generation failed: {response.error}")
        
#         # Load audio data
#         audio_data, sr = librosa.load(response.audio_path, sr=self.sample_rate)
        
#         return audio_data, response.audio_path

#     async def _calculate_evaluation_metrics(
#         self,
#         audio_data: np.ndarray,
#         test_case: TTSTestCase,
#         model_config: Dict[str, Any]
#     ) -> Dict[str, float]:
#         """
#         Calculate comprehensive evaluation metrics.
        
#         Args:
#             audio_data (np.ndarray): Generated audio data
#             test_case (TTSTestCase): Test case configuration
#             model_config (Dict[str, Any]): Model configuration
            
#         Returns:
#             Dict[str, float]: Evaluation metrics
#         """
#         metrics = {}
        
#         try:
#             # Audio duration metrics
#             duration = len(audio_data) / self.sample_rate
#             metrics['audio_duration_seconds'] = duration
            
#             # Text-to-audio ratio
#             text_length = len(test_case.text)
#             metrics['characters_per_second'] = text_length / duration if duration > 0 else 0
#             metrics['words_per_minute'] = (len(test_case.text.split()) * 60) / duration if duration > 0 else 0
            
#             # Audio level metrics
#             if len(audio_data) > 0:
#                 metrics['rms_level'] = float(np.sqrt(np.mean(audio_data ** 2)))
#                 metrics['peak_level'] = float(np.max(np.abs(audio_data)))
#                 metrics['dynamic_range'] = float(metrics['peak_level'] - metrics['rms_level'])
                
#                 # Zero crossing rate (indicator of speech characteristics)
#                 zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
#                 metrics['zero_crossing_rate_mean'] = float(np.mean(zcr))
#                 metrics['zero_crossing_rate_std'] = float(np.std(zcr))
                
#                 # Spectral features
#                 spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
#                 metrics['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
#                 metrics['spectral_centroid_std'] = float(np.std(spectral_centroids))
                
#                 # Mel-frequency cepstral coefficients (MFCC)
#                 mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
#                 metrics['mfcc_mean'] = float(np.mean(mfccs))
#                 metrics['mfcc_std'] = float(np.std(mfccs))
                
#             logger.debug(f"Calculated {len(metrics)} evaluation metrics")
            
#         except Exception as e:
#             logger.error(f"Error calculating evaluation metrics: {e}")
#             metrics['calculation_error'] = str(e)
        
#         return metrics

#     async def _calculate_quality_scores(
#         self,
#         audio_data: np.ndarray,
#         test_case: TTSTestCase
#     ) -> Dict[str, float]:
#         """
#         Calculate audio quality scores.
        
#         Args:
#             audio_data (np.ndarray): Generated audio data
#             test_case (TTSTestCase): Test case configuration
            
#         Returns:
#             Dict[str, float]: Quality scores
#         """
#         scores = {}
        
#         try:
#             if len(audio_data) == 0:
#                 logger.warning("Empty audio data, cannot calculate quality scores")
#                 return {'audio_empty_error': 1.0}
            
#             # Signal-to-Noise Ratio estimation
#             scores['estimated_snr_db'] = self._estimate_snr(audio_data)
            
#             # Audio clarity (high frequency content)
#             scores['clarity_score'] = self._calculate_clarity_score(audio_data)
            
#             # Naturalness score (based on spectral characteristics)
#             scores['naturalness_score'] = self._calculate_naturalness_score(audio_data)
            
#             # Intelligibility score (based on formant analysis)
#             scores['intelligibility_score'] = self._calculate_intelligibility_score(audio_data)
            
#             # Overall quality score (weighted combination)
#             scores['overall_quality_score'] = self._calculate_overall_quality(scores)
            
#             # Quality classification
#             scores['quality_rating'] = self._classify_quality(scores['overall_quality_score'])
            
#             logger.debug(f"Calculated quality scores: {list(scores.keys())}")
            
#         except Exception as e:
#             logger.error(f"Error calculating quality scores: {e}")
#             scores['calculation_error'] = str(e)
        
#         return scores

#     def _estimate_snr(self, audio_data: np.ndarray) -> float:
#         """Estimate Signal-to-Noise Ratio."""
#         try:
#             # Simple SNR estimation using signal power vs noise floor
#             signal_power = np.mean(audio_data ** 2)
            
#             # Estimate noise as the lower 10th percentile of signal power
#             frame_powers = []
#             frame_size = int(0.025 * self.sample_rate)  # 25ms frames
            
#             for i in range(0, len(audio_data) - frame_size, frame_size):
#                 frame = audio_data[i:i + frame_size]
#                 frame_powers.append(np.mean(frame ** 2))
            
#             if frame_powers:
#                 noise_power = np.percentile(frame_powers, 10)
#                 snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
#                 return float(np.clip(snr_db, -20, 60))  # Reasonable SNR range
            
#             return 0.0
            
#         except Exception as e:
#             logger.warning(f"SNR estimation failed: {e}")
#             return 0.0

#     def _calculate_clarity_score(self, audio_data: np.ndarray) -> float:
#         """Calculate audio clarity score based on high-frequency content."""
#         try:
#             # Compute spectrogram
#             stft = librosa.stft(audio_data, hop_length=self.hop_length, n_fft=self.frame_length)
#             magnitude = np.abs(stft)
            
#             # Calculate energy in different frequency bands
#             freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.frame_length)
            
#             # High frequency band (2kHz - 8kHz for speech clarity)
#             high_freq_mask = (freqs >= 2000) & (freqs <= 8000)
#             total_freq_mask = freqs <= 8000
            
#             high_freq_energy = np.mean(magnitude[high_freq_mask, :])
#             total_energy = np.mean(magnitude[total_freq_mask, :])
            
#             clarity = high_freq_energy / (total_energy + 1e-10)
#             return float(np.clip(clarity, 0, 1))
            
#         except Exception as e:
#             logger.warning(f"Clarity score calculation failed: {e}")
#             return 0.0

#     def _calculate_naturalness_score(self, audio_data: np.ndarray) -> float:
#         """Calculate naturalness score based on spectral characteristics."""
#         try:
#             # Extract mel-scale spectrogram
#             mel_spec = librosa.feature.melspectrogram(
#                 y=audio_data, 
#                 sr=self.sample_rate,
#                 n_mels=self.n_mels,
#                 hop_length=self.hop_length
#             )
            
#             # Convert to log scale
#             log_mel_spec = librosa.power_to_db(mel_spec)
            
#             # Calculate spectral smoothness (naturalness indicator)
#             spectral_smoothness = []
#             for frame in log_mel_spec.T:
#                 # Calculate variance of spectral differences
#                 diff_var = np.var(np.diff(frame))
#                 spectral_smoothness.append(diff_var)
            
#             # Lower variance indicates smoother, more natural speech
#             avg_smoothness = np.mean(spectral_smoothness)
            
#             # Normalize to 0-1 scale (lower variance = higher naturalness)
#             naturalness = 1.0 / (1.0 + avg_smoothness / 100.0)
#             return float(np.clip(naturalness, 0, 1))
            
#         except Exception as e:
#             logger.warning(f"Naturalness score calculation failed: {e}")
#             return 0.0

#     def _calculate_intelligibility_score(self, audio_data: np.ndarray) -> float:
#         """Calculate intelligibility score based on formant analysis."""
#         try:
#             # Extract MFCC features (related to speech intelligibility)
#             mfccs = librosa.feature.mfcc(
#                 y=audio_data,
#                 sr=self.sample_rate,
#                 n_mfcc=13,
#                 hop_length=self.hop_length
#             )
            
#             # Calculate stability of MFCC features (more stable = more intelligible)
#             mfcc_stability = []
#             for mfcc_coeff in mfccs:
#                 # Calculate coefficient of variation
#                 cv = np.std(mfcc_coeff) / (np.abs(np.mean(mfcc_coeff)) + 1e-10)
#                 mfcc_stability.append(cv)
            
#             # Average stability across coefficients
#             avg_stability = np.mean(mfcc_stability)
            
#             # Convert to intelligibility score (lower variation = higher intelligibility)
#             intelligibility = 1.0 / (1.0 + avg_stability)
#             return float(np.clip(intelligibility, 0, 1))
            
#         except Exception as e:
#             logger.warning(f"Intelligibility score calculation failed: {e}")
#             return 0.0

#     def _calculate_overall_quality(self, scores: Dict[str, float]) -> float:
#         """Calculate overall quality score from individual metrics."""
#         try:
#             # Define weights for different quality aspects
#             weights = {
#                 'naturalness_score': 0.3,
#                 'intelligibility_score': 0.3,
#                 'clarity_score': 0.2,
#                 'estimated_snr_db': 0.2
#             }
            
#             weighted_sum = 0.0
#             total_weight = 0.0
            
#             for metric, weight in weights.items():
#                 if metric in scores and isinstance(scores[metric], (int, float)):
#                     if metric == 'estimated_snr_db':
#                         # Normalize SNR to 0-1 scale (assuming 20dB is excellent)
#                         normalized_value = np.clip(scores[metric] / 20.0, 0, 1)
#                     else:
#                         normalized_value = scores[metric]
                    
#                     weighted_sum += normalized_value * weight
#                     total_weight += weight
            
#             if total_weight > 0:
#                 overall_score = weighted_sum / total_weight
#                 return float(np.clip(overall_score, 0, 1))
            
#             return 0.0
            
#         except Exception as e:
#             logger.warning(f"Overall quality calculation failed: {e}")
#             return 0.0

#     def _classify_quality(self, score: float) -> str:
#         """Classify quality score into categories."""
#         if score >= 0.8:
#             return "excellent"
#         elif score >= 0.6:
#             return "good"
#         elif score >= 0.4:
#             return "fair"
#         elif score >= 0.2:
#             return "poor"
#         else:
#             return "very_poor"

#     async def _analyze_audio_properties(self, audio_data: np.ndarray) -> Dict[str, Any]:
#         """Analyze audio properties for metadata."""
#         try:
#             properties = {
#                 'length_samples': len(audio_data),
#                 'length_seconds': len(audio_data) / self.sample_rate,
#                 'sample_rate': self.sample_rate,
#                 'channels': 1,  # Assuming mono
#                 'dtype': str(audio_data.dtype),
#                 'min_value': float(np.min(audio_data)),
#                 'max_value': float(np.max(audio_data)),
#                 'mean_value': float(np.mean(audio_data)),
#                 'std_value': float(np.std(audio_data))
#             }
            
#             # Detect silence segments
#             silence_threshold = 0.01 * np.max(np.abs(audio_data))
#             silence_frames = np.abs(audio_data) < silence_threshold
#             properties['silence_percentage'] = float(np.mean(silence_frames) * 100)
            
#             return properties
            
#         except Exception as e:
#             logger.warning(f"Audio properties analysis failed: {e}")
#             return {}

#     def get_evaluation_summary(self) -> Dict[str, Any]:
#         """
#         Get comprehensive evaluation summary.
        
#         Returns:
#             Dict[str, Any]: Evaluation summary with statistics
#         """
#         if not self.results:
#             return {"error": "No evaluation results available"}
        
#         summary = {
#             "evaluation_id": self.evaluation_id,
#             "total_test_cases": len(self.results),
#             "successful_evaluations": sum(1 for r in self.results if r.success),
#             "failed_evaluations": sum(1 for r in self.results if not r.success),
#             "models_evaluated": len(set(r.model_id for r in self.results)),
#             "timestamp": datetime.now(timezone.utc).isoformat()
#         }
        
#         # Calculate aggregate statistics
#         successful_results = [r for r in self.results if r.success]
        
#         if successful_results:
#             # Performance statistics
#             generation_times = [r.performance_metrics.get('generation_time_seconds', 0) 
#                               for r in successful_results]
#             summary["avg_generation_time_seconds"] = float(np.mean(generation_times))
#             summary["min_generation_time_seconds"] = float(np.min(generation_times))
#             summary["max_generation_time_seconds"] = float(np.max(generation_times))
            
#             # Quality statistics
#             quality_scores = [r.quality_scores.get('overall_quality_score', 0) 
#                             for r in successful_results]
#             summary["avg_quality_score"] = float(np.mean(quality_scores))
#             summary["min_quality_score"] = float(np.min(quality_scores))
#             summary["max_quality_score"] = float(np.max(quality_scores))
            
#             # Quality distribution
#             quality_ratings = [r.quality_scores.get('quality_rating', 'unknown') 
#                              for r in successful_results]
#             rating_counts = {}
#             for rating in quality_ratings:
#                 rating_counts[rating] = rating_counts.get(rating, 0) + 1
#             summary["quality_distribution"] = rating_counts
        
#         return summary

#     def export_results(self, format_type: str = "json") -> str:
#         """
#         Export evaluation results in specified format.
        
#         Args:
#             format_type (str): Export format ('json', 'yaml', 'csv')
            
#         Returns:
#             str: Serialized results
#         """
#         if format_type.lower() == "json":
#             results_dict = [asdict(result) for result in self.results]
#             return json.dumps({
#                 "evaluation_summary": self.get_evaluation_summary(),
#                 "results": results_dict
#             }, indent=2, default=str)
        
#         elif format_type.lower() == "yaml":
#             import yaml
#             results_dict = [asdict(result) for result in self.results]
#             return yaml.dump({
#                 "evaluation_summary": self.get_evaluation_summary(),
#                 "results": results_dict
#             }, default_flow_style=False)
        
#         else:
#             raise ValueError(f"Unsupported export format: {format_type}")

#     def clear_results(self):
#         """Clear all evaluation results."""
#         self.results.clear()
#         logger.info("Evaluation results cleared")

# # Module-level utility functions
# def create_test_cases_from_config(config: Dict[str, Any]) -> List[TTSTestCase]:
#     """
#     Create test cases from configuration.
    
#     Args:
#         config (Dict[str, Any]): Configuration dictionary
        
#     Returns:
#         List[TTSTestCase]: Generated test cases
#     """
#     test_cases = []
    
#     try:
#         tts_config = config.get('testing', {}).get('tts_test_params', {})
#         test_texts = tts_config.get('test_texts', {})
        
#         for language, texts in test_texts.items():
#             for i, text in enumerate(texts):
#                 test_case = TTSTestCase(
#                     text=text,
#                     language=language,
#                     test_id=f"{language}_test_{i+1}"
#                 )
#                 test_cases.append(test_case)
        
#         logger.info(f"Created {len(test_cases)} test cases from configuration")
        
#     except Exception as e:
#         logger.error(f"Error creating test cases from config: {e}")
    
#     return test_cases


"""
TTS/STT Testing Framework - TTS Evaluator

This module provides comprehensive evaluation capabilities for Text-to-Speech models.
It includes quality assessment, performance metrics, and detailed analysis.

Author: TTS/STT Testing Framework Team
Version: 1.0.0
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

from utils.logger import get_logger
from utils.audio_utils import AudioProcessor
from utils.metrics_utils import MetricsCalculator
from utils.file_utils import FileManager


@dataclass
class TTSEvaluationResult:
    """Data class for TTS evaluation results."""
    test_id: str
    provider: str
    model_name: str
    input_text: str
    success: bool
    audio_file: Optional[str] = None
    quality_score: float = 0.0
    execution_time: float = 0.0
    metrics: Dict[str, Any] = None
    errors: List[str] = None
    warnings: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


class TTSEvaluator:
    """
    Comprehensive TTS model evaluator.
    
    This class provides evaluation capabilities for TTS models including:
    - Audio quality assessment
    - Performance metrics calculation
    - Intelligibility analysis
    - Comparative evaluation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the TTS evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.audio_processor = AudioProcessor()
        self.metrics_calculator = MetricsCalculator()
        self.file_manager = FileManager()
        
        # Setup output directories
        self.output_dir = Path(config.get('output_data_dir', 'data/outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("TTSEvaluator initialized")
    
    async def evaluate_audio(self, audio_file: str, reference_text: str, 
                           model_name: str, **kwargs) -> Dict[str, Any]:
        """
        Evaluate generated TTS audio.
        
        Args:
            audio_file: Path to generated audio file
            reference_text: Original text that was synthesized
            model_name: Name of the TTS model
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        try:
            start_time = time.time()
            
            # Validate audio file
            validation_result = self.audio_processor.validate_audio_file(audio_file)
            if not validation_result['is_valid']:
                return {
                    'success': False,
                    'errors': validation_result['errors'],
                    'warnings': validation_result['warnings']
                }
            
            # Analyze audio quality
            quality_analysis = self.audio_processor.analyze_audio_quality(audio_file)
            
            # Extract audio features
            features = self.audio_processor.extract_audio_features(audio_file)
            
            # Analyze intelligibility (if STT is available)
            intelligibility = await self._analyze_intelligibility(
                audio_file, reference_text, **kwargs
            )
            
            # Calculate overall quality score
            quality_score = self._calculate_quality_score(
                quality_analysis, features, intelligibility
            )
            
            # Compile metrics
            metrics = {
                'quality_analysis': quality_analysis,
                'audio_features': features,
                'intelligibility': intelligibility,
                'quality_score': quality_score,
                'reference_text_length': len(reference_text),
                'audio_duration': validation_result['audio_info']['duration']
            }
            
            execution_time = time.time() - start_time
            
            return {
                'success': True,
                'metrics': metrics,
                'execution_time': execution_time,
                'warnings': validation_result.get('warnings', [])
            }
            
        except Exception as e:
            self.logger.error(f"TTS evaluation failed: {e}")
            return {
                'success': False,
                'errors': [str(e)],
                'execution_time': time.time() - start_time if 'start_time' in locals() else 0
            }
    
    async def evaluate_batch(self, client, test_cases: List[Dict[str, Any]]) -> List[TTSEvaluationResult]:
        """
        Evaluate multiple TTS test cases.
        
        Args:
            client: TTS client instance
            test_cases: List of test case dictionaries
            
        Returns:
            List[TTSEvaluationResult]: List of evaluation results
        """
        results = []
        
        for i, test_case in enumerate(test_cases):
            try:
                test_id = test_case.get('test_id', f'tts_test_{i}')
                text = test_case['text']
                voice = test_case.get('voice', 'default')
                language = test_case.get('language', 'en')
                
                self.logger.info(f"Evaluating TTS test: {test_id}")
                
                # Generate audio using client
                start_time = time.time()
                tts_response = client.text_to_speech(text, voice=voice, language=language)
                
                if not tts_response.success:
                    result = TTSEvaluationResult(
                        test_id=test_id,
                        provider=client.provider,
                        model_name=client.model_name,
                        input_text=text,
                        success=False,
                        execution_time=time.time() - start_time,
                        errors=[tts_response.error_message or "TTS generation failed"]
                    )
                    results.append(result)
                    continue
                
                # Save audio to file
                audio_file = self.output_dir / f"{test_id}_{client.provider}.wav"
                with open(audio_file, 'wb') as f:
                    f.write(tts_response.audio_data)
                
                # Evaluate generated audio
                evaluation = await self.evaluate_audio(
                    str(audio_file), text, client.model_name
                )
                
                # Create result object
                result = TTSEvaluationResult(
                    test_id=test_id,
                    provider=client.provider,
                    model_name=client.model_name,
                    input_text=text,
                    success=evaluation['success'],
                    audio_file=str(audio_file),
                    quality_score=evaluation.get('metrics', {}).get('quality_score', 0),
                    execution_time=evaluation.get('execution_time', 0),
                    metrics=evaluation.get('metrics', {}),
                    errors=evaluation.get('errors', []),
                    warnings=evaluation.get('warnings', []),
                    metadata={
                        'voice': voice,
                        'language': language,
                        'audio_format': tts_response.audio_format,
                        'sample_rate': tts_response.sample_rate
                    }
                )
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate test case {i}: {e}")
                result = TTSEvaluationResult(
                    test_id=test_case.get('test_id', f'tts_test_{i}'),
                    provider=getattr(client, 'provider', 'unknown'),
                    model_name=getattr(client, 'model_name', 'unknown'),
                    input_text=test_case.get('text', ''),
                    success=False,
                    errors=[str(e)]
                )
                results.append(result)
        
        return results
    
    async def _analyze_intelligibility(self, audio_file: str, reference_text: str, 
                                     **kwargs) -> Dict[str, Any]:
        """
        Analyze audio intelligibility using STT.
        
        Args:
            audio_file: Path to audio file
            reference_text: Original reference text
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Intelligibility analysis results
        """
        try:
            # This would use an STT service to transcribe the audio
            # and compare with reference text
            # For now, return mock data
            
            return {
                'intelligibility_score': 0.85,
                'word_error_rate': 0.15,
                'transcription_confidence': 0.90,
                'transcribed_text': reference_text,  # Mock - would be actual transcription
                'analysis_method': 'mock_stt'
            }
            
        except Exception as e:
            self.logger.warning(f"Intelligibility analysis failed: {e}")
            return {
                'intelligibility_score': 0.0,
                'word_error_rate': 1.0,
                'transcription_confidence': 0.0,
                'error': str(e)
            }
    
    def _calculate_quality_score(self, quality_analysis: Dict[str, Any], 
                               features: Dict[str, Any], 
                               intelligibility: Dict[str, Any]) -> float:
        """
        Calculate overall quality score from various metrics.
        
        Args:
            quality_analysis: Audio quality analysis results
            features: Audio feature analysis results
            intelligibility: Intelligibility analysis results
            
        Returns:
            float: Overall quality score (0-100)
        """
        try:
            # Base score from audio quality
            base_score = quality_analysis.get('quality_metrics', {}).get('overall_score', 50.0)
            
            # Adjust based on intelligibility
            intelligibility_score = intelligibility.get('intelligibility_score', 0.5)
            intelligibility_weight = 0.4
            
            # Adjust based on audio features
            feature_score = 50.0  # Default
            if features.get('success'):
                # Simple feature-based scoring
                feature_score = 70.0  # Mock score
            
            # Weighted combination
            final_score = (
                base_score * 0.5 +
                intelligibility_score * 100 * intelligibility_weight +
                feature_score * 0.1
            )
            
            return max(0.0, min(100.0, final_score))
            
        except Exception as e:
            self.logger.error(f"Quality score calculation failed: {e}")
            return 50.0  # Default moderate score
    
    async def generate_report(self, results: List[TTSEvaluationResult], 
                            format_type: str = 'json', 
                            output_dir: Optional[str] = None) -> Optional[str]:
        """
        Generate evaluation report.
        
        Args:
            results: List of evaluation results
            format_type: Report format ('json', 'yaml', 'html')
            output_dir: Output directory for report
            
        Returns:
            Optional[str]: Path to generated report file
        """
        try:
            if not output_dir:
                output_dir = self.output_dir
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Prepare report data
            report_data = {
                'evaluation_type': 'tts',
                'timestamp': time.time(),
                'total_tests': len(results),
                'successful_tests': sum(1 for r in results if r.success),
                'failed_tests': sum(1 for r in results if not r.success),
                'results': [self._result_to_dict(r) for r in results]
            }
            
            # Generate report based on format
            if format_type == 'json':
                import json
                report_file = output_path / 'tts_evaluation_report.json'
                with open(report_file, 'w') as f:
                    json.dump(report_data, f, indent=2, default=str)
                
            elif format_type == 'yaml':
                import yaml
                report_file = output_path / 'tts_evaluation_report.yaml'
                with open(report_file, 'w') as f:
                    yaml.dump(report_data, f, default_flow_style=False)
                
            elif format_type == 'html':
                report_file = output_path / 'tts_evaluation_report.html'
                html_content = self._generate_html_report(report_data)
                with open(report_file, 'w') as f:
                    f.write(html_content)
            
            else:
                raise ValueError(f"Unsupported report format: {format_type}")
            
            self.logger.info(f"TTS evaluation report generated: {report_file}")
            return str(report_file)
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            return None
    
    def _result_to_dict(self, result: TTSEvaluationResult) -> Dict[str, Any]:
        """Convert TTSEvaluationResult to dictionary."""
        return {
            'test_id': result.test_id,
            'provider': result.provider,
            'model_name': result.model_name,
            'input_text': result.input_text,
            'success': result.success,
            'audio_file': result.audio_file,
            'quality_score': result.quality_score,
            'execution_time': result.execution_time,
            'metrics': result.metrics,
            'errors': result.errors,
            'warnings': result.warnings,
            'metadata': result.metadata
        }
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML report content."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>TTS Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .summary { background: #f0f0f0; padding: 15px; margin-bottom: 20px; }
                .result { border: 1px solid #ddd; margin: 10px 0; padding: 10px; }
                .success { border-left: 5px solid green; }
                .failed { border-left: 5px solid red; }
            </style>
        </head>
        <body>
            <h1>TTS Evaluation Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p>Total Tests: {total_tests}</p>
                <p>Successful: {successful_tests}</p>
                <p>Failed: {failed_tests}</p>
                <p>Success Rate: {success_rate:.1%}</p>
            </div>
            <h2>Detailed Results</h2>
            {results_html}
        </body>
        </html>
        """
        
        # Generate results HTML
        results_html = ""
        for result in report_data['results']:
            status_class = "success" if result['success'] else "failed"
            results_html += f"""
            <div class="result {status_class}">
                <h3>{result['test_id']} - {result['provider']}</h3>
                <p>Model: {result['model_name']}</p>
                <p>Quality Score: {result['quality_score']:.1f}</p>
                <p>Execution Time: {result['execution_time']:.2f}s</p>
                <p>Status: {'Success' if result['success'] else 'Failed'}</p>
            </div>
            """
        
        success_rate = report_data['successful_tests'] / report_data['total_tests'] if report_data['total_tests'] > 0 else 0
        
        return html_template.format(
            total_tests=report_data['total_tests'],
            successful_tests=report_data['successful_tests'],
            failed_tests=report_data['failed_tests'],
            success_rate=success_rate,
            results_html=results_html
        )
