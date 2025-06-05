"""
TTS/STT Testing Framework - TTS Evaluator Tests
==============================================

Comprehensive test suite for TTS evaluation functionality.
Tests include quality metrics, audio analysis, and performance evaluation.

Author: TTS/STT Testing Framework Team
Version: 1.0.0
Created: 2024-06-04
"""

import os
import sys
import pytest
import asyncio
import json
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, Any, List, Optional
import tempfile
import time
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.tts_evaluator import TTSEvaluator, TTSEvaluationResult
from core.evaluator_factory import EvaluatorFactory
from clients.base_client import ClientResponse
from utils.logger import get_logger, setup_logging
from utils.audio_utils import AudioProcessor
from utils.metrics_utils import MetricsCalculator

class TestTTSEvaluator:
    """Test suite for TTSEvaluator class."""
    
    @pytest.fixture(autouse=True)
    def setup_logging(self):
        """Setup logging for tests."""
        setup_logging(log_level="DEBUG", enable_file=False)
        self.logger = get_logger(__name__)
    
    @pytest.fixture
    def sample_config(self):
        """Sample TTS evaluation configuration."""
        return {
            'audio_quality_metrics': {
                'enabled': True,
                'sample_rate': 44100,
                'bit_depth': 16,
                'min_duration': 0.5,
                'max_duration': 300.0
            },
            'naturalness_metrics': {
                'enabled': True,
                'prosody_analysis': True,
                'emotion_detection': True,
                'speech_rate_analysis': True
            },
            'intelligibility_metrics': {
                'enabled': True,
                'clarity_threshold': 0.8,
                'pronunciation_accuracy': True
            },
            'performance_metrics': {
                'enabled': True,
                'latency_threshold': 5.0,
                'throughput_measurement': True
            },
            'output_formats': ['json', 'yaml', 'html'],
            'save_audio_files': True,
            'audio_output_dir': 'test_audio_output'
        }
    
    @pytest.fixture
    def tts_evaluator(self, sample_config):
        """Create TTSEvaluator instance for testing."""
        return TTSEvaluator(sample_config)
    
    @pytest.fixture
    def mock_client(self):
        """Mock TTS client for testing."""
        client = Mock()
        client.client_name = "test_client"
        return client
    
    @pytest.fixture
    def sample_test_cases(self):
        """Sample test cases for TTS evaluation."""
        return [
            {
                'text': 'Hello world',
                'voice': 'default',
                'language': 'en',
                'expected_duration': 1.5,
                'quality_threshold': 0.8
            },
            {
                'text': 'This is a longer sentence to test speech synthesis quality.',
                'voice': 'female',
                'language': 'en',
                'expected_duration': 4.0,
                'quality_threshold': 0.8
            },
            {
                'text': 'Quick brown fox jumps over the lazy dog.',
                'voice': 'male',
                'language': 'en',
                'expected_duration': 3.5,
                'quality_threshold': 0.8
            }
        ]
    
    def test_evaluator_initialization(self, tts_evaluator, sample_config):
        """Test TTS evaluator initialization."""
        assert tts_evaluator.config == sample_config
        assert isinstance(tts_evaluator.audio_processor, AudioProcessor)
        assert isinstance(tts_evaluator.metrics_calculator, MetricsCalculator)
        assert tts_evaluator.evaluation_id is not None
        assert len(tts_evaluator.evaluation_id) == 36  # UUID length
    
    @pytest.mark.asyncio
    async def test_single_evaluation_success(self, tts_evaluator, mock_client):
        """Test successful single TTS evaluation."""
        # Mock successful TTS response
        mock_response = ClientResponse(
            success=True,
            audio_url="/tmp/test_audio.wav",
            metadata={
                'voice': 'default',
                'language': 'en',
                'duration': 1.5,
                'model': 'test_model'
            }
        )
        
        mock_client.text_to_speech = Mock(return_value=mock_response)
        
        # Mock audio analysis
        with patch.object(tts_evaluator.audio_processor, 'validate_audio_file') as mock_validate, \
             patch.object(tts_evaluator.audio_processor, 'analyze_audio_quality') as mock_analyze:
            
            mock_validate.return_value = {
                'is_valid': True,
                'audio_info': {
                    'duration': 1.5,
                    'sample_rate': 44100,
                    'channels': 1
                }
            }
            
            mock_analyze.return_value = {
                'success': True,
                'quality_metrics': {
                    'overall_score': 85.0,
                    'audio_quality': 0.85,
                    'naturalness_score': 0.80,
                    'intelligibility_score': 0.90
                }
            }
            
            result = await tts_evaluator.evaluate_single(
                client=mock_client,
                text="Hello world",
                voice="default",
                language="en"
            )
            
            assert isinstance(result, TTSEvaluationResult)
            assert result.success is True
            assert result.text == "Hello world"
            assert result.voice == "default"
            assert result.audio_url == "/tmp/test_audio.wav"
            assert result.quality_score == 85.0
            assert 'audio_quality' in result.metrics
    
    @pytest.mark.asyncio
    async def test_single_evaluation_client_failure(self, tts_evaluator, mock_client):
        """Test TTS evaluation with client failure."""
        # Mock failed TTS response
        mock_response = ClientResponse(
            success=False,
            error_message="API rate limit exceeded",
            error_code="RATE_LIMIT"
        )
        
        mock_client.text_to_speech = Mock(return_value=mock_response)
        
        result = await tts_evaluator.evaluate_single(
            client=mock_client,
            text="Hello world",
            voice="default",
            language="en"
        )
        
        assert isinstance(result, TTSEvaluationResult)
        assert result.success is False
        assert "API rate limit exceeded" in result.error_message
        assert result.quality_score == 0.0
    
    @pytest.mark.asyncio
    async def test_batch_evaluation_success(self, tts_evaluator, mock_client, sample_test_cases):
        """Test successful batch TTS evaluation."""
        # Mock successful TTS responses
        def mock_tts_response(text, voice, language, **kwargs):
            return ClientResponse(
                success=True,
                audio_url=f"/tmp/test_{hash(text)}.wav",
                metadata={
                    'voice': voice,
                    'language': language,
                    'duration': len(text) * 0.1,  # Simple duration estimation
                    'model': 'test_model'
                }
            )
        
        mock_client.text_to_speech = Mock(side_effect=mock_tts_response)
        
        # Mock audio processing
        with patch.object(tts_evaluator.audio_processor, 'validate_audio_file') as mock_validate, \
             patch.object(tts_evaluator.audio_processor, 'analyze_audio_quality') as mock_analyze:
            
            mock_validate.return_value = {
                'is_valid': True,
                'audio_info': {'duration': 1.5, 'sample_rate': 44100, 'channels': 1}
            }
            
            mock_analyze.return_value = {
                'success': True,
                'quality_metrics': {
                    'overall_score': 85.0,
                    'audio_quality': 0.85,
                    'naturalness_score': 0.80,
                    'intelligibility_score': 0.90
                }
            }
            
            results = await tts_evaluator.evaluate_batch(
                client=mock_client,
                test_cases=sample_test_cases
            )
            
            assert len(results) == len(sample_test_cases)
            assert all(isinstance(result, TTSEvaluationResult) for result in results)
            assert all(result.success for result in results)
            
            # Check that all test cases were processed
            processed_texts = [result.text for result in results]
            expected_texts = [case['text'] for case in sample_test_cases]
            assert set(processed_texts) == set(expected_texts)
    
    @pytest.mark.asyncio
    async def test_batch_evaluation_mixed_results(self, tts_evaluator, mock_client, sample_test_cases):
        """Test batch evaluation with mixed success/failure results."""
        call_count = 0
        
        def mock_tts_response(text, voice, language, **kwargs):
            nonlocal call_count
            call_count += 1
            
            # Fail every second request
            if call_count % 2 == 0:
                return ClientResponse(
                    success=False,
                    error_message="Mock failure",
                    error_code="TEST_ERROR"
                )
            else:
                return ClientResponse(
                    success=True,
                    audio_url=f"/tmp/test_{hash(text)}.wav",
                    metadata={'voice': voice, 'language': language, 'duration': 1.5}
                )
        
        mock_client.text_to_speech = Mock(side_effect=mock_tts_response)
        
        # Mock audio processing for successful cases
        with patch.object(tts_evaluator.audio_processor, 'validate_audio_file') as mock_validate, \
             patch.object(tts_evaluator.audio_processor, 'analyze_audio_quality') as mock_analyze:
            
            mock_validate.return_value = {
                'is_valid': True,
                'audio_info': {'duration': 1.5, 'sample_rate': 44100, 'channels': 1}
            }
            
            mock_analyze.return_value = {
                'success': True,
                'quality_metrics': {
                    'overall_score': 85.0,
                    'audio_quality': 0.85,
                    'naturalness_score': 0.80,
                    'intelligibility_score': 0.90
                }
            }
            
            results = await tts_evaluator.evaluate_batch(
                client=mock_client,
                test_cases=sample_test_cases
            )
            
            assert len(results) == len(sample_test_cases)
            
            # Check that we have both successful and failed results
            successful_results = [r for r in results if r.success]
            failed_results = [r for r in results if not r.success]
            
            assert len(successful_results) > 0
            assert len(failed_results) > 0
    
    def test_audio_quality_analysis(self, tts_evaluator):
        """Test audio quality analysis functionality."""
        # Mock audio file path
        audio_path = "/tmp/test_audio.wav"
        
        # Mock audio processor response
        mock_quality_result = {
            'success': True,
            'quality_metrics': {
                'amplitude_analysis': {
                    'max_amplitude': 0.95,
                    'rms_energy': 0.3,
                    'dynamic_range_db': 40.0
                },
                'frequency_analysis': {
                    'spectral_centroid': 2000.0,
                    'spectral_bandwidth': 1500.0,
                    'zero_crossing_rate': 0.05
                },
                'quality_indicators': {
                    'signal_to_noise_ratio': 25.0,
                    'total_harmonic_distortion': 2.5,
                    'silence_percentage': 5.0,
                    'clipping_percentage': 0.1
                },
                'quality_score': 85.0,
                'recommendations': ["Audio quality is good"]
            }
        }
        
        with patch.object(tts_evaluator.audio_processor, 'analyze_audio_quality', return_value=mock_quality_result):
            metrics = tts_evaluator._analyze_audio_quality(audio_path)
            
            assert metrics['audio_quality'] == 0.85  # Score / 100
            assert metrics['signal_to_noise_ratio'] == 25.0
            assert metrics['dynamic_range'] == 40.0
            assert metrics['has_clipping'] is False  # < 1% clipping
            assert 'recommendations' in metrics
    
    def test_naturalness_analysis(self, tts_evaluator):
        """Test naturalness analysis functionality."""
        audio_path = "/tmp/test_audio.wav"
        text = "Hello world"
        
        # Mock audio features
        mock_features = {
            'success': True,
            'features': {
                'basic_features': {
                    'duration': 1.5,
                    'sample_rate': 44100
                },
                'spectral_features': {
                    'mfcc': [[1.0, 2.0, 3.0]] * 13,
                    'spectral_centroid': [[2000.0]],
                    'zero_crossing_rate': [[0.05]]
                },
                'temporal_features': {
                    'tempo': 120.0
                }
            }
        }
        
        with patch.object(tts_evaluator.audio_processor, 'extract_audio_features', return_value=mock_features):
            metrics = tts_evaluator._analyze_naturalness(audio_path, text)
            
            assert 'naturalness_score' in metrics
            assert 'prosody_score' in metrics
            assert 'speech_rate' in metrics
            assert 'rhythm_regularity' in metrics
            assert isinstance(metrics['naturalness_score'], float)
            assert 0.0 <= metrics['naturalness_score'] <= 1.0
    
    def test_intelligibility_analysis(self, tts_evaluator):
        """Test intelligibility analysis functionality."""
        audio_path = "/tmp/test_audio.wav"
        original_text = "Hello world"
        
        # Mock STT result for intelligibility testing
        with patch('clients.openai_client.OpenAIClient') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            mock_stt_response = ClientResponse(
                success=True,
                text="Hello world",
                metadata={'confidence': 0.95}
            )
            mock_client.speech_to_text = Mock(return_value=mock_stt_response)
            
            # Mock WER calculation
            with patch.object(tts_evaluator.metrics_calculator, 'calculate_wer') as mock_wer:
                mock_wer.return_value = 0.0  # Perfect match
                
                metrics = tts_evaluator._analyze_intelligibility(audio_path, original_text)
                
                assert 'intelligibility_score' in metrics
                assert 'word_error_rate' in metrics
                assert 'transcription_confidence' in metrics
                assert metrics['word_error_rate'] == 0.0
                assert metrics['transcription_confidence'] == 0.95
    
    def test_performance_analysis(self, tts_evaluator):
        """Test performance analysis functionality."""
        start_time = time.time()
        # Simulate some processing time
        time.sleep(0.1)
        end_time = time.time()
        
        audio_size = 1024 * 1024  # 1MB
        text_length = 100
        
        metrics = tts_evaluator._analyze_performance(
            start_time=start_time,
            end_time=end_time,
            audio_size=audio_size,
            text_length=text_length
        )
        
        assert 'latency' in metrics
        assert 'throughput_chars_per_second' in metrics
        assert 'audio_generation_rate' in metrics
        assert metrics['latency'] >= 0.1  # At least 100ms
        assert metrics['throughput_chars_per_second'] > 0
    
    @pytest.mark.asyncio
    async def test_generate_report_json(self, tts_evaluator, sample_test_cases):
        """Test JSON report generation."""
        # Create mock evaluation results
        mock_results = []
        for i, case in enumerate(sample_test_cases):
            result = TTSEvaluationResult(
                success=True,
                text=case['text'],
                voice=case['voice'],
                language=case['language'],
                audio_url=f"/tmp/test_{i}.wav",
                quality_score=85.0 + i,
                latency=1.5 + i * 0.1,
                metrics={
                    'audio_quality': 0.85,
                    'naturalness_score': 0.80,
                    'intelligibility_score': 0.90
                }
            )
            mock_results.append(result)
        
        with patch('utils.file_utils.save_json', return_value=True) as mock_save:
            report_path = await tts_evaluator.generate_report(
                results=mock_results,
                output_format='json',
                output_dir='/tmp/test_reports'
            )
            
            assert report_path is not None
            mock_save.assert_called_once()
            
            # Check the structure of saved data
            call_args = mock_save.call_args[0]
            report_data = call_args[0]
            
            assert 'evaluation_summary' in report_data
            assert 'individual_results' in report_data
            assert 'statistics' in report_data
            assert len(report_data['individual_results']) == len(sample_test_cases)
    
    @pytest.mark.asyncio
    async def test_generate_report_html(self, tts_evaluator, sample_test_cases):
        """Test HTML report generation."""
        # Create mock evaluation results
        mock_results = []
        for i, case in enumerate(sample_test_cases):
            result = TTSEvaluationResult(
                success=True,
                text=case['text'],
                voice=case['voice'],
                language=case['language'],
                audio_url=f"/tmp/test_{i}.wav",
                quality_score=85.0 + i,
                latency=1.5 + i * 0.1,
                metrics={
                    'audio_quality': 0.85,
                    'naturalness_score': 0.80,
                    'intelligibility_score': 0.90
                }
            )
            mock_results.append(result)
        
        with patch('builtins.open', create=True) as mock_open, \
             patch('pathlib.Path.write_text') as mock_write:
            
            report_path = await tts_evaluator.generate_report(
                results=mock_results,
                output_format='html',
                output_dir='/tmp/test_reports'
            )
            
            assert report_path is not None
            assert '.html' in str(report_path)
    
    def test_calculate_overall_score(self, tts_evaluator):
        """Test overall score calculation."""
        metrics = {
            'audio_quality': 0.85,
            'naturalness_score': 0.80,
            'intelligibility_score': 0.90,
            'latency': 2.0,  # seconds
            'signal_to_noise_ratio': 25.0
        }
        
        score = tts_evaluator._calculate_overall_score(metrics)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        valid_config = {
            'audio_quality_metrics': {'enabled': True},
            'naturalness_metrics': {'enabled': True},
            'intelligibility_metrics': {'enabled': True},
            'performance_metrics': {'enabled': True},
            'output_formats': ['json'],
            'save_audio_files': True
        }
        
        evaluator = TTSEvaluator(valid_config)
        assert evaluator.config == valid_config
        
        # Invalid config - missing required fields
        invalid_config = {
            'audio_quality_metrics': {'enabled': True}
            # Missing other required fields
        }
        
        with pytest.raises(ValueError, match="Missing required configuration"):
            TTSEvaluator(invalid_config)

class TestTTSEvaluationResult:
    """Test suite for TTSEvaluationResult class."""
    
    def test_result_creation_success(self):
        """Test successful evaluation result creation."""
        result = TTSEvaluationResult(
            success=True,
            text="Hello world",
            voice="default",
            language="en",
            audio_url="/tmp/test.wav",
            quality_score=85.0,
            latency=1.5,
            metrics={
                'audio_quality': 0.85,
                'naturalness_score': 0.80
            }
        )
        
        assert result.success is True
        assert result.text == "Hello world"
        assert result.voice == "default"
        assert result.language == "en"
        assert result.audio_url == "/tmp/test.wav"
        assert result.quality_score == 85.0
        assert result.latency == 1.5
        assert result.metrics['audio_quality'] == 0.85
        assert isinstance(result.timestamp, str)
        assert result.error_message is None
    
    def test_result_creation_failure(self):
        """Test failed evaluation result creation."""
        result = TTSEvaluationResult(
            success=False,
            text="Hello world",
            voice="default",
            language="en",
            error_message="TTS generation failed",
            error_code="TTS_ERROR"
        )
        
        assert result.success is False
        assert result.text == "Hello world"
        assert result.error_message == "TTS generation failed"
        assert result.error_code == "TTS_ERROR"
        assert result.quality_score == 0.0
        assert result.latency == 0.0
        assert result.audio_url is None
    
    def test_result_to_dict(self):
        """Test conversion of result to dictionary."""
        result = TTSEvaluationResult(
            success=True,
            text="Hello world",
            voice="default",
            language="en",
            audio_url="/tmp/test.wav",
            quality_score=85.0,
            latency=1.5,
            metrics={'audio_quality': 0.85}
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['success'] is True
        assert result_dict['text'] == "Hello world"
        assert result_dict['quality_score'] == 85.0
        assert 'timestamp' in result_dict
        assert 'metrics' in result_dict

class TestPerformanceAndIntegration:
    """Performance and integration tests for TTS evaluator."""
    
    @pytest.fixture(autouse=True)
    def setup_logging(self):
        """Setup logging for tests."""
        setup_logging(log_level="DEBUG", enable_file=False)
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_large_batch_evaluation(self):
        """Test evaluation with a large batch of test cases."""
        config = {
            'audio_quality_metrics': {'enabled': True},
            'naturalness_metrics': {'enabled': True},
            'intelligibility_metrics': {'enabled': True},
            'performance_metrics': {'enabled': True},
            'output_formats': ['json'],
            'save_audio_files': False  # Disable to speed up test
        }
        
        evaluator = TTSEvaluator(config)
        mock_client = Mock()
        
        # Create large test case list
        test_cases = [
            {
                'text': f"Test sentence number {i} for evaluation",
                'voice': 'default',
                'language': 'en'
            }
            for i in range(50)  # 50 test cases
        ]
        
        # Mock successful responses
        def mock_tts_response(text, voice, language, **kwargs):
            return ClientResponse(
                success=True,
                audio_url=f"/tmp/test_{hash(text)}.wav",
                metadata={'voice': voice, 'language': language, 'duration': 2.0}
            )
        
        mock_client.text_to_speech = Mock(side_effect=mock_tts_response)
        
        # Mock audio processing
        with patch.object(evaluator.audio_processor, 'validate_audio_file') as mock_validate, \
             patch.object(evaluator.audio_processor, 'analyze_audio_quality') as mock_analyze:
            
            mock_validate.return_value = {
                'is_valid': True,
                'audio_info': {'duration': 2.0, 'sample_rate': 44100, 'channels': 1}
            }
            
            mock_analyze.return_value = {
                'success': True,
                'quality_metrics': {
                    'overall_score': 85.0,
                    'audio_quality': 0.85,
                    'naturalness_score': 0.80,
                    'intelligibility_score': 0.90
                }
            }
            
            start_time = time.time()
            results = await evaluator.evaluate_batch(mock_client, test_cases)
            end_time = time.time()
            
            # Verify results
            assert len(results) == 50
            assert all(result.success for result in results)
            
            # Performance check - should complete within reasonable time
            total_time = end_time - start_time
            assert total_time < 60  # Should complete within 1 minute
            
            # Calculate average processing time per case
            avg_time_per_case = total_time / len(test_cases)
            assert avg_time_per_case < 2.0  # Less than 2 seconds per case
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_end_to_end_evaluation_workflow(self):
        """Test complete end-to-end evaluation workflow."""
        config = {
            'audio_quality_metrics': {'enabled': True},
            'naturalness_metrics': {'enabled': True},
            'intelligibility_metrics': {'enabled': True},
            'performance_metrics': {'enabled': True},
            'output_formats': ['json', 'yaml', 'html'],
            'save_audio_files': True,
            'audio_output_dir': '/tmp/test_tts_audio'
        }
        
        evaluator = TTSEvaluator(config)
        mock_client = Mock()
        mock_client.client_name = "test_client"
        
        test_cases = [
            {
                'text': 'Hello world',
                'voice': 'default',
                'language': 'en'
            },
            {
                'text': 'This is a test of text-to-speech synthesis.',
                'voice': 'female',
                'language': 'en'
            }
        ]
        
        # Mock TTS responses
        def mock_tts_response(text, voice, language, **kwargs):
            return ClientResponse(
                success=True,
                audio_url=f"/tmp/test_{hash(text)}.wav",
                metadata={'voice': voice, 'language': language, 'duration': len(text) * 0.1}
            )
        
        mock_client.text_to_speech = Mock(side_effect=mock_tts_response)
        
        # Mock all audio processing steps
        with patch.object(evaluator.audio_processor, 'validate_audio_file') as mock_validate, \
             patch.object(evaluator.audio_processor, 'analyze_audio_quality') as mock_analyze, \
             patch.object(evaluator.audio_processor, 'extract_audio_features') as mock_features, \
             patch.object(evaluator, '_analyze_intelligibility') as mock_intelligibility:
            
            mock_validate.return_value = {
                'is_valid': True,
                'audio_info': {'duration': 2.0, 'sample_rate': 44100, 'channels': 1}
            }
            
            mock_analyze.return_value = {
                'success': True,
                'quality_metrics': {
                    'overall_score': 85.0,
                    'amplitude_analysis': {'rms_energy': 0.3},
                    'frequency_analysis': {'spectral_centroid': 2000.0},
                    'quality_indicators': {'signal_to_noise_ratio': 25.0}
                }
            }
            
            mock_features.return_value = {
                'success': True,
                'features': {
                    'temporal_features': {'tempo': 120.0},
                    'spectral_features': {'mfcc': [[1.0]] * 13}
                }
            }
            
            mock_intelligibility.return_value = {
                'intelligibility_score': 0.90,
                'word_error_rate': 0.05,
                'transcription_confidence': 0.95
            }
            
            # Mock report generation
            with patch('utils.file_utils.save_json', return_value=True), \
                 patch('utils.file_utils.save_yaml', return_value=True), \
                 patch('pathlib.Path.write_text'), \
                 patch('pathlib.Path.mkdir'):
                
                # Execute full workflow
                results = await evaluator.evaluate_batch(mock_client, test_cases)
                
                # Generate reports
                json_report = await evaluator.generate_report(results, 'json', '/tmp/reports')
                yaml_report = await evaluator.generate_report(results, 'yaml', '/tmp/reports')
                html_report = await evaluator.generate_report(results, 'html', '/tmp/reports')
                
                # Verify results
                assert len(results) == 2
                assert all(result.success for result in results)
                assert all(result.quality_score > 0 for result in results)
                
                # Verify reports were generated
                assert json_report is not None
                assert yaml_report is not None
                assert html_report is not None

# Test utilities and fixtures

@pytest.fixture(scope="session")
def sample_wav_file():
    """Create a sample WAV file for testing."""
    import wave
    import struct
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        # Create a simple WAV file
        with wave.open(f.name, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(44100)  # 44.1kHz
            
            # Generate 1 second of sine wave
            duration = 1.0
            frequency = 440.0  # A4 note
            frames = int(44100 * duration)
            
            for i in range(frames):
                value = int(32767 * np.sin(2 * np.pi * frequency * i / 44100))
                wav_file.writeframes(struct.pack('<h', value))
        
        return f.name

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main(["-v", "--tb=short", __file__])