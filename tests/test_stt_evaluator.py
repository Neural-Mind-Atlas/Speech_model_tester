"""
TTS/STT Testing Framework - STT Evaluator Tests
==============================================

Comprehensive test suite for STT evaluation functionality.
Tests include accuracy metrics, performance evaluation, and error analysis.

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

from core.stt_evaluator import STTEvaluator, STTEvaluationResult
from core.evaluator_factory import EvaluatorFactory
from clients.base_client import ClientResponse
from utils.logger import get_logger, setup_logging
from utils.audio_utils import AudioProcessor
from utils.metrics_utils import MetricsCalculator, MetricResult

class TestSTTEvaluator:
    """Test suite for STTEvaluator class."""
    
    @pytest.fixture(autouse=True)
    def setup_logging(self):
        """Setup logging for tests."""
        setup_logging(log_level="DEBUG", enable_file=False)
        self.logger = get_logger(__name__)
    
    @pytest.fixture
    def sample_config(self):
        """Sample STT evaluation configuration."""
        return {
            'accuracy_metrics': {
                'enabled': True,
                'calculate_wer': True,
                'calculate_cer': True,
                'calculate_bleu': True,
                'semantic_similarity': True
            },
            'performance_metrics': {
                'enabled': True,
                'latency_threshold': 10.0,
                'throughput_measurement': True,
                'real_time_factor': True
            },
            'audio_analysis': {
                'enabled': True,
                'validate_audio_format': True,
                'extract_audio_features': True
            },
            'confidence_analysis': {
                'enabled': True,
                'confidence_threshold': 0.8,
                'analyze_low_confidence_segments': True
            },
            'error_analysis': {
                'enabled': True,
                'detailed_error_categorization': True,
                'common_error_patterns': True
            },
            'output_formats': ['json', 'yaml', 'html'],
            'save_transcriptions': True,
            'transcription_output_dir': 'test_stt_output'
        }
    
    @pytest.fixture
    def stt_evaluator(self, sample_config):
        """Create STTEvaluator instance for testing."""
        return STTEvaluator(sample_config)
    
    @pytest.fixture
    def mock_client(self):
        """Mock STT client for testing."""
        client = Mock()
        client.client_name = "test_stt_client"
        return client
    
    @pytest.fixture
    def sample_test_cases(self):
        """Sample test cases for STT evaluation."""
        return [
            {
                'audio_path': '/tmp/test_audio_1.wav',
                'reference_text': 'Hello world',
                'language': 'en',
                'expected_confidence': 0.9
            },
            {
                'audio_path': '/tmp/test_audio_2.wav',
                'reference_text': 'This is a longer sentence to test speech recognition accuracy.',
                'language': 'en',
                'expected_confidence': 0.85
            },
            {
                'audio_path': '/tmp/test_audio_3.wav',
                'reference_text': 'Quick brown fox jumps over the lazy dog.',
                'language': 'en',
                'expected_confidence': 0.8
            }
        ]
    
    def test_evaluator_initialization(self, stt_evaluator, sample_config):
        """Test STT evaluator initialization."""
        assert stt_evaluator.config == sample_config
        assert isinstance(stt_evaluator.audio_processor, AudioProcessor)
        assert isinstance(stt_evaluator.metrics_calculator, MetricsCalculator)
        assert stt_evaluator.evaluation_id is not None
        assert len(stt_evaluator.evaluation_id) == 36  # UUID length
    
    @pytest.mark.asyncio
    async def test_single_evaluation_success(self, stt_evaluator, mock_client):
        """Test successful single STT evaluation."""
        # Mock successful STT response
        mock_response = ClientResponse(
            success=True,
            text="Hello world",
            metadata={
                'confidence': 0.95,
                'language': 'en',
                'duration': 1.5,
                'model': 'test_model'
            }
        )
        
        mock_client.speech_to_text = Mock(return_value=mock_response)
        
        # Mock audio validation
        with patch.object(stt_evaluator.audio_processor, 'validate_audio_file') as mock_validate, \
             patch.object(stt_evaluator.metrics_calculator, 'calculate_wer') as mock_wer, \
             patch.object(stt_evaluator.metrics_calculator, 'calculate_cer') as mock_cer:
            
            mock_validate.return_value = {
                'is_valid': True,
                'audio_info': {
                    'duration': 1.5,
                    'sample_rate': 16000,
                    'channels': 1
                }
            }
            
            # Mock WER calculation
            mock_wer.return_value = MetricResult(
                value=0.0,  # Perfect match
                details={
                    'reference_text': 'Hello world',
                    'hypothesis_text': 'Hello world',
                    'substitutions': 0,
                    'insertions': 0,
                    'deletions': 0,
                    'correct_words': 2
                },
                timestamp=datetime.now().isoformat()
            )
            
            # Mock CER calculation
            mock_cer.return_value = MetricResult(
                value=0.0,  # Perfect match
                details={
                    'reference_text': 'Hello world',
                    'hypothesis_text': 'Hello world',
                    'edit_distance': 0
                },
                timestamp=datetime.now().isoformat()
            )
            
            result = await stt_evaluator.evaluate_single(
                client=mock_client,
                audio_path="/tmp/test_audio.wav",
                reference_text="Hello world",
                language="en"
            )
            
            assert isinstance(result, STTEvaluationResult)
            assert result.success is True
            assert result.audio_path == "/tmp/test_audio.wav"
            assert result.reference_text == "Hello world"
            assert result.predicted_text == "Hello world"
            assert result.word_error_rate == 0.0
            assert result.character_error_rate == 0.0
            assert result.confidence_score == 0.95
            assert 'wer_details' in result.metrics
    
    @pytest.mark.asyncio
    async def test_single_evaluation_client_failure(self, stt_evaluator, mock_client):
        """Test STT evaluation with client failure."""
        # Mock failed STT response
        mock_response = ClientResponse(
            success=False,
            error_message="Audio format not supported",
            error_code="UNSUPPORTED_FORMAT"
        )
        
        mock_client.speech_to_text = Mock(return_value=mock_response)
        
        result = await stt_evaluator.evaluate_single(
            client=mock_client,
            audio_path="/tmp/test_audio.wav",
            reference_text="Hello world",
            language="en"
        )
        
        assert isinstance(result, STTEvaluationResult)
        assert result.success is False
        assert "Audio format not supported" in result.error_message
        assert result.word_error_rate == float('inf')
        assert result.character_error_rate == float('inf')
        assert result.accuracy_score == 0.0
    
    @pytest.mark.asyncio
    async def test_single_evaluation_audio_validation_failure(self, stt_evaluator, mock_client):
        """Test STT evaluation with audio validation failure."""
        # Mock audio validation failure
        with patch.object(stt_evaluator.audio_processor, 'validate_audio_file') as mock_validate:
            mock_validate.return_value = {
                'is_valid': False,
                'errors': ['File does not exist'],
                'audio_info': None
            }
            
            result = await stt_evaluator.evaluate_single(
                client=mock_client,
                audio_path="/tmp/nonexistent.wav",
                reference_text="Hello world",
                language="en"
            )
            
            assert result.success is False
            assert "Audio validation failed" in result.error_message
    
    @pytest.mark.asyncio
    async def test_batch_evaluation_success(self, stt_evaluator, mock_client, sample_test_cases):
        """Test successful batch STT evaluation."""
        # Mock successful STT responses
        def mock_stt_response(audio_path, language, **kwargs):
            # Simple mapping of test files to expected responses
            if "test_audio_1" in audio_path:
                return ClientResponse(
                    success=True,
                    text="Hello world",
                    metadata={'confidence': 0.95, 'language': language}
                )
            elif "test_audio_2" in audio_path:
                return ClientResponse(
                    success=True,
                    text="This is a longer sentence to test speech recognition accuracy.",
                    metadata={'confidence': 0.90, 'language': language}
                )
            elif "test_audio_3" in audio_path:
                return ClientResponse(
                    success=True,
                    text="Quick brown fox jumps over the lazy dog.",
                    metadata={'confidence': 0.85, 'language': language}
                )
            else:
                return ClientResponse(
                    success=False,
                    error_message="Unknown test file"
                )
        
        mock_client.speech_to_text = Mock(side_effect=mock_stt_response)
        
        # Mock audio processing and metrics
        with patch.object(stt_evaluator.audio_processor, 'validate_audio_file') as mock_validate, \
             patch.object(stt_evaluator.metrics_calculator, 'calculate_wer') as mock_wer, \
             patch.object(stt_evaluator.metrics_calculator, 'calculate_cer') as mock_cer:
            
            mock_validate.return_value = {
                'is_valid': True,
                'audio_info': {'duration': 2.0, 'sample_rate': 16000, 'channels': 1}
            }
            
            # Mock perfect matches for simplicity
            mock_wer.return_value = MetricResult(
                value=0.0,
                details={'substitutions': 0, 'insertions': 0, 'deletions': 0},
                timestamp=datetime.now().isoformat()
            )
            
            mock_cer.return_value = MetricResult(
                value=0.0,
                details={'edit_distance': 0},
                timestamp=datetime.now().isoformat()
            )
            
            results = await stt_evaluator.evaluate_batch(
                client=mock_client,
                test_cases=sample_test_cases
            )
            
            assert len(results) == len(sample_test_cases)
            assert all(isinstance(result, STTEvaluationResult) for result in results)
            assert all(result.success for result in results)
            
            # Check that all test cases were processed
            processed_paths = [result.audio_path for result in results]
            expected_paths = [case['audio_path'] for case in sample_test_cases]
            assert set(processed_paths) == set(expected_paths)
    
    @pytest.mark.asyncio
    async def test_batch_evaluation_mixed_results(self, stt_evaluator, mock_client, sample_test_cases):
        """Test batch evaluation with mixed success/failure results."""
        call_count = 0
        
        def mock_stt_response(audio_path, language, **kwargs):
            nonlocal call_count
            call_count += 1
            
            # Fail every second request
            if call_count % 2 == 0:
                return ClientResponse(
                    success=False,
                    error_message="Mock transcription failure",
                    error_code="STT_ERROR"
                )
            else:
                return ClientResponse(
                    success=True,
                    text="Mock transcription",
                    metadata={'confidence': 0.8, 'language': language}
                )
        
        mock_client.speech_to_text = Mock(side_effect=mock_stt_response)
        
        # Mock audio processing for successful cases
        with patch.object(stt_evaluator.audio_processor, 'validate_audio_file') as mock_validate, \
             patch.object(stt_evaluator.metrics_calculator, 'calculate_wer') as mock_wer, \
             patch.object(stt_evaluator.metrics_calculator, 'calculate_cer') as mock_cer:
            
            mock_validate.return_value = {
                'is_valid': True,
                'audio_info': {'duration': 2.0, 'sample_rate': 16000, 'channels': 1}
            }
            
            mock_wer.return_value = MetricResult(
                value=0.5,  # 50% WER
                details={'substitutions': 1, 'insertions': 0, 'deletions': 0},
                timestamp=datetime.now().isoformat()
            )
            
            mock_cer.return_value = MetricResult(
                value=0.3,  # 30% CER
                details={'edit_distance': 3},
                timestamp=datetime.now().isoformat()
            )
            
            results = await stt_evaluator.evaluate_batch(
                client=mock_client,
                test_cases=sample_test_cases
            )
            
            assert len(results) == len(sample_test_cases)
            
            # Check that we have both successful and failed results
            successful_results = [r for r in results if r.success]
            failed_results = [r for r in results if not r.success]
            
            assert len(successful_results) > 0
            assert len(failed_results) > 0
    
    def test_wer_calculation(self, stt_evaluator):
        """Test Word Error Rate calculation."""
        reference = "Hello world this is a test"
        hypothesis = "Hello world this was a test"  # 1 substitution
        
        with patch.object(stt_evaluator.metrics_calculator, 'calculate_wer') as mock_wer:
            mock_wer.return_value = MetricResult(
                value=1/6,  # 1 error out of 6 words
                details={
                    'reference_text': reference,
                    'hypothesis_text': hypothesis,
                    'reference_word_count': 6,
                    'hypothesis_word_count': 6,
                    'substitutions': 1,
                    'insertions': 0,
                    'deletions': 0,
                    'correct_words': 5,
                    'edit_distance': 1
                },
                timestamp=datetime.now().isoformat()
            )
            
            metrics = stt_evaluator._calculate_accuracy_metrics(reference, hypothesis)
            
            assert metrics['word_error_rate'] == 1/6
            assert metrics['word_accuracy'] == 5/6
            assert 'wer_details' in metrics
    
    def test_cer_calculation(self, stt_evaluator):
        """Test Character Error Rate calculation."""
        reference = "hello"
        hypothesis = "helo"  # 1 deletion
        
        with patch.object(stt_evaluator.metrics_calculator, 'calculate_cer') as mock_cer:
            mock_cer.return_value = MetricResult(
                value=1/5,  # 1 error out of 5 characters
                details={
                    'reference_text': reference,
                    'hypothesis_text': hypothesis,
                    'reference_char_count': 5,
                    'hypothesis_char_count': 4,
                    'substitutions': 0,
                    'insertions': 0,
                    'deletions': 1,
                    'edit_distance': 1
                },
                timestamp=datetime.now().isoformat()
            )
            
            metrics = stt_evaluator._calculate_accuracy_metrics(reference, hypothesis)
            
            assert metrics['character_error_rate'] == 1/5
            assert metrics['character_accuracy'] == 4/5
            assert 'cer_details' in metrics
    
    def test_bleu_score_calculation(self, stt_evaluator):
        """Test BLEU score calculation."""
        reference = "Hello world this is a test"
        hypothesis = "Hello world this is a test"  # Perfect match
        
        with patch.object(stt_evaluator.metrics_calculator, 'calculate_bleu_score') as mock_bleu:
            mock_bleu.return_value = MetricResult(
                value=1.0,  # Perfect BLEU score
                details={
                    'reference_text': reference,
                    'hypothesis_text': hypothesis,
                    'precisions': [1.0, 1.0, 1.0, 1.0],
                    'brevity_penalty': 1.0
                },
                timestamp=datetime.now().isoformat()
            )
            
            metrics = stt_evaluator._calculate_accuracy_metrics(reference, hypothesis)
            
            assert metrics['bleu_score'] == 1.0
            assert 'bleu_details' in metrics
    
    def test_semantic_similarity_calculation(self, stt_evaluator):
        """Test semantic similarity calculation."""
        reference = "Hello world"
        hypothesis = "Hi world"  # Similar meaning, different words
        
        with patch.object(stt_evaluator.metrics_calculator, 'calculate_semantic_similarity') as mock_sim:
            mock_sim.return_value = MetricResult(
                value=0.75,  # Good semantic similarity
                details={
                    'reference_text': reference,
                    'hypothesis_text': hypothesis,
                    'method': 'jaccard',
                    'common_words': ['world'],
                    'reference_word_count': 2,
                    'hypothesis_word_count': 2
                },
                timestamp=datetime.now().isoformat()
            )
            
            metrics = stt_evaluator._calculate_accuracy_metrics(reference, hypothesis)
            
            assert metrics['semantic_similarity'] == 0.75
            assert 'similarity_details' in metrics
    
    def test_confidence_analysis(self, stt_evaluator):
        """Test confidence score analysis."""
        predicted_text = "Hello world this is a test"
        confidence_score = 0.85
        
        # Mock word-level confidence scores
        word_confidences = [0.9, 0.95, 0.8, 0.7, 0.9, 0.85]
        
        metrics = stt_evaluator._analyze_confidence(
            predicted_text, confidence_score, word_confidences
        )
        
        assert metrics['overall_confidence'] == 0.85
        assert metrics['average_word_confidence'] == sum(word_confidences) / len(word_confidences)
        assert metrics['min_word_confidence'] == min(word_confidences)
        assert metrics['max_word_confidence'] == max(word_confidences)
        assert metrics['low_confidence_word_count'] == 1  # 0.7 < 0.8 threshold
        assert 'confidence_distribution' in metrics
    
    def test_error_analysis(self, stt_evaluator):
        """Test detailed error analysis."""
        reference = "Hello world this is a test"
        hypothesis = "Hello word this was the test"  # Multiple errors
        
        # Mock WER details with specific error types
        wer_details = {
            'alignment': [
                {'operation': 'match', 'reference': 'Hello', 'hypothesis': 'Hello'},
                {'operation': 'substitute', 'reference': 'world', 'hypothesis': 'word'},
                {'operation': 'match', 'reference': 'this', 'hypothesis': 'this'},
                {'operation': 'substitute', 'reference': 'is', 'hypothesis': 'was'},
                {'operation': 'substitute', 'reference': 'a', 'hypothesis': 'the'},
                {'operation': 'match', 'reference': 'test', 'hypothesis': 'test'}
            ],
            'substitutions': 3,
            'insertions': 0,
            'deletions': 0
        }
        
        error_analysis = stt_evaluator._analyze_errors(reference, hypothesis, wer_details)
        
        assert error_analysis['total_errors'] == 3
        assert error_analysis['substitution_errors'] == 3
        assert error_analysis['insertion_errors'] == 0
        assert error_analysis['deletion_errors'] == 0
        assert 'error_patterns' in error_analysis
        assert 'error_positions' in error_analysis
    
    def test_performance_analysis(self, stt_evaluator):
        """Test performance metrics analysis."""
        start_time = time.time()
        time.sleep(0.1)  # Simulate processing time
        end_time = time.time()
        
        audio_duration = 5.0  # 5 seconds of audio
        
        metrics = stt_evaluator._analyze_performance(
            start_time=start_time,
            end_time=end_time,
            audio_duration=audio_duration
        )
        
        assert 'latency' in metrics
        assert 'real_time_factor' in metrics
        assert 'processing_speed' in metrics
        assert metrics['latency'] >= 0.1  # At least 100ms
        assert metrics['real_time_factor'] > 0
        assert metrics['processing_speed'] > 0
    
    @pytest.mark.asyncio
    async def test_generate_report_json(self, stt_evaluator, sample_test_cases):
        """Test JSON report generation."""
        # Create mock evaluation results
        mock_results = []
        for i, case in enumerate(sample_test_cases):
            result = STTEvaluationResult(
                success=True,
                audio_path=case['audio_path'],
                reference_text=case['reference_text'],
                predicted_text=case['reference_text'],  # Perfect match for simplicity
                language=case['language'],
                word_error_rate=0.0,
                character_error_rate=0.0,
                confidence_score=0.9 + i * 0.01,
                accuracy_score=100.0,
                latency=2.0 + i * 0.1,
                metrics={
                    'bleu_score': 1.0,
                    'semantic_similarity': 1.0,
                    'word_accuracy': 1.0
                }
            )
            mock_results.append(result)
        
        with patch('utils.file_utils.save_json', return_value=True) as mock_save:
            report_path = await stt_evaluator.generate_report(
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
            assert 'error_analysis' in report_data
            assert len(report_data['individual_results']) == len(sample_test_cases)
    
    @pytest.mark.asyncio
    async def test_generate_report_html(self, stt_evaluator, sample_test_cases):
        """Test HTML report generation."""
        # Create mock evaluation results
        mock_results = []
        for i, case in enumerate(sample_test_cases):
            result = STTEvaluationResult(
                success=True,
                audio_path=case['audio_path'],
                reference_text=case['reference_text'],
                predicted_text=case['reference_text'] + " extra",  # Small difference
                language=case['language'],
                word_error_rate=0.1,
                character_error_rate=0.05,
                confidence_score=0.85,
                accuracy_score=90.0,
                latency=2.5,
                metrics={
                    'bleu_score': 0.9,
                    'semantic_similarity': 0.95,
                    'word_accuracy': 0.9
                }
            )
            mock_results.append(result)
        
        with patch('pathlib.Path.write_text') as mock_write:
            report_path = await stt_evaluator.generate_report(
                results=mock_results,
                output_format='html',
                output_dir='/tmp/test_reports'
            )
            
            assert report_path is not None
            assert '.html' in str(report_path)
            mock_write.assert_called_once()
            
            # Check that HTML content was generated
            html_content = mock_write.call_args[0][0]
            assert '<html>' in html_content
            assert 'STT Evaluation Report' in html_content
    
    def test_calculate_overall_accuracy_score(self, stt_evaluator):
        """Test overall accuracy score calculation."""
        metrics = {
            'word_error_rate': 0.1,  # 10% WER
            'character_error_rate': 0.05,  # 5% CER
            'bleu_score': 0.85,
            'semantic_similarity': 0.9,
            'confidence_score': 0.8
        }
        
        score = stt_evaluator._calculate_overall_accuracy_score(metrics)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0
        # With 10% WER, the accuracy should be around 90%
        assert 80.0 <= score <= 95.0
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        valid_config = {
            'accuracy_metrics': {'enabled': True},
            'performance_metrics': {'enabled': True},
            'audio_analysis': {'enabled': True},
            'confidence_analysis': {'enabled': True},
            'error_analysis': {'enabled': True},
            'output_formats': ['json'],
            'save_transcriptions': True
        }
        
        evaluator = STTEvaluator(valid_config)
        assert evaluator.config == valid_config
        
        # Invalid config - missing required fields
        invalid_config = {
            'accuracy_metrics': {'enabled': True}
            # Missing other required fields
        }
        
        with pytest.raises(ValueError, match="Missing required configuration"):
            STTEvaluator(invalid_config)

class TestSTTEvaluationResult:
    """Test suite for STTEvaluationResult class."""
    
    def test_result_creation_success(self):
        """Test successful evaluation result creation."""
        result = STTEvaluationResult(
            success=True,
            audio_path="/tmp/test.wav",
            reference_text="Hello world",
            predicted_text="Hello world",
            language="en",
            word_error_rate=0.0,
            character_error_rate=0.0,
            confidence_score=0.95,
            accuracy_score=100.0,
            latency=2.5,
            metrics={
                'bleu_score': 1.0,
                'semantic_similarity': 1.0
            }
        )
        
        assert result.success is True
        assert result.audio_path == "/tmp/test.wav"
        assert result.reference_text == "Hello world"
        assert result.predicted_text == "Hello world"
        assert result.word_error_rate == 0.0
        assert result.character_error_rate == 0.0
        assert result.confidence_score == 0.95
        assert result.accuracy_score == 100.0
        assert result.latency == 2.5
        assert result.metrics['bleu_score'] == 1.0
        assert isinstance(result.timestamp, str)
        assert result.error_message is None
    
    def test_result_creation_failure(self):
        """Test failed evaluation result creation."""
        result = STTEvaluationResult(
            success=False,
            audio_path="/tmp/test.wav",
            reference_text="Hello world",
            predicted_text=None,
            language="en",
            error_message="STT transcription failed",
            error_code="STT_ERROR"
        )
        
        assert result.success is False
        assert result.audio_path == "/tmp/test.wav"
        assert result.reference_text == "Hello world"
        assert result.predicted_text is None
        assert result.error_message == "STT transcription failed"
        assert result.error_code == "STT_ERROR"
        assert result.word_error_rate == float('inf')
        assert result.character_error_rate == float('inf')
        assert result.accuracy_score == 0.0
    
    def test_result_to_dict(self):
        """Test conversion of result to dictionary."""
        result = STTEvaluationResult(
            success=True,
            audio_path="/tmp/test.wav",
            reference_text="Hello world",
            predicted_text="Hello world",
            language="en",
            word_error_rate=0.0,
            character_error_rate=0.0,
            confidence_score=0.95,
            accuracy_score=100.0,
            latency=2.5,
            metrics={'bleu_score': 1.0}
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['success'] is True
        assert result_dict['audio_path'] == "/tmp/test.wav"
        assert result_dict['word_error_rate'] == 0.0
        assert result_dict['accuracy_score'] == 100.0
        assert 'timestamp' in result_dict
        assert 'metrics' in result_dict

class TestPerformanceAndIntegration:
    """Performance and integration tests for STT evaluator."""
    
    @pytest.fixture(autouse=True)
    def setup_logging(self):
        """Setup logging for tests."""
        setup_logging(log_level="DEBUG", enable_file=False)
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_large_batch_evaluation(self):
        """Test evaluation with a large batch of test cases."""
        config = {
            'accuracy_metrics': {'enabled': True, 'calculate_wer': True, 'calculate_cer': True},
            'performance_metrics': {'enabled': True},
            'audio_analysis': {'enabled': True},
            'confidence_analysis': {'enabled': True},
            'error_analysis': {'enabled': True},
            'output_formats': ['json'],
            'save_transcriptions': False  # Disable to speed up test
        }
        
        evaluator = STTEvaluator(config)
        mock_client = Mock()
        
        # Create large test case list
        test_cases = [
            {
                'audio_path': f"/tmp/test_audio_{i}.wav",
                'reference_text': f"Test sentence number {i} for speech recognition evaluation",
                'language': 'en'
            }
            for i in range(30)  # 30 test cases
        ]
        
        # Mock successful responses
        def mock_stt_response(audio_path, language, **kwargs):
            # Extract number from path for consistent responses
            import re
            match = re.search(r'test_audio_(\d+)', audio_path)
            if match:
                num = int(match.group(1))
                return ClientResponse(
                    success=True,
                    text=f"Test sentence number {num} for speech recognition evaluation",
                    metadata={'confidence': 0.9, 'language': language}
                )
            return ClientResponse(success=False, error_message="Unknown file")
        
        mock_client.speech_to_text = Mock(side_effect=mock_stt_response)
        
        # Mock audio processing and metrics
        with patch.object(evaluator.audio_processor, 'validate_audio_file') as mock_validate, \
             patch.object(evaluator.metrics_calculator, 'calculate_wer') as mock_wer, \
             patch.object(evaluator.metrics_calculator, 'calculate_cer') as mock_cer:
            
            mock_validate.return_value = {
                'is_valid': True,
                'audio_info': {'duration': 3.0, 'sample_rate': 16000, 'channels': 1}
            }
            
            mock_wer.return_value = MetricResult(
                value=0.0,
                details={'substitutions': 0, 'insertions': 0, 'deletions': 0},
                timestamp=datetime.now().isoformat()
            )
            
            mock_cer.return_value = MetricResult(
                value=0.0,
                details={'edit_distance': 0},
                timestamp=datetime.now().isoformat()
            )
            
            start_time = time.time()
            results = await evaluator.evaluate_batch(mock_client, test_cases)
            end_time = time.time()
            
            # Verify results
            assert len(results) == 30
            assert all(result.success for result in results)
            
            # Performance check - should complete within reasonable time
            total_time = end_time - start_time
            assert total_time < 60  # Should complete within 1 minute
            
            # Calculate average processing time per case
            avg_time_per_case = total_time / len(test_cases)
            assert avg_time_per_case < 3.0  # Less than 3 seconds per case
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_end_to_end_evaluation_workflow(self):
        """Test complete end-to-end STT evaluation workflow."""
        config = {
            'accuracy_metrics': {
                'enabled': True,
                'calculate_wer': True,
                'calculate_cer': True,
                'calculate_bleu': True,
                'semantic_similarity': True
            },
            'performance_metrics': {'enabled': True},
            'audio_analysis': {'enabled': True},
            'confidence_analysis': {'enabled': True},
            'error_analysis': {'enabled': True},
            'output_formats': ['json', 'yaml', 'html'],
            'save_transcriptions': True,
            'transcription_output_dir': '/tmp/test_stt_output'
        }
        
        evaluator = STTEvaluator(config)
        mock_client = Mock()
        mock_client.client_name = "test_stt_client"
        
        test_cases = [
            {
                'audio_path': '/tmp/test1.wav',
                'reference_text': 'Hello world',
                'language': 'en'
            },
            {
                'audio_path': '/tmp/test2.wav',
                'reference_text': 'This is a test of speech recognition.',
                'language': 'en'
            }
        ]
        
        # Mock STT responses with slight errors
        def mock_stt_response(audio_path, language, **kwargs):
            if "test1" in audio_path:
                return ClientResponse(
                    success=True,
                    text="Hello world",  # Perfect match
                    metadata={'confidence': 0.95, 'language': language}
                )
            elif "test2" in audio_path:
                return ClientResponse(
                    success=True,
                    text="This is a test of speech recognition",  # Missing period
                    metadata={'confidence': 0.85, 'language': language}
                )
            else:
                return ClientResponse(success=False, error_message="Unknown file")
        
        mock_client.speech_to_text = Mock(side_effect=mock_stt_response)
        
        # Mock all processing steps
        with patch.object(evaluator.audio_processor, 'validate_audio_file') as mock_validate, \
             patch.object(evaluator.metrics_calculator, 'calculate_wer') as mock_wer, \
             patch.object(evaluator.metrics_calculator, 'calculate_cer') as mock_cer, \
             patch.object(evaluator.metrics_calculator, 'calculate_bleu_score') as mock_bleu, \
             patch.object(evaluator.metrics_calculator, 'calculate_semantic_similarity') as mock_sim:
            
            mock_validate.return_value = {
                'is_valid': True,
                'audio_info': {'duration': 2.0, 'sample_rate': 16000, 'channels': 1}
            }
            
            mock_wer.return_value = MetricResult(
                value=0.05,  # 5% WER
                details={'substitutions': 0, 'insertions': 0, 'deletions': 1},
                timestamp=datetime.now().isoformat()
            )
            
            mock_cer.return_value = MetricResult(
                value=0.02,  # 2% CER
                details={'edit_distance': 1},
                timestamp=datetime.now().isoformat()
            )
            
            mock_bleu.return_value = MetricResult(
                value=0.95,
                details={'precisions': [1.0, 0.9, 0.95, 0.9]},
                timestamp=datetime.now().isoformat()
            )
            
            mock_sim.return_value = MetricResult(
                value=0.98,
                details={'method': 'jaccard', 'common_words': ['this', 'is', 'test']},
                timestamp=datetime.now().isoformat()
            )
            
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
                assert all(result.accuracy_score > 90 for result in results)
                
                # Verify reports were generated
                assert json_report is not None
                assert yaml_report is not None
                assert html_report is not None

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main(["-v", "--tb=short", __file__])