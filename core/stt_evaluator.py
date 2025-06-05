"""
Speech-to-Text (STT) Evaluator
==============================

Comprehensive evaluation module for STT models with support for accuracy metrics,
performance benchmarking, and detailed error analysis.

Author: AI Testing Team
Version: 1.0.0
"""

import logging
import time
import hashlib
import json
import re
import difflib
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import asyncio
import librosa
import soundfile as sf
import numpy as np
from jiwer import wer, cer, mer, wil, wip

# Handle optional editdistance import
try:
    import editdistance
    EDITDISTANCE_AVAILABLE = True
except ImportError:
    EDITDISTANCE_AVAILABLE = False
    # Create a fallback function
    def editdistance_eval(s1, s2):
        """Fallback edit distance calculation using difflib."""
        return len(list(difflib.unified_diff(s1, s2)))
    
    class MockEditDistance:
        @staticmethod
        def eval(s1, s2):
            return editdistance_eval(s1, s2)
    
    editdistance = MockEditDistance()

# Configure logging
logger = logging.getLogger(__name__)

if not EDITDISTANCE_AVAILABLE:
    logger.warning("editdistance package not available, using fallback implementation")

@dataclass
class STTEvaluationResult:
    """Data class for STT evaluation results."""
    model_id: str
    audio_input_path: str
    reference_text: str
    transcribed_text: str
    accuracy_metrics: Dict[str, float]
    error_analysis: Dict[str, Any]
    performance_metrics: Dict[str, float]
    confidence_metrics: Dict[str, float]
    metadata: Dict[str, Any]
    timestamp: str
    success: bool
    error_message: Optional[str] = None

@dataclass
class STTTestCase:
    """Data class for STT test case configuration."""
    audio_path: str
    reference_text: str
    language: str
    test_scenario: str
    expected_accuracy_threshold: float = 0.8
    test_id: Optional[str] = None
    audio_metadata: Optional[Dict[str, Any]] = None

class STTEvaluator:
    """
    Comprehensive STT model evaluator with advanced accuracy metrics and error analysis.
    
    This class provides functionality to evaluate STT models across multiple dimensions:
    - Word Error Rate (WER) and Character Error Rate (CER)
    - Match Error Rate (MER) and Word Information Lost (WIL)
    - Confidence score analysis
    - Performance metrics (latency, throughput)
    - Detailed error categorization and analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize STT evaluator with configuration.
        
        Args:
            config (Dict[str, Any]): Evaluation configuration
        """
        self.config = config
        self.evaluation_id = self._generate_evaluation_id()
        self.results: List[STTEvaluationResult] = []
        
        # Initialize evaluation parameters
        self.accuracy_thresholds = config.get('testing', {}).get('stt', {}).get('accuracy_thresholds', {})
        self.metrics_config = config.get('testing', {}).get('stt', {}).get('metrics', [])
        
        # Text processing parameters
        self.normalize_text = True
        self.case_sensitive = False
        self.remove_punctuation = True
        
        logger.info(f"STT Evaluator initialized with ID: {self.evaluation_id}")
        logger.debug(f"Metrics: {self.metrics_config}")

    def _generate_evaluation_id(self) -> str:
        """Generate unique evaluation ID."""
        timestamp = datetime.now(timezone.utc).isoformat()
        content = f"stt_evaluation_{timestamp}_{id(self)}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    async def evaluate_model(
        self, 
        client, 
        test_cases: List[STTTestCase],
        model_config: Dict[str, Any]
    ) -> List[STTEvaluationResult]:
        """
        Evaluate an STT model with multiple test cases.
        
        Args:
            client: STT client instance
            test_cases (List[STTTestCase]): Test cases to evaluate
            model_config (Dict[str, Any]): Model configuration
            
        Returns:
            List[STTEvaluationResult]: Evaluation results for all test cases
        """
        model_id = model_config.get('model_id', 'unknown')
        logger.info(f"Starting STT evaluation for model: {model_id}")
        logger.info(f"Evaluating {len(test_cases)} test cases")
        
        model_results = []
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"Evaluating test case {i}/{len(test_cases)}: {test_case.test_id or 'unnamed'}")
            
            try:
                result = await self._evaluate_single_test_case(
                    client, test_case, model_config
                )
                model_results.append(result)
                self.results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to evaluate test case {i}: {e}")
                
                # Create failed result
                failed_result = STTEvaluationResult(
                    model_id=model_id,
                    audio_input_path=test_case.audio_path,
                    reference_text=test_case.reference_text,
                    transcribed_text="",
                    accuracy_metrics={},
                    error_analysis={},
                    performance_metrics={},
                    confidence_metrics={},
                    metadata={
                        'test_case_id': test_case.test_id,
                        'language': test_case.language,
                        'test_scenario': test_case.test_scenario
                    },
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    success=False,
                    error_message=str(e)
                )
                model_results.append(failed_result)
                self.results.append(failed_result)
        
        logger.info(f"Completed STT evaluation for model: {model_id}")
        return model_results

    async def _evaluate_single_test_case(
        self,
        client,
        test_case: STTTestCase,
        model_config: Dict[str, Any]
    ) -> STTEvaluationResult:
        """
        Evaluate a single STT test case.
        
        Args:
            client: STT client instance
            test_case (STTTestCase): Test case to evaluate
            model_config (Dict[str, Any]): Model configuration
            
        Returns:
            STTEvaluationResult: Evaluation result
        """
        start_time = time.time()
        model_id = model_config.get('model_id', 'unknown')
        
        try:
            # Validate audio file
            if not Path(test_case.audio_path).exists():
                raise FileNotFoundError(f"Audio file not found: {test_case.audio_path}")
            
            # Transcribe audio
            logger.debug(f"Transcribing audio: {test_case.audio_path}")
            transcription_start = time.time()
            
            transcription_result = await self._transcribe_audio(
                client, test_case, model_config
            )
            
            transcription_time = time.time() - transcription_start
            logger.debug(f"Transcription completed in {transcription_time:.2f}s")
            
            # Perform comprehensive evaluation
            accuracy_metrics = await self._calculate_accuracy_metrics(
                test_case.reference_text, 
                transcription_result['text'],
                test_case.language
            )
            
            error_analysis = await self._perform_error_analysis(
                test_case.reference_text,
                transcription_result['text']
            )
            
            confidence_metrics = self._analyze_confidence_scores(
                transcription_result.get('confidence_scores', [])
            )
            
            # Get audio metadata
            audio_metadata = await self._analyze_audio_metadata(test_case.audio_path)
            
            performance_metrics = {
                'transcription_time_seconds': transcription_time,
                'total_evaluation_time_seconds': time.time() - start_time,
                'audio_duration_seconds': audio_metadata.get('duration_seconds', 0),
                'real_time_factor': transcription_time / audio_metadata.get('duration_seconds', 1),
                'reference_text_length_characters': len(test_case.reference_text),
                'reference_text_length_words': len(test_case.reference_text.split()),
                'transcribed_text_length_characters': len(transcription_result['text']),
                'transcribed_text_length_words': len(transcription_result['text'].split()),
                'audio_file_size_bytes': Path(test_case.audio_path).stat().st_size
            }
            
            # Compile metadata
            metadata = {
                'test_case_id': test_case.test_id,
                'language': test_case.language,
                'test_scenario': test_case.test_scenario,
                'model_config': model_config,
                'evaluation_id': self.evaluation_id,
                'audio_metadata': audio_metadata,
                'transcription_metadata': transcription_result.get('metadata', {})
            }
            
            result = STTEvaluationResult(
                model_id=model_id,
                audio_input_path=test_case.audio_path,
                reference_text=test_case.reference_text,
                transcribed_text=transcription_result['text'],
                accuracy_metrics=accuracy_metrics,
                error_analysis=error_analysis,
                performance_metrics=performance_metrics,
                confidence_metrics=confidence_metrics,
                metadata=metadata,
                timestamp=datetime.now(timezone.utc).isoformat(),
                success=True
            )
            
            logger.debug(f"Successfully evaluated test case for model: {model_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating test case for model {model_id}: {e}")
            raise

    async def _transcribe_audio(
        self,
        client,
        test_case: STTTestCase,
        model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Transcribe audio using the STT client.
        
        Args:
            client: STT client instance
            test_case (STTTestCase): Test case configuration
            model_config (Dict[str, Any]): Model configuration
            
        Returns:
            Dict[str, Any]: Transcription result with metadata
        """
        # Prepare transcription parameters
        params = {
            'audio_path': test_case.audio_path,
            'language': test_case.language,
            'model_id': model_config.get('model_id')
        }
        
        # Add model-specific parameters
        if 'default_params' in model_config:
            params.update(model_config['default_params'])
        
        # Transcribe audio
        response = await client.speech_to_text(**params)
        
        if not response.success:
            raise RuntimeError(f"STT transcription failed: {response.error}")
        
        return {
            'text': response.text,
            'confidence_scores': getattr(response, 'confidence_scores', []),
            'word_timestamps': getattr(response, 'word_timestamps', []),
            'metadata': getattr(response, 'metadata', {})
        }

    async def _calculate_accuracy_metrics(
        self,
        reference_text: str,
        transcribed_text: str,
        language: str
    ) -> Dict[str, float]:
        """
        Calculate comprehensive accuracy metrics.
        
        Args:
            reference_text (str): Ground truth text
            transcribed_text (str): Transcribed text
            language (str): Language code
            
        Returns:
            Dict[str, float]: Accuracy metrics
        """
        metrics = {}
        
        try:
            # Normalize texts for comparison
            ref_normalized = self._normalize_text(reference_text)
            trans_normalized = self._normalize_text(transcribed_text)
            
            # Basic accuracy metrics using jiwer
            if ref_normalized and trans_normalized:
                # Word Error Rate (WER)
                metrics['word_error_rate'] = float(wer(ref_normalized, trans_normalized))
                metrics['word_accuracy'] = 1.0 - metrics['word_error_rate']
                
                # Character Error Rate (CER)
                metrics['character_error_rate'] = float(cer(ref_normalized, trans_normalized))
                metrics['character_accuracy'] = 1.0 - metrics['character_error_rate']
                
                # Match Error Rate (MER)
                metrics['match_error_rate'] = float(mer(ref_normalized, trans_normalized))
                
                # Word Information Lost (WIL)
                metrics['word_information_lost'] = float(wil(ref_normalized, trans_normalized))
                
                # Word Information Preserved (WIP)
                metrics['word_information_preserved'] = float(wip(ref_normalized, trans_normalized))
                
            else:
                # Handle empty texts
                if not ref_normalized and not trans_normalized:
                    metrics['word_error_rate'] = 0.0
                    metrics['character_error_rate'] = 0.0
                elif not ref_normalized:
                    metrics['word_error_rate'] = float('inf') if trans_normalized else 0.0
                    metrics['character_error_rate'] = float('inf') if trans_normalized else 0.0
                else:
                    metrics['word_error_rate'] = 1.0
                    metrics['character_error_rate'] = 1.0
                
                metrics['word_accuracy'] = max(0.0, 1.0 - metrics.get('word_error_rate', 1.0))
                metrics['character_accuracy'] = max(0.0, 1.0 - metrics.get('character_error_rate', 1.0))
            
            # Additional custom metrics
            metrics.update(self._calculate_custom_metrics(ref_normalized, trans_normalized))
            
            logger.debug(f"Calculated {len(metrics)} accuracy metrics")
            
        except Exception as e:
            logger.error(f"Error calculating accuracy metrics: {e}")
            metrics['calculation_error'] = str(e)
        
        return metrics

    def _calculate_custom_metrics(self, reference: str, transcribed: str) -> Dict[str, float]:
        """Calculate additional custom accuracy metrics."""
        metrics = {}
        
        try:
            ref_words = reference.split()
            trans_words = transcribed.split()
            
            # Exact match accuracy
            if ref_words:
                exact_matches = sum(1 for r, t in zip(ref_words, trans_words) if r == t)
                metrics['exact_word_match_rate'] = exact_matches / len(ref_words)
            else:
                metrics['exact_word_match_rate'] = 1.0 if not trans_words else 0.0
            
            # Edit distance metrics
            edit_dist = editdistance.eval(reference, transcribed)
            max_len = max(len(reference), len(transcribed))
            metrics['normalized_edit_distance'] = edit_dist / max_len if max_len > 0 else 0.0
            
            # Length-based metrics
            ref_len = len(ref_words)
            trans_len = len(trans_words)
            
            if ref_len > 0:
                metrics['length_ratio'] = trans_len / ref_len
                metrics['length_difference_ratio'] = abs(trans_len - ref_len) / ref_len
            else:
                metrics['length_ratio'] = float('inf') if trans_len > 0 else 1.0
                metrics['length_difference_ratio'] = float('inf') if trans_len > 0 else 0.0
            
            # Semantic similarity (basic word overlap)
            ref_set = set(ref_words)
            trans_set = set(trans_words)
            
            if ref_set:
                jaccard_similarity = len(ref_set & trans_set) / len(ref_set | trans_set)
                metrics['jaccard_word_similarity'] = jaccard_similarity
                
                recall = len(ref_set & trans_set) / len(ref_set)
                precision = len(ref_set & trans_set) / len(trans_set) if trans_set else 0.0
                
                metrics['word_recall'] = recall
                metrics['word_precision'] = precision
                
                if precision + recall > 0:
                    metrics['word_f1_score'] = 2 * (precision * recall) / (precision + recall)
                else:
                    metrics['word_f1_score'] = 0.0
            else:
                metrics['jaccard_word_similarity'] = 1.0 if not trans_set else 0.0
                metrics['word_recall'] = 1.0 if not trans_set else 0.0
                metrics['word_precision'] = 1.0 if not trans_set else 0.0
                metrics['word_f1_score'] = 1.0 if not trans_set else 0.0
            
        except Exception as e:
            logger.warning(f"Custom metrics calculation failed: {e}")
        
        return metrics

    async def _perform_error_analysis(
        self,
        reference_text: str,
        transcribed_text: str
    ) -> Dict[str, Any]:
        """
        Perform detailed error analysis.
        
        Args:
            reference_text (str): Ground truth text
            transcribed_text (str): Transcribed text
            
        Returns:
            Dict[str, Any]: Error analysis results
        """
        analysis = {}
        
        try:
            ref_normalized = self._normalize_text(reference_text)
            trans_normalized = self._normalize_text(transcribed_text)
            
            ref_words = ref_normalized.split()
            trans_words = trans_normalized.split()
            
            # Calculate edit operations
            operations = self._get_edit_operations(ref_words, trans_words)
            
            # Count error types
            error_counts = {
                'substitutions': operations.count('substitute'),
                'insertions': operations.count('insert'),
                'deletions': operations.count('delete'),
                'correct': operations.count('equal')
            }
            
            analysis['error_counts'] = error_counts
            analysis['total_errors'] = sum(error_counts.values()) - error_counts['correct']
            
            # Error rate by type
            total_ref_words = len(ref_words)
            if total_ref_words > 0:
                analysis['substitution_rate'] = error_counts['substitutions'] / total_ref_words
                analysis['insertion_rate'] = error_counts['insertions'] / total_ref_words
                analysis['deletion_rate'] = error_counts['deletions'] / total_ref_words
            
            # Detailed error analysis
            analysis['error_details'] = self._analyze_specific_errors(ref_words, trans_words)
            
            # Character-level error analysis
            analysis['character_errors'] = self._analyze_character_errors(
                ref_normalized, trans_normalized
            )
            
            # Pattern analysis
            analysis['error_patterns'] = self._identify_error_patterns(ref_words, trans_words)
            
        except Exception as e:
            logger.error(f"Error analysis failed: {e}")
            analysis['analysis_error'] = str(e)
        
        return analysis

    def _get_edit_operations(self, ref_words: List[str], trans_words: List[str]) -> List[str]:
        """Get sequence of edit operations between reference and transcribed text."""
        operations = []
        
        try:
            # Use difflib to get edit operations
            matcher = difflib.SequenceMatcher(None, ref_words, trans_words)
            
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'equal':
                    operations.extend(['equal'] * (i2 - i1))
                elif tag == 'replace':
                    operations.extend(['substitute'] * (i2 - i1))
                elif tag == 'delete':
                    operations.extend(['delete'] * (i2 - i1))
                elif tag == 'insert':
                    operations.extend(['insert'] * (j2 - j1))
                    
        except Exception as e:
            logger.warning(f"Edit operations calculation failed: {e}")
        
        return operations

    def _analyze_specific_errors(
        self, 
        ref_words: List[str], 
        trans_words: List[str]
    ) -> Dict[str, Any]:
        """Analyze specific word-level errors."""
        errors = {
            'substitutions': [],
            'insertions': [],
            'deletions': [],
            'frequent_errors': {}
        }
        
        try:
            # Use difflib for detailed comparison
            matcher = difflib.SequenceMatcher(None, ref_words, trans_words)
            
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'replace':
                    for ref_word, trans_word in zip(ref_words[i1:i2], trans_words[j1:j2]):
                        error_pair = (ref_word, trans_word)
                        errors['substitutions'].append(error_pair)
                        
                        # Track frequent substitution errors
                        if error_pair not in errors['frequent_errors']:
                            errors['frequent_errors'][error_pair] = 0
                        errors['frequent_errors'][error_pair] += 1
                        
                elif tag == 'delete':
                    for ref_word in ref_words[i1:i2]:
                        errors['deletions'].append(ref_word)
                        
                elif tag == 'insert':
                    for trans_word in trans_words[j1:j2]:
                        errors['insertions'].append(trans_word)
            
            # Sort frequent errors by count
            errors['frequent_errors'] = dict(
                sorted(errors['frequent_errors'].items(), 
                      key=lambda x: x[1], reverse=True)[:10]  # Top 10
            )
            
        except Exception as e:
            logger.warning(f"Specific error analysis failed: {e}")
        
        return errors

    def _analyze_character_errors(self, reference: str, transcribed: str) -> Dict[str, Any]:
        """Analyze character-level errors."""
        char_errors = {
            'substitutions': [],
            'insertions': [],
            'deletions': [],
            'character_accuracy_by_position': []
        }
        
        try:
            # Character-level comparison
            matcher = difflib.SequenceMatcher(None, reference, transcribed)
            
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'replace':
                    for ref_char, trans_char in zip(reference[i1:i2], transcribed[j1:j2]):
                        char_errors['substitutions'].append((ref_char, trans_char))
                elif tag == 'delete':
                    for ref_char in reference[i1:i2]:
                        char_errors['deletions'].append(ref_char)
                elif tag == 'insert':
                    for trans_char in transcribed[j1:j2]:
                        char_errors['insertions'].append(trans_char)
            
            # Calculate positional accuracy
            min_len = min(len(reference), len(transcribed))
            for i in range(min_len):
                char_errors['character_accuracy_by_position'].append(
                    1.0 if reference[i] == transcribed[i] else 0.0
                )
                
        except Exception as e:
            logger.warning(f"Character error analysis failed: {e}")
        
        return char_errors

    def _identify_error_patterns(
        self, 
        ref_words: List[str], 
        trans_words: List[str]
    ) -> Dict[str, Any]:
        """Identify common error patterns."""
        patterns = {
            'phonetically_similar_errors': [],
            'length_based_errors': {},
            'capitalization_errors': 0,
            'punctuation_errors': 0
        }
        
        try:
            # Find phonetically similar substitutions (basic implementation)
            matcher = difflib.SequenceMatcher(None, ref_words, trans_words)
            
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'replace' and i2-i1 == 1 and j2-j1 == 1:
                    ref_word = ref_words[i1]
                    trans_word = trans_words[j1]
                    
                    # Check if words are phonetically similar (simple heuristic)
                    if self._are_phonetically_similar(ref_word, trans_word):
                        patterns['phonetically_similar_errors'].append((ref_word, trans_word))
                    
                    # Length-based error analysis
                    length_diff = len(trans_word) - len(ref_word)
                    if length_diff not in patterns['length_based_errors']:
                        patterns['length_based_errors'][length_diff] = 0
                    patterns['length_based_errors'][length_diff] += 1
            
        except Exception as e:
            logger.warning(f"Error pattern identification failed: {e}")
        
        return patterns

    def _are_phonetically_similar(self, word1: str, word2: str) -> bool:
        """
        Simple heuristic to check if two words are phonetically similar.
        This is a basic implementation - could be enhanced with proper phonetic algorithms.
        """
        if len(word1) == 0 or len(word2) == 0:
            return False
        
        # Calculate character overlap ratio
        set1, set2 = set(word1.lower()), set(word2.lower())
        overlap = len(set1 & set2)
        total = len(set1 | set2)
        
        overlap_ratio = overlap / total if total > 0 else 0
        
        # Consider phonetically similar if high character overlap and similar length
        length_ratio = min(len(word1), len(word2)) / max(len(word1), len(word2))
        
        return overlap_ratio > 0.6 and length_ratio > 0.7

    def _analyze_confidence_scores(self, confidence_scores: List[float]) -> Dict[str, float]:
        """Analyze confidence scores if available."""
        metrics = {}
        
        try:
            if confidence_scores:
                metrics['average_confidence'] = float(np.mean(confidence_scores))
                metrics['min_confidence'] = float(np.min(confidence_scores))
                metrics['max_confidence'] = float(np.max(confidence_scores))
                metrics['confidence_std'] = float(np.std(confidence_scores))
                
                # Confidence distribution
                high_conf = sum(1 for c in confidence_scores if c > 0.8)
                medium_conf = sum(1 for c in confidence_scores if 0.5 <= c <= 0.8)
                low_conf = sum(1 for c in confidence_scores if c < 0.5)
                
                total = len(confidence_scores)
                metrics['high_confidence_ratio'] = high_conf / total
                metrics['medium_confidence_ratio'] = medium_conf / total
                metrics['low_confidence_ratio'] = low_conf / total
            else:
                metrics['confidence_available'] = False
                
        except Exception as e:
            logger.warning(f"Confidence analysis failed: {e}")
            metrics['confidence_analysis_error'] = str(e)
        
        return metrics

    async def _analyze_audio_metadata(self, audio_path: str) -> Dict[str, Any]:
        """Analyze audio file metadata."""
        metadata = {}
        
        try:
            # Load audio file
            audio_data, sample_rate = librosa.load(audio_path, sr=None)
            
            metadata.update({
                'duration_seconds': float(len(audio_data) / sample_rate),
                'sample_rate': int(sample_rate),
                'channels': 1,  # librosa loads as mono by default
                'samples': len(audio_data),
                'file_size_bytes': Path(audio_path).stat().st_size,
                'format': Path(audio_path).suffix.lower()
            })
            
            # Audio quality metrics
            if len(audio_data) > 0:
                metadata['rms_level'] = float(np.sqrt(np.mean(audio_data ** 2)))
                metadata['peak_level'] = float(np.max(np.abs(audio_data)))
                metadata['dynamic_range'] = float(metadata['peak_level'] - metadata['rms_level'])
                
                # Silence detection
                silence_threshold = 0.01 * metadata['peak_level']
                silence_frames = np.abs(audio_data) < silence_threshold
                metadata['silence_percentage'] = float(np.mean(silence_frames) * 100)
                
        except Exception as e:
            logger.warning(f"Audio metadata analysis failed: {e}")
            metadata['analysis_error'] = str(e)
        
        return metadata

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        try:
            normalized = text.strip()
            
            if not self.case_sensitive:
                normalized = normalized.lower()
            
            if self.remove_punctuation:
                # Remove punctuation but keep apostrophes in contractions
                normalized = re.sub(r"[^\w\s']", "", normalized)
                normalized = re.sub(r"'\s", " ", normalized)  # Remove hanging apostrophes
            
            # Normalize whitespace
            normalized = re.sub(r'\s+', ' ', normalized).strip()
            
            return normalized
            
        except Exception as e:
            logger.warning(f"Text normalization failed: {e}")
            return text

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive evaluation summary.
        
        Returns:
            Dict[str, Any]: Evaluation summary with statistics
        """
        if not self.results:
            return {"error": "No evaluation results available"}
        
        summary = {
            "evaluation_id": self.evaluation_id,
            "total_test_cases": len(self.results),
            "successful_evaluations": sum(1 for r in self.results if r.success),
            "failed_evaluations": sum(1 for r in self.results if not r.success),
            "models_evaluated": len(set(r.model_id for r in self.results)),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Calculate aggregate statistics
        successful_results = [r for r in self.results if r.success]
        
        if successful_results:
            # Accuracy statistics
            wer_scores = [r.accuracy_metrics.get('word_error_rate', 1.0) 
                         for r in successful_results]
            word_accuracy_scores = [r.accuracy_metrics.get('word_accuracy', 0.0) 
                                  for r in successful_results]
            
            summary["avg_word_error_rate"] = float(np.mean(wer_scores))
            summary["min_word_error_rate"] = float(np.min(wer_scores))
            summary["max_word_error_rate"] = float(np.max(wer_scores))
            summary["avg_word_accuracy"] = float(np.mean(word_accuracy_scores))
            summary["min_word_accuracy"] = float(np.min(word_accuracy_scores))
            summary["max_word_accuracy"] = float(np.max(word_accuracy_scores))
            
            # Performance statistics
            transcription_times = [r.performance_metrics.get('transcription_time_seconds', 0) 
                                 for r in successful_results]
            summary["avg_transcription_time_seconds"] = float(np.mean(transcription_times))
            summary["min_transcription_time_seconds"] = float(np.min(transcription_times))
            summary["max_transcription_time_seconds"] = float(np.max(transcription_times))
            
            # Accuracy distribution
            accuracy_ranges = {
                "excellent (>95%)": sum(1 for acc in word_accuracy_scores if acc > 0.95),
                "good (85-95%)": sum(1 for acc in word_accuracy_scores if 0.85 <= acc <= 0.95),
                "fair (70-85%)": sum(1 for acc in word_accuracy_scores if 0.70 <= acc < 0.85),
                "poor (<70%)": sum(1 for acc in word_accuracy_scores if acc < 0.70)
            }
            summary["accuracy_distribution"] = accuracy_ranges
        
        return summary

    def export_results(self, format_type: str = "json") -> str:
        """
        Export evaluation results in specified format.
        
        Args:
            format_type (str): Export format ('json', 'yaml', 'csv')
            
        Returns:
            str: Serialized results
        """
        if format_type.lower() == "json":
            results_dict = [asdict(result) for result in self.results]
            return json.dumps({
                "evaluation_summary": self.get_evaluation_summary(),
                "results": results_dict
            }, indent=2, default=str)
        
        elif format_type.lower() == "yaml":
            import yaml
            results_dict = [asdict(result) for result in self.results]
            return yaml.dump({
                "evaluation_summary": self.get_evaluation_summary(),
                "results": results_dict
            }, default_flow_style=False)
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}")

    def clear_results(self):
        """Clear all evaluation results."""
        self.results.clear()
        logger.info("Evaluation results cleared")

# Module-level utility functions
def create_test_cases_from_config(config: Dict[str, Any]) -> List[STTTestCase]:
    """
    Create test cases from configuration.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        List[STTTestCase]: Generated test cases
    """
    test_cases = []
    
    try:
        stt_config = config.get('testing', {}).get('stt_test_params', {})
        test_audio_files = stt_config.get('test_audio_files', {})
        
        for scenario, audio_files in test_audio_files.items():
            for i, audio_path in enumerate(audio_files):
                # Try to get reference text from filename or config
                reference_text = f"Reference text for {Path(audio_path).stem}"
                
                test_case = STTTestCase(
                    audio_path=audio_path,
                    reference_text=reference_text,
                    language="en",  # Default language
                    test_scenario=scenario,
                    test_id=f"{scenario}_test_{i+1}"
                )
                test_cases.append(test_case)
        
        logger.info(f"Created {len(test_cases)} test cases from configuration")
        
    except Exception as e:
        logger.error(f"Error creating test cases from config: {e}")
    
    return test_cases