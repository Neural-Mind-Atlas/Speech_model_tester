"""
TTS/STT Testing Framework - Metrics Utilities
============================================

This module provides comprehensive metrics calculation utilities for the TTS/STT testing framework.
It includes WER, CER, BLEU, perceptual metrics, and statistical analysis functionality.

Author: TTS/STT Testing Framework Team
Version: 1.0.0
Created: 2024-06-04
"""

import re
import math
import string
import difflib
import statistics
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
from collections import Counter, defaultdict
import unicodedata
from dataclasses import dataclass

from .logger import get_logger, log_function_call

@dataclass
class MetricResult:
    """Data class for metric calculation results."""
    value: float
    details: Dict[str, Any]
    timestamp: str
    
class MetricsCalculator:
    """
    Comprehensive metrics calculation class for the TTS/STT testing framework.
    
    Features:
    - Word Error Rate (WER) and Character Error Rate (CER)
    - BLEU and ROUGE scores
    - Perceptual and semantic similarity metrics
    - Statistical analysis and aggregation
    - Custom metric definitions and calculations
    """
    
    def __init__(self, case_sensitive: bool = False, normalize_text: bool = True):
        """
        Initialize the metrics calculator.
        
        Args:
            case_sensitive: Whether to perform case-sensitive comparisons
            normalize_text: Whether to normalize text before comparison
        """
        self.logger = get_logger(__name__)
        self.case_sensitive = case_sensitive
        self.normalize_text = normalize_text
        
        # Common words for filtering (can be expanded)
        self.stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'
        }
        
        self.logger.info(f"MetricsCalculator initialized - Case sensitive: {case_sensitive}")
    
    @log_function_call
    def calculate_wer(
        self,
        reference: str,
        hypothesis: str,
        return_details: bool = True
    ) -> Union[float, MetricResult]:
        """
        Calculate Word Error Rate (WER) between reference and hypothesis texts.
        
        Args:
            reference: Reference (ground truth) text
            hypothesis: Hypothesis (predicted) text
            return_details: Whether to return detailed results
            
        Returns:
            Union[float, MetricResult]: WER value or detailed result
        """
        try:
            # Normalize texts
            ref_words = self._normalize_and_tokenize(reference)
            hyp_words = self._normalize_and_tokenize(hypothesis)
            
            # Calculate edit distance
            distance_matrix, operations = self._calculate_edit_distance(ref_words, hyp_words)
            
            # Extract error counts
            total_words = len(ref_words)
            if total_words == 0:
                wer_value = 0.0 if len(hyp_words) == 0 else float('inf')
            else:
                edit_distance = distance_matrix[-1][-1]
                wer_value = edit_distance / total_words
            
            if not return_details:
                return wer_value
            
            # Calculate detailed statistics
            substitutions = operations.count('substitute')
            insertions = operations.count('insert')
            deletions = operations.count('delete')
            
            details = {
                'reference_text': reference,
                'hypothesis_text': hypothesis,
                'reference_words': ref_words,
                'hypothesis_words': hyp_words,
                'reference_word_count': len(ref_words),
                'hypothesis_word_count': len(hyp_words),
                'edit_distance': edit_distance,
                'substitutions': substitutions,
                'insertions': insertions,
                'deletions': deletions,
                'correct_words': total_words - substitutions - deletions,
                'accuracy': max(0, 1 - wer_value),
                'operations': operations,
                'alignment': self._get_alignment(ref_words, hyp_words, operations)
            }
            
            result = MetricResult(
                value=wer_value,
                details=details,
                timestamp=datetime.now().isoformat()
            )
            
            self.logger.debug(f"WER calculated: {wer_value:.4f}")
            return result
            
        except Exception as e:
            self.logger.error("WER calculation failed", e)
            if return_details:
                return MetricResult(
                    value=float('inf'),
                    details={'error': str(e)},
                    timestamp=datetime.now().isoformat()
                )
            return float('inf')
    
    @log_function_call
    def calculate_cer(
        self,
        reference: str,
        hypothesis: str,
        return_details: bool = True
    ) -> Union[float, MetricResult]:
        """
        Calculate Character Error Rate (CER) between reference and hypothesis texts.
        
        Args:
            reference: Reference (ground truth) text
            hypothesis: Hypothesis (predicted) text
            return_details: Whether to return detailed results
            
        Returns:
            Union[float, MetricResult]: CER value or detailed result
        """
        try:
            # Normalize texts
            ref_text = self._normalize_text(reference)
            hyp_text = self._normalize_text(hypothesis)
            
            # Convert to character lists
            ref_chars = list(ref_text)
            hyp_chars = list(hyp_text)
            
            # Calculate edit distance
            distance_matrix, operations = self._calculate_edit_distance(ref_chars, hyp_chars)
            
            # Calculate CER
            total_chars = len(ref_chars)
            if total_chars == 0:
                cer_value = 0.0 if len(hyp_chars) == 0 else float('inf')
            else:
                edit_distance = distance_matrix[-1][-1]
                cer_value = edit_distance / total_chars
            
            if not return_details:
                return cer_value
            
            # Calculate detailed statistics
            substitutions = operations.count('substitute')
            insertions = operations.count('insert')
            deletions = operations.count('delete')
            
            details = {
                'reference_text': reference,
                'hypothesis_text': hypothesis,
                'reference_chars': ref_chars,
                'hypothesis_chars': hyp_chars,
                'reference_char_count': len(ref_chars),
                'hypothesis_char_count': len(hyp_chars),
                'edit_distance': edit_distance,
                'substitutions': substitutions,
                'insertions': insertions,
                'deletions': deletions,
                'correct_chars': total_chars - substitutions - deletions,
                'accuracy': max(0, 1 - cer_value),
                'operations': operations
            }
            
            result = MetricResult(
                value=cer_value,
                details=details,
                timestamp=datetime.now().isoformat()
            )
            
            self.logger.debug(f"CER calculated: {cer_value:.4f}")
            return result
            
        except Exception as e:
            self.logger.error("CER calculation failed", e)
            if return_details:
                return MetricResult(
                    value=float('inf'),
                    details={'error': str(e)},
                    timestamp=datetime.now().isoformat()
                )
            return float('inf')
    
    @log_function_call
    def calculate_bleu_score(
        self,
        reference: str,
        hypothesis: str,
        n_grams: int = 4,
        return_details: bool = True
    ) -> Union[float, MetricResult]:
        """
        Calculate BLEU score between reference and hypothesis texts.
        
        Args:
            reference: Reference (ground truth) text
            hypothesis: Hypothesis (predicted) text
            n_grams: Maximum n-gram order for BLEU calculation
            return_details: Whether to return detailed results
            
        Returns:
            Union[float, MetricResult]: BLEU score or detailed result
        """
        try:
            # Normalize and tokenize
            ref_words = self._normalize_and_tokenize(reference)
            hyp_words = self._normalize_and_tokenize(hypothesis)
            
            if len(hyp_words) == 0:
                bleu_value = 0.0
            else:
                # Calculate n-gram precisions
                precisions = []
                for n in range(1, n_grams + 1):
                    ref_ngrams = self._get_ngrams(ref_words, n)
                    hyp_ngrams = self._get_ngrams(hyp_words, n)
                    
                    if len(hyp_ngrams) == 0:
                        precisions.append(0.0)
                        continue
                    
                    # Count matches
                    matches = 0
                    for ngram in hyp_ngrams:
                        if ngram in ref_ngrams:
                            matches += min(hyp_ngrams[ngram], ref_ngrams[ngram])
                    
                    precision = matches / sum(hyp_ngrams.values())
                    precisions.append(precision)
                
                # Calculate brevity penalty
                ref_length = len(ref_words)
                hyp_length = len(hyp_words)
                
                if hyp_length > ref_length:
                    brevity_penalty = 1.0
                else:
                    brevity_penalty = math.exp(1 - ref_length / hyp_length) if hyp_length > 0 else 0.0
                
                # Calculate BLEU score
                if all(p > 0 for p in precisions):
                    log_precision_sum = sum(math.log(p) for p in precisions)
                    bleu_value = brevity_penalty * math.exp(log_precision_sum / len(precisions))
                else:
                    bleu_value = 0.0
            
            if not return_details:
                return bleu_value
            
            details = {
                'reference_text': reference,
                'hypothesis_text': hypothesis,
                'reference_words': ref_words,
                'hypothesis_words': hyp_words,
                'reference_length': len(ref_words),
                'hypothesis_length': len(hyp_words),
                'n_grams': n_grams,
                'precisions': precisions,
                'brevity_penalty': brevity_penalty if len(hyp_words) > 0 else 0.0,
                'geometric_mean': math.exp(sum(math.log(p + 1e-10) for p in precisions) / len(precisions))
            }
            
            result = MetricResult(
                value=bleu_value,
                details=details,
                timestamp=datetime.now().isoformat()
            )
            
            self.logger.debug(f"BLEU score calculated: {bleu_value:.4f}")
            return result
            
        except Exception as e:
            self.logger.error("BLEU calculation failed", e)
            if return_details:
                return MetricResult(
                    value=0.0,
                    details={'error': str(e)},
                    timestamp=datetime.now().isoformat()
                )
            return 0.0
    
    @log_function_call
    def calculate_semantic_similarity(
        self,
        reference: str,
        hypothesis: str,
        method: str = 'jaccard',
        return_details: bool = True
    ) -> Union[float, MetricResult]:
        """
        Calculate semantic similarity between texts.
        
        Args:
            reference: Reference text
            hypothesis: Hypothesis text
            method: Similarity method ('jaccard', 'cosine', 'overlap')
            return_details: Whether to return detailed results
            
        Returns:
            Union[float, MetricResult]: Similarity score or detailed result
        """
        try:
            # Normalize and tokenize
            ref_words = set(self._normalize_and_tokenize(reference))
            hyp_words = set(self._normalize_and_tokenize(hypothesis))
            
            # Calculate similarity based on method
            if method == 'jaccard':
                intersection = len(ref_words.intersection(hyp_words))
                union = len(ref_words.union(hyp_words))
                similarity = intersection / union if union > 0 else 0.0
                
            elif method == 'cosine':
                # Simple cosine similarity based on word presence
                all_words = ref_words.union(hyp_words)
                ref_vector = [1 if word in ref_words else 0 for word in all_words]
                hyp_vector = [1 if word in hyp_words else 0 for word in all_words]
                
                dot_product = sum(r * h for r, h in zip(ref_vector, hyp_vector))
                ref_magnitude = math.sqrt(sum(r * r for r in ref_vector))
                hyp_magnitude = math.sqrt(sum(h * h for h in hyp_vector))
                
                if ref_magnitude > 0 and hyp_magnitude > 0:
                    similarity = dot_product / (ref_magnitude * hyp_magnitude)
                else:
                    similarity = 0.0
                    
            elif method == 'overlap':
                intersection = len(ref_words.intersection(hyp_words))
                min_length = min(len(ref_words), len(hyp_words))
                similarity = intersection / min_length if min_length > 0 else 0.0
                
            else:
                raise ValueError(f"Unknown similarity method: {method}")
            
            if not return_details:
                return similarity
            
            details = {
                'reference_text': reference,
                'hypothesis_text': hypothesis,
                'method': method,
                'reference_words': list(ref_words),
                'hypothesis_words': list(hyp_words),
                'common_words': list(ref_words.intersection(hyp_words)),
                'unique_to_reference': list(ref_words - hyp_words),
                'unique_to_hypothesis': list(hyp_words - ref_words),
                'reference_word_count': len(ref_words),
                'hypothesis_word_count': len(hyp_words),
                'common_word_count': len(ref_words.intersection(hyp_words))
            }
            
            result = MetricResult(
                value=similarity,
                details=details,
                timestamp=datetime.now().isoformat()
            )
            
            self.logger.debug(f"Semantic similarity calculated: {similarity:.4f}")
            return result
            
        except Exception as e:
            self.logger.error("Semantic similarity calculation failed", e)
            if return_details:
                return MetricResult(
                    value=0.0,
                    details={'error': str(e)},
                    timestamp=datetime.now().isoformat()
                )
            return 0.0
    
    @log_function_call
    def calculate_aggregated_metrics(
        self,
        metric_results: List[Dict[str, Any]],
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Calculate aggregated metrics from multiple individual results.
        
        Args:
            metric_results: List of individual metric results
            weights: Optional weights for different metrics
            
        Returns:
            Dict[str, Any]: Aggregated metrics and statistics
        """
        try:
            if not metric_results:
                return {'error': 'No metric results provided'}
            
            # Default weights
            if weights is None:
                weights = {'wer': 1.0, 'cer': 1.0, 'bleu': 1.0, 'similarity': 1.0}
            
            # Collect values by metric type
            metrics_by_type = defaultdict(list)
            for result in metric_results:
                for metric_name, metric_value in result.items():
                    if isinstance(metric_value, (int, float)) and not math.isnan(metric_value):
                        metrics_by_type[metric_name].append(metric_value)
            
            # Calculate statistics for each metric
            aggregated = {}
            for metric_name, values in metrics_by_type.items():
                if values:
                    aggregated[metric_name] = {
                        'count': len(values),
                        'mean': statistics.mean(values),
                        'median': statistics.median(values),
                        'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0,
                        'min': min(values),
                        'max': max(values),
                        'percentiles': {
                            'p25': self._percentile(values, 25),
                            'p75': self._percentile(values, 75),
                            'p90': self._percentile(values, 90),
                            'p95': self._percentile(values, 95)
                        }
                    }
            
            # Calculate weighted overall score
            overall_score = 0.0
            total_weight = 0.0
            
            for metric_name, weight in weights.items():
                if metric_name in aggregated:
                    # For error rates (WER, CER), lower is better, so invert
                    if metric_name in ['wer', 'cer']:
                        score = max(0, 1 - aggregated[metric_name]['mean'])
                    else:
                        score = aggregated[metric_name]['mean']
                    
                    overall_score += score * weight
                    total_weight += weight
            
            if total_weight > 0:
                overall_score /= total_weight
            
            result = {
                'metrics': aggregated,
                'overall_score': overall_score,
                'total_samples': len(metric_results),
                'weights': weights,
                'calculation_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Aggregated metrics calculated for {len(metric_results)} samples")
            return result
            
        except Exception as e:
            self.logger.error("Aggregated metrics calculation failed", e)
            return {'error': str(e)}
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not self.normalize_text:
            return text
        
        # Convert to lowercase if not case sensitive
        if not self.case_sensitive:
            text = text.lower()
        
        # Normalize Unicode characters
        text = unicodedata.normalize('NFD', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    def _normalize_and_tokenize(self, text: str) -> List[str]:
        """Normalize text and split into tokens."""
        normalized = self._normalize_text(text)
        
        # Remove punctuation
        normalized = normalized.translate(str.maketrans('', '', string.punctuation))
        
        # Split into words
        words = normalized.split()
        
        return words
    
    def _calculate_edit_distance(
        self,
        sequence1: List[str],
        sequence2: List[str]
    ) -> Tuple[List[List[int]], List[str]]:
        """Calculate edit distance with operation tracking."""
        m, n = len(sequence1), len(sequence2)
        
        # Initialize distance matrix
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize first row and column
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill the matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if sequence1[i-1] == sequence2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # deletion
                        dp[i][j-1],    # insertion
                        dp[i-1][j-1]   # substitution
                    )
        
        # Backtrack to get operations
        operations = []
        i, j = m, n
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and sequence1[i-1] == sequence2[j-1]:
                operations.append('match')
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
                operations.append('substitute')
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
                operations.append('delete')
                i -= 1
            else:
                operations.append('insert')
                j -= 1
        
        operations.reverse()
        return dp, operations
    
    def _get_alignment(
        self,
        ref_words: List[str],
        hyp_words: List[str],
        operations: List[str]
    ) -> List[Dict[str, str]]:
        """Get word alignment from operations."""
        alignment = []
        ref_idx = 0
        hyp_idx = 0
        
        for op in operations:
            if op == 'match':
                alignment.append({
                    'operation': 'match',
                    'reference': ref_words[ref_idx],
                    'hypothesis': hyp_words[hyp_idx]
                })
                ref_idx += 1
                hyp_idx += 1
            elif op == 'substitute':
                alignment.append({
                    'operation': 'substitute',
                    'reference': ref_words[ref_idx],
                    'hypothesis': hyp_words[hyp_idx]
                })
                ref_idx += 1
                hyp_idx += 1
            elif op == 'delete':
                alignment.append({
                    'operation': 'delete',
                    'reference': ref_words[ref_idx],
                    'hypothesis': ''
                })
                ref_idx += 1
            elif op == 'insert':
                alignment.append({
                    'operation': 'insert',
                    'reference': '',
                    'hypothesis': hyp_words[hyp_idx]
                })
                hyp_idx += 1
        
        return alignment
    
    def _get_ngrams(self, words: List[str], n: int) -> Counter:
        """Get n-grams from word list."""
        ngrams = Counter()
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i+n])
            ngrams[ngram] += 1
        return ngrams
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * (percentile / 100)
        f = math.floor(k)
        c = math.ceil(k)
        
        if f == c:
            return sorted_values[int(k)]
        else:
            return sorted_values[int(f)] * (c - k) + sorted_values[int(c)] * (k - f)

# Convenience functions
def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate WER using default MetricsCalculator."""
    calculator = MetricsCalculator()
    return calculator.calculate_wer(reference, hypothesis, return_details=False)

def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate CER using default MetricsCalculator."""
    calculator = MetricsCalculator()
    return calculator.calculate_cer(reference, hypothesis, return_details=False)

def calculate_bleu(reference: str, hypothesis: str) -> float:
    """Calculate BLEU score using default MetricsCalculator."""
    calculator = MetricsCalculator()
    return calculator.calculate_bleu_score(reference, hypothesis, return_details=False)

def calculate_similarity(reference: str, hypothesis: str, method: str = 'jaccard') -> float:
    """Calculate semantic similarity using default MetricsCalculator."""
    calculator = MetricsCalculator()
    return calculator.calculate_semantic_similarity(reference, hypothesis, method, return_details=False)