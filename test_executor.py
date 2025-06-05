"""
Test Executor for TTS/STT Testing Framework

This module orchestrates the execution of TTS and STT evaluation tests,
managing test suites, coordinating evaluations, and generating comprehensive reports.

Author: TTS/STT Testing Framework Team
Version: 1.0.0
"""

import asyncio
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import structlog
import yaml
from pydantic import BaseModel, Field, validator

# Framework imports
from clients.client_factory import ClientFactory
from configs.base_config import BaseConfig
from core.evaluator_factory import EvaluatorFactory
from utils.audio_utils import AudioValidator
from utils.logger import setup_logging
from utils.metrics import MetricsCalculator
from utils.report_generator import ReportGenerator

# Configure structured logging
logger = structlog.get_logger(__name__)


class TestExecutionConfig(BaseModel):
    """Configuration for test execution parameters."""
    
    max_workers: int = Field(default=4, ge=1, le=16)
    timeout_seconds: int = Field(default=300, ge=30)
    retry_attempts: int = Field(default=3, ge=1, le=5)
    retry_delay: float = Field(default=1.0, ge=0.1)
    enable_parallel: bool = Field(default=True)
    memory_limit_mb: int = Field(default=2048, ge=512)
    
    @validator('max_workers')
    def validate_workers(cls, v):
        """Validate worker count against system capabilities."""
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        if v > cpu_count * 2:
            logger.warning(
                "Worker count exceeds recommended limit",
                workers=v,
                cpu_count=cpu_count,
                recommended=cpu_count * 2
            )
        return v


@dataclass
class TestResult:
    """Individual test result container."""
    
    test_id: str
    provider: str
    model_type: str  # 'tts' or 'stt'
    model_name: str
    success: bool
    execution_time: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert test result to dictionary."""
        return {
            'test_id': self.test_id,
            'provider': self.provider,
            'model_type': self.model_type,
            'model_name': self.model_name,
            'success': self.success,
            'execution_time': self.execution_time,
            'metrics': self.metrics,
            'errors': self.errors,
            'warnings': self.warnings,
            'metadata': self.metadata
        }


@dataclass
class TestSuite:
    """Test suite containing multiple test configurations."""
    
    name: str
    description: str
    tests: List[Dict[str, Any]] = field(default_factory=list)
    setup_hooks: List[str] = field(default_factory=list)
    teardown_hooks: List[str] = field(default_factory=list)
    parallel_execution: bool = True
    
    def add_test(self, provider: str, model_type: str, model_name: str, 
                 test_data: Dict[str, Any]) -> None:
        """Add a test configuration to the suite."""
        test_config = {
            'provider': provider,
            'model_type': model_type,
            'model_name': model_name,
            'test_data': test_data,
            'test_id': f"{provider}_{model_type}_{model_name}_{len(self.tests)}"
        }
        self.tests.append(test_config)
        
        logger.debug(
            "Added test to suite",
            suite_name=self.name,
            test_id=test_config['test_id'],
            provider=provider,
            model_type=model_type
        )


class TestExecutor:
    """
    Main test execution orchestrator for TTS/STT framework.
    
    This class manages the execution of evaluation tests across multiple
    providers, handles parallel execution, and coordinates result collection.
    """
    
    def __init__(self, config: Optional[BaseConfig] = None):
        """
        Initialize the test executor.
        
        Args:
            config: Base configuration object, if None loads from environment
        """
        self.config = config or BaseConfig()
        self.execution_config = TestExecutionConfig()
        
        # Initialize components
        self.client_factory = ClientFactory(self.config)
        self.evaluator_factory = EvaluatorFactory(self.config)
        self.metrics_calculator = MetricsCalculator()
        self.report_generator = ReportGenerator(self.config)
        self.audio_validator = AudioValidator()
        
        # Execution state
        self.test_suites: List[TestSuite] = []
        self.results: List[TestResult] = []
        self.execution_start_time: Optional[datetime] = None
        self.execution_end_time: Optional[datetime] = None
        
        # Resource management
        self.executor: Optional[ThreadPoolExecutor] = None
        self.active_tasks: Set[str] = set()
        
        logger.info(
            "Test executor initialized",
            max_workers=self.execution_config.max_workers,
            parallel_enabled=self.execution_config.enable_parallel
        )
    
    def add_test_suite(self, test_suite: TestSuite) -> None:
        """Add a test suite to the executor."""
        self.test_suites.append(test_suite)
        logger.info(
            "Test suite added",
            suite_name=test_suite.name,
            test_count=len(test_suite.tests)
        )
    
    def create_standard_test_suite(self) -> TestSuite:
        """Create a standard test suite with common test scenarios."""
        suite = TestSuite(
            name="Standard TTS/STT Evaluation",
            description="Comprehensive evaluation of all enabled TTS/STT models"
        )
        
        # Sample test data
        test_texts = [
            "Hello, this is a test of text-to-speech conversion.",
            "The quick brown fox jumps over the lazy dog.",
            "Testing speech recognition accuracy with various sentence structures.",
            "Numbers and dates: 123, January 1st, 2024, $45.67."
        ]
        
        # Add TTS tests for enabled providers
        enabled_tts_providers = self._get_enabled_providers('tts')
        for provider in enabled_tts_providers:
            try:
                client = self.client_factory.create_client(provider)
                models = client.get_available_models('tts')
                
                for model in models:
                    for i, text in enumerate(test_texts):
                        test_data = {
                            'text': text,
                            'test_index': i,
                            'expected_duration': len(text.split()) * 0.5  # Rough estimate
                        }
                        suite.add_test(provider, 'tts', model, test_data)
                        
            except Exception as e:
                logger.error(
                    "Failed to add TTS tests for provider",
                    provider=provider,
                    error=str(e)
                )
        
        # Add STT tests for enabled providers
        enabled_stt_providers = self._get_enabled_providers('stt')
        audio_files = self._get_test_audio_files()
        
        for provider in enabled_stt_providers:
            try:
                client = self.client_factory.create_client(provider)
                models = client.get_available_models('stt')
                
                for model in models:
                    for audio_file in audio_files:
                        test_data = {
                            'audio_file': audio_file,
                            'reference_text': self._get_reference_text(audio_file),
                            'expected_accuracy': 0.95  # Target accuracy
                        }
                        suite.add_test(provider, 'stt', model, test_data)
                        
            except Exception as e:
                logger.error(
                    "Failed to add STT tests for provider",
                    provider=provider,
                    error=str(e)
                )
        
        return suite
    
    async def execute_all_suites(self) -> Dict[str, Any]:
        """
        Execute all test suites and return comprehensive results.
        
        Returns:
            Dictionary containing execution results and metadata
        """
        self.execution_start_time = datetime.now()
        
        logger.info(
            "Starting test execution",
            suite_count=len(self.test_suites),
            start_time=self.execution_start_time.isoformat()
        )
        
        try:
            # Initialize thread pool executor
            if self.execution_config.enable_parallel:
                self.executor = ThreadPoolExecutor(
                    max_workers=self.execution_config.max_workers
                )
            
            # Execute each test suite
            suite_results = {}
            
            for suite in self.test_suites:
                logger.info(
                    "Executing test suite",
                    suite_name=suite.name,
                    test_count=len(suite.tests)
                )
                
                suite_start_time = time.time()
                suite_results[suite.name] = await self._execute_test_suite(suite)
                suite_execution_time = time.time() - suite_start_time
                
                logger.info(
                    "Test suite completed",
                    suite_name=suite.name,
                    execution_time=suite_execution_time,
                    success_count=sum(1 for r in suite_results[suite.name] if r.success),
                    failure_count=sum(1 for r in suite_results[suite.name] if not r.success)
                )
            
            self.execution_end_time = datetime.now()
            
            # Generate comprehensive results
            execution_results = self._compile_execution_results(suite_results)
            
            # Generate reports
            await self._generate_reports(execution_results)
            
            logger.info(
                "Test execution completed",
                total_duration=str(self.execution_end_time - self.execution_start_time),
                total_tests=len(self.results),
                success_rate=self._calculate_success_rate()
            )
            
            return execution_results
            
        except Exception as e:
            logger.error(
                "Test execution failed",
                error=str(e),
                traceback=traceback.format_exc()
            )
            raise
        
        finally:
            # Cleanup resources
            if self.executor:
                self.executor.shutdown(wait=True)
                self.executor = None
    
    async def _execute_test_suite(self, suite: TestSuite) -> List[TestResult]:
        """Execute a single test suite."""
        suite_results = []
        
        # Run setup hooks
        await self._run_hooks(suite.setup_hooks, "setup")
        
        try:
            if suite.parallel_execution and self.execution_config.enable_parallel:
                # Parallel execution
                suite_results = await self._execute_tests_parallel(suite.tests)
            else:
                # Sequential execution
                suite_results = await self._execute_tests_sequential(suite.tests)
                
        finally:
            # Run teardown hooks
            await self._run_hooks(suite.teardown_hooks, "teardown")
        
        self.results.extend(suite_results)
        return suite_results
    
    async def _execute_tests_parallel(self, tests: List[Dict[str, Any]]) -> List[TestResult]:
        """Execute tests in parallel using thread pool."""
        futures = []
        
        for test_config in tests:
            if self.executor:
                future = self.executor.submit(self._execute_single_test, test_config)
                futures.append((future, test_config['test_id']))
        
        results = []
        for future, test_id in futures:
            try:
                result = future.result(timeout=self.execution_config.timeout_seconds)
                results.append(result)
                
                logger.debug(
                    "Parallel test completed",
                    test_id=test_id,
                    success=result.success,
                    execution_time=result.execution_time
                )
                
            except Exception as e:
                # Create failed result for timeout/error
                error_result = TestResult(
                    test_id=test_id,
                    provider="unknown",
                    model_type="unknown",
                    model_name="unknown",
                    success=False,
                    execution_time=self.execution_config.timeout_seconds,
                    errors=[f"Test execution failed: {str(e)}"]
                )
                results.append(error_result)
                
                logger.error(
                    "Parallel test failed",
                    test_id=test_id,
                    error=str(e)
                )
        
        return results
    
    async def _execute_tests_sequential(self, tests: List[Dict[str, Any]]) -> List[TestResult]:
        """Execute tests sequentially."""
        results = []
        
        for test_config in tests:
            try:
                result = self._execute_single_test(test_config)
                results.append(result)
                
                logger.debug(
                    "Sequential test completed",
                    test_id=test_config['test_id'],
                    success=result.success,
                    execution_time=result.execution_time
                )
                
            except Exception as e:
                # Create failed result
                error_result = TestResult(
                    test_id=test_config['test_id'],
                    provider=test_config.get('provider', 'unknown'),
                    model_type=test_config.get('model_type', 'unknown'),
                    model_name=test_config.get('model_name', 'unknown'),
                    success=False,
                    execution_time=0.0,
                    errors=[f"Test execution failed: {str(e)}"]
                )
                results.append(error_result)
                
                logger.error(
                    "Sequential test failed",
                    test_id=test_config['test_id'],
                    error=str(e)
                )
        
        return results
    
    def _execute_single_test(self, test_config: Dict[str, Any]) -> TestResult:
        """Execute a single test and return results."""
        test_id = test_config['test_id']
        provider = test_config['provider']
        model_type = test_config['model_type']
        model_name = test_config['model_name']
        test_data = test_config['test_data']
        
        start_time = time.time()
        
        logger.debug(
            "Starting single test execution",
            test_id=test_id,
            provider=provider,
            model_type=model_type
        )
        
        try:
            # Get client and evaluator
            client = self.client_factory.create_client(provider)
            evaluator = self.evaluator_factory.create_evaluator(model_type)
            
            # Execute test based on model type
            if model_type == 'tts':
                result = self._execute_tts_test(
                    client, evaluator, model_name, test_data
                )
            elif model_type == 'stt':
                result = self._execute_stt_test(
                    client, evaluator, model_name, test_data
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            execution_time = time.time() - start_time
            
            # Create successful result
            test_result = TestResult(
                test_id=test_id,
                provider=provider,
                model_type=model_type,
                model_name=model_name,
                success=True,
                execution_time=execution_time,
                metrics=result.get('metrics', {}),
                warnings=result.get('warnings', []),
                metadata=result.get('metadata', {})
            )
            
            logger.debug(
                "Single test completed successfully",
                test_id=test_id,
                execution_time=execution_time
            )
            
            return test_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Create failed result
            test_result = TestResult(
                test_id=test_id,
                provider=provider,
                model_type=model_type,
                model_name=model_name,
                success=False,
                execution_time=execution_time,
                errors=[str(e)],
                metadata={'exception_type': type(e).__name__}
            )
            
            logger.error(
                "Single test failed",
                test_id=test_id,
                error=str(e),
                execution_time=execution_time
            )
            
            return test_result
    
    def _execute_tts_test(self, client, evaluator, model_name: str, 
                         test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a TTS test."""
        text = test_data['text']
        
        # Generate audio using client
        audio_result = client.text_to_speech(text, model_name)
        
        if not audio_result.success:
            raise Exception(f"TTS generation failed: {audio_result.error}")
        
        # Evaluate generated audio
        evaluation_result = evaluator.evaluate_audio(
            audio_file=audio_result.audio_file,
            reference_text=text,
            model_name=model_name
        )
        
        return evaluation_result
    
    def _execute_stt_test(self, client, evaluator, model_name: str, 
                         test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a STT test."""
        audio_file = test_data['audio_file']
        reference_text = test_data['reference_text']
        
        # Transcribe audio using client
        transcription_result = client.speech_to_text(audio_file, model_name)
        
        if not transcription_result.success:
            raise Exception(f"STT transcription failed: {transcription_result.error}")
        
        # Evaluate transcription accuracy
        evaluation_result = evaluator.evaluate_transcription(
            predicted_text=transcription_result.text,
            reference_text=reference_text,
            model_name=model_name,
            audio_file=audio_file
        )
        
        return evaluation_result
    
    def _compile_execution_results(self, suite_results: Dict[str, List[TestResult]]) -> Dict[str, Any]:
        """Compile comprehensive execution results."""
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - successful_tests
        
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics()
        
        # Provider performance summary
        provider_summary = self._calculate_provider_summary()
        
        # Model type summary
        model_type_summary = self._calculate_model_type_summary()
        
        execution_duration = (
            self.execution_end_time - self.execution_start_time
            if self.execution_end_time and self.execution_start_time
            else timedelta(0)
        )
        
        return {
            'metadata': {
                'framework_version': '1.0.0',
                'execution_timestamp': self.execution_start_time.isoformat() if self.execution_start_time else None,
                'execution_duration': str(execution_duration),
                'total_test_suites': len(self.test_suites),
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'failed_tests': failed_tests,
                'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
                'configuration': {
                    'max_workers': self.execution_config.max_workers,
                    'parallel_execution': self.execution_config.enable_parallel,
                    'timeout_seconds': self.execution_config.timeout_seconds
                }
            },
            'suite_results': {
                suite_name: [result.to_dict() for result in results]
                for suite_name, results in suite_results.items()
            },
            'aggregate_metrics': aggregate_metrics,
            'provider_summary': provider_summary,
            'model_type_summary': model_type_summary,
            'detailed_results': [result.to_dict() for result in self.results]
        }
    
    def _calculate_aggregate_metrics(self) -> Dict[str, Any]:
        """Calculate aggregate metrics across all tests."""
        successful_results = [r for r in self.results if r.success]
        
        if not successful_results:
            return {}
        
        # Collect metrics by type
        tts_metrics = [r.metrics for r in successful_results if r.model_type == 'tts']
        stt_metrics = [r.metrics for r in successful_results if r.model_type == 'stt']
        
        aggregate = {}
        
        # TTS aggregate metrics
        if tts_metrics:
            aggregate['tts'] = self.metrics_calculator.calculate_aggregate_metrics(
                tts_metrics, 'tts'
            )
        
        # STT aggregate metrics
        if stt_metrics:
            aggregate['stt'] = self.metrics_calculator.calculate_aggregate_metrics(
                stt_metrics, 'stt'
            )
        
        # Overall performance metrics
        execution_times = [r.execution_time for r in successful_results]
        aggregate['performance'] = {
            'avg_execution_time': sum(execution_times) / len(execution_times),
            'min_execution_time': min(execution_times),
            'max_execution_time': max(execution_times),
            'total_execution_time': sum(execution_times)
        }
        
        return aggregate
    
    def _calculate_provider_summary(self) -> Dict[str, Any]:
        """Calculate performance summary by provider."""
        provider_summary = {}
        
        for provider in set(r.provider for r in self.results):
            provider_results = [r for r in self.results if r.provider == provider]
            successful = [r for r in provider_results if r.success]
            
            provider_summary[provider] = {
                'total_tests': len(provider_results),
                'successful_tests': len(successful),
                'failed_tests': len(provider_results) - len(successful),
                'success_rate': len(successful) / len(provider_results) if provider_results else 0,
                'avg_execution_time': (
                    sum(r.execution_time for r in successful) / len(successful)
                    if successful else 0
                ),
                'models_tested': list(set(r.model_name for r in provider_results))
            }
        
        return provider_summary
    
    def _calculate_model_type_summary(self) -> Dict[str, Any]:
        """Calculate performance summary by model type."""
        model_type_summary = {}
        
        for model_type in set(r.model_type for r in self.results):
            type_results = [r for r in self.results if r.model_type == model_type]
            successful = [r for r in type_results if r.success]
            
            model_type_summary[model_type] = {
                'total_tests': len(type_results),
                'successful_tests': len(successful),
                'failed_tests': len(type_results) - len(successful),
                'success_rate': len(successful) / len(type_results) if type_results else 0,
                'providers_tested': list(set(r.provider for r in type_results))
            }
        
        return model_type_summary
    
    async def _generate_reports(self, execution_results: Dict[str, Any]) -> None:
        """Generate comprehensive reports in multiple formats."""
        try:
            # Generate HTML report
            if self.config.enable_html_reports:
                html_report = await self.report_generator.generate_html_report(
                    execution_results
                )
                logger.info("HTML report generated", file_path=html_report)
            
            # Generate JSON report
            if self.config.enable_json_reports:
                json_report = await self.report_generator.generate_json_report(
                    execution_results
                )
                logger.info("JSON report generated", file_path=json_report)
            
            # Generate YAML report
            if self.config.enable_yaml_reports:
                yaml_report = await self.report_generator.generate_yaml_report(
                    execution_results
                )
                logger.info("YAML report generated", file_path=yaml_report)
                
        except Exception as e:
            logger.error(
                "Report generation failed",
                error=str(e),
                traceback=traceback.format_exc()
            )
    
    async def _run_hooks(self, hooks: List[str], hook_type: str) -> None:
        """Run setup or teardown hooks."""
        for hook in hooks:
            try:
                logger.debug(f"Running {hook_type} hook", hook=hook)
                # Hook execution logic would go here
                # This could involve running shell commands, Python functions, etc.
                pass
            except Exception as e:
                logger.error(
                    f"{hook_type.title()} hook failed",
                    hook=hook,
                    error=str(e)
                )
    
    def _get_enabled_providers(self, model_type: str) -> List[str]:
        """Get list of enabled providers for the specified model type."""
        providers = []
        
        env_prefix = f"ENABLE_{model_type.upper()}_"
        
        for key, value in os.environ.items():
            if key.startswith(env_prefix) and value.lower() == 'true':
                provider = key.replace(env_prefix, '').lower()
                providers.append(provider)
        
        return providers
    
    def _get_test_audio_files(self) -> List[str]:
        """Get list of test audio files for STT evaluation."""
        test_data_dir = Path(self.config.test_data_dir)
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        
        audio_files = []
        
        if test_data_dir.exists():
            for ext in audio_extensions:
                audio_files.extend(test_data_dir.glob(f"*{ext}"))
        
        return [str(f) for f in audio_files]
    
    def _get_reference_text(self, audio_file: str) -> str:
        """Get reference text for an audio file."""
        # Look for corresponding .txt file
        audio_path = Path(audio_file)
        txt_file = audio_path.with_suffix('.txt')
        
        if txt_file.exists():
            return txt_file.read_text(encoding='utf-8').strip()
        
        # Default reference text if not found
        return "Reference text not available"
    
    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate."""
        if not self.results:
            return 0.0
        
        successful = sum(1 for r in self.results if r.success)
        return successful / len(self.results)
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of the current execution state."""
        return {
            'test_suites_count': len(self.test_suites),
            'total_tests': len(self.results),
            'completed_tests': len([r for r in self.results if r.success or r.errors]),
            'active_tasks': len(self.active_tasks),
            'execution_status': 'running' if self.execution_start_time and not self.execution_end_time else 'completed'
        }


# CLI interface functions
def create_test_executor_from_args(args: Dict[str, Any]) -> TestExecutor:
    """Create test executor from command line arguments."""
    config = BaseConfig()
    
    # Override config with command line arguments
    if 'max_workers' in args:
        config.max_workers = args['max_workers']
    
    if 'timeout' in args:
        config.timeout_seconds = args['timeout']
    
    executor = TestExecutor(config)
    
    # Configure execution parameters
    if 'parallel' in args:
        executor.execution_config.enable_parallel = args['parallel']
    
    return executor


async def main():
    """Main entry point for test executor."""
    # Setup logging
    setup_logging()
    
    logger.info("Starting TTS/STT Test Executor")
    
    try:
        # Create test executor
        executor = TestExecutor()
        
        # Create standard test suite
        standard_suite = executor.create_standard_test_suite()
        executor.add_test_suite(standard_suite)
        
        # Execute all tests
        results = await executor.execute_all_suites()
        
        # Print summary
        print("\n" + "="*60)
        print("TEST EXECUTION SUMMARY")
        print("="*60)
        print(f"Total Tests: {results['metadata']['total_tests']}")
        print(f"Successful: {results['metadata']['successful_tests']}")
        print(f"Failed: {results['metadata']['failed_tests']}")
        print(f"Success Rate: {results['metadata']['success_rate']:.2%}")
        print(f"Execution Duration: {results['metadata']['execution_duration']}")
        print("="*60)
        
        return results
        
    except Exception as e:
        logger.error(
            "Test executor failed",
            error=str(e),
            traceback=traceback.format_exc()
        )
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())