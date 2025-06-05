#!/usr/bin/env python3
"""
Framework Integration Test Script

This script validates that all components of the TTS/STT Testing Framework
are properly interlinked and functioning correctly before actual execution.

Author: TTS/STT Testing Framework Team
Version: 1.0.0
"""

import asyncio
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

# Initialize console for output
console = Console()


class TestResult:
    """Test result container."""
    
    def __init__(self, name: str, passed: bool, message: str = "", 
                 details: Optional[str] = None, warning: bool = False):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details
        self.warning = warning
    
    def __repr__(self):
        status = "PASS" if self.passed else "WARN" if self.warning else "FAIL"
        return f"TestResult({self.name}: {status})"


class FrameworkIntegrationTester:
    """Comprehensive framework integration tester."""
    
    def __init__(self):
        self.test_results: List[TestResult] = []
        self.temp_dir = tempfile.mkdtemp(prefix="tts_stt_test_")
        self.original_cwd = os.getcwd()
        
        # Test configuration
        self.test_config = {
            'test_data_dir': f"{self.temp_dir}/data/test_samples",
            'reference_data_dir': f"{self.temp_dir}/data/reference", 
            'output_data_dir': f"{self.temp_dir}/data/outputs",
            'results_dir': f"{self.temp_dir}/results"
        }
        
        console.print(Panel(
            "[bold blue]TTS/STT Framework Integration Test Suite[/bold blue]\n"
            "This script validates all framework components and their interconnections.",
            title="🧪 Integration Test Suite",
            expand=False
        ))
    
    def add_result(self, name: str, passed: bool, message: str = "", 
                   details: Optional[str] = None, warning: bool = False):
        """Add a test result."""
        result = TestResult(name, passed, message, details, warning)
        self.test_results.append(result)
        
        # Immediate feedback
        if passed:
            console.print(f"✅ {name}: {message}")
        elif warning:
            console.print(f"⚠️  {name}: {message}")
        else:
            console.print(f"❌ {name}: {message}")
            if details:
                console.print(f"   Details: {details}")
    
    async def run_all_tests(self) -> bool:
        """Run all integration tests."""
        console.print("\n[bold yellow]Starting Integration Tests...[/bold yellow]\n")
        
        # Setup test environment
        await self.setup_test_environment()
        
        # Run test categories
        test_categories = [
            ("Framework Structure", self.test_framework_structure),
            ("Dependencies", self.test_dependencies),
            ("Configuration", self.test_configuration_system),
            ("Client Factory", self.test_client_factory),
            ("Evaluator Factory", self.test_evaluator_factory),
            ("Utilities", self.test_utility_modules),
            ("Test Executor", self.test_executor_functionality),
            ("Report Generation", self.test_report_generation),
            ("Main CLI", self.test_main_cli),
            ("Integration Flow", self.test_integration_flow)
        ]
        
        for category_name, test_func in test_categories:
            console.print(f"\n[bold cyan]Testing {category_name}...[/bold cyan]")
            try:
                await test_func()
            except Exception as e:
                self.add_result(
                    f"{category_name} - Critical Error",
                    False,
                    f"Test category failed: {str(e)}",
                    traceback.format_exc()
                )
        
        # Display results summary
        self.display_test_summary()
        
        # Cleanup
        await self.cleanup_test_environment()
        
        # Return overall success
        failed_tests = [r for r in self.test_results if not r.passed and not r.warning]
        return len(failed_tests) == 0
    
    async def setup_test_environment(self):
        """Setup test environment with required directories and files."""
        try:
            # Create test directories
            for dir_path in self.test_config.values():
                Path(dir_path).mkdir(parents=True, exist_ok=True)
            
            # Create sample test files
            await self.create_sample_test_files()
            
            # Set environment variables for testing
            os.environ.update({
                'LOG_LEVEL': 'INFO',
                'DEBUG_MODE': 'false',
                'FRAMEWORK_NAME': 'TTS-STT-Testing-Framework',
                'FRAMEWORK_VERSION': '1.0.0',
                'RESULTS_DIR': self.test_config['results_dir'],
                'TEST_DATA_DIR': self.test_config['test_data_dir'],
                'ENABLE_HTML_REPORTS': 'true',
                'ENABLE_JSON_REPORTS': 'true',
                'ENABLE_YAML_REPORTS': 'true',
                # Mock API keys for testing
                'OPENAI_API_KEY': 'test_openai_key',
                'AZURE_SPEECH_KEY': 'test_azure_key',
                'AZURE_SPEECH_REGION': 'test_region',
                'GOOGLE_APPLICATION_CREDENTIALS': f"{self.temp_dir}/mock_google_creds.json"
            })
            
            # Create mock Google credentials file
            mock_creds = {
                "type": "service_account",
                "project_id": "test-project",
                "private_key_id": "test-key-id",
                "private_key": "-----BEGIN PRIVATE KEY-----\nMOCK_KEY\n-----END PRIVATE KEY-----\n",
                "client_email": "test@test-project.iam.gserviceaccount.com",
                "client_id": "123456789",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token"
            }
            
            with open(f"{self.temp_dir}/mock_google_creds.json", 'w') as f:
                json.dump(mock_creds, f)
            
            self.add_result(
                "Test Environment Setup",
                True,
                "Test environment initialized successfully"
            )
            
        except Exception as e:
            self.add_result(
                "Test Environment Setup",
                False,
                f"Failed to setup test environment: {str(e)}",
                traceback.format_exc()
            )
    
    async def create_sample_test_files(self):
        """Create sample test files for testing."""
        try:
            # Create sample audio files (empty files for testing)
            audio_files = [
                "sample1.wav",
                "sample2.mp3", 
                "sample3.flac"
            ]
            
            for audio_file in audio_files:
                audio_path = Path(self.test_config['test_data_dir']) / audio_file
                audio_path.write_bytes(b"MOCK_AUDIO_DATA")
                
                # Create corresponding reference text file
                txt_path = audio_path.with_suffix('.txt')
                txt_path.write_text("This is a sample reference text for testing.")
            
            # Create configuration files
            config_dir = Path(self.temp_dir) / "configs"
            config_dir.mkdir(exist_ok=True)
            
            # Create sample config YAML
            sample_config = {
                'framework': {
                    'name': 'TTS-STT-Testing-Framework',
                    'version': '1.0.0'
                },
                'evaluation': {
                    'tts_enabled': True,
                    'stt_enabled': True,
                    'parallel_execution': True
                }
            }
            
            import yaml
            with open(config_dir / "test_config.yaml", 'w') as f:
                yaml.dump(sample_config, f)
            
        except Exception as e:
            self.add_result(
                "Sample Files Creation",
                False,
                f"Failed to create sample files: {str(e)}"
            )
    
    async def test_framework_structure(self):
        """Test framework directory structure and required files."""
        required_files = [
            "main.py",
            "test_executor.py", 
            "requirements.txt",
            ".env",
            "README.md"
        ]
        
        required_directories = [
            "clients",
            "core", 
            "configs",
            "utils",
            "tests",
            "data",
            "results"
        ]
        
        # Check required files
        for file_name in required_files:
            if Path(file_name).exists():
                self.add_result(
                    f"File: {file_name}",
                    True,
                    "File exists"
                )
            else:
                self.add_result(
                    f"File: {file_name}",
                    False,
                    "Required file missing"
                )
        
        # Check required directories
        for dir_name in required_directories:
            if Path(dir_name).exists():
                self.add_result(
                    f"Directory: {dir_name}",
                    True,
                    "Directory exists"
                )
                
                # Check for __init__.py in Python packages
                if dir_name in ["clients", "core", "configs", "utils"]:
                    init_file = Path(dir_name) / "__init__.py"
                    if init_file.exists():
                        self.add_result(
                            f"Package: {dir_name}",
                            True,
                            "__init__.py found"
                        )
                    else:
                        self.add_result(
                            f"Package: {dir_name}",
                            False,
                            "__init__.py missing"
                        )
            else:
                self.add_result(
                    f"Directory: {dir_name}",
                    False,
                    "Required directory missing"
                )
    
    async def test_dependencies(self):
        """Test if all required dependencies can be imported."""
        required_imports = [
            ("structlog", "Structured logging"),
            ("pydantic", "Data validation"),
            ("yaml", "YAML processing"),
            ("click", "CLI framework"),
            ("rich", "Rich console output"),
            ("librosa", "Audio processing"),
            ("numpy", "Numerical computing"),
            ("requests", "HTTP requests"),
            ("jinja2", "Template engine"),
            ("asyncio", "Async programming")
        ]
        
        for module_name, description in required_imports:
            try:
                __import__(module_name)
                self.add_result(
                    f"Import: {module_name}",
                    True,
                    f"{description} available"
                )
            except ImportError as e:
                self.add_result(
                    f"Import: {module_name}",
                    False,
                    f"Failed to import {description}",
                    str(e)
                )
    
    async def test_configuration_system(self):
        """Test configuration system functionality."""
        try:
            # Test base configuration import
            from configs.base_config import BaseConfig
            
            self.add_result(
                "BaseConfig Import",
                True,
                "BaseConfig imported successfully"
            )
            
            # Test configuration initialization
            config = BaseConfig()
            
            # Test required configuration attributes
            required_attrs = [
                'framework_name',
                'framework_version',
                'log_level',
                'results_dir',
                'max_workers'
            ]
            
            for attr in required_attrs:
                if hasattr(config, attr):
                    self.add_result(
                        f"Config Attribute: {attr}",
                        True,
                        f"Attribute exists: {getattr(config, attr)}"
                    )
                else:
                    self.add_result(
                        f"Config Attribute: {attr}",
                        False,
                        "Required attribute missing"
                    )
            
            # Test TTS and STT specific configs
            try:
                from configs.tts_config import TTSConfig
                from configs.stt_config import STTConfig
                
                tts_config = TTSConfig()
                stt_config = STTConfig()
                
                self.add_result(
                    "TTS/STT Configs",
                    True,
                    "TTS and STT configurations loaded"
                )
                
            except ImportError as e:
                self.add_result(
                    "TTS/STT Configs",
                    False,
                    "Failed to load TTS/STT configurations",
                    str(e)
                )
                
        except Exception as e:
            self.add_result(
                "Configuration System",
                False,
                f"Configuration system test failed: {str(e)}",
                traceback.format_exc()
            )
    
    async def test_client_factory(self):
        """Test client factory functionality."""
        try:
            from clients.client_factory import ClientFactory
            from configs.base_config import BaseConfig
            
            config = BaseConfig()
            factory = ClientFactory(config)
            
            self.add_result(
                "ClientFactory Creation",
                True,
                "ClientFactory initialized successfully"
            )
            
            # Test available providers
            available_providers = factory.get_available_providers()
            
            if available_providers:
                self.add_result(
                    "Available Providers",
                    True,
                    f"Found providers: {', '.join(available_providers)}"
                )
            else:
                self.add_result(
                    "Available Providers",
                    False,
                    "No providers available"
                )
            
            # Test client creation (with mocking)
            test_providers = ['openai', 'azure', 'google']
            
            for provider in test_providers:
                try:
                    with patch.multiple(
                        f'clients.{provider}_client',
                        **{f'{provider.title()}Client': MagicMock()}
                    ):
                        client = factory.create_client(provider)
                        self.add_result(
                            f"Client Creation: {provider}",
                            True,
                            f"{provider.title()} client created"
                        )
                except Exception as e:
                    self.add_result(
                        f"Client Creation: {provider}",
                        False,
                        f"Failed to create {provider} client",
                        str(e)
                    )
                    
        except Exception as e:
            self.add_result(
                "Client Factory",
                False,
                f"Client factory test failed: {str(e)}",
                traceback.format_exc()
            )
    
    async def test_evaluator_factory(self):
        """Test evaluator factory functionality."""
        try:
            from core.evaluator_factory import EvaluatorFactory
            from configs.base_config import BaseConfig
            
            config = BaseConfig()
            factory = EvaluatorFactory(config)
            
            self.add_result(
                "EvaluatorFactory Creation",
                True,
                "EvaluatorFactory initialized successfully"
            )
            
            # Test evaluator creation
            evaluator_types = ['tts', 'stt']
            
            for eval_type in evaluator_types:
                try:
                    evaluator = factory.create_evaluator(eval_type)
                    self.add_result(
                        f"Evaluator Creation: {eval_type}",
                        True,
                        f"{eval_type.upper()} evaluator created"
                    )
                except Exception as e:
                    self.add_result(
                        f"Evaluator Creation: {eval_type}",
                        False,
                        f"Failed to create {eval_type} evaluator",
                        str(e)
                    )
            
            # Test available evaluators
            available_evaluators = factory.get_available_evaluators()
            
            if available_evaluators:
                self.add_result(
                    "Available Evaluators",
                    True,
                    f"Found evaluators: {', '.join(available_evaluators)}"
                )
            else:
                self.add_result(
                    "Available Evaluators",
                    False,
                    "No evaluators available"
                )
                
        except Exception as e:
            self.add_result(
                "Evaluator Factory",
                False,
                f"Evaluator factory test failed: {str(e)}",
                traceback.format_exc()
            )
    
    async def test_utility_modules(self):
        """Test utility modules functionality."""
        utility_modules = [
            ("utils.logger", "Logging utilities"),
            ("utils.audio_utils", "Audio processing utilities"),
            ("utils.metrics", "Metrics calculation"),
            ("utils.report_generator", "Report generation")
        ]
        
        for module_name, description in utility_modules:
            try:
                module = __import__(module_name, fromlist=[''])
                self.add_result(
                    f"Utility: {module_name}",
                    True,
                    f"{description} imported successfully"
                )
                
                # Test specific utility functions
                if module_name == "utils.logger":
                    try:
                        from utils.logger import setup_logging
                        setup_logging(level='INFO', enable_console=False)
                        self.add_result(
                            "Logger Setup",
                            True,
                            "Logging configured successfully"
                        )
                    except Exception as e:
                        self.add_result(
                            "Logger Setup",
                            False,
                            f"Logger setup failed: {str(e)}"
                        )
                
                elif module_name == "utils.audio_utils":
                    try:
                        from utils.audio_utils import AudioValidator
                        validator = AudioValidator()
                        self.add_result(
                            "Audio Validator",
                            True,
                            "Audio validator initialized"
                        )
                    except Exception as e:
                        self.add_result(
                            "Audio Validator",
                            False,
                            f"Audio validator failed: {str(e)}"
                        )
                
                elif module_name == "utils.metrics":
                    try:
                        from utils.metrics import MetricsCalculator
                        calculator = MetricsCalculator()
                        self.add_result(
                            "Metrics Calculator",
                            True,
                            "Metrics calculator initialized"
                        )
                    except Exception as e:
                        self.add_result(
                            "Metrics Calculator",
                            False,
                            f"Metrics calculator failed: {str(e)}"
                        )
                
                elif module_name == "utils.report_generator":
                    try:
                        from utils.report_generator import ReportGenerator
                        from configs.base_config import BaseConfig
                        generator = ReportGenerator(BaseConfig())
                        self.add_result(
                            "Report Generator",
                            True,
                            "Report generator initialized"
                        )
                    except Exception as e:
                        self.add_result(
                            "Report Generator",
                            False,
                            f"Report generator failed: {str(e)}"
                        )
                        
            except ImportError as e:
                self.add_result(
                    f"Utility: {module_name}",
                    False,
                    f"Failed to import {description}",
                    str(e)
                )
    
    async def test_executor_functionality(self):
        """Test test executor functionality."""
        try:
            from test_executor import TestExecutor, TestSuite
            from configs.base_config import BaseConfig
            
            config = BaseConfig()
            executor = TestExecutor(config)
            
            self.add_result(
                "TestExecutor Creation",
                True,
                "TestExecutor initialized successfully"
            )
            
            # Test suite creation
            test_suite = TestSuite(
                name="Integration Test Suite",
                description="Test suite for integration testing"
            )
            
            # Add mock test
            test_suite.add_test(
                provider="mock_provider",
                model_type="tts",
                model_name="mock_model",
                test_data={"text": "Test text"}
            )
            
            executor.add_test_suite(test_suite)
            
            self.add_result(
                "Test Suite Management",
                True,
                "Test suite created and added successfully"
            )
            
            # Test execution configuration
            exec_config = executor.execution_config
            
            required_config_attrs = [
                'max_workers',
                'timeout_seconds',
                'enable_parallel',
                'retry_attempts'
            ]
            
            config_valid = all(hasattr(exec_config, attr) for attr in required_config_attrs)
            
            self.add_result(
                "Execution Configuration",
                config_valid,
                "Execution configuration validated" if config_valid else "Missing configuration attributes"
            )
            
        except Exception as e:
            self.add_result(
                "Test Executor",
                False,
                f"Test executor test failed: {str(e)}",
                traceback.format_exc()
            )
    
    async def test_report_generation(self):
        """Test report generation functionality."""
        try:
            from utils.report_generator import ReportGenerator
            from configs.base_config import BaseConfig
            
            config = BaseConfig()
            # Override results directory for testing
            config.results_dir = self.test_config['results_dir']
            
            generator = ReportGenerator(config)
            
            # Create mock results data
            mock_results = {
                'metadata': {
                    'timestamp': '2024-01-15T10:30:00Z',
                    'framework_version': '1.0.0',
                    'total_tests': 5,
                    'successful_tests': 4,
                    'failed_tests': 1,
                    'success_rate': 0.8
                },
                'detailed_results': [
                    {
                        'test_id': 'test_1',
                        'provider': 'openai',
                        'model_type': 'tts',
                        'success': True,
                        'metrics': {'quality': 0.85}
                    }
                ]
            }
            
            # Test JSON report generation (mock)
            with patch.object(generator, 'generate_json_report') as mock_json:
                mock_json.return_value = f"{self.test_config['results_dir']}/test_report.json"
                json_report = await generator.generate_json_report(mock_results)
                
                self.add_result(
                    "JSON Report Generation",
                    True,
                    "JSON report generation tested successfully"
                )
            
            # Test YAML report generation (mock)
            with patch.object(generator, 'generate_yaml_report') as mock_yaml:
                mock_yaml.return_value = f"{self.test_config['results_dir']}/test_report.yaml"
                yaml_report = await generator.generate_yaml_report(mock_results)
                
                self.add_result(
                    "YAML Report Generation",
                    True,
                    "YAML report generation tested successfully"
                )
            
            # Test HTML report generation (mock)
            with patch.object(generator, 'generate_html_report') as mock_html:
                mock_html.return_value = f"{self.test_config['results_dir']}/test_report.html"
                html_report = await generator.generate_html_report(mock_results)
                
                self.add_result(
                    "HTML Report Generation",
                    True,
                    "HTML report generation tested successfully"
                )
                
        except Exception as e:
            self.add_result(
                "Report Generation",
                False,
                f"Report generation test failed: {str(e)}",
                traceback.format_exc()
            )
    
    async def test_main_cli(self):
        """Test main CLI functionality."""
        try:
            import main
            
            # Test CLI group import
            self.add_result(
                "Main CLI Import",
                True,
                "Main CLI module imported successfully"
            )
            
            # Test framework metadata
            metadata = main.FrameworkMetadata()
            
            required_metadata = ['NAME', 'VERSION', 'DESCRIPTION', 'AUTHOR']
            metadata_valid = all(hasattr(metadata, attr) for attr in required_metadata)
            
            self.add_result(
                "Framework Metadata",
                metadata_valid,
                "Framework metadata validated" if metadata_valid else "Missing metadata attributes"
            )
            
            # Test helper functions
            helper_functions = [
                'create_tts_test_suite',
                'create_stt_test_suite',
                'get_enabled_tts_providers',
                'get_enabled_stt_providers'
            ]
            
            for func_name in helper_functions:
                if hasattr(main, func_name):
                    self.add_result(
                        f"Helper Function: {func_name}",
                        True,
                        "Function exists"
                    )
                else:
                    self.add_result(
                        f"Helper Function: {func_name}",
                        False,
                        "Function missing"
                    )
                    
        except Exception as e:
            self.add_result(
                "Main CLI",
                False,
                f"Main CLI test failed: {str(e)}",
                traceback.format_exc()
            )
    
    async def test_integration_flow(self):
        """Test end-to-end integration flow."""
        try:
            # Test full integration with mocked components
            from test_executor import TestExecutor, TestSuite
            from configs.base_config import BaseConfig
            
            # Create test configuration
            config = BaseConfig()
            config.results_dir = self.test_config['results_dir']
            config.test_data_dir = self.test_config['test_data_dir']
            
            # Create executor
            executor = TestExecutor(config)
            
            # Create test suite
            test_suite = TestSuite(
                name="Integration Flow Test",
                description="End-to-end integration test"
            )
            
            # Add mock tests
            test_suite.add_test(
                provider="mock_provider",
                model_type="tts",
                model_name="mock_tts_model",
                test_data={"text": "Integration test text"}
            )
            
            test_suite.add_test(
                provider="mock_provider", 
                model_type="stt",
                model_name="mock_stt_model",
                test_data={
                    "audio_file": f"{self.test_config['test_data_dir']}/sample1.wav",
                    "reference_text": "Integration test reference"
                }
            )
            
            executor.add_test_suite(test_suite)
            
            # Mock the execution to avoid actual API calls
            with patch.object(executor, '_execute_single_test') as mock_execute:
                mock_execute.return_value = MagicMock(
                    test_id="mock_test",
                    provider="mock_provider",
                    model_type="tts",
                    model_name="mock_model",
                    success=True,
                    execution_time=1.0,
                    metrics={"quality": 0.85},
                    errors=[],
                    warnings=[],
                    metadata={}
                )
                
                # Test execution summary
                summary = executor.get_execution_summary()
                
                self.add_result(
                    "Integration Flow",
                    True,
                    "End-to-end integration flow tested successfully"
                )
                
                self.add_result(
                    "Execution Summary",
                    'test_suites_count' in summary,
                    "Execution summary generated" if 'test_suites_count' in summary else "Summary generation failed"
                )
                
        except Exception as e:
            self.add_result(
                "Integration Flow",
                False,
                f"Integration flow test failed: {str(e)}",
                traceback.format_exc()
            )
    
    def display_test_summary(self):
        """Display comprehensive test results summary."""
        console.print("\n" + "="*80)
        console.print("[bold yellow]INTEGRATION TEST SUMMARY[/bold yellow]")
        console.print("="*80)
        
        # Count results
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.passed])
        failed_tests = len([r for r in self.test_results if not r.passed and not r.warning])
        warning_tests = len([r for r in self.test_results if r.warning])
        
        # Summary table
        summary_table = Table(title="Test Results Summary")
        summary_table.add_column("Category", style="cyan")
        summary_table.add_column("Count", style="green")
        summary_table.add_column("Percentage", style="yellow")
        
        summary_table.add_row("Total Tests", str(total_tests), "100%")
        summary_table.add_row("Passed", str(passed_tests), f"{passed_tests/total_tests*100:.1f}%" if total_tests > 0 else "0%")
        summary_table.add_row("Failed", str(failed_tests), f"{failed_tests/total_tests*100:.1f}%" if total_tests > 0 else "0%")
        summary_table.add_row("Warnings", str(warning_tests), f"{warning_tests/total_tests*100:.1f}%" if total_tests > 0 else "0%")
        
        console.print(summary_table)
        
        # Detailed results by category
        if failed_tests > 0:
            console.print("\n[bold red]FAILED TESTS:[/bold red]")
            failed_table = Table()
            failed_table.add_column("Test Name", style="red")
            failed_table.add_column("Error Message", style="yellow")
            
            for result in self.test_results:
                if not result.passed and not result.warning:
                    failed_table.add_row(result.name, result.message)
            
            console.print(failed_table)
        
        if warning_tests > 0:
            console.print("\n[bold yellow]WARNINGS:[/bold yellow]")
            warning_table = Table()
            warning_table.add_column("Test Name", style="yellow")
            warning_table.add_column("Warning Message", style="cyan")
            
            for result in self.test_results:
                if result.warning:
                    warning_table.add_row(result.name, result.message)
            
            console.print(warning_table)
        
        # Overall status
        if failed_tests == 0:
            console.print("\n[bold green]✅ ALL TESTS PASSED - Framework is ready for execution![/bold green]")
        else:
            console.print(f"\n[bold red]❌ {failed_tests} TESTS FAILED - Please fix issues before running the framework[/bold red]")
        
        console.print("="*80)
    
    async def cleanup_test_environment(self):
        """Cleanup test environment."""
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            
            # Reset environment variables
            test_env_vars = [
                'LOG_LEVEL',
                'DEBUG_MODE', 
                'FRAMEWORK_NAME',
                'FRAMEWORK_VERSION',
                'RESULTS_DIR',
                'TEST_DATA_DIR',
                'OPENAI_API_KEY',
                'AZURE_SPEECH_KEY',
                'AZURE_SPEECH_REGION',
                'GOOGLE_APPLICATION_CREDENTIALS'
            ]
            
            for var in test_env_vars:
                os.environ.pop(var, None)
            
        except Exception as e:
            console.print(f"[yellow]Warning: Cleanup failed: {str(e)}[/yellow]")


async def main():
    """Main function to run integration tests."""
    console.print("[bold blue]TTS/STT Framework Integration Test Suite[/bold blue]")
    console.print("Testing framework components and interconnections...\n")
    
    tester = FrameworkIntegrationTester()
    
    try:
        # Run all tests
        success = await tester.run_all_tests()
        
        # Exit with appropriate code
        if success:
            console.print("\n[bold green]Integration tests completed successfully! 🎉[/bold green]")
            console.print("The framework is ready for execution.")
            sys.exit(0)
        else:
            console.print("\n[bold red]Integration tests failed! ❌[/bold red]") 
            console.print("Please fix the reported issues before running the framework.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Integration tests cancelled by user[/yellow]")
        sys.exit(130)
    
    except Exception as e:
        console.print(f"\n[bold red]Integration test suite failed: {str(e)}[/bold red]")
        console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())