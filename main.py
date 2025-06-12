#!/usr/bin/env python3
"""
Main Entry Point for TTS/STT Testing Framework

This module serves as the primary entry point for the TTS/STT evaluation framework,
providing command-line interface, configuration management, and orchestration
of the entire testing pipeline.

Author: TTS/STT Testing Framework Team
Version: 1.0.0
"""

import argparse
import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import structlog
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Framework imports
from configs.base_config import BaseConfig
from test_executor import TestExecutor, TestSuite
from utils.logger import setup_logging
from utils.report_generator import ReportGenerator

# Initialize rich console for beautiful output
console = Console()
logger = structlog.get_logger(__name__)


class FrameworkMetadata:
    """Framework metadata and version information."""
    
    NAME = "TTS/STT Testing Framework"
    VERSION = "1.0.0"
    DESCRIPTION = "Comprehensive evaluation framework for Text-to-Speech and Speech-to-Text models"
    AUTHOR = "TTS/STT Testing Framework Team"
    
    @classmethod
    def display_banner(cls):
        """Display framework banner."""
        banner_text = f"""
        {cls.NAME}
        Version: {cls.VERSION}
        {cls.DESCRIPTION}
        
        Author: {cls.AUTHOR}
        """
        
        console.print(
            Panel(
                banner_text,
                title="🎤 TTS/STT Framework 🔊",
                style="bold blue",
                expand=False
            )
        )


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--log-level', default='INFO', help='Logging level')
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.option('--quiet', is_flag=True, help='Suppress output')
@click.pass_context
def cli(ctx, config, log_level, debug, quiet):
    """TTS/STT Testing Framework - Comprehensive model evaluation tool."""
    
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Setup logging
    if debug:
        log_level = 'DEBUG'
    elif quiet:
        log_level = 'ERROR'
    
    setup_logging(level=log_level, enable_console=not quiet)
    
    # Load configuration
    if config:
        ctx.obj['config'] = BaseConfig.from_file(config)
    else:
        ctx.obj['config'] = BaseConfig()
    
    ctx.obj['debug'] = debug
    ctx.obj['quiet'] = quiet
    
    if not quiet:
        FrameworkMetadata.display_banner()


@cli.command()
@click.option('--mode', type=click.Choice(['all', 'tts', 'stt']), default='all', 
              help='Evaluation mode')
@click.option('--providers', help='Comma-separated list of providers to test')
@click.option('--models', help='Comma-separated list of models to test')
@click.option('--parallel/--sequential', default=True, help='Execution mode')
@click.option('--max-workers', type=int, default=4, help='Maximum worker threads')
@click.option('--timeout', type=int, default=300, help='Test timeout in seconds')
@click.option('--output-dir', type=click.Path(), help='Output directory for results')
@click.option('--report-format', multiple=True, 
              type=click.Choice(['html', 'json', 'yaml']), 
              default=['html', 'json', 'yaml'], help='Report formats')
@click.pass_context
async def evaluate(ctx, mode, providers, models, parallel, max_workers, 
                  timeout, output_dir, report_format):
    """Run comprehensive TTS/STT model evaluation."""
    
    config = ctx.obj['config']
    debug = ctx.obj['debug']
    quiet = ctx.obj['quiet']
    
    try:
        if not quiet:
            console.print("\n[bold green]Starting TTS/STT Evaluation[/bold green]")
        
        # Create test executor
        executor = TestExecutor(config)
        
        # Configure execution parameters
        executor.execution_config.enable_parallel = parallel
        executor.execution_config.max_workers = max_workers
        executor.execution_config.timeout_seconds = timeout
        
        # Override output directory if specified
        if output_dir:
            config.results_dir = output_dir
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Filter providers and models if specified
        if providers:
            provider_list = [p.strip() for p in providers.split(',')]
            config.enabled_providers = provider_list
        
        if models:
            model_list = [m.strip() for m in models.split(',')]
            config.enabled_models = model_list
        
        # Create test suite based on mode
        if mode == 'all':
            test_suite = executor.create_standard_test_suite()
        elif mode == 'tts':
            test_suite = create_tts_test_suite(config)
        elif mode == 'stt':
            test_suite = create_stt_test_suite(config)
        
        executor.add_test_suite(test_suite)
        
        # Execute tests with progress indicator
        if not quiet:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Running evaluations...", total=None)
                results = await executor.execute_all_suites()
                progress.update(task, completed=True)
        else:
            results = await executor.execute_all_suites()
        
        # Display results summary
        if not quiet:
            display_results_summary(results)
        
        # Generate reports in specified formats
        await generate_reports(results, config, report_format, quiet)
        
        logger.info(
            "Evaluation completed successfully",
            total_tests=results['metadata']['total_tests'],
            success_rate=results['metadata']['success_rate']
        )
    
    except Exception as e:
        logger.error(
            "Evaluation failed",
            error=str(e),
            traceback=traceback.format_exc()
        )
        
        if not quiet:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
        
        sys.exit(1)


@cli.command()
@click.option('--provider', required=True, help='Provider to test')
@click.option('--model-type', type=click.Choice(['tts', 'stt']), required=True,
              help='Model type to test')
@click.option('--model-name', help='Specific model name')
@click.option('--test-input', required=True, help='Test input (text for TTS, audio file for STT)')
@click.pass_context
async def test_single(ctx, provider, model_type, model_name, test_input):
    """Test a single model with specific input."""
    
    config = ctx.obj['config']
    quiet = ctx.obj['quiet']
    
    try:
        if not quiet:
            console.print(f"\n[bold blue]Testing {provider} {model_type.upper()} model[/bold blue]")
        
        # Create test executor
        executor = TestExecutor(config)
        
        # Create single test suite
        suite = TestSuite(
            name=f"Single {model_type.upper()} Test",
            description=f"Testing {provider} {model_type} model with specific input"
        )
        
        # Add single test
        if model_type == 'tts':
            test_data = {'text': test_input}
        else:  # stt
            test_data = {
                'audio_file': test_input,
                'reference_text': 'Reference text for comparison'
            }
        
        suite.add_test(provider, model_type, model_name or 'default', test_data)
        executor.add_test_suite(suite)
        
        # Execute test
        results = await executor.execute_all_suites()
        
        # Display results
        if not quiet:
            display_single_test_results(results)
        
        logger.info("Single test completed", provider=provider, model_type=model_type)
    
    except Exception as e:
        logger.error(
            "Single test failed",
            error=str(e),
            provider=provider,
            model_type=model_type
        )
        
        if not quiet:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
        
        sys.exit(1)


@cli.command()
@click.option('--input-file', required=True, type=click.Path(exists=True),
              help='Input results file (JSON or YAML)')
@click.option('--output-dir', type=click.Path(), help='Output directory')
@click.option('--format', 'formats', multiple=True,
              type=click.Choice(['html', 'json', 'yaml']),
              default=['html'], help='Report formats to generate')
@click.pass_context
async def generate_report(ctx, input_file, output_dir, formats):
    """Generate reports from existing results file."""
    
    config = ctx.obj['config']
    quiet = ctx.obj['quiet']
    
    try:
        if not quiet:
            console.print("\n[bold blue]Generating reports from existing results[/bold blue]")
        
        # Load results from file
        results = load_results_file(input_file)
        
        # Override output directory if specified
        if output_dir:
            config.results_dir = output_dir
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate reports
        await generate_reports(results, config, formats, quiet)
        
        logger.info("Report generation completed", input_file=input_file)
    
    except Exception as e:
        logger.error(
            "Report generation failed",
            error=str(e),
            input_file=input_file
        )
        
        if not quiet:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
        
        sys.exit(1)


@cli.command()
@click.pass_context
def list_providers(ctx):
    """List available providers and models."""
    
    config = ctx.obj['config']
    quiet = ctx.obj['quiet']
    
    try:
        from clients.client_factory import ClientFactory
        
        factory = ClientFactory(config)
        available_providers = factory.get_available_providers()
        
        if not quiet:
            table = Table(title="Available Providers and Models")
            table.add_column("Provider", style="cyan")
            table.add_column("TTS Models", style="green")
            table.add_column("STT Models", style="yellow")
            table.add_column("Status", style="magenta")
            
            for provider in available_providers:
                try:
                    client = factory.create_client(provider)
                    tts_models = client.get_available_models('tts')
                    stt_models = client.get_available_models('stt')
                    status = "✅ Available"
                except Exception as e:
                    tts_models = []
                    stt_models = []
                    status = f"❌ Error: {str(e)[:30]}..."
                
                table.add_row(
                    provider,
                    ", ".join(tts_models) if tts_models else "None",
                    ", ".join(stt_models) if stt_models else "None",
                    status
                )
            
            console.print(table)
        else:
            # JSON output for programmatic use
            provider_info = {}
            for provider in available_providers:
                try:
                    client = factory.create_client(provider)
                    provider_info[provider] = {
                        'tts_models': client.get_available_models('tts'),
                        'stt_models': client.get_available_models('stt'),
                        'available': True
                    }
                except Exception as e:
                    provider_info[provider] = {
                        'tts_models': [],
                        'stt_models': [],
                        'available': False,
                        'error': str(e)
                    }
            
            print(json.dumps(provider_info, indent=2))
    
    except Exception as e:
        logger.error("Failed to list providers", error=str(e))
        sys.exit(1)


@cli.command()
@click.option('--check-all', is_flag=True, help='Check all components')
@click.option('--check-providers', is_flag=True, help='Check provider connectivity')
@click.option('--check-data', is_flag=True, help='Check test data availability')
@click.option('--check-config', is_flag=True, help='Check configuration validity')
@click.pass_context
async def health_check(ctx, check_all, check_providers, check_data, check_config):
    """Perform system health checks."""
    
    config = ctx.obj['config']
    quiet = ctx.obj['quiet']
    
    checks_to_run = []
    
    if check_all:
        checks_to_run = ['providers', 'data', 'config']
    else:
        if check_providers:
            checks_to_run.append('providers')
        if check_data:
            checks_to_run.append('data')
        if check_config:
            checks_to_run.append('config')
    
    if not checks_to_run:
        checks_to_run = ['providers', 'data', 'config']  # Default to all
    
    try:
        if not quiet:
            console.print("\n[bold blue]Running Health Checks[/bold blue]")
        
        health_results = {}
        
        # Configuration check
        if 'config' in checks_to_run:
            health_results['config'] = check_configuration_health(config)
        
        # Provider connectivity check
        if 'providers' in checks_to_run:
            health_results['providers'] = await check_provider_health(config)
        
        # Test data availability check
        if 'data' in checks_to_run:
            health_results['data'] = check_data_availability(config)
        
        # Display results
        if not quiet:
            display_health_check_results(health_results)
        else:
            print(json.dumps(health_results, indent=2))
        
        # Determine overall health
        overall_health = all(
            result.get('status') == 'healthy'
            for result in health_results.values()
        )
        
        if not overall_health:
            sys.exit(1)
    
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        sys.exit(1)


# Helper functions

def create_tts_test_suite(config: BaseConfig) -> TestSuite:
    """Create a TTS-specific test suite."""
    suite = TestSuite(
        name="TTS Evaluation Suite",
        description="Comprehensive Text-to-Speech model evaluation"
    )
    
    # Sample texts for TTS evaluation
    test_texts = [
        "Hello, this is a test of text-to-speech conversion.",
        "The quick brown fox jumps over the lazy dog.",
        "Testing pronunciation of numbers: 1, 2, 3, 4, 5.",
        "How are you doing today? I hope everything is going well!",
        "This sentence contains various punctuation marks: commas, periods, and exclamation points!"
    ]
    
    enabled_providers = get_enabled_tts_providers()
    
    for provider in enabled_providers:
        from clients.client_factory import ClientFactory
        
        try:
            factory = ClientFactory(config)
            client = factory.create_client(provider)
            models = client.get_available_models('tts')
            
            for model in models:
                for i, text in enumerate(test_texts):
                    test_data = {
                        'text': text,
                        'test_index': i
                    }
                    suite.add_test(provider, 'tts', model, test_data)
        
        except Exception as e:
            logger.warning(
                "Failed to add TTS tests for provider",
                provider=provider,
                error=str(e)
            )
    
    return suite


def create_stt_test_suite(config: BaseConfig) -> TestSuite:
    """Create a STT-specific test suite."""
    suite = TestSuite(
        name="STT Evaluation Suite",
        description="Comprehensive Speech-to-Text model evaluation"
    )
    
    # Get test audio files
    test_audio_files = get_test_audio_files(config)
    enabled_providers = get_enabled_stt_providers()
    
    for provider in enabled_providers:
        from clients.client_factory import ClientFactory
        
        try:
            factory = ClientFactory(config)
            client = factory.create_client(provider)
            models = client.get_available_models('stt')
            
            for model in models:
                for audio_file in test_audio_files:
                    test_data = {
                        'audio_file': audio_file,
                        'reference_text': get_reference_text_for_audio(audio_file)
                    }
                    suite.add_test(provider, 'stt', model, test_data)
        
        except Exception as e:
            logger.warning(
                "Failed to add STT tests for provider",
                provider=provider,
                error=str(e)
            )
    
    return suite


def get_enabled_tts_providers() -> List[str]:
    """Get list of enabled TTS providers from environment."""
    providers = []
    env_vars = [
        'ENABLE_OPENAI_TTS',
        'ENABLE_AZURE_TTS',
        'ENABLE_GOOGLE_TTS',
        'ENABLE_SARVAM_TTS',
        'ENABLE_CHATTERBOX_TTS',
        'ENABLE_ELEVENLABS_TTS'  # Added ElevenLabs TTS support
    ]
    
    for env_var in env_vars:
        if os.getenv(env_var, 'false').lower() == 'true':
            provider = env_var.replace('ENABLE_', '').replace('_TTS', '').lower()
            providers.append(provider)
    
    return providers


def get_enabled_stt_providers() -> List[str]:
    """Get list of enabled STT providers from environment."""
    providers = []
    env_vars = [
        'ENABLE_OPENAI_STT',
        'ENABLE_AZURE_STT',
        'ENABLE_GOOGLE_STT',
        'ENABLE_SARVAM_STT',
        'ENABLE_CHATTERBOX_STT',
        'ENABLE_ELEVENLABS_STT'  # Added ElevenLabs STT support
    ]
    
    for env_var in env_vars:
        if os.getenv(env_var, 'false').lower() == 'true':
            provider = env_var.replace('ENABLE_', '').replace('_STT', '').lower()
            providers.append(provider)
    
    return providers


def get_test_audio_files(config: BaseConfig) -> List[str]:
    """Get list of available test audio files."""
    test_data_dir = Path(config.test_data_dir)
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
    
    audio_files = []
    
    if test_data_dir.exists():
        for ext in audio_extensions:
            audio_files.extend(test_data_dir.glob(f"*{ext}"))
    
    return [str(f) for f in audio_files[:5]]  # Limit to first 5 files


def get_reference_text_for_audio(audio_file: str) -> str:
    """Get reference text for an audio file."""
    audio_path = Path(audio_file)
    txt_file = audio_path.with_suffix('.txt')
    
    if txt_file.exists():
        return txt_file.read_text(encoding='utf-8').strip()
    
    return "Reference text not available"


def load_results_file(file_path: str) -> Dict[str, Any]:
    """Load results from JSON or YAML file."""
    file_path = Path(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.suffix.lower() == '.json':
            return json.load(f)
        elif file_path.suffix.lower() in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")


async def generate_reports(results: Dict[str, Any], config: BaseConfig,
                          formats: List[str], quiet: bool = False) -> None:
    """Generate reports in specified formats."""
    report_generator = ReportGenerator(config)
    
    generated_reports = []
    
    for format_type in formats:
        try:
            if format_type == 'html':
                report_file = await report_generator.generate_html_report(results)
            elif format_type == 'json':
                report_file = await report_generator.generate_json_report(results)
            elif format_type == 'yaml':
                report_file = await report_generator.generate_yaml_report(results)
            else:
                continue
            
            generated_reports.append((format_type.upper(), report_file))
        
        except Exception as e:
            logger.error(
                "Failed to generate report",
                format=format_type,
                error=str(e)
            )
    
    if not quiet and generated_reports:
        console.print("\n[bold green]Reports Generated:[/bold green]")
        for format_name, report_file in generated_reports:
            console.print(f"  {format_name}: {report_file}")


def display_results_summary(results: Dict[str, Any]) -> None:
    """Display evaluation results summary."""
    metadata = results['metadata']
    
    # Main summary table
    summary_table = Table(title="Evaluation Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    
    summary_table.add_row("Total Tests", str(metadata['total_tests']))
    summary_table.add_row("Successful Tests", str(metadata['successful_tests']))
    summary_table.add_row("Failed Tests", str(metadata['failed_tests']))
    summary_table.add_row("Success Rate", f"{metadata['success_rate']:.2%}")
    summary_table.add_row("Execution Duration", metadata['execution_duration'])
    
    console.print(summary_table)
    
    # Provider summary
    if 'provider_summary' in results:
        provider_table = Table(title="Provider Performance")
        provider_table.add_column("Provider", style="cyan")
        provider_table.add_column("Tests", style="yellow")
        provider_table.add_column("Success Rate", style="green")
        provider_table.add_column("Avg Time (s)", style="magenta")
        
        for provider, stats in results['provider_summary'].items():
            provider_table.add_row(
                provider,
                str(stats['total_tests']),
                f"{stats['success_rate']:.2%}",
                f"{stats['avg_execution_time']:.2f}"
            )
        
        console.print(provider_table)


def display_single_test_results(results: Dict[str, Any]) -> None:
    """Display results for a single test."""
    detailed_results = results.get('detailed_results', [])
    
    if detailed_results:
        result = detailed_results[0]
        
        table = Table(title="Single Test Results")
        table.add_column("Attribute", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Test ID", result['test_id'])
        table.add_row("Provider", result['provider'])
        table.add_row("Model Type", result['model_type'])
        table.add_row("Model Name", result['model_name'])
        table.add_row("Success", "✅ Yes" if result['success'] else "❌ No")
        table.add_row("Execution Time", f"{result['execution_time']:.2f}s")
        
        if result['metrics']:
            table.add_row("Metrics", json.dumps(result['metrics'], indent=2))
        
        if result['errors']:
            table.add_row("Errors", "\n".join(result['errors']))
        
        console.print(table)


def check_configuration_health(config: BaseConfig) -> Dict[str, Any]:
    """Check configuration health."""
    issues = []
    
    # Check required directories
    required_dirs = ['test_data_dir', 'results_dir', 'output_data_dir']
    for dir_attr in required_dirs:
        dir_path = getattr(config, dir_attr, None)
        if dir_path:
            path = Path(dir_path)
            if not path.exists():
                issues.append(f"Directory does not exist: {dir_path}")
    
    # Check API keys
    required_env_vars = [
        'OPENAI_API_KEY',
        'AZURE_SPEECH_KEY',
        'GOOGLE_APPLICATION_CREDENTIALS'
    ]
    
    for env_var in required_env_vars:
        if not os.getenv(env_var):
            issues.append(f"Missing environment variable: {env_var}")
    
    return {
        'status': 'healthy' if not issues else 'unhealthy',
        'issues': issues,
        'checked_items': len(required_dirs) + len(required_env_vars)
    }


async def check_provider_health(config: BaseConfig) -> Dict[str, Any]:
    """Check provider connectivity health."""
    from clients.client_factory import ClientFactory
    
    factory = ClientFactory(config)
    provider_status = {}
    
    enabled_providers = get_enabled_tts_providers() + get_enabled_stt_providers()
    enabled_providers = list(set(enabled_providers))  # Remove duplicates
    
    for provider in enabled_providers:
        try:
            client = factory.create_client(provider)
            health_check = await client.health_check()
            provider_status[provider] = {
                'status': 'healthy' if health_check.get('healthy', False) else 'unhealthy',
                'response_time': health_check.get('response_time', 0),
                'error': health_check.get('error')
            }
        except Exception as e:
            provider_status[provider] = {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    overall_healthy = all(
        status['status'] == 'healthy'
        for status in provider_status.values()
    )
    
    return {
        'status': 'healthy' if overall_healthy else 'unhealthy',
        'providers': provider_status
    }


def check_data_availability(config: BaseConfig) -> Dict[str, Any]:
    """Check test data availability."""
    issues = []
    
    # Check test data directory
    test_data_dir = Path(config.test_data_dir)
    if not test_data_dir.exists():
        issues.append(f"Test data directory does not exist: {test_data_dir}")
    else:
        # Check for audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(test_data_dir.glob(f"*{ext}"))
        
        if not audio_files:
            issues.append("No audio files found in test data directory")
    
    # Check reference data directory
    reference_dir = Path(config.reference_data_dir)
    if not reference_dir.exists():
        issues.append(f"Reference data directory does not exist: {reference_dir}")
    
    return {
        'status': 'healthy' if not issues else 'unhealthy',
        'issues': issues,
        'test_data_dir': str(test_data_dir),
        'reference_data_dir': str(reference_dir) if hasattr(config, 'reference_data_dir') else None
    }


def display_health_check_results(health_results: Dict[str, Any]) -> None:
    """Display health check results."""
    table = Table(title="System Health Check")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="yellow")
    
    for component, result in health_results.items():
        status = result['status']
        status_icon = "✅" if status == 'healthy' else "❌"
        
        details = []
        if 'issues' in result and result['issues']:
            details.extend(result['issues'])
        
        if 'providers' in result:
            for provider, provider_status in result['providers'].items():
                if provider_status['status'] == 'unhealthy':
                    details.append(f"{provider}: {provider_status.get('error', 'Unknown error')}")
        
        table.add_row(
            component.title(),
            f"{status_icon} {status.title()}",
            "\n".join(details) if details else "All checks passed"
        )
    
    console.print(table)


# Synchronous wrapper for async CLI commands
def async_command(f):
    """Decorator to run async commands in sync context."""
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper


# Apply async wrapper to async commands
evaluate.callback = async_command(evaluate.callback)
test_single.callback = async_command(test_single.callback)
generate_report.callback = async_command(generate_report.callback)
health_check.callback = async_command(health_check.callback)


if __name__ == "__main__":
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        logger.error("Unexpected error", error=str(e))
        console.print(f"[bold red]Unexpected error:[/bold red] {str(e)}")
        sys.exit(1)