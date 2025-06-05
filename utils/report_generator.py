"""
TTS/STT Testing Framework - Report Generator
===========================================

This module provides report generation functionality for the TTS/STT testing framework.

Author: TTS/STT Testing Framework Team
Version: 1.0.0
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from jinja2 import Template

from .logger import get_logger
from .file_utils import FileManager


class ReportGenerator:
    """
    Report generator for TTS/STT evaluation results.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the report generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.file_manager = FileManager()
        
        # Ensure results directory exists
        results_dir = getattr(config, 'results_dir', 'results')
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"ReportGenerator initialized with results dir: {self.results_dir}")
    
    async def generate_json_report(self, results: Dict[str, Any]) -> str:
        """
        Generate JSON report from results.
        
        Args:
            results: Evaluation results
            
        Returns:
            str: Path to generated report
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.results_dir / f"evaluation_report_{timestamp}.json"
            
            # Prepare report data
            report_data = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'framework_version': '1.0.0',
                    'report_type': 'json'
                },
                'evaluation_results': results
            }
            
            # Save JSON report
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"JSON report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate JSON report: {e}")
            raise
    
    async def generate_yaml_report(self, results: Dict[str, Any]) -> str:
        """
        Generate YAML report from results.
        
        Args:
            results: Evaluation results
            
        Returns:
            str: Path to generated report
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.results_dir / f"evaluation_report_{timestamp}.yaml"
            
            # Prepare report data
            report_data = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'framework_version': '1.0.0',
                    'report_type': 'yaml'
                },
                'evaluation_results': results
            }
            
            # Save YAML report
            with open(report_path, 'w', encoding='utf-8') as f:
                yaml.dump(report_data, f, default_flow_style=False, allow_unicode=True)
            
            self.logger.info(f"YAML report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate YAML report: {e}")
            raise
    
    async def generate_html_report(self, results: Dict[str, Any]) -> str:
        """
        Generate HTML report from results.
        
        Args:
            results: Evaluation results
            
        Returns:
            str: Path to generated report
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.results_dir / f"evaluation_report_{timestamp}.html"
            
            # Simple HTML template
            html_template = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>TTS/STT Evaluation Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .header { background-color: #f0f0f0; padding: 20px; }
                    .section { margin: 20px 0; }
                    .metric { margin: 10px 0; }
                    table { border-collapse: collapse; width: 100%; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>TTS/STT Evaluation Report</h1>
                    <p>Generated: {{ timestamp }}</p>
                </div>
                
                <div class="section">
                    <h2>Summary</h2>
                    <div class="metric">Total Tests: {{ summary.total_tests }}</div>
                    <div class="metric">Successful: {{ summary.successful_tests }}</div>
                    <div class="metric">Failed: {{ summary.failed_tests }}</div>
                    <div class="metric">Success Rate: {{ summary.success_rate }}%</div>
                </div>
                
                <div class="section">
                    <h2>Detailed Results</h2>
                    <p>See JSON/YAML reports for detailed results.</p>
                </div>
            </body>
            </html>
            """
            
            # Prepare template data
            metadata = results.get('metadata', {})
            template_data = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'summary': {
                    'total_tests': metadata.get('total_tests', 0),
                    'successful_tests': metadata.get('successful_tests', 0),
                    'failed_tests': metadata.get('failed_tests', 0),
                    'success_rate': round(metadata.get('success_rate', 0) * 100, 2)
                }
            }
            
            # Render template
            template = Template(html_template)
            html_content = template.render(**template_data)
            
            # Save HTML report
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"HTML report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate HTML report: {e}")
            raise