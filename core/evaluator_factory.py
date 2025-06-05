"""
Evaluator Factory
================

Factory pattern implementation for creating and managing TTS and STT evaluators.
Provides a centralized interface for evaluator creation with configuration validation
and dependency management.

Author: AI Testing Team
Version: 1.0.0
"""

import logging
from typing import Dict, Any, Optional, Union, List, Type
from pathlib import Path
import importlib.util
from abc import ABC, abstractmethod

from .tts_evaluator import TTSEvaluator, TTSTestCase
from .stt_evaluator import STTEvaluator, STTTestCase

# Configure logging
logger = logging.getLogger(__name__)

class EvaluatorInterface(ABC):
    """Abstract base class for evaluators."""
    
    @abstractmethod
    async def evaluate_model(self, client, test_cases: List, model_config: Dict[str, Any]) -> List:
        """Evaluate a model with test cases."""
        pass
    
    @abstractmethod
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get evaluation summary."""
        pass
    
    @abstractmethod
    def export_results(self, format_type: str = "json") -> str:
        """Export results in specified format."""
        pass
    
    @abstractmethod
    def clear_results(self):
        """Clear evaluation results."""
        pass

class EvaluatorValidationError(Exception):
    """Custom exception for evaluator validation errors."""
    pass

class EvaluatorFactory:
    """
    Factory class for creating and managing TTS and STT evaluators.
    
    This factory provides:
    - Centralized evaluator creation
    - Configuration validation
    - Dependency checking
    - Plugin support for custom evaluators
    - Resource management
    """
    
    # Supported evaluator types
    SUPPORTED_TYPES = {
        'tts': TTSEvaluator,
        'stt': STTEvaluator
    }
    
    def __init__(self, base_config: Dict[str, Any]):
        """
        Initialize the evaluator factory.
        
        Args:
            base_config (Dict[str, Any]): Base configuration for all evaluators
        """
        self.base_config = base_config
        self.created_evaluators: Dict[str, EvaluatorInterface] = {}
        self.custom_evaluators: Dict[str, Type[EvaluatorInterface]] = {}
        
        # Validate base configuration
        self._validate_base_config()
        
        logger.info("Evaluator factory initialized successfully")
        logger.debug(f"Supported evaluator types: {list(self.SUPPORTED_TYPES.keys())}")

    def _validate_base_config(self):
        """Validate the base configuration."""
        try:
            required_sections = ['testing', 'output', 'environment']
            
            for section in required_sections:
                if section not in self.base_config:
                    raise EvaluatorValidationError(f"Missing required configuration section: {section}")
            
            # Validate testing configuration
            testing_config = self.base_config.get('testing', {})
            
            if 'tts' not in testing_config and 'stt' not in testing_config:
                raise EvaluatorValidationError("No TTS or STT testing configuration found")
            
            # Validate output configuration
            output_config = self.base_config.get('output', {})
            if 'formats' not in output_config:
                logger.warning("No output formats specified in configuration")
            
            logger.debug("Base configuration validation successful")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise EvaluatorValidationError(f"Configuration validation failed: {e}")

    def create_evaluator(
        self, 
        evaluator_type: str, 
        evaluator_id: Optional[str] = None,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> EvaluatorInterface:
        """
        Create an evaluator instance.
        
        Args:
            evaluator_type (str): Type of evaluator ('tts' or 'stt')
            evaluator_id (Optional[str]): Unique identifier for the evaluator
            custom_config (Optional[Dict[str, Any]]): Custom configuration overrides
            
        Returns:
            EvaluatorInterface: Created evaluator instance
            
        Raises:
            EvaluatorValidationError: If evaluator type is not supported or configuration is invalid
        """
        try:
            # Validate evaluator type
            if evaluator_type not in self.SUPPORTED_TYPES and evaluator_type not in self.custom_evaluators:
                raise EvaluatorValidationError(
                    f"Unsupported evaluator type: {evaluator_type}. "
                    f"Supported types: {list(self.SUPPORTED_TYPES.keys()) + list(self.custom_evaluators.keys())}"
                )
            
            # Generate evaluator ID if not provided
            if evaluator_id is None:
                evaluator_id = f"{evaluator_type}_evaluator_{len(self.created_evaluators) + 1}"
            
            # Check if evaluator already exists
            if evaluator_id in self.created_evaluators:
                logger.warning(f"Evaluator {evaluator_id} already exists, returning existing instance")
                return self.created_evaluators[evaluator_id]
            
            # Merge configurations
            merged_config = self._merge_configurations(evaluator_type, custom_config)
            
            # Validate merged configuration
            self._validate_evaluator_config(evaluator_type, merged_config)
            
            # Create evaluator instance
            if evaluator_type in self.SUPPORTED_TYPES:
                evaluator_class = self.SUPPORTED_TYPES[evaluator_type]
            else:
                evaluator_class = self.custom_evaluators[evaluator_type]
            
            evaluator = evaluator_class(merged_config)
            
            # Store evaluator reference
            self.created_evaluators[evaluator_id] = evaluator
            
            logger.info(f"Created {evaluator_type} evaluator with ID: {evaluator_id}")
            return evaluator
            
        except Exception as e:
            logger.error(f"Failed to create {evaluator_type} evaluator: {e}")
            raise EvaluatorValidationError(f"Evaluator creation failed: {e}")

    def _merge_configurations(
        self, 
        evaluator_type: str, 
        custom_config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Merge base configuration with custom configuration.
        
        Args:
            evaluator_type (str): Type of evaluator
            custom_config (Optional[Dict[str, Any]]): Custom configuration
            
        Returns:
            Dict[str, Any]: Merged configuration
        """
        try:
            # Start with base configuration
            merged_config = self.base_config.copy()
            
            # Apply custom configuration if provided
            if custom_config:
                merged_config = self._deep_merge_dict(merged_config, custom_config)
                logger.debug(f"Applied custom configuration for {evaluator_type} evaluator")
            
            return merged_config
            
        except Exception as e:
            logger.error(f"Configuration merge failed: {e}")
            raise EvaluatorValidationError(f"Configuration merge failed: {e}")

    def _deep_merge_dict(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_dict(result[key], value)
            else:
                result[key] = value
        
        return result

    def _validate_evaluator_config(self, evaluator_type: str, config: Dict[str, Any]):
        """
        Validate evaluator-specific configuration.
        
        Args:
            evaluator_type (str): Type of evaluator
            config (Dict[str, Any]): Configuration to validate
            
        Raises:
            EvaluatorValidationError: If configuration is invalid
        """
        try:
            testing_config = config.get('testing', {})
            
            if evaluator_type == 'tts':
                tts_config = testing_config.get('tts', {})
                
                # Check required TTS configuration
                if 'metrics' not in tts_config:
                    logger.warning("No TTS metrics specified in configuration")
                
                # Validate audio configuration
                audio_config = config.get('audio', {})
                if not audio_config:
                    logger.warning("No audio configuration found")
                
            elif evaluator_type == 'stt':
                stt_config = testing_config.get('stt', {})
                
                # Check required STT configuration
                if 'metrics' not in stt_config:
                    logger.warning("No STT metrics specified in configuration")
                
                # Validate accuracy thresholds
                if 'accuracy_thresholds' not in stt_config:
                    logger.warning("No accuracy thresholds specified")
            
            logger.debug(f"Configuration validation successful for {evaluator_type} evaluator")
            
        except Exception as e:
            logger.error(f"Evaluator configuration validation failed: {e}")
            raise EvaluatorValidationError(f"Evaluator configuration validation failed: {e}")

    def get_evaluator(self, evaluator_id: str) -> Optional[EvaluatorInterface]:
        """
        Get an existing evaluator by ID.
        
        Args:
            evaluator_id (str): Evaluator identifier
            
        Returns:
            Optional[EvaluatorInterface]: Evaluator instance or None if not found
        """
        return self.created_evaluators.get(evaluator_id)

    def list_evaluators(self) -> Dict[str, str]:
        """
        List all created evaluators.
        
        Returns:
            Dict[str, str]: Dictionary mapping evaluator IDs to their types
        """
        evaluator_info = {}
        
        for evaluator_id, evaluator in self.created_evaluators.items():
            evaluator_type = type(evaluator).__name__.lower().replace('evaluator', '')
            evaluator_info[evaluator_id] = evaluator_type
        
        return evaluator_info

    def remove_evaluator(self, evaluator_id: str) -> bool:
        """
        Remove an evaluator instance.
        
        Args:
            evaluator_id (str): Evaluator identifier
            
        Returns:
            bool: True if removed successfully, False if not found
        """
        if evaluator_id in self.created_evaluators:
            # Clear evaluator results before removal
            self.created_evaluators[evaluator_id].clear_results()
            del self.created_evaluators[evaluator_id]
            
            logger.info(f"Removed evaluator: {evaluator_id}")
            return True
        
        logger.warning(f"Evaluator not found for removal: {evaluator_id}")
        return False

    def clear_all_evaluators(self):
        """Clear all created evaluators."""
        for evaluator_id in list(self.created_evaluators.keys()):
            self.remove_evaluator(evaluator_id)
        
        logger.info("Cleared all evaluators")

    def register_custom_evaluator(
        self, 
        evaluator_type: str, 
        evaluator_class: Type[EvaluatorInterface]
    ):
        """
        Register a custom evaluator class.
        
        Args:
            evaluator_type (str): Custom evaluator type identifier
            evaluator_class (Type[EvaluatorInterface]): Evaluator class implementing EvaluatorInterface
            
        Raises:
            EvaluatorValidationError: If evaluator class is invalid
        """
        try:
            # Validate that class implements EvaluatorInterface
            if not issubclass(evaluator_class, EvaluatorInterface):
                raise EvaluatorValidationError(
                    f"Custom evaluator class must implement EvaluatorInterface"
                )
            
            # Check for name conflicts
            if evaluator_type in self.SUPPORTED_TYPES:
                raise EvaluatorValidationError(
                    f"Evaluator type '{evaluator_type}' conflicts with built-in type"
                )
            
            self.custom_evaluators[evaluator_type] = evaluator_class
            logger.info(f"Registered custom evaluator: {evaluator_type}")
            
        except Exception as e:
            logger.error(f"Failed to register custom evaluator: {e}")
            raise EvaluatorValidationError(f"Custom evaluator registration failed: {e}")

    def load_evaluator_plugin(self, plugin_path: str, evaluator_type: str):
        """
        Load an evaluator from a plugin file.
        
        Args:
            plugin_path (str): Path to the plugin Python file
            evaluator_type (str): Type identifier for the custom evaluator
            
        Raises:
            EvaluatorValidationError: If plugin loading fails
        """
        try:
            plugin_path = Path(plugin_path)
            
            if not plugin_path.exists():
                raise EvaluatorValidationError(f"Plugin file not found: {plugin_path}")
            
            # Load module from file
            spec = importlib.util.spec_from_file_location("evaluator_plugin", plugin_path)
            if spec is None or spec.loader is None:
                raise EvaluatorValidationError(f"Could not load plugin spec: {plugin_path}")
            
            plugin_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(plugin_module)
            
            # Look for evaluator class in module
            evaluator_class = None
            for attr_name in dir(plugin_module):
                attr = getattr(plugin_module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, EvaluatorInterface) and 
                    attr != EvaluatorInterface):
                    evaluator_class = attr
                    break
            
            if evaluator_class is None:
                raise EvaluatorValidationError(
                    f"No evaluator class found in plugin: {plugin_path}"
                )
            
            # Register the custom evaluator
            self.register_custom_evaluator(evaluator_type, evaluator_class)
            
            logger.info(f"Loaded evaluator plugin: {plugin_path} as {evaluator_type}")
            
        except Exception as e:
            logger.error(f"Failed to load evaluator plugin: {e}")
            raise EvaluatorValidationError(f"Plugin loading failed: {e}")

    def create_test_cases(
        self, 
        evaluator_type: str, 
        config_override: Optional[Dict[str, Any]] = None
    ) -> List[Union[TTSTestCase, STTTestCase]]:
        """
        Create test cases for a specific evaluator type.
        
        Args:
            evaluator_type (str): Type of evaluator ('tts' or 'stt')
            config_override (Optional[Dict[str, Any]]): Configuration overrides
            
        Returns:
            List[Union[TTSTestCase, STTTestCase]]: Generated test cases
            
        Raises:
            EvaluatorValidationError: If evaluator type is not supported
        """
        try:
            if evaluator_type not in ['tts', 'stt']:
                raise EvaluatorValidationError(f"Unsupported evaluator type for test case creation: {evaluator_type}")
            
            # Merge configuration
            config = self._merge_configurations(evaluator_type, config_override)
            
            if evaluator_type == 'tts':
                from .tts_evaluator import create_test_cases_from_config
                test_cases = create_test_cases_from_config(config)
            else:  # stt
                from .stt_evaluator import create_test_cases_from_config
                test_cases = create_test_cases_from_config(config)
            
            logger.info(f"Created {len(test_cases)} test cases for {evaluator_type} evaluator")
            return test_cases
            
        except Exception as e:
            logger.error(f"Failed to create test cases for {evaluator_type}: {e}")
            raise EvaluatorValidationError(f"Test case creation failed: {e}")

    def get_factory_info(self) -> Dict[str, Any]:
        """
        Get information about the factory and its capabilities.
        
        Returns:
            Dict[str, Any]: Factory information
        """
        return {
            "supported_evaluator_types": list(self.SUPPORTED_TYPES.keys()),
            "custom_evaluator_types": list(self.custom_evaluators.keys()),
            "created_evaluators_count": len(self.created_evaluators),
            "created_evaluators": self.list_evaluators(),
            "factory_version": "1.0.0"
        }

    def validate_dependencies(self) -> Dict[str, bool]:
        """
        Validate that all required dependencies are available.
        
        Returns:
            Dict[str, bool]: Dependency validation results
        """
        dependencies = {
            'librosa': False,
            'soundfile': False,
            'jiwer': False,
            'editdistance': False,
            'numpy': False,
            'scipy': False
        }
        
        for dependency in dependencies:
            try:
                importlib.import_module(dependency)
                dependencies[dependency] = True
            except ImportError:
                logger.warning(f"Missing dependency: {dependency}")
        
        all_available = all(dependencies.values())
        if all_available:
            logger.info("All dependencies are available")
        else:
            missing = [dep for dep, available in dependencies.items() if not available]
            logger.warning(f"Missing dependencies: {missing}")
        
        return dependencies

    def __del__(self):
        """Cleanup when factory is destroyed."""
        try:
            self.clear_all_evaluators()
        except Exception as e:
            logger.error(f"Error during factory cleanup: {e}")

# Factory singleton instance management
_factory_instance: Optional[EvaluatorFactory] = None

def get_factory(config: Optional[Dict[str, Any]] = None) -> EvaluatorFactory:
    """
    Get or create the global factory instance.
    
    Args:
        config (Optional[Dict[str, Any]]): Configuration for factory initialization
        
    Returns:
        EvaluatorFactory: Factory instance
    """
    global _factory_instance
    
    if _factory_instance is None:
        if config is None:
            raise EvaluatorValidationError("Configuration required for factory initialization")
        _factory_instance = EvaluatorFactory(config)
    
    return _factory_instance

def reset_factory():
    """Reset the global factory instance."""
    global _factory_instance
    
    if _factory_instance is not None:
        _factory_instance.clear_all_evaluators()
        _factory_instance = None
    
    logger.info("Factory instance reset")

# Module-level convenience functions
def create_tts_evaluator(config: Dict[str, Any], evaluator_id: Optional[str] = None) -> TTSEvaluator:
    """
    Convenience function to create a TTS evaluator.
    
    Args:
        config (Dict[str, Any]): Configuration
        evaluator_id (Optional[str]): Evaluator identifier
        
    Returns:
        TTSEvaluator: TTS evaluator instance
    """
    factory = get_factory(config)
    return factory.create_evaluator('tts', evaluator_id)

def create_stt_evaluator(config: Dict[str, Any], evaluator_id: Optional[str] = None) -> STTEvaluator:
    """
    Convenience function to create an STT evaluator.
    
    Args:
        config (Dict[str, Any]): Configuration
        evaluator_id (Optional[str]): Evaluator identifier
        
    Returns:
        STTEvaluator: STT evaluator instance
    """
    factory = get_factory(config)
    return factory.create_evaluator('stt', evaluator_id)