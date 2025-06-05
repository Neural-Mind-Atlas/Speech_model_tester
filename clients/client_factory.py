"""
Client factory for creating TTS/STT service clients
Provides centralized client instantiation with configuration management
"""

import os
import importlib
from typing import Dict, Any, Optional, List, Type
import logging
import traceback

from .base_client import BaseTTSSTTClient
from .sarvam_client import SarvamClient
from .chatterbox_client import ChatterboxClient
from .openai_client import OpenAIClient
from .azure_client import AzureClient
from .google_client import GoogleClient

logger = logging.getLogger(__name__)

class ClientFactory:
    """
    Factory class for creating TTS/STT model clients based on configuration.
    Manages client creation for different provider types and models.
    """

    # Map of provider names to client classes
    _CLIENT_CLASSES = {
        "sarvam": SarvamClient,
        "chatterbox": ChatterboxClient,
        "openai": OpenAIClient,
        "azure": AzureClient,
        "google": GoogleClient
    }

    def __init__(self, config_dir: str = "configs/models"):
        """
        Initialize the client factory.

        Args:
            config_dir: Directory containing model configuration files
        """
        self.config_dir = config_dir
        self.model_configs = {}
        self.available_models = []
        self._load_model_configs()
        
        logger.info(f"ClientFactory initialized with {len(self.model_configs)} model configurations")

    def _load_model_configs(self) -> None:
        """
        Load all model configurations from the config directory.
        Only loads models that are not commented out in configuration files.
        """
        if not os.path.exists(self.config_dir):
            logger.warning(f"Config directory {self.config_dir} does not exist")
            return

        config_files = [f for f in os.listdir(self.config_dir) if f.endswith('.yaml')]
        
        for config_file in config_files:
            config_path = os.path.join(self.config_dir, config_file)
            provider_name = config_file.replace('.yaml', '')
            
            try:
                config_data = self._load_yaml_config(config_path)
                
                if not config_data or 'models' not in config_data:
                    logger.warning(f"No models found in {config_file}")
                    continue
                
                # Load only active (non-commented) models
                for model_name, model_config in config_data['models'].items():
                    # Skip if model is explicitly disabled
                    if model_config.get('enabled', True) is False:
                        logger.debug(f"Skipping disabled model: {model_name}")
                        continue
                    
                    # Add provider info if not present
                    if 'provider' not in model_config:
                        model_config['provider'] = provider_name
                    
                    # Merge with provider-level defaults
                    provider_defaults = config_data.get('defaults', {})
                    merged_config = {**provider_defaults, **model_config}
                    
                    self.model_configs[model_name] = merged_config
                    self.available_models.append(model_name)
                    
                    logger.debug(f"Loaded model configuration: {model_name} ({provider_name})")
                    
            except Exception as e:
                logger.error(f"Error loading config from {config_path}: {str(e)}")
                logger.debug(traceback.format_exc())

        logger.info(f"Loaded {len(self.model_configs)} active model configurations")

    def _load_yaml_config(self, file_path: str) -> Dict[str, Any]:
        """
        Load YAML configuration file with comment filtering.
        
        Args:
            file_path: Path to YAML file
            
        Returns:
            Parsed configuration dictionary
        """
        try:
            import yaml
            
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Parse YAML while preserving the ability to comment out models
            config = yaml.safe_load(content)
            return config or {}
            
        except ImportError:
            logger.error("PyYAML is required for configuration loading")
            raise
        except Exception as e:
            logger.error(f"Error parsing YAML file {file_path}: {str(e)}")
            raise

    def get_available_models(self) -> List[str]:
        """
        Get list of all available configured models.

        Returns:
            List of model names
        """
        return self.available_models.copy()

    def get_models_by_provider(self, provider: str) -> List[str]:
        """
        Get list of models for a specific provider.

        Args:
            provider: Provider name (e.g., "openai", "azure", "sarvam")

        Returns:
            List of model names for the provider
        """
        return [
            model_name for model_name, config in self.model_configs.items()
            if config.get("provider", "").lower() == provider.lower()
        ]

    def get_models_by_capability(self, capability: str) -> List[str]:
        """
        Get list of models that support a specific capability.
        
        Args:
            capability: Capability name ("tts" or "stt")
            
        Returns:
            List of model names supporting the capability
        """
        capability_key = f"supports_{capability.lower()}"
        return [
            model_name for model_name, config in self.model_configs.items()
            if config.get(capability_key, False)
        ]

    def create_client(self, model_name: str) -> BaseTTSSTTClient:
        """
        Create a client for the specified model.

        Args:
            model_name: Name of the model to create a client for

        Returns:
            Appropriate client instance for the model

        Raises:
            ValueError: If model is not found or client creation fails
        """
        if model_name not in self.model_configs:
            available = ", ".join(self.available_models[:5])
            if len(self.available_models) > 5:
                available += f", ... ({len(self.available_models)} total)"
            raise ValueError(
                f"Model '{model_name}' not found in configurations. "
                f"Available models: {available}"
            )

        model_config = self.model_configs[model_name].copy()
        provider = model_config.get("provider", "unknown").lower()

        # Get the appropriate client class
        client_class = self._CLIENT_CLASSES.get(provider)
        if not client_class:
            available_providers = ", ".join(self._CLIENT_CLASSES.keys())
            raise ValueError(
                f"No client implementation for provider '{provider}'. "
                f"Available providers: {available_providers}"
            )

        try:
            # Validate required configuration
            self._validate_model_config(model_name, model_config)
            
            # Create and return client instance
            client = client_class(model_name, model_config)
            
            logger.info(f"Created {provider} client for model: {model_name}")
            
            # Perform health check if enabled
            if model_config.get('health_check_on_init', False):
                if not client.health_check():
                    logger.warning(f"Health check failed for {model_name}, but client created")
            
            return client
            
        except Exception as e:
            logger.error(f"Error creating client for model {model_name}: {str(e)}")
            logger.debug(traceback.format_exc())
            raise ValueError(f"Failed to create client for model {model_name}: {str(e)}")

    def create_clients_for_provider(self, provider: str) -> Dict[str, BaseTTSSTTClient]:
        """
        Create clients for all models from a specific provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Dictionary mapping model names to client instances
        """
        models = self.get_models_by_provider(provider)
        clients = {}
        
        for model_name in models:
            try:
                clients[model_name] = self.create_client(model_name)
            except Exception as e:
                logger.error(f"Skipping model {model_name} due to error: {str(e)}")
        
        return clients

    def create_clients_for_capability(self, capability: str) -> Dict[str, BaseTTSSTTClient]:
        """
        Create clients for all models supporting a specific capability.
        
        Args:
            capability: Capability name ("tts" or "stt")
            
        Returns:
            Dictionary mapping model names to client instances
        """
        models = self.get_models_by_capability(capability)
        clients = {}
        
        for model_name in models:
            try:
                clients[model_name] = self.create_client(model_name)
            except Exception as e:
                logger.error(f"Skipping model {model_name} due to error: {str(e)}")
        
        return clients

    def create_all_clients(self) -> Dict[str, BaseTTSSTTClient]:
        """
        Create clients for all configured models.

        Returns:
            Dictionary mapping model names to client instances
        """
        clients = {}
        
        for model_name in self.available_models:
            try:
                clients[model_name] = self.create_client(model_name)
            except Exception as e:
                logger.error(f"Skipping model {model_name} due to error: {str(e)}")
        
        logger.info(f"Created {len(clients)} clients out of {len(self.available_models)} configured models")
        return clients

    def _validate_model_config(self, model_name: str, config: Dict[str, Any]) -> None:
        """
        Validate model configuration before client creation.
        
        Args:
            model_name: Name of the model
            config: Model configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = ['provider']
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in configuration for {model_name}")
        
        # Validate provider-specific requirements
        provider = config['provider'].lower()
        
        if provider == 'azure':
            if 'region' not in config and 'speech_region' not in config:
                raise ValueError(f"Azure models require 'region' or 'speech_region' in configuration")
        
        elif provider == 'google':
            if 'credentials_path' not in config and 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
                logger.warning(f"Google models may require 'credentials_path' or GOOGLE_APPLICATION_CREDENTIALS")
        
        # Validate capabilities
        if not config.get('supports_tts', False) and not config.get('supports_stt', False):
            logger.warning(f"Model {model_name} has no TTS or STT capabilities enabled")

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary containing model information
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Model '{model_name}' not found")
        
        config = self.model_configs[model_name]
        
        return {
            'name': model_name,
            'provider': config.get('provider'),
            'supports_tts': config.get('supports_tts', False),
            'supports_stt': config.get('supports_stt', False),
            'supported_languages': config.get('supported_languages', []),
            'supported_voices': config.get('supported_voices', []),
            'max_text_length': config.get('max_text_length'),
            'max_audio_size': config.get('max_audio_size'),
            'description': config.get('description', ''),
            'enabled': config.get('enabled', True)
        }

    def list_all_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all available models.
        
        Returns:
            Dictionary mapping model names to their information
        """
        return {
            model_name: self.get_model_info(model_name)
            for model_name in self.available_models
        }

    def health_check_all(self) -> Dict[str, bool]:
        """
        Perform health checks on all available models.
        
        Returns:
            Dictionary mapping model names to health status
        """
        health_status = {}
        
        for model_name in self.available_models:
            try:
                client = self.create_client(model_name)
                health_status[model_name] = client.health_check()
            except Exception as e:
                logger.error(f"Health check failed for {model_name}: {str(e)}")
                health_status[model_name] = False
        
        return health_status

    def register_custom_client(self, provider: str, client_class: Type[BaseTTSSTTClient]) -> None:
        """
        Register a custom client class for a provider.
        
        Args:
            provider: Provider name
            client_class: Client class implementing BaseTTSSTTClient
        """
        if not issubclass(client_class, BaseTTSSTTClient):
            raise ValueError("Client class must inherit from BaseTTSSTTClient")
        
        self._CLIENT_CLASSES[provider.lower()] = client_class
        logger.info(f"Registered custom client for provider: {provider}")