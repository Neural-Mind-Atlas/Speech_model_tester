# # """
# # Client factory for creating TTS/STT service clients
# # Provides centralized client instantiation with configuration management
# # """

# # import os
# # import importlib
# # from typing import Dict, Any, Optional, List, Type
# # import logging
# # import traceback

# # from .base_client import BaseTTSSTTClient
# # from .sarvam_client import SarvamClient
# # from .chatterbox_client import ChatterboxClient
# # from .openai_client import OpenAIClient
# # from .azure_client import AzureClient
# # from .google_client import GoogleClient

# # logger = logging.getLogger(__name__)

# # class ClientFactory:
# #     """
# #     Factory class for creating TTS/STT model clients based on configuration.
# #     Manages client creation for different provider types and models.
# #     """

# #     # Map of provider names to client classes
# #     _CLIENT_CLASSES = {
# #         "sarvam": SarvamClient,
# #         "chatterbox": ChatterboxClient,
# #         "openai": OpenAIClient,
# #         "azure": AzureClient,
# #         "google": GoogleClient
# #     }

# #     def __init__(self, config_dir: str = "configs/models"):
# #         """
# #         Initialize the client factory.

# #         Args:
# #             config_dir: Directory containing model configuration files
# #         """
# #         self.config_dir = config_dir
# #         self.model_configs = {}
# #         self.available_models = []
# #         self._load_model_configs()
        
# #         logger.info(f"ClientFactory initialized with {len(self.model_configs)} model configurations")

# #     def _load_model_configs(self) -> None:
# #         """
# #         Load all model configurations from the config directory.
# #         Only loads models that are not commented out in configuration files.
# #         """
# #         if not os.path.exists(self.config_dir):
# #             logger.warning(f"Config directory {self.config_dir} does not exist")
# #             return

# #         config_files = [f for f in os.listdir(self.config_dir) if f.endswith('.yaml')]
        
# #         for config_file in config_files:
# #             config_path = os.path.join(self.config_dir, config_file)
# #             provider_name = config_file.replace('.yaml', '')
            
# #             try:
# #                 config_data = self._load_yaml_config(config_path)
                
# #                 if not config_data or 'models' not in config_data:
# #                     logger.warning(f"No models found in {config_file}")
# #                     continue
                
# #                 # Load only active (non-commented) models
# #                 for model_name, model_config in config_data['models'].items():
# #                     # Skip if model is explicitly disabled
# #                     if model_config.get('enabled', True) is False:
# #                         logger.debug(f"Skipping disabled model: {model_name}")
# #                         continue
                    
# #                     # Add provider info if not present
# #                     if 'provider' not in model_config:
# #                         model_config['provider'] = provider_name
                    
# #                     # Merge with provider-level defaults
# #                     provider_defaults = config_data.get('defaults', {})
# #                     merged_config = {**provider_defaults, **model_config}
                    
# #                     self.model_configs[model_name] = merged_config
# #                     self.available_models.append(model_name)
                    
# #                     logger.debug(f"Loaded model configuration: {model_name} ({provider_name})")
                    
# #             except Exception as e:
# #                 logger.error(f"Error loading config from {config_path}: {str(e)}")
# #                 logger.debug(traceback.format_exc())

# #         logger.info(f"Loaded {len(self.model_configs)} active model configurations")

# #     def _load_yaml_config(self, file_path: str) -> Dict[str, Any]:
# #         """
# #         Load YAML configuration file with comment filtering.
        
# #         Args:
# #             file_path: Path to YAML file
            
# #         Returns:
# #             Parsed configuration dictionary
# #         """
# #         try:
# #             import yaml
            
# #             with open(file_path, 'r', encoding='utf-8') as file:
# #                 content = file.read()
            
# #             # Parse YAML while preserving the ability to comment out models
# #             config = yaml.safe_load(content)
# #             return config or {}
            
# #         except ImportError:
# #             logger.error("PyYAML is required for configuration loading")
# #             raise
# #         except Exception as e:
# #             logger.error(f"Error parsing YAML file {file_path}: {str(e)}")
# #             raise

# #     def get_available_models(self) -> List[str]:
# #         """
# #         Get list of all available configured models.

# #         Returns:
# #             List of model names
# #         """
# #         return self.available_models.copy()

# #     def get_models_by_provider(self, provider: str) -> List[str]:
# #         """
# #         Get list of models for a specific provider.

# #         Args:
# #             provider: Provider name (e.g., "openai", "azure", "sarvam")

# #         Returns:
# #             List of model names for the provider
# #         """
# #         return [
# #             model_name for model_name, config in self.model_configs.items()
# #             if config.get("provider", "").lower() == provider.lower()
# #         ]

# #     def get_models_by_capability(self, capability: str) -> List[str]:
# #         """
# #         Get list of models that support a specific capability.
        
# #         Args:
# #             capability: Capability name ("tts" or "stt")
            
# #         Returns:
# #             List of model names supporting the capability
# #         """
# #         capability_key = f"supports_{capability.lower()}"
# #         return [
# #             model_name for model_name, config in self.model_configs.items()
# #             if config.get(capability_key, False)
# #         ]

# #     def create_client(self, model_name: str) -> BaseTTSSTTClient:
# #         """
# #         Create a client for the specified model.

# #         Args:
# #             model_name: Name of the model to create a client for

# #         Returns:
# #             Appropriate client instance for the model

# #         Raises:
# #             ValueError: If model is not found or client creation fails
# #         """
# #         if model_name not in self.model_configs:
# #             available = ", ".join(self.available_models[:5])
# #             if len(self.available_models) > 5:
# #                 available += f", ... ({len(self.available_models)} total)"
# #             raise ValueError(
# #                 f"Model '{model_name}' not found in configurations. "
# #                 f"Available models: {available}"
# #             )

# #         model_config = self.model_configs[model_name].copy()
# #         provider = model_config.get("provider", "unknown").lower()

# #         # Get the appropriate client class
# #         client_class = self._CLIENT_CLASSES.get(provider)
# #         if not client_class:
# #             available_providers = ", ".join(self._CLIENT_CLASSES.keys())
# #             raise ValueError(
# #                 f"No client implementation for provider '{provider}'. "
# #                 f"Available providers: {available_providers}"
# #             )

# #         try:
# #             # Validate required configuration
# #             self._validate_model_config(model_name, model_config)
            
# #             # Create and return client instance
# #             client = client_class(model_name, model_config)
            
# #             logger.info(f"Created {provider} client for model: {model_name}")
            
# #             # Perform health check if enabled
# #             if model_config.get('health_check_on_init', False):
# #                 if not client.health_check():
# #                     logger.warning(f"Health check failed for {model_name}, but client created")
            
# #             return client
            
# #         except Exception as e:
# #             logger.error(f"Error creating client for model {model_name}: {str(e)}")
# #             logger.debug(traceback.format_exc())
# #             raise ValueError(f"Failed to create client for model {model_name}: {str(e)}")

# #     def create_clients_for_provider(self, provider: str) -> Dict[str, BaseTTSSTTClient]:
# #         """
# #         Create clients for all models from a specific provider.
        
# #         Args:
# #             provider: Provider name
            
# #         Returns:
# #             Dictionary mapping model names to client instances
# #         """
# #         models = self.get_models_by_provider(provider)
# #         clients = {}
        
# #         for model_name in models:
# #             try:
# #                 clients[model_name] = self.create_client(model_name)
# #             except Exception as e:
# #                 logger.error(f"Skipping model {model_name} due to error: {str(e)}")
        
# #         return clients

# #     def create_clients_for_capability(self, capability: str) -> Dict[str, BaseTTSSTTClient]:
# #         """
# #         Create clients for all models supporting a specific capability.
        
# #         Args:
# #             capability: Capability name ("tts" or "stt")
            
# #         Returns:
# #             Dictionary mapping model names to client instances
# #         """
# #         models = self.get_models_by_capability(capability)
# #         clients = {}
        
# #         for model_name in models:
# #             try:
# #                 clients[model_name] = self.create_client(model_name)
# #             except Exception as e:
# #                 logger.error(f"Skipping model {model_name} due to error: {str(e)}")
        
# #         return clients

# #     def create_all_clients(self) -> Dict[str, BaseTTSSTTClient]:
# #         """
# #         Create clients for all configured models.

# #         Returns:
# #             Dictionary mapping model names to client instances
# #         """
# #         clients = {}
        
# #         for model_name in self.available_models:
# #             try:
# #                 clients[model_name] = self.create_client(model_name)
# #             except Exception as e:
# #                 logger.error(f"Skipping model {model_name} due to error: {str(e)}")
        
# #         logger.info(f"Created {len(clients)} clients out of {len(self.available_models)} configured models")
# #         return clients

# #     def _validate_model_config(self, model_name: str, config: Dict[str, Any]) -> None:
# #         """
# #         Validate model configuration before client creation.
        
# #         Args:
# #             model_name: Name of the model
# #             config: Model configuration dictionary
            
# #         Raises:
# #             ValueError: If configuration is invalid
# #         """
# #         required_fields = ['provider']
        
# #         for field in required_fields:
# #             if field not in config:
# #                 raise ValueError(f"Missing required field '{field}' in configuration for {model_name}")
        
# #         # Validate provider-specific requirements
# #         provider = config['provider'].lower()
        
# #         if provider == 'azure':
# #             if 'region' not in config and 'speech_region' not in config:
# #                 raise ValueError(f"Azure models require 'region' or 'speech_region' in configuration")
        
# #         elif provider == 'google':
# #             if 'credentials_path' not in config and 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
# #                 logger.warning(f"Google models may require 'credentials_path' or GOOGLE_APPLICATION_CREDENTIALS")
        
# #         # Validate capabilities
# #         if not config.get('supports_tts', False) and not config.get('supports_stt', False):
# #             logger.warning(f"Model {model_name} has no TTS or STT capabilities enabled")

# #     def get_model_info(self, model_name: str) -> Dict[str, Any]:
# #         """
# #         Get detailed information about a model.
        
# #         Args:
# #             model_name: Name of the model
            
# #         Returns:
# #             Dictionary containing model information
# #         """
# #         if model_name not in self.model_configs:
# #             raise ValueError(f"Model '{model_name}' not found")
        
# #         config = self.model_configs[model_name]
        
# #         return {
# #             'name': model_name,
# #             'provider': config.get('provider'),
# #             'supports_tts': config.get('supports_tts', False),
# #             'supports_stt': config.get('supports_stt', False),
# #             'supported_languages': config.get('supported_languages', []),
# #             'supported_voices': config.get('supported_voices', []),
# #             'max_text_length': config.get('max_text_length'),
# #             'max_audio_size': config.get('max_audio_size'),
# #             'description': config.get('description', ''),
# #             'enabled': config.get('enabled', True)
# #         }

# #     def list_all_models(self) -> Dict[str, Dict[str, Any]]:
# #         """
# #         Get information about all available models.
        
# #         Returns:
# #             Dictionary mapping model names to their information
# #         """
# #         return {
# #             model_name: self.get_model_info(model_name)
# #             for model_name in self.available_models
# #         }

# #     def health_check_all(self) -> Dict[str, bool]:
# #         """
# #         Perform health checks on all available models.
        
# #         Returns:
# #             Dictionary mapping model names to health status
# #         """
# #         health_status = {}
        
# #         for model_name in self.available_models:
# #             try:
# #                 client = self.create_client(model_name)
# #                 health_status[model_name] = client.health_check()
# #             except Exception as e:
# #                 logger.error(f"Health check failed for {model_name}: {str(e)}")
# #                 health_status[model_name] = False
        
# #         return health_status

# #     def register_custom_client(self, provider: str, client_class: Type[BaseTTSSTTClient]) -> None:
# #         """
# #         Register a custom client class for a provider.
        
# #         Args:
# #             provider: Provider name
# #             client_class: Client class implementing BaseTTSSTTClient
# #         """
# #         if not issubclass(client_class, BaseTTSSTTClient):
# #             raise ValueError("Client class must inherit from BaseTTSSTTClient")
        
# #         self._CLIENT_CLASSES[provider.lower()] = client_class
# #         logger.info(f"Registered custom client for provider: {provider}")


# """
# Client Factory for TTS/STT Testing Framework

# This module provides a factory pattern implementation for creating
# TTS/STT client instances based on provider configuration.

# Author: TTS/STT Testing Framework Team
# Version: 1.0.0
# """

# import logging
# from typing import Dict, Any, List, Optional, Type
# from pathlib import Path

# from .base_client import BaseTTSSTTClient
# from configs.base_config import BaseConfig

# logger = logging.getLogger(__name__)

# class ClientFactory:
#     """
#     Factory class for creating TTS/STT client instances.
    
#     This factory handles the instantiation of different provider clients
#     based on configuration and availability.
#     """
    
#     def __init__(self, config: BaseConfig):
#         """
#         Initialize the client factory.
        
#         Args:
#             config: Base configuration object
#         """
#         self.config = config
#         self._client_registry = {}
#         self._register_available_clients()
        
#         logger.info(f"ClientFactory initialized with {len(self._client_registry)} available providers")
    
#     def _register_available_clients(self):
#         """Register all available client classes."""
#         # Import clients that are available
#         try:
#             from .openai_client import OpenAIClient
#             self._client_registry['openai'] = OpenAIClient
#             logger.debug("Registered OpenAI client")
#         except ImportError as e:
#             logger.warning(f"OpenAI client not available: {e}")
        
#         try:
#             from .azure_client import AzureClient
#             self._client_registry['azure'] = AzureClient
#             logger.debug("Registered Azure client")
#         except ImportError as e:
#             logger.warning(f"Azure client not available: {e}")
        
#         try:
#             from .google_client import GoogleClient
#             self._client_registry['google'] = GoogleClient
#             logger.debug("Registered Google client")
#         except ImportError as e:
#             logger.warning(f"Google client not available: {e}")
        
#         try:
#             from .sarvam_client import SarvamClient
#             self._client_registry['sarvam'] = SarvamClient
#             logger.debug("Registered Sarvam client")
#         except ImportError as e:
#             logger.warning(f"Sarvam client not available: {e}")
        
#         try:
#             from .chatterbox_client import ChatterboxClient
#             self._client_registry['chatterbox'] = ChatterboxClient
#             logger.debug("Registered Chatterbox client")
#         except ImportError as e:
#             logger.warning(f"Chatterbox client not available: {e}")
    
#     def create_client(self, provider: str, model_config: Optional[Dict[str, Any]] = None) -> BaseTTSSTTClient:
#         """
#         Create a client instance for the specified provider.
        
#         Args:
#             provider: Provider name (e.g., 'openai', 'azure', 'google')
#             model_config: Optional model-specific configuration
            
#         Returns:
#             Client instance for the provider
            
#         Raises:
#             ValueError: If provider is not supported or available
#         """
#         provider = provider.lower()
        
#         if provider not in self._client_registry:
#             available_providers = list(self._client_registry.keys())
#             raise ValueError(
#                 f"Provider '{provider}' is not available. "
#                 f"Available providers: {available_providers}"
#             )
        
#         client_class = self._client_registry[provider]
        
#         # Load provider-specific configuration
#         provider_config = self._load_provider_config(provider)
        
#         # Merge with model-specific config if provided
#         if model_config:
#             provider_config.update(model_config)
        
#         try:
#             # Create client instance
#             client = client_class(
#                 model_name=provider_config.get('default_model', 'default'),
#                 config=provider_config
#             )
            
#             logger.info(f"Created {provider} client successfully")
#             return client
            
#         except Exception as e:
#             logger.error(f"Failed to create {provider} client: {e}")
#             raise ValueError(f"Failed to create {provider} client: {e}")
    
#     def _load_provider_config(self, provider: str) -> Dict[str, Any]:
#         """
#         Load configuration for a specific provider.
        
#         Args:
#             provider: Provider name
            
#         Returns:
#             Provider configuration dictionary
#         """
#         config_file = Path(f"configs/models/{provider}.yaml")
        
#         if config_file.exists():
#             try:
#                 import yaml
#                 with open(config_file, 'r') as f:
#                     provider_config = yaml.safe_load(f)
#                 logger.debug(f"Loaded configuration for {provider} from {config_file}")
#                 return provider_config
#             except Exception as e:
#                 logger.warning(f"Failed to load config file for {provider}: {e}")
        
#         # Return default configuration
#         default_config = {
#             'provider': provider,
#             'supports_tts': True,
#             'supports_stt': True,
#             'max_retries': 3,
#             'retry_delay': 1.0
#         }
        
#         logger.debug(f"Using default configuration for {provider}")
#         return default_config
    
#     def get_available_providers(self) -> List[str]:
#         """
#         Get list of available providers.
        
#         Returns:
#             List of provider names
#         """
#         return list(self._client_registry.keys())
    
#     def is_provider_available(self, provider: str) -> bool:
#         """
#         Check if a provider is available.
        
#         Args:
#             provider: Provider name
            
#         Returns:
#             True if provider is available, False otherwise
#         """
#         return provider.lower() in self._client_registry
    
#     def get_provider_capabilities(self, provider: str) -> Dict[str, Any]:
#         """
#         Get capabilities of a specific provider.
        
#         Args:
#             provider: Provider name
            
#         Returns:
#             Dictionary containing provider capabilities
#         """
#         if not self.is_provider_available(provider):
#             return {}
        
#         try:
#             # Create a temporary client to get capabilities
#             temp_config = self._load_provider_config(provider)
#             client_class = self._client_registry[provider.lower()]
            
#             # Get capabilities without full initialization
#             capabilities = {
#                 'supports_tts': temp_config.get('supports_tts', False),
#                 'supports_stt': temp_config.get('supports_stt', False),
#                 'supported_languages': temp_config.get('supported_languages', []),
#                 'supported_voices': temp_config.get('supported_voices', []),
#                 'max_text_length': temp_config.get('max_text_length', 1000),
#                 'supported_audio_formats': temp_config.get('supported_audio_formats', ['wav'])
#             }
            
#             return capabilities
            
#         except Exception as e:
#             logger.error(f"Failed to get capabilities for {provider}: {e}")
#             return {}


# def create_client_factory(config: Optional[BaseConfig] = None) -> ClientFactory:
#     """
#     Convenience function to create a client factory.
    
#     Args:
#         config: Optional base configuration
        
#     Returns:
#         ClientFactory instance
#     """
#     if config is None:
#         config = BaseConfig()
    
#     return ClientFactory(config)


"""
Client factory for creating TTS/STT service clients
Provides centralized client instantiation with configuration management
"""

import os
import importlib
from typing import Dict, Any, Optional, List, Type, Tuple
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
    
    def __init__(self, config: Any):
        """
        Initialize the ClientFactory with configuration.
        
        Args:
            config: Configuration object containing client settings
        """
        self.config = config
        self.logger = logger
        
        # Registry of available client classes
        self.client_registry = {
            'sarvam': SarvamClient,
            'chatterbox': ChatterboxClient,
            'openai': OpenAIClient,
            'azure': AzureClient,
            'google': GoogleClient
        }
        
        # Cache for created clients
        self._client_cache = {}
        
        self.logger.info("ClientFactory initialized successfully")
    
    def create_client(self, provider: str, model_name: Optional[str] = None, **kwargs) -> BaseTTSSTTClient:
        """
        Create a client instance for the specified provider.
        
        Args:
            provider: Provider name (e.g., 'openai', 'azure', 'google')
            model_name: Optional model name override
            **kwargs: Additional configuration parameters
            
        Returns:
            BaseTTSSTTClient: Configured client instance
            
        Raises:
            ValueError: If provider is not supported
            Exception: If client creation fails
        """
        try:
            provider = provider.lower()
            
            if provider not in self.client_registry:
                available_providers = list(self.client_registry.keys())
                raise ValueError(f"Unsupported provider: {provider}. Available: {available_providers}")
            
            # Check cache first
            cache_key = f"{provider}_{model_name or 'default'}"
            if cache_key in self._client_cache:
                self.logger.debug(f"Returning cached client for {provider}")
                return self._client_cache[cache_key]
            
            # Get client class
            client_class = self.client_registry[provider]
            
            # Get provider configuration
            provider_config = self._get_provider_config(provider)
            
            # Override with kwargs
            if kwargs:
                provider_config.update(kwargs)
            
            # Override model name if specified
            if model_name:
                provider_config['model_name'] = model_name
            
            # Create client instance
            client = client_class(**provider_config)
            
            # Cache the client
            self._client_cache[cache_key] = client
            
            self.logger.info(f"Successfully created {provider} client")
            return client
            
        except Exception as e:
            self.logger.error(f"Failed to create {provider} client: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise
    
    def get_available_providers(self) -> List[str]:
        """
        Get list of available provider names.
        
        Returns:
            List[str]: List of supported provider names
        """
        return list(self.client_registry.keys())
    
    def is_provider_available(self, provider: str) -> bool:
        """
        Check if a provider is available.
        
        Args:
            provider: Provider name to check
            
        Returns:
            bool: True if provider is available
        """
        return provider.lower() in self.client_registry
    
    def get_provider_info(self, provider: str) -> Dict[str, Any]:
        """
        Get information about a specific provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Dict[str, Any]: Provider information
        """
        provider = provider.lower()
        if provider not in self.client_registry:
            return {}
        
        client_class = self.client_registry[provider]
        return {
            'name': provider,
            'class': client_class.__name__,
            'module': client_class.__module__,
            'available': True
        }
    
    def _get_provider_config(self, provider: str) -> Dict[str, Any]:
        """
        Get configuration for a specific provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Dict[str, Any]: Provider configuration
        """
        try:
            # Try to get provider config from the main config
            if hasattr(self.config, 'to_dict'):
                config_dict = self.config.to_dict()
            elif hasattr(self.config, 'dict'):
                config_dict = self.config.dict()
            else:
                config_dict = dict(self.config) if self.config else {}
            
            provider_config = config_dict.get(provider, {})
            
            # Add common configuration
            common_config = {
                'timeout': config_dict.get('timeout', 30),
                'max_retries': config_dict.get('max_retries', 3),
                'enable_logging': config_dict.get('enable_logging', True)
            }
            
            # Merge configurations
            final_config = {**common_config, **provider_config}
            
            return final_config
            
        except Exception as e:
            self.logger.warning(f"Failed to get config for {provider}: {str(e)}")
            return {}
    
    def clear_cache(self):
        """Clear the client cache."""
        self._client_cache.clear()
        self.logger.info("Client cache cleared")
    
    def get_cached_clients(self) -> Dict[str, BaseTTSSTTClient]:
        """
        Get all cached clients.
        
        Returns:
            Dict[str, BaseTTSSTTClient]: Dictionary of cached clients
        """
        return self._client_cache.copy()
    
    async def health_check_all(self) -> Dict[str, bool]:
        """
        Perform health check on all available providers.
        
        Returns:
            Dict[str, bool]: Health status for each provider
        """
        health_status = {}
        
        for provider in self.get_available_providers():
            try:
                client = self.create_client(provider)
                if hasattr(client, 'health_check'):
                    if asyncio.iscoroutinefunction(client.health_check):
                        health_status[provider] = await client.health_check()
                    else:
                        health_status[provider] = client.health_check()
                else:
                    health_status[provider] = True  # Assume healthy if no health check method
            except Exception as e:
                self.logger.error(f"Health check failed for {provider}: {str(e)}")
                health_status[provider] = False
        
        return health_status