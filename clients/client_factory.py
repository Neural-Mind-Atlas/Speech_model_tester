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
        
        # Registry of available client classes - populated dynamically
        self.client_registry = {}
        self._initialize_client_registry()
        
        # Cache for created clients
        self._client_cache = {}
        
        self.logger.info(f"ClientFactory initialized with {len(self.client_registry)} available clients")
    
    def _initialize_client_registry(self):
        """Initialize the client registry with available clients."""
        # Static mapping to fix provider.title() issue
        client_imports = [
            ('sarvam', '.sarvam_client', 'SarvamClient'),
            ('chatterbox', '.chatterbox_client', 'ChatterboxClient'),
            ('openai', '.openai_client', 'OpenAIClient'),  # Fixed case sensitivity
            ('azure', '.azure_client', 'AzureClient'),
            ('google', '.google_client', 'GoogleClient')
        ]
        
        for provider_name, module_path, class_name in client_imports:
            try:
                module = importlib.import_module(module_path, package=__package__)
                client_class = getattr(module, class_name)
                self.client_registry[provider_name] = client_class
                self.logger.info(f"Successfully registered {provider_name} client")
            except ImportError as e:
                self.logger.warning(f"Failed to import {provider_name} client: {e}")
            except AttributeError as e:
                self.logger.error(f"Client class {class_name} not found in {module_path}: {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error importing {provider_name} client: {e}")
    
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
            
            # Get client class from registry (no more provider.title() issues!)
            client_class = self.client_registry[provider]
            
            # Get provider configuration
            provider_config = self._get_provider_config(provider)
            
            # Override with kwargs (but filter out factory-specific params)
            filtered_kwargs = {k: v for k, v in kwargs.items() 
                              if k not in ['timeout', 'max_retries', 'enable_logging']}
            if filtered_kwargs:
                provider_config.update(filtered_kwargs)
            
            # Override model name if specified
            if model_name:
                provider_config['model_name'] = model_name
            
            # Create client instance with proper constructor signature
            if provider == 'azure':
                # Azure client expects (model_name, config)
                model = provider_config.pop('model_name', 'default')
                client = client_class(model, provider_config)
            elif provider == 'openai':
                # OpenAI client expects (config) - model_name is in config
                client = client_class(provider_config)
            elif provider == 'google':
                # Google client expects (config) - fixed constructor
                client = client_class(provider_config)
            else:
                # Other clients typically expect (config) or (**config)
                try:
                    client = client_class(provider_config)
                except TypeError:
                    # Try with **kwargs if single dict param doesn't work
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
            
            # Add API key from environment if not in config
            if 'api_key' not in provider_config:
                env_key_map = {
                    'openai': 'OPENAI_API_KEY',
                    'azure': 'AZURE_SPEECH_KEY',
                    'google': 'GOOGLE_APPLICATION_CREDENTIALS',
                    'sarvam': 'SARVAM_API_KEY',
                    'chatterbox': 'CHATTERBOX_API_KEY'
                }
                
                if provider in env_key_map:
                    env_key = os.getenv(env_key_map[provider])
                    if env_key:
                        provider_config['api_key'] = env_key
                    else:
                        # Set a default for testing
                        provider_config['api_key'] = 'test_key'
            
            # Add region for Azure if not present
            if provider == 'azure' and 'region' not in provider_config:
                provider_config['region'] = os.getenv('AZURE_SPEECH_REGION', 'eastus')
            
            # Ensure model_name is set
            if 'model_name' not in provider_config:
                default_models = {
                    'openai': 'tts-1',
                    'azure': 'en-US-JennyNeural', 
                    'google': 'standard',
                    'sarvam': 'bulbul',
                    'chatterbox': 'default'
                }
                provider_config['model_name'] = default_models.get(provider, 'default')
            
            return provider_config
            
        except Exception as e:
            self.logger.warning(f"Failed to get config for {provider}: {str(e)}")
            # Return minimal working config
            return {
                'api_key': 'test_key',
                'model_name': 'default'
            }
    
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
                    import asyncio
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

    def get_client_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the client factory.
        
        Returns:
            Dict[str, Any]: Factory statistics
        """
        return {
            'total_registered_clients': len(self.client_registry),
            'available_providers': list(self.client_registry.keys()),
            'cached_clients': len(self._client_cache),
            'cache_keys': list(self._client_cache.keys())
        }