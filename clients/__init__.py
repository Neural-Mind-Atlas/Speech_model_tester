"""
TTS/STT Testing Framework - Clients Module
Provides unified interface for various TTS/STT service providers
"""

from .base_client import BaseTTSSTTClient
from .sarvam_client import SarvamClient
from .chatterbox_client import ChatterboxClient
from .openai_client import OpenAIClient
from .azure_client import AzureClient
from .google_client import GoogleClient
from .client_factory import ClientFactory
import logging

logger = logging.getLogger(__name__)

# __all__ = [
#     'BaseTTSSTTClient',
#     'SarvamClient',
#     'ChatterboxClient',
#     'OpenAIClient',
#     'AzureClient',
#     'GoogleClient',
#     'ClientFactory'
# ]

# __version__ = "1.0.0"


"""
TTS/STT Testing Framework - Clients Module
Provides unified interface for various TTS/STT service providers
"""

import logging

logger = logging.getLogger(__name__)

# Import base client (should always work)
try:
    from .base_client import BaseTTSSTTClient
    logger.info("Successfully imported BaseTTSSTTClient")
except ImportError as e:
    logger.error(f"Failed to import BaseTTSSTTClient: {e}")
    raise ImportError(f"Critical client module import failed: {e}")

# Import other clients with graceful error handling
_available_clients = ['BaseTTSSTTClient']
_import_errors = []

try:
    from .sarvam_client import SarvamClient
    _available_clients.append('SarvamClient')
    logger.info("Successfully imported SarvamClient")
except ImportError as e:
    logger.warning(f"Failed to import SarvamClient: {e}")
    _import_errors.append(('SarvamClient', str(e)))

try:
    from .chatterbox_client import ChatterboxClient
    _available_clients.append('ChatterboxClient')
    logger.info("Successfully imported ChatterboxClient")
except ImportError as e:
    logger.warning(f"Failed to import ChatterboxClient: {e}")
    _import_errors.append(('ChatterboxClient', str(e)))

try:
    from .openai_client import OpenAIClient
    _available_clients.append('OpenAIClient')
    logger.info("Successfully imported OpenAIClient")
except ImportError as e:
    logger.warning(f"Failed to import OpenAIClient: {e}")
    _import_errors.append(('OpenAIClient', str(e)))

try:
    from .azure_client import AzureClient
    _available_clients.append('AzureClient')
    logger.info("Successfully imported AzureClient")
except ImportError as e:
    logger.warning(f"Failed to import AzureClient (Azure SDK may not be installed): {e}")
    _import_errors.append(('AzureClient', str(e)))
    # Create a placeholder for AzureClient
    AzureClient = None

try:
    from .google_client import GoogleClient
    _available_clients.append('GoogleClient')
    logger.info("Successfully imported GoogleClient")
except ImportError as e:
    logger.warning(f"Failed to import GoogleClient: {e}")
    _import_errors.append(('GoogleClient', str(e)))

try:
    from .client_factory import ClientFactory
    _available_clients.append('ClientFactory')
    logger.info("Successfully imported ClientFactory")
except ImportError as e:
    logger.warning(f"Failed to import ClientFactory: {e}")
    _import_errors.append(('ClientFactory', str(e)))

# Export only available clients
__all__ = _available_clients

__version__ = "1.0.0"

def get_available_clients():
    """Get list of successfully imported clients."""
    return _available_clients.copy()

def get_import_errors():
    """Get list of import errors."""
    return _import_errors.copy()

logger.info(f"Clients module initialized with {len(_available_clients)} available clients")
if _import_errors:
    logger.warning(f"Some clients failed to import: {[err[0] for err in _import_errors]}")