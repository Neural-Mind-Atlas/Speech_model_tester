"""
TTS/STT Testing Framework - Client Tests
======================================

Comprehensive test suite for all client implementations in the TTS/STT testing framework.
Tests include unit tests, integration tests, error handling, and performance benchmarks.

Author: TTS/STT Testing Framework Team
Version: 1.0.0
Created: 2024-06-04
"""

import os
import sys
import pytest
import asyncio
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path
from typing import Dict, Any, List, Optional
import tempfile
import time
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from clients.base_client import BaseClient, ClientResponse, ClientError
from clients.client_factory import ClientFactory
from clients.sarvam_client import SarvamClient
from clients.chatterbox_client import ChatterboxClient
from clients.openai_client import OpenAIClient
from clients.azure_client import AzureClient
from clients.google_client import GoogleClient
from utils.logger import get_logger, setup_logging

class TestBaseClient:
    """Test suite for BaseClient abstract class and common functionality."""
    
    @pytest.fixture(autouse=True)
    def setup_logging(self):
        """Setup logging for tests."""
        setup_logging(log_level="DEBUG", enable_file=False)
        self.logger = get_logger(__name__)
    
    @pytest.fixture
    def base_client(self):
        """Create a concrete implementation of BaseClient for testing."""
        class TestClient(BaseClient):
            def __init__(self):
                super().__init__(
                    client_name="test_client",
                    api_key="test_key",
                    base_url="https://test.api.com"
                )
            
            async def text_to_speech(self, text: str, voice: str = "default", **kwargs) -> ClientResponse:
                return ClientResponse(
                    success=True,
                    audio_url="https://test.com/audio.wav",
                    metadata={"voice": voice, "text": text}
                )
            
            async def speech_to_text(self, audio_path: str, **kwargs) -> ClientResponse:
                return ClientResponse(
                    success=True,
                    text="transcribed text",
                    metadata={"audio_path": audio_path}
                )
        
        return TestClient()
    
    def test_client_initialization(self, base_client):
        """Test client initialization and properties."""
        assert base_client.client_name == "test_client"
        assert base_client.api_key == "test_key"
        assert base_client.base_url == "https://test.api.com"
        assert base_client.is_healthy is True
        assert isinstance(base_client.stats, dict)
    
    def test_client_response_creation(self):
        """Test ClientResponse creation and properties."""
        response = ClientResponse(
            success=True,
            text="test text",
            audio_url="test_url",
            metadata={"key": "value"}
        )
        
        assert response.success is True
        assert response.text == "test text"
        assert response.audio_url == "test_url"
        assert response.metadata["key"] == "value"
        assert isinstance(response.timestamp, str)
    
    def test_client_error_creation(self):
        """Test ClientError creation and properties."""
        error = ClientError(
            message="Test error",
            error_code="TEST_001",
            details={"detail": "value"}
        )
        
        assert str(error) == "Test error"
        assert error.error_code == "TEST_001"
        assert error.details["detail"] == "value"
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, base_client):
        """Test successful health check."""
        with patch.object(base_client, 'text_to_speech') as mock_tts:
            mock_tts.return_value = ClientResponse(success=True)
            
            result = await base_client.health_check()
            assert result is True
            assert base_client.is_healthy is True
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, base_client):
        """Test failed health check."""
        with patch.object(base_client, 'text_to_speech') as mock_tts:
            mock_tts.side_effect = Exception("Connection failed")
            
            result = await base_client.health_check()
            assert result is False
            assert base_client.is_healthy is False
    
    def test_get_stats(self, base_client):
        """Test statistics retrieval."""
        stats = base_client.get_stats()
        
        required_keys = [
            'client_name', 'total_requests', 'successful_requests',
            'failed_requests', 'average_response_time', 'last_request_time'
        ]
        
        for key in required_keys:
            assert key in stats
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, base_client):
        """Test rate limiting functionality."""
        # Set a very low rate limit for testing
        base_client.rate_limit = 1  # 1 request per second
        
        start_time = time.time()
        
        # Make two requests quickly
        await base_client.text_to_speech("test1")
        await base_client.text_to_speech("test2")
        
        end_time = time.time()
        
        # Should take at least 1 second due to rate limiting
        assert end_time - start_time >= 1.0

class TestClientFactory:
    """Test suite for ClientFactory."""
    
    @pytest.fixture(autouse=True)
    def setup_logging(self):
        """Setup logging for tests."""
        setup_logging(log_level="DEBUG", enable_file=False)
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            'sarvam': {
                'api_key': 'test_sarvam_key',
                'base_url': 'https://api.sarvam.ai',
                'enabled': True
            },
            'openai': {
                'api_key': 'test_openai_key',
                'organization': 'test_org',
                'enabled': True
            },
            'azure': {
                'api_key': 'test_azure_key',
                'region': 'eastus',
                'enabled': False
            }
        }
    
    def test_create_client_success(self, sample_config):
        """Test successful client creation."""
        with patch('clients.sarvam_client.SarvamClient') as mock_sarvam:
            mock_instance = Mock()
            mock_sarvam.return_value = mock_instance
            
            client = ClientFactory.create_client('sarvam', sample_config['sarvam'])
            
            assert client == mock_instance
            mock_sarvam.assert_called_once()
    
    def test_create_client_invalid_type(self, sample_config):
        """Test client creation with invalid type."""
        with pytest.raises(ValueError, match="Unsupported client type"):
            ClientFactory.create_client('invalid_client', sample_config['sarvam'])
    
    def test_create_client_missing_config(self):
        """Test client creation with missing configuration."""
        with pytest.raises(ValueError, match="Client configuration is required"):
            ClientFactory.create_client('sarvam', None)
    
    def test_create_all_clients(self, sample_config):
        """Test creating all configured clients."""
        with patch('clients.sarvam_client.SarvamClient') as mock_sarvam, \
             patch('clients.openai_client.OpenAIClient') as mock_openai:
            
            mock_sarvam.return_value = Mock()
            mock_openai.return_value = Mock()
            
            clients = ClientFactory.create_all_clients(sample_config)
            
            assert len(clients) == 2  # Only enabled clients
            assert 'sarvam' in clients
            assert 'openai' in clients
            assert 'azure' not in clients  # Disabled
    
    def test_get_available_clients(self):
        """Test getting list of available client types."""
        available = ClientFactory.get_available_clients()
        
        expected_clients = ['sarvam', 'chatterbox', 'openai', 'azure', 'google']
        for client in expected_clients:
            assert client in available

class TestSarvamClient:
    """Test suite for SarvamClient."""
    
    @pytest.fixture(autouse=True)
    def setup_logging(self):
        """Setup logging for tests."""
        setup_logging(log_level="DEBUG", enable_file=False)
    
    @pytest.fixture
    def sarvam_client(self):
        """Create SarvamClient for testing."""
        config = {
            'api_key': 'test_key',
            'base_url': 'https://api.sarvam.ai'
        }
        return SarvamClient(config)
    
    @pytest.mark.asyncio
    async def test_text_to_speech_success(self, sarvam_client):
        """Test successful TTS request."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            'audio_url': 'https://sarvam.ai/audio/test.wav',
            'duration': 5.2,
            'voice': 'meera'
        })
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await sarvam_client.text_to_speech(
                text="Hello world",
                voice="meera",
                language="hi"
            )
            
            assert result.success is True
            assert result.audio_url == 'https://sarvam.ai/audio/test.wav'
            assert result.metadata['duration'] == 5.2
            assert result.metadata['voice'] == 'meera'
    
    @pytest.mark.asyncio
    async def test_text_to_speech_api_error(self, sarvam_client):
        """Test TTS request with API error."""
        mock_response = Mock()
        mock_response.status = 400
        mock_response.json = AsyncMock(return_value={
            'error': 'Invalid voice parameter'
        })
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await sarvam_client.text_to_speech(
                text="Hello world",
                voice="invalid_voice"
            )
            
            assert result.success is False
            assert "Invalid voice parameter" in result.error_message
    
    @pytest.mark.asyncio
    async def test_speech_to_text_success(self, sarvam_client):
        """Test successful STT request."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            'transcript': 'Hello world',
            'confidence': 0.95,
            'language': 'en'
        })
        
        with patch('aiohttp.ClientSession.post') as mock_post, \
             patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open_audio_file()):
            
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await sarvam_client.speech_to_text(
                audio_path="test_audio.wav",
                language="en"
            )
            
            assert result.success is True
            assert result.text == 'Hello world'
            assert result.metadata['confidence'] == 0.95
            assert result.metadata['language'] == 'en'
    
    @pytest.mark.asyncio
    async def test_speech_to_text_file_not_found(self, sarvam_client):
        """Test STT request with missing audio file."""
        with patch('pathlib.Path.exists', return_value=False):
            result = await sarvam_client.speech_to_text("nonexistent.wav")
            
            assert result.success is False
            assert "not found" in result.error_message.lower()
    
    def test_get_available_voices(self, sarvam_client):
        """Test getting available voices."""
        voices = sarvam_client.get_available_voices()
        
        assert isinstance(voices, list)
        assert len(voices) > 0
        
        # Check voice structure
        for voice in voices:
            assert 'name' in voice
            assert 'language' in voice
            assert 'gender' in voice

class TestOpenAIClient:
    """Test suite for OpenAIClient."""
    
    @pytest.fixture(autouse=True)
    def setup_logging(self):
        """Setup logging for tests."""
        setup_logging(log_level="DEBUG", enable_file=False)
    
    @pytest.fixture
    def openai_client(self):
        """Create OpenAIClient for testing."""
        config = {
            'api_key': 'test_key',
            'organization': 'test_org'
        }
        return OpenAIClient(config)
    
    @pytest.mark.asyncio
    async def test_text_to_speech_success(self, openai_client):
        """Test successful TTS request."""
        mock_response = Mock()
        mock_response.content = b"fake_audio_data"
        
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            mock_client.audio.speech.create = AsyncMock(return_value=mock_response)
            
            with patch('tempfile.NamedTemporaryFile') as mock_temp:
                mock_temp.return_value.__enter__.return_value.name = "/tmp/test_audio.mp3"
                
                result = await openai_client.text_to_speech(
                    text="Hello world",
                    voice="alloy",
                    model="tts-1"
                )
                
                assert result.success is True
                assert result.audio_url == "/tmp/test_audio.mp3"
                assert result.metadata['voice'] == 'alloy'
                assert result.metadata['model'] == 'tts-1'
    
    @pytest.mark.asyncio
    async def test_speech_to_text_success(self, openai_client):
        """Test successful STT request."""
        mock_response = Mock()
        mock_response.text = "Hello world"
        
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)
            
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('builtins.open', mock_open_audio_file()):
                
                result = await openai_client.speech_to_text(
                    audio_path="test_audio.wav",
                    model="whisper-1"
                )
                
                assert result.success is True
                assert result.text == "Hello world"
                assert result.metadata['model'] == 'whisper-1'
    
    def test_get_available_voices(self, openai_client):
        """Test getting available voices."""
        voices = openai_client.get_available_voices()
        
        expected_voices = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
        for voice in expected_voices:
            assert voice in [v['name'] for v in voices]

class TestAzureClient:
    """Test suite for AzureClient."""
    
    @pytest.fixture(autouse=True)
    def setup_logging(self):
        """Setup logging for tests."""
        setup_logging(log_level="DEBUG", enable_file=False)
    
    @pytest.fixture
    def azure_client(self):
        """Create AzureClient for testing."""
        config = {
            'api_key': 'test_key',
            'region': 'eastus'
        }
        return AzureClient(config)
    
    @pytest.mark.asyncio
    async def test_text_to_speech_success(self, azure_client):
        """Test successful TTS request."""
        with patch('azure.cognitiveservices.speech.SpeechSynthesizer') as mock_synth:
            mock_result = Mock()
            mock_result.reason = Mock()
            mock_result.reason.name = "SynthesizingAudioCompleted"
            mock_result.audio_data = b"fake_audio_data"
            
            mock_synthesizer = Mock()
            mock_synthesizer.speak_ssml_async.return_value.get.return_value = mock_result
            mock_synth.return_value = mock_synthesizer
            
            with patch('tempfile.NamedTemporaryFile') as mock_temp:
                mock_temp.return_value.__enter__.return_value.name = "/tmp/test_azure.wav"
                
                result = await azure_client.text_to_speech(
                    text="Hello world",
                    voice="en-US-AriaNeural",
                    language="en-US"
                )
                
                assert result.success is True
                assert result.audio_url == "/tmp/test_azure.wav"
                assert result.metadata['voice'] == 'en-US-AriaNeural'
    
    @pytest.mark.asyncio
    async def test_speech_to_text_success(self, azure_client):
        """Test successful STT request."""
        with patch('azure.cognitiveservices.speech.SpeechRecognizer') as mock_recognizer:
            mock_result = Mock()
            mock_result.reason.name = "RecognizedSpeech"
            mock_result.text = "Hello world"
            
            mock_rec = Mock()
            mock_rec.recognize_once_async.return_value.get.return_value = mock_result
            mock_recognizer.return_value = mock_rec
            
            with patch('pathlib.Path.exists', return_value=True):
                result = await azure_client.speech_to_text(
                    audio_path="test_audio.wav",
                    language="en-US"
                )
                
                assert result.success is True
                assert result.text == "Hello world"
                assert result.metadata['language'] == 'en-US'

class TestGoogleClient:
    """Test suite for GoogleClient."""
    
    @pytest.fixture(autouse=True)
    def setup_logging(self):
        """Setup logging for tests."""
        setup_logging(log_level="DEBUG", enable_file=False)
    
    @pytest.fixture
    def google_client(self):
        """Create GoogleClient for testing."""
        config = {
            'credentials_path': '/path/to/credentials.json',
            'project_id': 'test_project'
        }
        return GoogleClient(config)
    
    @pytest.mark.asyncio
    async def test_text_to_speech_success(self, google_client):
        """Test successful TTS request."""
        with patch('google.cloud.texttospeech.TextToSpeechClient') as mock_client:
            mock_response = Mock()
            mock_response.audio_content = b"fake_audio_data"
            
            mock_instance = Mock()
            mock_instance.synthesize_speech.return_value = mock_response
            mock_client.return_value = mock_instance
            
            with patch('tempfile.NamedTemporaryFile') as mock_temp:
                mock_temp.return_value.__enter__.return_value.name = "/tmp/test_google.mp3"
                
                result = await google_client.text_to_speech(
                    text="Hello world",
                    voice="en-US-Wavenet-D",
                    language="en-US"
                )
                
                assert result.success is True
                assert result.audio_url == "/tmp/test_google.mp3"
                assert result.metadata['voice'] == 'en-US-Wavenet-D'
    
    @pytest.mark.asyncio
    async def test_speech_to_text_success(self, google_client):
        """Test successful STT request."""
        with patch('google.cloud.speech.SpeechClient') as mock_client:
            mock_response = Mock()
            mock_alternative = Mock()
            mock_alternative.transcript = "Hello world"
            mock_alternative.confidence = 0.95
            
            mock_result = Mock()
            mock_result.alternatives = [mock_alternative]
            
            mock_response.results = [mock_result]
            
            mock_instance = Mock()
            mock_instance.recognize.return_value = mock_response
            mock_client.return_value = mock_instance
            
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('builtins.open', mock_open_audio_file()):
                
                result = await google_client.speech_to_text(
                    audio_path="test_audio.wav",
                    language="en-US"
                )
                
                assert result.success is True
                assert result.text == "Hello world"
                assert result.metadata['confidence'] == 0.95

class TestPerformance:
    """Performance tests for clients."""
    
    @pytest.fixture(autouse=True)
    def setup_logging(self):
        """Setup logging for tests."""
        setup_logging(log_level="DEBUG", enable_file=False)
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        config = {'api_key': 'test_key', 'base_url': 'https://test.api.com'}
        client = SarvamClient(config)
        
        # Mock successful responses
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={'audio_url': 'test.wav'})
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            # Make 10 concurrent requests
            tasks = [
                client.text_to_speech(f"Text {i}", voice="test")
                for i in range(10)
            ]
            
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            # All requests should succeed
            assert all(result.success for result in results)
            
            # Should complete within reasonable time (accounting for rate limiting)
            assert end_time - start_time < 30  # 30 seconds max
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_memory_usage(self):
        """Test memory usage during multiple requests."""
        import psutil
        import gc
        
        config = {'api_key': 'test_key', 'base_url': 'https://test.api.com'}
        client = SarvamClient(config)
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Mock responses
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={'audio_url': 'test.wav'})
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            # Make many requests
            for i in range(100):
                await client.text_to_speech(f"Text {i}", voice="test")
                
                # Force garbage collection every 10 requests
                if i % 10 == 0:
                    gc.collect()
        
        # Check final memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024

# Helper functions for mocking

def mock_open_audio_file():
    """Mock for opening audio files."""
    from unittest.mock import mock_open
    return mock_open(read_data=b"fake_audio_data")

# Test fixtures and utilities

@pytest.fixture(scope="session")
def sample_audio_file():
    """Create a temporary audio file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        # Write minimal WAV header
        f.write(b'RIFF')
        f.write((36).to_bytes(4, 'little'))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write((16).to_bytes(4, 'little'))
        f.write((1).to_bytes(2, 'little'))
        f.write((1).to_bytes(2, 'little'))
        f.write((44100).to_bytes(4, 'little'))
        f.write((88200).to_bytes(4, 'little'))
        f.write((2).to_bytes(2, 'little'))
        f.write((16).to_bytes(2, 'little'))
        f.write(b'data')
        f.write((0).to_bytes(4, 'little'))
        
        return f.name

@pytest.fixture(scope="session")
def sample_text_data():
    """Sample text data for testing."""
    return [
        "Hello world",
        "This is a test sentence",
        "Quick brown fox jumps over the lazy dog",
        "Python is a great programming language",
        "Text to speech conversion test"
    ]

# Test configuration

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )

# Test collection customization

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add slow marker to performance tests
        if "performance" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Add integration marker to integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main(["-v", "--tb=short", __file__])