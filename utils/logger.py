"""
TTS/STT Testing Framework - Logging Utilities
============================================

This module provides comprehensive logging functionality for the TTS/STT testing framework.
It includes configurable logging levels, file rotation, formatting, and error tracking.

Author: TTS/STT Testing Framework Team
Version: 1.0.0
Created: 2024-06-04
"""

import os
import sys
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union
import json
import traceback
from enum import Enum

class LogLevel(Enum):
    """Enumeration for logging levels."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

class FrameworkLogger:
    """
    Advanced logging class for the TTS/STT testing framework.
    
    Features:
    - Multiple output formats (console, file, JSON)
    - Automatic log rotation
    - Error tracking and reporting
    - Performance monitoring
    - Structured logging
    """
    
    def __init__(
        self,
        name: str,
        log_level: Union[str, int, LogLevel] = LogLevel.INFO,
        log_dir: Optional[str] = None,
        max_file_size: int = 10485760,  # 10MB
        backup_count: int = 5,
        enable_console: bool = True,
        enable_file: bool = True,
        enable_json: bool = False
    ):
        """
        Initialize the framework logger.
        
        Args:
            name: Logger name (typically module name)
            log_level: Logging level
            log_dir: Directory for log files
            max_file_size: Maximum size per log file in bytes
            backup_count: Number of backup files to keep
            enable_console: Enable console logging
            enable_file: Enable file logging
            enable_json: Enable JSON structured logging
        """
        self.name = name
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_json = enable_json
        
        # Convert log level to integer if needed
        if isinstance(log_level, LogLevel):
            self.log_level = log_level.value
        elif isinstance(log_level, str):
            self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        else:
            self.log_level = log_level
        
        # Initialize logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_logger()
        
        # Error tracking
        self.error_count = 0
        self.warning_count = 0
        self.last_error = None
        
    def _setup_logger(self) -> None:
        """Set up logger handlers and formatters."""
        try:
            # Create log directory if it doesn't exist
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # Console handler
            if self.enable_console:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(self.log_level)
                console_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                console_handler.setFormatter(console_formatter)
                self.logger.addHandler(console_handler)
            
            # File handler with rotation
            if self.enable_file:
                log_file = self.log_dir / f"{self.name}.log"
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=self.max_file_size,
                    backupCount=self.backup_count,
                    encoding='utf-8'
                )
                file_handler.setLevel(self.log_level)
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)
            
            # JSON handler for structured logging
            if self.enable_json:
                json_file = self.log_dir / f"{self.name}_structured.json"
                json_handler = logging.handlers.RotatingFileHandler(
                    json_file,
                    maxBytes=self.max_file_size,
                    backupCount=self.backup_count,
                    encoding='utf-8'
                )
                json_handler.setLevel(self.log_level)
                json_handler.setFormatter(JSONFormatter())
                self.logger.addHandler(json_handler)
                
        except Exception as e:
            # Fallback to basic console logging
            print(f"Failed to setup logger {self.name}: {e}")
            basic_handler = logging.StreamHandler()
            basic_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
            self.logger.addHandler(basic_handler)
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, extra)
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log info message."""
        self._log(logging.INFO, message, extra)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log warning message."""
        self.warning_count += 1
        self._log(logging.WARNING, message, extra)
    
    def error(self, message: str, exception: Optional[Exception] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log error message with optional exception details."""
        self.error_count += 1
        self.last_error = {
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'exception': str(exception) if exception else None,
            'traceback': traceback.format_exc() if exception else None
        }
        
        if exception:
            message = f"{message} - Exception: {exception}"
        
        self._log(logging.ERROR, message, extra)
    
    def critical(self, message: str, exception: Optional[Exception] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log critical message with optional exception details."""
        self.error_count += 1
        self.last_error = {
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'exception': str(exception) if exception else None,
            'traceback': traceback.format_exc() if exception else None
        }
        
        if exception:
            message = f"{message} - Exception: {exception}"
        
        self._log(logging.CRITICAL, message, extra)
    
    def _log(self, level: int, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Internal logging method with extra data support."""
        try:
            if extra:
                # Add extra data to log record
                self.logger.log(level, message, extra=extra)
            else:
                self.logger.log(level, message)
        except Exception as e:
            # Fallback logging
            print(f"Logging failed for {self.name}: {e}")
            print(f"Original message: {message}")
    
    def log_performance(self, operation: str, duration: float, success: bool = True, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log performance metrics for operations."""
        perf_data = {
            'operation': operation,
            'duration_seconds': duration,
            'success': success,
            'timestamp': datetime.now().isoformat()
        }
        
        if extra:
            perf_data.update(extra)
        
        message = f"Performance: {operation} - {duration:.3f}s - {'SUCCESS' if success else 'FAILED'}"
        self._log(logging.INFO, message, {'performance': perf_data})
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        return {
            'logger_name': self.name,
            'log_level': logging.getLevelName(self.log_level),
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'last_error': self.last_error,
            'handlers_count': len(self.logger.handlers),
            'log_directory': str(self.log_dir)
        }

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
            'thread': record.thread,
            'process': record.process
        }
        
        # Add extra data if present
        if hasattr(record, 'performance'):
            log_data['performance'] = record.performance
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': self.formatException(record.exc_info) if record.exc_info else None
            }
        
        return json.dumps(log_data, ensure_ascii=False)

# Global logger instance
_global_logger: Optional[FrameworkLogger] = None

def setup_logging(
    log_level: Union[str, int, LogLevel] = LogLevel.INFO,
    log_dir: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = True,
    enable_json: bool = False
) -> None:
    """
    Set up global logging configuration.
    
    Args:
        log_level: Global logging level
        log_dir: Directory for log files
        enable_console: Enable console logging
        enable_file: Enable file logging
        enable_json: Enable JSON structured logging
    """
    global _global_logger
    
    try:
        _global_logger = FrameworkLogger(
            name="tts_stt_framework",
            log_level=log_level,
            log_dir=log_dir,
            enable_console=enable_console,
            enable_file=enable_file,
            enable_json=enable_json
        )
        _global_logger.info("Global logging setup completed successfully")
    except Exception as e:
        print(f"Failed to setup global logging: {e}")
        # Create minimal fallback logger
        _global_logger = FrameworkLogger(
            name="tts_stt_framework_fallback",
            log_level=LogLevel.INFO,
            enable_file=False,
            enable_json=False
        )

def get_logger(name: str) -> FrameworkLogger:
    """
    Get a logger instance for the specified name.
    
    Args:
        name: Logger name (typically module name)
        
    Returns:
        FrameworkLogger: Logger instance
    """
    if _global_logger is None:
        setup_logging()
    
    return FrameworkLogger(
        name=name,
        log_level=_global_logger.log_level if _global_logger else LogLevel.INFO,
        log_dir=str(_global_logger.log_dir) if _global_logger else None,
        enable_console=_global_logger.enable_console if _global_logger else True,
        enable_file=_global_logger.enable_file if _global_logger else True,
        enable_json=_global_logger.enable_json if _global_logger else False
    )

def log_function_call(func):
    """
    Decorator to log function calls with parameters and execution time.
    
    Args:
        func: Function to be decorated
        
    Returns:
        Wrapped function with logging
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = datetime.now()
        
        try:
            logger.debug(f"Calling function: {func.__name__} with args: {args[:3]}... kwargs: {list(kwargs.keys())}")
            result = func(*args, **kwargs)
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.log_performance(f"{func.__name__}", duration, True)
            
            return result
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.log_performance(f"{func.__name__}", duration, False)
            logger.error(f"Function {func.__name__} failed", e)
            raise
    
    return wrapper