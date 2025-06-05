"""
TTS/STT Testing Framework - File Utilities
=========================================

This module provides comprehensive file handling utilities for the TTS/STT testing framework.
It includes file validation, path management, data persistence, and backup functionality.

Author: TTS/STT Testing Framework Team
Version: 1.0.0
Created: 2024-06-04
"""

import os
import shutil
import json
import yaml
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import hashlib
import tempfile
import zipfile
import mimetypes

from .logger import get_logger, log_function_call

class FileManager:
    """
    Comprehensive file management class for the TTS/STT testing framework.
    
    Features:
    - File validation and type checking
    - Data persistence in multiple formats
    - Backup and recovery functionality
    - Path management and directory operations
    - File integrity verification
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize the file manager.
        
        Args:
            base_dir: Base directory for file operations
        """
        self.logger = get_logger(__name__)
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.supported_formats = {
            'json': ['.json'],
            'yaml': ['.yaml', '.yml'],
            'csv': ['.csv'],
            'text': ['.txt', '.log'],
            'audio': ['.wav', '.mp3', '.flac', '.ogg', '.m4a'],
            'config': ['.env', '.ini', '.cfg', '.conf']
        }
        
        self.logger.info(f"FileManager initialized with base directory: {self.base_dir}")
    
    @log_function_call
    def validate_file_path(self, file_path: Union[str, Path], must_exist: bool = True) -> bool:
        """
        Validate file path and check existence.
        
        Args:
            file_path: Path to the file
            must_exist: Whether the file must exist
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            path = Path(file_path)
            
            # Check if path is absolute or relative to base_dir
            if not path.is_absolute():
                path = self.base_dir / path
            
            # Check existence if required
            if must_exist and not path.exists():
                self.logger.error(f"File does not exist: {path}")
                return False
            
            # Check if parent directory exists
            if not path.parent.exists():
                self.logger.error(f"Parent directory does not exist: {path.parent}")
                return False
            
            # Check permissions
            if path.exists():
                if not os.access(path, os.R_OK):
                    self.logger.error(f"File is not readable: {path}")
                    return False
            
            self.logger.debug(f"File path validation successful: {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"File path validation failed: {file_path}", e)
            return False
    
    @log_function_call
    def ensure_directory(self, dir_path: Union[str, Path]) -> bool:
        """
        Ensure directory exists, create if necessary.
        
        Args:
            dir_path: Path to the directory
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            path = Path(dir_path)
            
            # Make absolute path if relative
            if not path.is_absolute():
                path = self.base_dir / path
            
            # Create directory if it doesn't exist
            path.mkdir(parents=True, exist_ok=True)
            
            # Verify directory creation
            if not path.exists() or not path.is_dir():
                self.logger.error(f"Failed to create directory: {path}")
                return False
            
            self.logger.debug(f"Directory ensured: {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to ensure directory: {dir_path}", e)
            return False
    
    @log_function_call
    def save_data(
        self,
        data: Any,
        file_path: Union[str, Path],
        format_type: str = 'auto',
        backup: bool = True,
        indent: int = 2
    ) -> bool:
        """
        Save data to file in specified format.
        
        Args:
            data: Data to save
            file_path: Path to save the file
            format_type: Format type ('json', 'yaml', 'csv', 'auto')
            backup: Create backup if file exists
            indent: Indentation for formatted output
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            path = Path(file_path)
            
            # Make absolute path if relative
            if not path.is_absolute():
                path = self.base_dir / path
            
            # Ensure parent directory exists
            if not self.ensure_directory(path.parent):
                return False
            
            # Auto-detect format from extension
            if format_type == 'auto':
                format_type = self._detect_format(path)
            
            # Create backup if requested and file exists
            if backup and path.exists():
                if not self._create_backup(path):
                    self.logger.warning(f"Failed to create backup for: {path}")
            
            # Save data based on format
            success = False
            if format_type == 'json':
                success = self._save_json(data, path, indent)
            elif format_type == 'yaml':
                success = self._save_yaml(data, path, indent)
            elif format_type == 'csv':
                success = self._save_csv(data, path)
            elif format_type == 'text':
                success = self._save_text(data, path)
            else:
                self.logger.error(f"Unsupported format type: {format_type}")
                return False
            
            if success:
                self.logger.info(f"Data saved successfully: {path}")
                return True
            else:
                self.logger.error(f"Failed to save data: {path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to save data to {file_path}", e)
            return False
    
    @log_function_call
    def load_data(
        self,
        file_path: Union[str, Path],
        format_type: str = 'auto',
        encoding: str = 'utf-8'
    ) -> Optional[Any]:
        """
        Load data from file.
        
        Args:
            file_path: Path to the file
            format_type: Format type ('json', 'yaml', 'csv', 'auto')
            encoding: File encoding
            
        Returns:
            Optional[Any]: Loaded data or None if failed
        """
        try:
            path = Path(file_path)
            
            # Make absolute path if relative
            if not path.is_absolute():
                path = self.base_dir / path
            
            # Validate file exists
            if not self.validate_file_path(path, must_exist=True):
                return None
            
            # Auto-detect format from extension
            if format_type == 'auto':
                format_type = self._detect_format(path)
            
            # Load data based on format
            data = None
            if format_type == 'json':
                data = self._load_json(path, encoding)
            elif format_type == 'yaml':
                data = self._load_yaml(path, encoding)
            elif format_type == 'csv':
                data = self._load_csv(path, encoding)
            elif format_type == 'text':
                data = self._load_text(path, encoding)
            else:
                self.logger.error(f"Unsupported format type: {format_type}")
                return None
            
            if data is not None:
                self.logger.debug(f"Data loaded successfully: {path}")
                return data
            else:
                self.logger.error(f"Failed to load data: {path}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to load data from {file_path}", e)
            return None
    
    @log_function_call
    def get_file_info(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive file information.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Optional[Dict[str, Any]]: File information or None if failed
        """
        try:
            path = Path(file_path)
            
            # Make absolute path if relative
            if not path.is_absolute():
                path = self.base_dir / path
            
            if not path.exists():
                self.logger.error(f"File does not exist: {path}")
                return None
            
            stat = path.stat()
            
            file_info = {
                'path': str(path),
                'name': path.name,
                'stem': path.stem,
                'suffix': path.suffix,
                'size_bytes': stat.st_size,
                'size_human': self._format_size(stat.st_size),
                'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'accessed': datetime.fromtimestamp(stat.st_atime).isoformat(),
                'is_file': path.is_file(),
                'is_dir': path.is_dir(),
                'is_symlink': path.is_symlink(),
                'permissions': oct(stat.st_mode)[-3:],
                'mime_type': mimetypes.guess_type(str(path))[0],
                'format_type': self._detect_format(path),
                'checksum_md5': self._calculate_checksum(path, 'md5') if path.is_file() else None
            }
            
            self.logger.debug(f"File info retrieved: {path}")
            return file_info
            
        except Exception as e:
            self.logger.error(f"Failed to get file info for {file_path}", e)
            return None
    
    @log_function_call
    def cleanup_old_files(
        self,
        directory: Union[str, Path],
        max_age_days: int = 30,
        pattern: str = "*",
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Clean up old files in a directory.
        
        Args:
            directory: Directory to clean
            max_age_days: Maximum age in days
            pattern: File pattern to match
            dry_run: If True, only simulate cleanup
            
        Returns:
            Dict[str, Any]: Cleanup results
        """
        try:
            dir_path = Path(directory)
            
            # Make absolute path if relative
            if not dir_path.is_absolute():
                dir_path = self.base_dir / dir_path
            
            if not dir_path.exists() or not dir_path.is_dir():
                self.logger.error(f"Directory does not exist: {dir_path}")
                return {'success': False, 'error': 'Directory not found'}
            
            cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 3600)
            files_to_delete = []
            total_size = 0
            
            # Find old files
            for file_path in dir_path.glob(pattern):
                if file_path.is_file():
                    stat = file_path.stat()
                    if stat.st_mtime < cutoff_time:
                        files_to_delete.append({
                            'path': str(file_path),
                            'size': stat.st_size,
                            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                        })
                        total_size += stat.st_size
            
            # Delete files if not dry run
            deleted_count = 0
            if not dry_run:
                for file_info in files_to_delete:
                    try:
                        Path(file_info['path']).unlink()
                        deleted_count += 1
                        self.logger.debug(f"Deleted old file: {file_info['path']}")
                    except Exception as e:
                        self.logger.error(f"Failed to delete file: {file_info['path']}", e)
            
            result = {
                'success': True,
                'directory': str(dir_path),
                'max_age_days': max_age_days,
                'pattern': pattern,
                'dry_run': dry_run,
                'files_found': len(files_to_delete),
                'files_deleted': deleted_count,
                'total_size_bytes': total_size,
                'total_size_human': self._format_size(total_size),
                'files': files_to_delete[:10]  # Limit to first 10 for brevity
            }
            
            self.logger.info(f"Cleanup completed: {deleted_count}/{len(files_to_delete)} files deleted")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old files in {directory}", e)
            return {'success': False, 'error': str(e)}
    
    def _detect_format(self, file_path: Path) -> str:
        """Detect file format from extension."""
        suffix = file_path.suffix.lower()
        
        for format_type, extensions in self.supported_formats.items():
            if suffix in extensions:
                return format_type
        
        return 'text'  # Default to text format
    
    def _create_backup(self, file_path: Path) -> bool:
        """Create backup of existing file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = file_path.with_suffix(f"{file_path.suffix}.backup_{timestamp}")
            shutil.copy2(file_path, backup_path)
            self.logger.debug(f"Backup created: {backup_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create backup for {file_path}", e)
            return False
    
    def _save_json(self, data: Any, path: Path, indent: int) -> bool:
        """Save data as JSON."""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save JSON: {path}", e)
            return False
    
    def _save_yaml(self, data: Any, path: Path, indent: int) -> bool:
        """Save data as YAML."""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, indent=indent, default_flow_style=False, allow_unicode=True)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save YAML: {path}", e)
            return False
    
    def _save_csv(self, data: Any, path: Path) -> bool:
        """Save data as CSV."""
        try:
            if isinstance(data, list) and data and isinstance(data[0], dict):
                # List of dictionaries
                with open(path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
                return True
            else:
                self.logger.error(f"Unsupported data type for CSV: {type(data)}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to save CSV: {path}", e)
            return False
    
    def _save_text(self, data: Any, path: Path) -> bool:
        """Save data as text."""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                if isinstance(data, str):
                    f.write(data)
                else:
                    f.write(str(data))
            return True
        except Exception as e:
            self.logger.error(f"Failed to save text: {path}", e)
            return False
    
    def _load_json(self, path: Path, encoding: str) -> Optional[Any]:
        """Load data from JSON."""
        try:
            with open(path, 'r', encoding=encoding) as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load JSON: {path}", e)
            return None
    
    def _load_yaml(self, path: Path, encoding: str) -> Optional[Any]:
        """Load data from YAML."""
        try:
            with open(path, 'r', encoding=encoding) as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load YAML: {path}", e)
            return None
    
    def _load_csv(self, path: Path, encoding: str) -> Optional[List[Dict[str, Any]]]:
        """Load data from CSV."""
        try:
            with open(path, 'r', newline='', encoding=encoding) as f:
                reader = csv.DictReader(f)
                return list(reader)
        except Exception as e:
            self.logger.error(f"Failed to load CSV: {path}", e)
            return None
    
    def _load_text(self, path: Path, encoding: str) -> Optional[str]:
        """Load data from text file."""
        try:
            with open(path, 'r', encoding=encoding) as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Failed to load text: {path}", e)
            return None
    
    def _calculate_checksum(self, file_path: Path, algorithm: str = 'md5') -> Optional[str]:
        """Calculate file checksum."""
        try:
            hash_func = hashlib.new(algorithm)
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_func.update(chunk)
            return hash_func.hexdigest()
        except Exception as e:
            self.logger.error(f"Failed to calculate checksum for {file_path}", e)
            return None
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"

# Convenience functions
def validate_file_path(file_path: Union[str, Path], must_exist: bool = True) -> bool:
    """Validate file path using default FileManager."""
    manager = FileManager()
    return manager.validate_file_path(file_path, must_exist)

def ensure_directory(dir_path: Union[str, Path]) -> bool:
    """Ensure directory exists using default FileManager."""
    manager = FileManager()
    return manager.ensure_directory(dir_path)

def save_json(data: Any, file_path: Union[str, Path], indent: int = 2) -> bool:
    """Save data as JSON using default FileManager."""
    manager = FileManager()
    return manager.save_data(data, file_path, 'json', indent=indent)

def load_json(file_path: Union[str, Path]) -> Optional[Any]:
    """Load data from JSON using default FileManager."""
    manager = FileManager()
    return manager.load_data(file_path, 'json')

def save_yaml(data: Any, file_path: Union[str, Path], indent: int = 2) -> bool:
    """Save data as YAML using default FileManager."""
    manager = FileManager()
    return manager.save_data(data, file_path, 'yaml', indent=indent)

def load_yaml(file_path: Union[str, Path]) -> Optional[Any]:
    """Load data from YAML using default FileManager."""
    manager = FileManager()
    return manager.load_data(file_path, 'yaml')