# =========================
# Temporary File Management
# =========================
import os
import tempfile
import shutil
import logging
import atexit
from typing import List, Optional
import glob

# Setup logger
logger = logging.getLogger(__name__)

class TempFileManager:
    """Centralized temporary file management with automatic cleanup"""
    
    def __init__(self):
        self.temp_files: List[str] = []
        self.temp_dirs: List[str] = []
        # Register cleanup on exit
        atexit.register(self.cleanup_all)
    
    def create_temp_file(self, suffix: str = "", prefix: str = "trading_", dir: Optional[str] = None) -> str:
        """Create a temporary file and track it for cleanup"""
        try:
            fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=dir)
            os.close(fd)  # Close the file descriptor
            self.temp_files.append(path)
            logger.debug(f"Created temp file: {path}")
            return path
        except Exception as e:
            logger.error(f"Error creating temp file: {e}")
            raise
    
    def create_temp_dir(self, prefix: str = "trading_") -> str:
        """Create a temporary directory and track it for cleanup"""
        try:
            path = tempfile.mkdtemp(prefix=prefix)
            self.temp_dirs.append(path)
            logger.debug(f"Created temp dir: {path}")
            return path
        except Exception as e:
            logger.error(f"Error creating temp dir: {e}")
            raise
    
    def create_chart_file(self, ticker: str, base_dir: Optional[str] = None) -> str:
        """Create a temporary chart file with standardized naming"""
        if base_dir is None:
            base_dir = self.create_temp_dir("charts_")
        
        import pandas as pd
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{ticker}_chart_{timestamp}.png"
        path = os.path.join(base_dir, filename)
        
        # Track for cleanup
        self.temp_files.append(path)
        logger.debug(f"Created chart file path: {path}")
        return path
    
    def cleanup_file(self, file_path: str) -> bool:
        """Safely remove a specific file"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Removed temp file: {file_path}")
                # Remove from tracking list
                if file_path in self.temp_files:
                    self.temp_files.remove(file_path)
                return True
            return False
        except Exception as e:
            logger.warning(f"Error removing file {file_path}: {e}")
            return False
    
    def cleanup_dir(self, dir_path: str) -> bool:
        """Safely remove a directory and all its contents"""
        try:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                logger.debug(f"Removed temp dir: {dir_path}")
                # Remove from tracking list
                if dir_path in self.temp_dirs:
                    self.temp_dirs.remove(dir_path)
                return True
            return False
        except Exception as e:
            logger.warning(f"Error removing directory {dir_path}: {e}")
            return False
    
    def cleanup_old_files(self, pattern: str = "trading_*", max_age_hours: int = 24):
        """Clean up old temporary files based on pattern and age"""
        try:
            import time
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            # Search in system temp directory
            temp_dir = tempfile.gettempdir()
            old_files = glob.glob(os.path.join(temp_dir, pattern))
            
            removed_count = 0
            for file_path in old_files:
                try:
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > max_age_seconds:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                        removed_count += 1
                        logger.debug(f"Removed old temp file: {file_path}")
                except Exception as e:
                    logger.warning(f"Error removing old file {file_path}: {e}")
            
            if removed_count > 0:
                logger.info(f"🧹 Cleaned up {removed_count} old temporary files")
                
        except Exception as e:
            logger.error(f"Error during old file cleanup: {e}")
    
    def cleanup_all(self):
        """Clean up all tracked temporary files and directories"""
        cleaned_files = 0
        cleaned_dirs = 0
        
        # Clean up files
        for file_path in self.temp_files.copy():
            if self.cleanup_file(file_path):
                cleaned_files += 1
        
        # Clean up directories
        for dir_path in self.temp_dirs.copy():
            if self.cleanup_dir(dir_path):
                cleaned_dirs += 1
        
        if cleaned_files > 0 or cleaned_dirs > 0:
            logger.info(f"🧹 Cleanup complete: {cleaned_files} files, {cleaned_dirs} directories")
    
    def get_stats(self) -> dict:
        """Get statistics about tracked temporary files"""
        return {
            "tracked_files": len(self.temp_files),
            "tracked_dirs": len(self.temp_dirs),
            "existing_files": len([f for f in self.temp_files if os.path.exists(f)]),
            "existing_dirs": len([d for d in self.temp_dirs if os.path.exists(d)])
        }

# Global instance for use across the application
temp_manager = TempFileManager()

# Convenience functions
def create_temp_chart_file(ticker: str) -> str:
    """Create a temporary chart file"""
    return temp_manager.create_chart_file(ticker)

def cleanup_temp_files():
    """Clean up all temporary files"""
    temp_manager.cleanup_all()

def cleanup_old_temp_files():
    """Clean up old temporary files"""
    temp_manager.cleanup_old_files()
