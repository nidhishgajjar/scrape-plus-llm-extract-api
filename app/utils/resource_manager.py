"""
Resource management utilities to prevent OOM and ensure stability
"""
import asyncio
import psutil
import os
from typing import Optional
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class ResourceManager:
    """Manages system resources to prevent crashes"""
    
    def __init__(self):
        self.max_concurrent_browsers = self._calculate_max_browsers()
        self.browser_semaphore = asyncio.Semaphore(self.max_concurrent_browsers)
        self.request_semaphore = asyncio.Semaphore(self.max_concurrent_browsers * 2)  # Allow 2x requests
        self.memory_threshold_percent = 80  # Alert at 80% memory usage
        
    def _calculate_max_browsers(self) -> int:
        """Calculate safe number of concurrent browsers based on available memory"""
        try:
            # Get total memory in GB
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            
            # Each browser needs ~800MB-1GB, be conservative
            # Reserve 1.5GB for system and other processes
            available_for_browsers = max(0, total_memory_gb - 1.5)
            max_browsers = max(1, int(available_for_browsers / 1.0))
            
            # Cap at reasonable limits
            if total_memory_gb <= 4:
                max_browsers = min(max_browsers, 2)  # Max 2 for small instances
            elif total_memory_gb <= 8:
                max_browsers = min(max_browsers, 4)  # Max 4 for medium instances
            else:
                max_browsers = min(max_browsers, 6)  # Max 6 for large instances
                
            logger.info(f"Resource Manager: {total_memory_gb:.1f}GB RAM detected, allowing {max_browsers} concurrent browsers")
            return max_browsers
        except Exception as e:
            logger.error(f"Error calculating max browsers: {e}, defaulting to 2")
            return 2
    
    def check_memory_usage(self) -> tuple[float, bool]:
        """Check current memory usage and return (percentage, is_safe)"""
        try:
            memory = psutil.virtual_memory()
            percent = memory.percent
            available_gb = memory.available / (1024**3)
            
            is_safe = percent < self.memory_threshold_percent and available_gb > 0.5
            
            if not is_safe:
                logger.warning(f"Memory pressure detected: {percent:.1f}% used, {available_gb:.2f}GB available")
            
            return percent, is_safe
        except Exception as e:
            logger.error(f"Error checking memory: {e}")
            return 100.0, False
    
    async def acquire_browser_slot(self, timeout: float = 60.0) -> bool:
        """Try to acquire a slot for browser launch - will wait in queue"""
        # First check memory
        _, is_safe = self.check_memory_usage()
        if not is_safe:
            logger.warning("Memory pressure detected, waiting for resources...")
            # Wait a bit for memory to free up
            await asyncio.sleep(2)
        
        # Wait in queue for a browser slot (with longer timeout)
        try:
            logger.debug(f"Waiting for browser slot (queue size: {self.max_concurrent_browsers - self.browser_semaphore._value})")
            await asyncio.wait_for(self.browser_semaphore.acquire(), timeout=timeout)
            logger.debug("Browser slot acquired")
            return True
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for browser slot after {timeout}s")
            return False
    
    def release_browser_slot(self):
        """Release a browser slot"""
        try:
            self.browser_semaphore.release()
        except ValueError:
            pass  # Semaphore already at max
    
    async def acquire_request_slot(self, timeout: float = 120.0) -> bool:
        """Try to acquire a slot for incoming request - will wait in queue"""
        try:
            queue_position = (self.max_concurrent_browsers * 2) - self.request_semaphore._value
            if queue_position > 0:
                logger.info(f"Request queued at position {queue_position}")
            await asyncio.wait_for(self.request_semaphore.acquire(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.warning(f"Request timeout after {timeout}s in queue")
            return False
    
    def release_request_slot(self):
        """Release a request slot"""
        try:
            self.request_semaphore.release()
        except ValueError:
            pass
    
    def get_status(self) -> dict:
        """Get current resource status"""
        memory = psutil.virtual_memory()
        return {
            "memory_percent": memory.percent,
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "active_browsers": self.max_concurrent_browsers - self.browser_semaphore._value,
            "max_browsers": self.max_concurrent_browsers,
            "active_requests": (self.max_concurrent_browsers * 2) - self.request_semaphore._value,
            "max_requests": self.max_concurrent_browsers * 2,
            "cpu_percent": psutil.cpu_percent(interval=0.1)
        }

# Global instance
resource_manager = ResourceManager()