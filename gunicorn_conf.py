import multiprocessing
import os
import sys
import psutil

def get_workers_count():
    try:
        # Get available memory in GB
        available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)
        
        # For 1 CPU / 4GB RAM setup with browser automation
        # Each worker needs ~1.2-1.5GB for browser instances
        # Leave 1.5GB for system/OS
        memory_based_workers = max(1, int((available_memory - 1.5) / 1.2))
        
        # CPU-based: Conservative for single CPU to avoid context switching overhead
        cpu_count = multiprocessing.cpu_count()
        if cpu_count == 1:
            cpu_based_workers = 2  # Max 2 workers for single CPU
        else:
            cpu_based_workers = min(cpu_count + 1, 4)  # Conservative scaling
        
        # Return the lower of the two to respect both constraints
        optimal_workers = min(memory_based_workers, cpu_based_workers)
        
        return max(1, optimal_workers)  # Ensure at least 1 worker
    except:
        # Safe default for 4GB/1CPU setup
        return 2


# Worker processes - dynamic based on available resources
workers = get_workers_count()
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000  # Reduced from 3000 for memory efficiency

# Disable user/group settings for containerized environments
user = None
group = None

# Necessary for deployment
umask = 0o000

# Worker restart settings - disabled to prevent mid-operation restarts
max_requests = 300  # Disabled - prevents "browser context closed" errors
max_requests_jitter = 100

# Timeout settings optimized for browser automation
timeout = 600          # 10 minutes - enough for complex scraping
graceful_timeout = 120 # 2 minutes grace period
keepalive = 120        # 2 minutes keep-alive

# Logging
errorlog = "-"
loglevel = "info"
accesslog = "-"
access_log_format = "%({X-Real-IP}i)s %(h)s %(l)s %(u)s %(t)s \"%(r)s\" %(s)s %(b)s \"%(f)s\" \"%(a)s\""

# Process naming
proc_name = "scrape-plus-llm-extract-api"

# Server mechanics
daemon = False

# Log the number of workers being used
print(f"Starting with {workers} workers based on available resources")