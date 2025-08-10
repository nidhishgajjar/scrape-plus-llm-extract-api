import multiprocessing
import os
import sys
import psutil

# Function to calculate optimal workers based on available memory
def get_workers_count():
    try:
        # Get available memory in GB
        available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)
        
        # Memory-based scaling: each worker needs ~500MB, leave 1GB for system
        memory_based_workers = max(1, int((available_memory - 1) / 0.5))
        
        # CPU-based scaling: typical formula
        cpu_based_workers = multiprocessing.cpu_count() * 2 + 1
        
        # Return the lower of the two to respect both CPU and memory constraints
        return min(memory_based_workers, cpu_based_workers)
    except:
        # Default to 7 workers if we can't calculate
        return 7


# Worker processes - dynamic based on available resources
workers = get_workers_count()
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 3000

# Disable user/group settings for containerized environments like Render
user = None
group = None

# Necessary for Render deployment
umask = 0o000

# Worker restart settings to manage memory over time
max_requests = 1000
max_requests_jitter = 100

# Timeout settings (solving worker timeout issues)
timeout = 600          # Worker timeout increased to 10 minutes
graceful_timeout = 300 # Grace period for workers to finish
keepalive = 300        # Keep-alive timeout

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