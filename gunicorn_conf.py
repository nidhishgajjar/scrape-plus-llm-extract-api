import multiprocessing
import os

# Server socket settings
bind = "0.0.0.0:10000"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000

# Timeout settings (more balanced values)
timeout = 300          # Reduced from 1000 to 300 seconds (5 minutes)
graceful_timeout = 300  # Reduced from 1000 to 300 seconds
keepalive = 120         # Reduced from 1000 to 120 seconds

# Performance settings
max_requests = 1000     # Restart workers after handling this many requests
max_requests_jitter = 200  # Add randomness to prevent all workers restarting simultaneously

# Security settings
umask = 0o027          # Set file creation mask
user = "www-data"       # Run workers as this user
group = "www-data"      # Run workers as this group

# Logging
errorlog = "-"
loglevel = "info"
accesslog = "-"
access_log_format = "%({X-Real-IP}i)s %(h)s %(l)s %(u)s %(t)s \"%(r)s\" %(s)s %(b)s \"%(f)s\" \"%(a)s\""

# Process naming
proc_name = "scrape-plus-llm-extract-api"

# Server mechanics
daemon = False