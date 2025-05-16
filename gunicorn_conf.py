import multiprocessing
import os

# Server socket settings
bind = "0.0.0.0:10000"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000

# Disable user/group settings for containerized environments like Render
user = None
group = None

# Necessary for Render deployment
umask = 0o000

# Timeout settings (solving worker timeout issues)
timeout = 300          # Worker timeout increased to 5 minutes
graceful_timeout = 120 # Grace period for workers to finish
keepalive = 120        # Keep-alive timeout

# Logging
errorlog = "-"
loglevel = "info"
accesslog = "-"
access_log_format = "%({X-Real-IP}i)s %(h)s %(l)s %(u)s %(t)s \"%(r)s\" %(s)s %(b)s \"%(f)s\" \"%(a)s\""

# Process naming
proc_name = "scrape-plus-llm-extract-api"

# Server mechanics
daemon = False