# Worker processes - dynamic based on available resources
workers = 5
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