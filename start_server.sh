#!/bin/bash

# Set environment variables
export DEBUG_MODE=false

# Start the server with appropriate timeout settings
exec gunicorn -c gunicorn_conf.py main:app 