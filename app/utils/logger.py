import logging
import sys
import json
from typing import Optional
from colorama import init, Fore, Style

# Initialize colorama for cross-platform color support
init()

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors"""
    
    COLORS = {
        'DEBUG': Fore.BLUE,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }

    def _format_json(self, data: str) -> str:
        """Format JSON string with indentation and colors"""
        try:
            # Try to parse as JSON
            if isinstance(data, str) and (data.startswith('{') or data.startswith('[')):
                parsed = json.loads(data)
                return '\n' + json.dumps(parsed, indent=2)
            return data
        except:
            return data

    def format(self, record):
        # Add color to the level name
        level_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{level_color}{record.levelname}{Style.RESET_ALL}"
        
        # Add a subtle color to the timestamp
        record.asctime = f"{Fore.CYAN}{self.formatTime(record)}{Style.RESET_ALL}"
        
        # Add color to the logger name
        record.name = f"{Fore.MAGENTA}{record.name}{Style.RESET_ALL}"

        # Color and format the message based on log level
        try:
            # If message is a string representation of dict/list, format it
            if isinstance(record.msg, (dict, list)):
                record.msg = json.dumps(record.msg, indent=2)
            
            # Format any JSON strings in the message
            message = str(record.msg)
            message = self._format_json(message)
            
            # Add color to the message based on log level
            record.msg = f"{level_color}{message}{Style.RESET_ALL}"
        except:
            # If any error occurs during message formatting, use original message
            pass

        return super().format(record)


def setup_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Setup a logger with custom formatting and colors
    """
    logger = logging.getLogger(name)
    
    if level:
        logger.setLevel(level)
    else:
        logger.setLevel(logging.DEBUG)  # Set default level to DEBUG
    
    # Create console handler if not exists
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        
        # Create colored formatter
        formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger