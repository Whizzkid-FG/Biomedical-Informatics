# File: utils/logger.py
import logging
from logging.handlers import RotatingFileHandler
from config.config import Config
import os

def setup_logger(name: str) -> logging.Logger:
    """Configure and return a logger instance with rotating file handler."""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
        
    logger = logging.getLogger(name)
    logger.setLevel(Config.LOG_LEVEL)
    
    # Create handlers
    file_handler = RotatingFileHandler(
        f'logs/{name}.log',
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    console_handler = logging.StreamHandler()
    
    # Create formatters and add it to handlers
    formatter = logging.Formatter(Config.LOG_FORMAT)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger