import logging
from .config import Config

def setup_logger(name: str) -> logging.Logger:
    """Set up logger for the application."""
    logger = logging.getLogger(name)
    logger.setLevel(Config.LOG_LEVEL)  # Use the log level from the config
    
    ch = logging.StreamHandler()  # Log to console
    ch.setLevel(Config.LOG_LEVEL)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    
    logger.addHandler(ch)
    
    return logger
