import logging
import os

# Create a 'logs' directory if it doesn't exist
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

def get_logger(module_name):
    """Returns a configured logger instance."""
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate logs if the logger is already configured
    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # 1. File Handler (saves logs to a file)
        file_handler = logging.FileHandler(os.path.join(log_dir, 'pipeline.log'))
        file_handler.setFormatter(formatter)

        # 2. Stream Handler (prints logs to terminal)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger