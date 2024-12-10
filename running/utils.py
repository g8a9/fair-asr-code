import logging

logger = logging.getLogger(__name__)


def log_arguments(func):
    def wrapper(*args, **kwargs):
        logger.info(f"Calling function '{func.__name__}' with arguments:")
        for i, arg in enumerate(args):
            logger.info(f"arg[{i}]: {arg}")
        for key, value in kwargs.items():
            logger.info(f"{key}: {value}")
        return func(*args, **kwargs)

    return wrapper
