import logging
from tqdm.auto import tqdm


class TqdmLoggingHandler(logging.Handler):
    """
    Custom logging handler that directs standard log messages 
    through tqdm.write() to prevent progress bar corruption.
    """
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logger(name: str = __name__) -> logging.Logger:
    """
    Configures and returns a logger equipped with the TqdmLoggingHandler.
    """
    custom_logger = logging.getLogger(name)
    custom_logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if the module is imported multiple times
    if not custom_logger.hasHandlers():
        handler = TqdmLoggingHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        handler.setFormatter(formatter)
        custom_logger.addHandler(handler)
        
        # Prevent log propagation to the root logger to avoid duplicate standard console prints
        custom_logger.propagate = False
        
    return custom_logger


# Instantiate the logger to be imported across the project
logger = setup_logger("imaging_classifier")