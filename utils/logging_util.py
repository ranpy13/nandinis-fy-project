import logging
import os

def get_logger(name: str, log_to_file: bool = False, log_dir: str = "logs") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s %(filename)s#%(funcName)s - [%(levelname)s] = %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # Optional: File handler
        if log_to_file:
            os.makedirs(log_dir, exist_ok=True)
            file_path = os.path.join(log_dir, f"{name}.log")

            fh = logging.FileHandler(file_path)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        logger.propagate = False  # Avoid duplicate logs
    return logger
