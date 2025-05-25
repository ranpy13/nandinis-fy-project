import logging
import logging.config
import yaml

def setup_logger(config_path="./utils/logging.yaml", logger_name="root") -> logging.Logger:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)

    return logging.getLogger(logger_name)
