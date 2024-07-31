import logging
import sys

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            # "format": "%(levelprefix)s %(asctime)s - %(name)s - %(threadName)s - %(message)s",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(message)s",
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
        },
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["default"],
            "level": "DEBUG",
        },
    },
}

# logging.config.dictConfig(LOGGING_CONFIG)
