import logging
import sys


class C:
    file = "blue"
    dir = "blue"


def configure_logger():
    class InfoFilter(logging.Filter):
        def filter(self, record):
            return record.levelno == logging.INFO

    info_handler = logging.StreamHandler(sys.stdout)
    info_handler.setLevel(logging.INFO)
    info_handler.addFilter(InfoFilter())

    class WarningFilter(logging.Filter):
        def filter(self, record):
            return record.levelno == logging.WARNING

    warning_handler = logging.StreamHandler(sys.stdout)
    warning_handler.setLevel(logging.WARNING)
    warning_handler.addFilter(WarningFilter())

    class ErrorFilter(logging.Filter):
        def filter(self, record):
            return record.levelno == logging.ERROR

    error_handler = logging.StreamHandler(sys.stderr)
    error_handler.setLevel(logging.ERROR)
    error_handler.addFilter(ErrorFilter())

    format = "[%(asctime)s %(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [
        info_handler,
        warning_handler,
        error_handler
    ]

    logging.basicConfig(
        level=logging.INFO,
        format=format,
        datefmt=datefmt,
        handlers=handlers
    )
