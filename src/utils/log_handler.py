import logging


class LogFormatHandler(logging.StreamHandler):
    # https://stackoverflow.com/questions/15870380/python-custom-logging-across-all-modules
    def __init__(self):
        logging.StreamHandler.__init__(self)
        # fmt = '[%(levelname)-2s%(asctime)s %(filename)-12s]: %(message)s'
        # fmt_date = '%Y-%m-%dT%T%Z'
        # formatter = logging.Formatter(fmt, fmt_date)
        self.setFormatter(ColorFormatter())


class ColorFormatter(logging.Formatter):
    # https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
    # https://github.com/herzog0/best_python_logger
    green = "\x1b[1;32m"
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "[%(asctime)s  - %(levelname)s - (%(filename)s:%(lineno)d)]: %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
