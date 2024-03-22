import logging

def get_logger(filename, verbosity=1, name=None):
    """write log"""
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )

    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    # file_handler: write log file
    file_handler = logging.FileHandler(filename, "w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # stream_handler: output log in the terminal
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger