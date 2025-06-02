# Adapted from
# https://github.com/skypilot-org/skypilot/blob/86dc0f6283a335e4aa37b3c10716f90999f48ab6/sky/sky_logging.py
"""Logging configuration for vLLM."""
import logging
import sys

_FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"


class NewLineFormatter(logging.Formatter):
    """Adds logging prefix to newlines to align multi-line messages."""

    def __init__(self, fmt, datefmt=None):
        super().__init__(fmt, datefmt)
        import socket
        self.hostname = socket.gethostname()

    def format(self, record):
        import torch.distributed as dist
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = "NA"

        # Add custom prefix to the message
        original_msg = record.msg
        import os
        pid = os.getpid()
        record.msg = f"[{self.hostname}][{pid}][r{rank}] {original_msg}"
        msg = super().format(record)
        record.msg = original_msg  # Restore the original message

        # Align multi-line messages
        if original_msg != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg


# Load logging configuration from YAML
import yaml
import logging.config
with open('logging_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    logging.config.dictConfig(config)


_root_logger = logging.getLogger("openrlhf")
_default_handler = None

file_handler = None


def _setup_logger():
    _root_logger.setLevel(logging.DEBUG)

    global _default_handler
    if _default_handler is None:
        _default_handler = logging.StreamHandler(sys.stdout)
        _default_handler.flush = sys.stdout.flush  # type: ignore
        _default_handler.setLevel(logging.INFO)
        _root_logger.addHandler(_default_handler)
    fmt = NewLineFormatter(_FORMAT, datefmt=_DATE_FORMAT)

    _default_handler.setFormatter(fmt)

    # Setting this will avoid the message
    # being propagated to the parent logger.
    _root_logger.propagate = False

# The logger is initialized when the module is imported.
# This is thread-safe as the module is only imported once,
# guaranteed by the Python GIL.
_setup_logger()


fmt = NewLineFormatter("[%(levelname)s %(asctime)s.%(msecs)03d %(filename)s:%(lineno)d] %(message)s", datefmt="%m-%d %H:%M:%S")

for handler in _root_logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        default_handler = handler
        default_handler.setFormatter(fmt)
    elif isinstance(handler, logging.FileHandler):
        file_handler = handler
        file_handler.setFormatter(fmt)


def init_logger(name: str):
    global _default_handler
    global file_handler

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if _default_handler:
        logger.addHandler(_default_handler)
    if file_handler:
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger

