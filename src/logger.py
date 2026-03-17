"""src/logger.py"""
import logging, os
from logging.handlers import RotatingFileHandler

def setup_logger(log_dir: str = "logs", level: int = logging.INFO) -> None:
    os.makedirs(log_dir, exist_ok=True)
    fmt  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    root = logging.getLogger()
    root.setLevel(level)
    if root.handlers:
        return
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(fmt, "%Y-%m-%d %H:%M:%S"))
    fh = RotatingFileHandler(
        os.path.join(log_dir, "pipeline.log"), maxBytes=5*1024*1024, backupCount=3
    )
    fh.setFormatter(logging.Formatter(fmt, "%Y-%m-%d %H:%M:%S"))
    root.addHandler(ch)
    root.addHandler(fh)
