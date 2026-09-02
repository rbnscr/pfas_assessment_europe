from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime


def configure_logging(
    log_dir: str | Path = "logs",
    level: int = logging.INFO,
    naming: str = "",
) -> Path:
    """Configure console and file logging and return the log-file path."""

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if naming == "":
        log_path = log_dir / f"{timestamp}.log"
    else:
        log_path = log_dir / f"{naming}.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Prevent duplicate messages if configuration is called more than once.
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return log_path
