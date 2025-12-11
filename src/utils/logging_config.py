from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = Path(os.getenv("LOG_DIR", Path(__file__).resolve().parent.parent / "logs"))
ERROR_LOG_FILE = LOG_DIR / os.getenv("ERROR_LOG_FILE", "error.log")


def setup_logger() -> None:
    """Configure loguru sinks and formats."""
    logger.remove()
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger.add(
        sys.stderr,
        level=LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )

    logger.add(
        LOG_DIR / "app.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )

    logger.add(
        ERROR_LOG_FILE,
        rotation="5 MB",
        retention="14 days",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )


setup_logger()

__all__ = ["logger"]

