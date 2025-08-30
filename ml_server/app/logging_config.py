import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
import os
from typing import Optional


def _level_from_env(default: str = "INFO") -> int:
    lvl = os.getenv("LOG_LEVEL", default).upper()
    return getattr(logging, lvl, logging.INFO)


def _make_formatter() -> logging.Formatter:
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S%z"
    return logging.Formatter(fmt=fmt, datefmt=datefmt)


def _handler_identity(h: logging.Handler) -> tuple:
    """Identify a handler by its class and target file (if any) to avoid duplicates."""
    filename: Optional[str] = None
    if isinstance(h, (logging.FileHandler, TimedRotatingFileHandler)):
        filename = getattr(h, "baseFilename", None)
    return (h.__class__.__name__, filename)


def setup_logging() -> None:
    """
    Configure logging:
    - logs/app.log: TimedRotatingFileHandler, rotate at local midnight, keep 365 backups.
    - logs/error.log: FileHandler for ERROR+ from all loggers (non-rotating).
    - StreamHandler to console for app logs.

    Idempotent: safe to call multiple times (e.g., under uvicorn --reload) without duplicating handlers.
    """
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    app_log_path = log_dir / "app.log"
    error_log_path = log_dir / "error.log"

    level = _level_from_env("INFO")
    formatter = _make_formatter()

    # Handlers
    rotating = TimedRotatingFileHandler(
        app_log_path, when="midnight", interval=1, backupCount=365, encoding="utf-8"
    )
    rotating.setFormatter(formatter)
    rotating.setLevel(level)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(level)

    error_file = logging.FileHandler(error_log_path, encoding="utf-8")
    error_file.setFormatter(formatter)
    error_file.setLevel(logging.ERROR)

    # Application logger
    app_logger = logging.getLogger("ml_server")
    app_logger.setLevel(level)

    # Ensure idempotency by comparing handler identities
    existing_identities = {_handler_identity(h) for h in app_logger.handlers}
    for h in (rotating, console, error_file):
        if _handler_identity(h) not in existing_identities:
            app_logger.addHandler(h)
            existing_identities.add(_handler_identity(h))

    # Avoid double-propagation to root; we already added our own console/file handlers
    app_logger.propagate = False

    # Root logger gets the error file handler so that ERROR+ from all loggers are captured
    root_logger = logging.getLogger()
    root_existing = {_handler_identity(h) for h in root_logger.handlers}
    if _handler_identity(error_file) not in root_existing:
        # Attach a separate instance to the root to avoid shared handler state
        root_error_file = logging.FileHandler(error_log_path, encoding="utf-8")
        root_error_file.setFormatter(formatter)
        root_error_file.setLevel(logging.ERROR)
        root_logger.addHandler(root_error_file)

    # Do not change root level; let other frameworks manage their own levels.
