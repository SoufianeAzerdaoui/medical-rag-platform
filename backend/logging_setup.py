from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from backend import config

_CONFIGURED = False


def _handler_exists(logger: logging.Logger, marker: str) -> bool:
    return any(getattr(handler, "_medical_rag_marker", "") == marker for handler in logger.handlers)


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    logger = logging.getLogger("medical_rag")
    level = getattr(logging, str(config.LOG_LEVEL or "INFO").upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    stream_marker = "medical_rag_stream"
    file_marker = "medical_rag_file"
    if not _handler_exists(logger, stream_marker):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        stream_handler._medical_rag_marker = stream_marker  # type: ignore[attr-defined]
        logger.addHandler(stream_handler)

    if not _handler_exists(logger, file_marker):
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            config.LOGS_DIR / config.LOG_FILE_NAME,
            maxBytes=max(1024 * 1024, int(config.LOG_MAX_BYTES)),
            backupCount=max(1, int(config.LOG_BACKUP_COUNT)),
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        file_handler._medical_rag_marker = file_marker  # type: ignore[attr-defined]
        logger.addHandler(file_handler)

    _CONFIGURED = True
