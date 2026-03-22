import sys
from loguru import logger as _logger
from config.settings import settings


_logger.remove()
_logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <blue>{name}:{function}:{line}</blue> | <b>{message}</b>",
    level="DEBUG" if settings.api.debug else "INFO",
    colorize=True,
)

logger = _logger

__all__ = ["logger"]

