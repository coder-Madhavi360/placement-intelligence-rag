import json
import logging
import sys
from datetime import UTC, datetime

from core.config import Settings


class JsonFormatter(logging.Formatter):
    """Small JSON formatter for production log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(settings: Settings) -> None:
    """Configure process-wide logging.

    Production deployments can switch to JSON logs by plugging a structured
    formatter here without changing application code.
    """
    handler = logging.StreamHandler(sys.stdout)
    if settings.log_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))

    logging.basicConfig(
        level=settings.log_level,
        handlers=[handler],
        force=True,
    )
