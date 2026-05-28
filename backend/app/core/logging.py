import logging
import sys

from app.core.config import Settings


def configure_logging(settings: Settings) -> None:
    """Configure process-wide logging.

    Production deployments can switch to JSON logs by plugging a structured
    formatter here without changing application code.
    """
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
