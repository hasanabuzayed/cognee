"""The few cognee helpers the MCP layer uses, without the SDK in API mode.

Importing any ``cognee.*`` module runs ``cognee/__init__.py``, which loads the
whole SDK (litellm, SQLAlchemy, the database adapters...). In API mode this
server is a thin HTTP bridge that never calls the SDK, so those imports were
its entire memory footprint. API mode is decided by the same inputs ``main()``
reads (``--api-url`` / ``COGNEE_BASE_URL``, or ``API_URL`` as the entrypoint
passes it), because these names are bound at import time, before argparse runs.
Direct mode keeps the real helpers.
"""

import json
import logging
import os
import sys
from datetime import date, datetime
from decimal import Decimal
from functools import wraps
from uuid import UUID

API_MODE = bool(os.getenv("COGNEE_BASE_URL") or os.getenv("API_URL")) or any(
    arg == "--api-url" or arg.startswith("--api-url=") for arg in sys.argv
)

if API_MODE:
    REMEDIATION_MARKER = " Fix: "

    class JSONEncoder(json.JSONEncoder):
        """Same rules as ``cognee.modules.storage.utils.JSONEncoder``."""

        def default(self, obj):
            if isinstance(obj, (datetime, date)):
                return obj.isoformat()
            if isinstance(obj, UUID):
                return str(obj)
            if isinstance(obj, Decimal):
                return float(obj)
            return super().default(obj)

    def find_remediation(message: str) -> str | None:
        return None

    def get_logger(name=None, level=None) -> logging.Logger:
        logger = logging.getLogger(name or "cognee-mcp")
        if level is not None:
            logger.setLevel(level)
        return logger

    def setup_logging(log_level=None, name=None) -> logging.Logger:
        logging.basicConfig(level=log_level or logging.INFO, stream=sys.stderr)
        return get_logger(name)

    def get_log_file_location() -> str | None:
        return None

    def log_usage(function_name: str | None = None, log_type: str = "function"):
        """No-op: the API server records usage for the calls this bridge forwards."""

        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)

            return wrapper

        return decorator

else:
    from cognee.modules.storage.utils import JSONEncoder
    from cognee.shared.logging_utils import (
        get_log_file_location,
        get_logger,
        setup_logging,
    )
    from cognee.shared.usage_logger import log_usage

    try:
        from cognee.exceptions.remediation import REMEDIATION_MARKER, find_remediation
    except ImportError:  # cognee-mcp pins a released cognee; older cores have no hint table
        REMEDIATION_MARKER = " Fix: "

        def find_remediation(message: str) -> str | None:
            return None
