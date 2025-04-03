import logging
import logging.config
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog
from asgi_correlation_id import correlation_id

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Receive, Scope, Send


def setup_logging():
    # use log level from web server
    webserver_levels = [
        logging.getLogger("gunicorn.error").level,
        logging.getLogger("uvicorn.error").level,
    ]
    relevant_levels = [logging.INFO] + [
        webserver_level
        for webserver_level in webserver_levels
        if webserver_level != logging.NOTSET
    ]

    effective_level = min(relevant_levels)

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": structlog.stdlib.ProcessorFormatter,
                    "processor": structlog.dev.ConsoleRenderer()
                    if effective_level <= logging.DEBUG
                    else structlog.processors.JSONRenderer(),
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": logging.getLevelName(effective_level),
                    "formatter": "json",
                },
            },
            "root": {
                "handlers": ["console"],
                "level": logging.getLevelName(effective_level),
            },
        }
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    if effective_level <= logging.DEBUG:
        # This httpx logger logs very low level stuff, which we don't want even in debug
        logging.getLogger("httpcore").setLevel(logging.INFO)


@dataclass
class RequestIdLoggingMiddleware:
    app: "ASGIApp"

    async def __call__(self, scope: "Scope", receive: "Receive", send: "Send") -> None:
        """Set the correlation ID for structlog"""
        structlog.contextvars.clear_contextvars()
        # These context vars will be added to all log entries emitted during the request
        request_id = correlation_id.get()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        await self.app(scope, receive, send)


@contextmanager
def log_time(
    logger: structlog.BoundLogger,
    *args,
    level="info",
    **kwargs,
):
    start_time = time.perf_counter()
    yield
    diff = time.perf_counter() - start_time
    getattr(logger, level)(*args, **kwargs, seconds=diff)
