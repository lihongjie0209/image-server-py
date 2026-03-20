"""Structured JSON audit logging for the image service.

Design notes
────────────
- One JSON object per line → easy to pipe into jq / ELK / Cloud Logging.
- A 12-char hex request ID is set in a ContextVar by AuditMiddleware so
  every log line emitted within the same request carries the same ID.
- File bodies are **never** read or logged; only envelope metadata is
  captured (method, path, client IP, status, timing).
- Detailed per-endpoint data (file name, size, params, output) is logged
  by the endpoint handlers via ``get_logger()``.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextvars import ContextVar
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# ── request-scoped state ──────────────────────────────────────────────────────

_request_id: ContextVar[str] = ContextVar("request_id", default="-")


def current_request_id() -> str:
    """Return the 12-char hex request ID for the current async context."""
    return _request_id.get()


# ── JSON log formatter ────────────────────────────────────────────────────────

# Attributes that are part of the standard LogRecord — we skip them to avoid
# polluting the JSON document with internal Python logging machinery.
_BUILTIN_ATTRS: frozenset[str] = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", (), None))
) | {"message", "asctime", "exc_text"}


class _JsonFormatter(logging.Formatter):
    """Emit every log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        record.message = record.getMessage()
        doc: dict = {
            "ts":         self.formatTime(record, "%Y-%m-%dT%H:%M:%S") + "Z",
            "level":      record.levelname,
            "logger":     record.name,
            "request_id": _request_id.get(),
            "msg":        record.message,
        }
        # Merge caller-supplied extra fields (skip built-ins)
        for key, val in record.__dict__.items():
            if key not in _BUILTIN_ATTRS and not key.startswith("_") and key not in doc:
                doc[key] = val
        if record.exc_info:
            doc["traceback"] = self.formatException(record.exc_info)
        return json.dumps(doc, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Install a JSON StreamHandler on the root logger (idempotent)."""
    root = logging.getLogger()
    already = any(
        isinstance(h, logging.StreamHandler) and isinstance(h.formatter, _JsonFormatter)
        for h in root.handlers
    )
    if not already:
        handler = logging.StreamHandler()
        handler.setFormatter(_JsonFormatter())
        root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def get_logger(name: str = "image_service") -> logging.Logger:
    """Return a named logger for use inside endpoint handlers."""
    return logging.getLogger(name)


# ── audit middleware ──────────────────────────────────────────────────────────

_audit = logging.getLogger("audit")

# Paths omitted from request_start / request_end noise
_SILENT_PATHS: frozenset[str] = frozenset({
    "/", "/health", "/docs", "/redoc",
    "/openapi.json", "/docs/oauth2-redirect",
})


class AuditMiddleware(BaseHTTPMiddleware):
    """
    Wrap every HTTP request in structured audit log entries.

    Emits:
      ``request_start`` — method, path, query, client IP, UA, content-type
      ``request_end``   — status, elapsed_ms, content-length, X-Timing payload
      ``request_error`` — unhandled exceptions (re-raised after logging)

    File bodies are **not** consumed here; detailed logging lives in each
    endpoint handler.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        rid = uuid.uuid4().hex[:12]
        _request_id.set(rid)
        t0 = time.perf_counter()
        silent = request.url.path in _SILENT_PATHS

        if not silent:
            _audit.info(
                "request_start",
                extra={
                    "event":        "request_start",
                    "method":       request.method,
                    "path":         request.url.path,
                    "query":        dict(request.query_params),
                    "client_ip":    request.client.host if request.client else None,
                    "user_agent":   request.headers.get("user-agent"),
                    "content_type": request.headers.get("content-type", "").split(";")[0].strip(),
                },
            )

        try:
            response = await call_next(request)
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            _audit.exception(
                "request_error",
                extra={
                    "event":      "request_error",
                    "method":     request.method,
                    "path":       request.url.path,
                    "elapsed_ms": elapsed_ms,
                    "error":      str(exc),
                },
            )
            raise

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        if not silent:
            x_timing: dict | str | None = None
            raw = response.headers.get("x-timing")
            if raw:
                try:
                    x_timing = json.loads(raw)
                except Exception:
                    x_timing = raw

            _audit.info(
                "request_end",
                extra={
                    "event":          "request_end",
                    "method":         request.method,
                    "path":           request.url.path,
                    "status":         response.status_code,
                    "elapsed_ms":     elapsed_ms,
                    "content_type":   response.headers.get("content-type"),
                    "content_length": int(response.headers.get("content-length") or 0),
                    "x_timing":       x_timing,
                },
            )

        return response
