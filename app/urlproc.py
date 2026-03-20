"""URL-based image processing helpers.

Provides base64url payload encoding/decoding and async HTTP image download
for the GET /compress/{payload} and GET /transform/{payload} endpoints.

Payload format
──────────────
The payload is a **base64url-encoded** (RFC 4648 §5, no padding) JSON object.

Compress payload fields:
  url      (str, required)   — public http/https image URL
  quality  (int, 1-100)      — default 80
  lossless (bool)            — default false  (WebP only)
  palette  (bool)            — default true   (PNG only)
  colors   (int, 2-256)      — default 256    (PNG palette)
  dither   (float, 0-1)      — default 1.0    (PNG dithering)
  effort   (int, 1-10)       — default 10     (PNG effort)

Transform payload fields:
  url      (str, required)   — public http/https image URL
  pipeline (list)            — default []  (array of pipeline ops)

Example (Python):
    import base64, json
    p = base64.urlsafe_b64encode(json.dumps({"url":"https://…","quality":75}).encode()).rstrip(b"=").decode()
    # → GET /compress/<p>

Example (JavaScript):
    const p = btoa(JSON.stringify({url:"https://…",quality:75}))
                .replace(/\\+/g,'-').replace(/\\//g,'_').replace(/=+$/,'');
    // → GET /compress/<p>
"""
from __future__ import annotations

import base64
import json
import os
import tempfile
from urllib.parse import urlparse

import httpx

# Safety limits
MAX_DOWNLOAD_BYTES: int = 50 * 1024 * 1024  # 50 MB
DOWNLOAD_TIMEOUT: float = 30.0


# ── codec ─────────────────────────────────────────────────────────────────────

def decode_payload(raw: str) -> dict:
    """Decode a base64url-encoded JSON string into a dict.

    Raises ``ValueError`` with a descriptive message on any decoding failure.
    """
    # Restore padding stripped by the client
    rem = len(raw) % 4
    if rem:
        raw += "=" * (4 - rem)
    try:
        data = base64.urlsafe_b64decode(raw)
    except Exception as exc:
        raise ValueError(f"Payload is not valid base64url: {exc}") from exc
    try:
        obj = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Payload is not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("Payload must be a JSON object (dict)")
    return obj


def encode_payload(obj: dict) -> str:
    """Encode a dict to a base64url string (no padding).  Useful in tests."""
    return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()


# ── URL validation ─────────────────────────────────────────────────────────────

def validate_source_url(url: str) -> None:
    """Raise ``ValueError`` unless *url* is an http/https URL."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"'url' scheme must be http or https, got {parsed.scheme!r}"
        )
    if not parsed.netloc:
        raise ValueError("'url' has no host")


# ── async download ────────────────────────────────────────────────────────────

async def download_to_temp(url: str) -> tuple[str, int]:
    """Download *url* to a fresh temp file.

    Returns ``(temp_path, bytes_written)``.
    The caller is responsible for deleting the file when done.

    Raises:
        ValueError — invalid URL scheme or download too large
        httpx.HTTPStatusError — 4xx/5xx from the image server
        httpx.TimeoutException — server too slow
    """
    validate_source_url(url)

    fd, path = tempfile.mkstemp()
    try:
        written = 0
        async with httpx.AsyncClient(
            timeout=DOWNLOAD_TIMEOUT,
            follow_redirects=True,
        ) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                with os.fdopen(fd, "wb") as f:
                    async for chunk in resp.aiter_bytes(65_536):
                        written += len(chunk)
                        if written > MAX_DOWNLOAD_BYTES:
                            raise ValueError(
                                f"Download exceeds {MAX_DOWNLOAD_BYTES // 1_048_576} MB limit"
                            )
                        f.write(chunk)
        return path, written
    except Exception:
        # Clean up temp file on any error
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
