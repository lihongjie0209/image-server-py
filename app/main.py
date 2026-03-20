import asyncio
import json
import os
import pathlib
import tempfile
import time
from typing import Annotated, Literal, Optional

import pyvips
from fastapi import FastAPI, File, Form, Query, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, ValidationError
from starlette.background import BackgroundTask

from app.audit import (
    AuditMiddleware,
    configure_logging,
    current_request_id,
    get_logger,
)
from app.compress import compress, detect_format, CONTENT_TYPES, EXTENSIONS
from app.transform import pipeline_validator, run_pipeline
from app.urlproc import decode_payload, download_to_temp

# ── bootstrap ────────────────────────────────────────────────────────────────

configure_logging()
_log = get_logger("image_service")

_START_TIME = time.time()

# ── app ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Image Compression Service",
    version="0.1.0",
    description=(
        "Compress JPEG / PNG / WebP images with a single POST request.  "
        "Transform images via a composable pipeline of resize / crop / flip / "
        "rotate / convert operations.  "
        "All parameters are passed as **multipart/form-data** fields alongside "
        "the uploaded file.  Per-phase timing metrics are exposed in the "
        "`X-Timing` response header (JSON)."
    ),
    license_info={"name": "MIT"},
)

# Order matters: CORS first so preflight replies don't bypass audit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Timing", "Content-Disposition"],
)
app.add_middleware(AuditMiddleware)

_DEMO_HTML = pathlib.Path(__file__).parent.parent / "demo.html"
_UPLOAD_CHUNK = 64 * 1024


# ── response models ───────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    version: str
    pyvips_version: str
    libvips_version: str


class ErrorResponse(BaseModel):
    detail: str


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def demo_page() -> HTMLResponse:
    return HTMLResponse(_DEMO_HTML.read_text(encoding="utf-8"))


@app.get(
    "/health",
    summary="Health check",
    tags=["ops"],
    response_model=HealthResponse,
)
async def health() -> HealthResponse:
    """Returns service health and version information."""
    lv = f"{pyvips.version(0)}.{pyvips.version(1)}.{pyvips.version(2)}"
    return HealthResponse(
        status="ok",
        uptime_seconds=round(time.time() - _START_TIME, 1),
        version=app.version,
        pyvips_version=pyvips.__version__,
        libvips_version=lv,
    )


# ── /compress ─────────────────────────────────────────────────────────────────

@app.post(
    "/compress",
    summary="Compress an image",
    tags=["compression"],
    response_class=FileResponse,
    responses={
        200: {
            "description": (
                "Compressed image stream.  "
                "`X-Timing` header carries per-phase durations (ms), "
                "before/after byte counts for PNG, and the request_id."
            ),
            "headers": {
                "X-Timing": {"schema": {"type": "string"}},
                "Content-Disposition": {"schema": {"type": "string"}},
            },
            "content": {"image/jpeg": {}, "image/png": {}, "image/webp": {}},
        },
        400: {"description": "Empty file", "model": ErrorResponse},
        415: {"description": "Unsupported format", "model": ErrorResponse},
        500: {"description": "Compression error", "model": ErrorResponse},
    },
)
async def compress_image(
    file: Annotated[UploadFile, File(description="Image to compress (JPEG/PNG/WebP)")],
    quality: Annotated[int, Form(ge=1, le=100, description="Output quality (1-100)")] = 80,
    lossless: Annotated[bool, Form(description="WebP only — force lossless")] = False,
    palette: Annotated[bool, Form(description="PNG only — quantise to palette")] = True,
    colors: Annotated[int, Form(ge=2, le=256, description="PNG palette colours")] = 256,
    dither: Annotated[float, Form(ge=0.0, le=1.0, description="PNG dithering (0-1)")] = 1.0,
    effort: Annotated[int, Form(ge=1, le=10, description="PNG effort (1-10)")] = 10,
) -> FileResponse:
    t_start = time.perf_counter()

    header = await file.read(12)
    if not header:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    try:
        fmt = detect_format(header)
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    await file.seek(0)

    t_upload0 = time.perf_counter()
    with tempfile.NamedTemporaryFile(delete=True) as in_tmp:
        while chunk := await file.read(_UPLOAD_CHUNK):
            in_tmp.write(chunk)
        in_tmp.flush()
        upload_ms = (time.perf_counter() - t_upload0) * 1000
        in_size = os.path.getsize(in_tmp.name)

        out_fd, out_path = tempfile.mkstemp(suffix=EXTENSIONS[fmt])
        os.close(out_fd)

        try:
            content_type, ext, compress_timings = await asyncio.to_thread(
                compress, in_tmp.name, out_path, quality,
                lossless, palette, colors, dither, effort,
            )
        except ValueError as exc:
            os.unlink(out_path)
            _log.warning(
                "compress_rejected",
                extra={"event": "compress_rejected", "file_name": file.filename,
                       "error": str(exc), "status": 415},
            )
            raise HTTPException(status_code=415, detail=str(exc)) from exc
        except Exception as exc:
            os.unlink(out_path)
            _log.error(
                "compress_error",
                extra={"event": "compress_error", "file_name": file.filename,
                       "error": str(exc)},
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail=f"Compression failed: {exc}") from exc

        out_size = os.path.getsize(out_path)

    total_ms = (time.perf_counter() - t_start) * 1000

    timing = {
        "request_id":  current_request_id(),
        "upload_ms":   round(upload_ms, 1),
        **compress_timings,
        "total_ms":    round(total_ms, 1),
    }

    _log.info(
        "compress_done",
        extra={
            "event": "compress_done",
            "input":  {"filename": file.filename, "content_type": file.content_type,
                       "format": fmt, "size_bytes": in_size},
            "output": {"format": fmt, "content_type": content_type,
                       "size_bytes": out_size, "filename": None},
            "params": {"quality": quality, "lossless": lossless,
                       "palette": palette, "colors": colors,
                       "dither": dither, "effort": effort},
            "timing": timing,
        },
    )

    original_name = file.filename or "image"
    stem = original_name.rsplit(".", 1)[0] if "." in original_name else original_name
    download_name = f"{stem}_compressed{ext}"

    return FileResponse(
        path=out_path,
        media_type=content_type,
        filename=download_name,
        headers={"X-Timing": json.dumps(timing)},
        background=BackgroundTask(os.unlink, out_path),
    )


# ── /transform ────────────────────────────────────────────────────────────────

_PIPELINE_DOC = """\
Ordered JSON array of pipeline operations. Available ops:

| op | required fields | optional fields |
|----|----------------|-----------------|
| `resize`  | — | `width`, `height`, `fit` (contain/cover/fill/scale-down), `smart_crop` (none/centre/entropy/attention) |
| `crop`    | `width`, `height` | `left` (0), `top` (0) |
| `flip`    | — | `direction` (h/v/both, default h) |
| `rotate`  | `angle` (-360..360) | — |
| `convert` | `format` (jpeg/png/webp) | `quality` (1-100, default 80), `lossless` (bool, default false) |

The **last** `convert` op determines the output format.
If no `convert` op is present the input format is preserved.

Example: `[{"op":"resize","width":800},{"op":"convert","format":"webp","quality":85}]`
"""


@app.post(
    "/transform",
    summary="Transform an image via a composable pipeline",
    tags=["transform"],
    response_class=FileResponse,
    responses={
        200: {
            "description": (
                "Transformed image stream. `X-Timing` carries per-step timings "
                "(ms), input/output dimensions, and the request_id."
            ),
            "headers": {
                "X-Timing": {"schema": {"type": "string"}},
                "Content-Disposition": {"schema": {"type": "string"}},
            },
            "content": {"image/jpeg": {}, "image/png": {}, "image/webp": {}},
        },
        400: {"description": "Empty file or invalid pipeline", "model": ErrorResponse},
        415: {"description": "Unsupported format", "model": ErrorResponse},
        500: {"description": "Transform error", "model": ErrorResponse},
    },
)
async def transform_image(
    file: Annotated[UploadFile, File(description="Source image (JPEG/PNG/WebP)")],
    pipeline: Annotated[str, Form(description=_PIPELINE_DOC)] = "[]",
) -> FileResponse:
    t_start = time.perf_counter()

    # 1. Detect input format
    header = await file.read(12)
    if not header:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    try:
        fmt_in = detect_format(header)
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    await file.seek(0)

    # 2. Parse + validate pipeline JSON
    try:
        ops_raw = json.loads(pipeline)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400, detail=f"pipeline is not valid JSON: {exc}"
        ) from exc
    if not isinstance(ops_raw, list):
        raise HTTPException(
            status_code=400, detail="pipeline must be a JSON array"
        )
    try:
        ops = pipeline_validator.validate_python(ops_raw)
    except ValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid pipeline operation: {exc.errors(include_url=False)}",
        ) from exc

    # 3. Determine output format (last ConvertOp wins; default = input format)
    fmt_out = fmt_in
    for op_raw in ops_raw:
        if isinstance(op_raw, dict) and op_raw.get("op") == "convert":
            fmt_out = op_raw.get("format", fmt_in)

    # 4. Stream upload to temp file
    t_upload0 = time.perf_counter()
    with tempfile.NamedTemporaryFile(delete=True) as in_tmp:
        while chunk := await file.read(_UPLOAD_CHUNK):
            in_tmp.write(chunk)
        in_tmp.flush()
        upload_ms = (time.perf_counter() - t_upload0) * 1000
        in_size = os.path.getsize(in_tmp.name)

        out_fd, out_path = tempfile.mkstemp(suffix=EXTENSIONS[fmt_out])
        os.close(out_fd)

        # 5. Run pipeline in a thread
        try:
            tf = await asyncio.to_thread(
                run_pipeline, in_tmp.name, out_path, ops, fmt_in,
            )
        except Exception as exc:
            os.unlink(out_path)
            _log.error(
                "transform_error",
                extra={"event": "transform_error", "file_name": file.filename,
                       "pipeline": ops_raw, "error": str(exc)},
                exc_info=True,
            )
            raise HTTPException(
                status_code=500, detail=f"Transform failed: {exc}"
            ) from exc

        out_size = os.path.getsize(out_path)

    total_ms = (time.perf_counter() - t_start) * 1000

    timing = {
        "request_id":  current_request_id(),
        "upload_ms":   round(upload_ms, 1),
        **tf,
        "total_ms":    round(total_ms, 1),
    }

    _log.info(
        "transform_done",
        extra={
            "event":   "transform_done",
            "input":   {"filename": file.filename, "content_type": file.content_type,
                        "format": fmt_in, "size_bytes": in_size},
            "output":  {"format": fmt_out, "content_type": CONTENT_TYPES[fmt_out],
                        "size_bytes": out_size,
                        "width": tf["output_width"], "height": tf["output_height"]},
            "pipeline_ops": ops_raw,
            "steps":   len(ops),
            "timing":  timing,
        },
    )

    original_name = file.filename or "image"
    stem = original_name.rsplit(".", 1)[0] if "." in original_name else original_name
    download_name = f"{stem}_transformed{EXTENSIONS[fmt_out]}"

    return FileResponse(
        path=out_path,
        media_type=CONTENT_TYPES[fmt_out],
        filename=download_name,
        headers={"X-Timing": json.dumps(timing)},
        background=BackgroundTask(os.unlink, out_path),
    )


# ── GET /compress/{payload} ───────────────────────────────────────────────────

_COMPRESS_URL_DOC = """\
Compress an image **by URL** without uploading a file.

`payload` is a **base64url-encoded** (no padding) JSON object:

```json
{
  "url":      "https://example.com/photo.jpg",
  "quality":  80,
  "lossless": false,
  "palette":  true,
  "colors":   256,
  "dither":   1.0,
  "effort":   10
}
```

Only `url` is required; all other fields are optional and fall back to the
same defaults as `POST /compress`.

**How to build the payload (JavaScript):**
```js
const payload = btoa(JSON.stringify({ url: "https://…", quality: 75 }))
  .replace(/\\+/g, '-').replace(/\\//g, '_').replace(/=+$/, '');
// → fetch(`/compress/${payload}`)
// → <img src="/compress/${payload}" />
```
"""


@app.get(
    "/compress/{payload}",
    summary="Compress an image by URL",
    tags=["compression"],
    response_class=FileResponse,
    responses={
        200: {
            "description": "Compressed image. `X-Timing` header carries per-phase durations.",
            "headers": {"X-Timing": {"schema": {"type": "string"}}},
            "content": {"image/jpeg": {}, "image/png": {}, "image/webp": {}},
        },
        400: {"description": "Invalid payload or download error", "model": ErrorResponse},
        415: {"description": "Unsupported image format",          "model": ErrorResponse},
        500: {"description": "Compression error",                 "model": ErrorResponse},
    },
)
async def compress_url(payload: str) -> FileResponse:
    t_start = time.perf_counter()

    # 1. Decode payload
    try:
        params = decode_payload(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    src_url: str = params.get("url", "")
    if not src_url:
        raise HTTPException(status_code=400, detail="payload must contain 'url'")

    quality  = int(params.get("quality",  80))
    lossless = bool(params.get("lossless", False))
    palette  = bool(params.get("palette",  True))
    colors   = int(params.get("colors",   256))
    dither   = float(params.get("dither",  1.0))
    effort   = int(params.get("effort",   10))

    # 2. Download source image
    t_dl0 = time.perf_counter()
    try:
        in_path, in_size = await download_to_temp(src_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Failed to download image: {exc}"
        ) from exc
    download_ms = round((time.perf_counter() - t_dl0) * 1000, 1)

    # 3. Detect format
    try:
        with open(in_path, "rb") as fh:
            header = fh.read(12)
        fmt = detect_format(header)
    except ValueError as exc:
        os.unlink(in_path)
        raise HTTPException(status_code=415, detail=str(exc)) from exc

    out_fd, out_path = tempfile.mkstemp(suffix=EXTENSIONS[fmt])
    os.close(out_fd)

    # 4. Compress
    try:
        content_type, ext, compress_timings = await asyncio.to_thread(
            compress, in_path, out_path, quality,
            lossless, palette, colors, dither, effort,
        )
    except ValueError as exc:
        os.unlink(in_path); os.unlink(out_path)
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except Exception as exc:
        os.unlink(in_path); os.unlink(out_path)
        _log.error("compress_url_error",
                   extra={"event": "compress_url_error", "source_url": src_url, "error": str(exc)},
                   exc_info=True)
        raise HTTPException(status_code=500, detail=f"Compression failed: {exc}") from exc
    finally:
        try:
            os.unlink(in_path)
        except OSError:
            pass

    out_size = os.path.getsize(out_path)
    total_ms = round((time.perf_counter() - t_start) * 1000, 1)

    timing = {
        "request_id":  current_request_id(),
        "download_ms": download_ms,
        **compress_timings,
        "total_ms":    total_ms,
    }

    _log.info("compress_url_done", extra={
        "event": "compress_url_done",
        "source_url": src_url,
        "output": {"format": fmt, "content_type": content_type, "size_bytes": out_size},
        "params": {"quality": quality, "lossless": lossless,
                   "palette": palette, "colors": colors, "dither": dither, "effort": effort},
        "timing": timing,
    })

    stem = src_url.rstrip("/").rsplit("/", 1)[-1].rsplit(".", 1)[0] or "image"
    download_name = f"{stem}_compressed{ext}"

    return FileResponse(
        path=out_path,
        media_type=content_type,
        filename=download_name,
        headers={"X-Timing": json.dumps(timing)},
        background=BackgroundTask(os.unlink, out_path),
    )


# ── GET /transform/{payload} ──────────────────────────────────────────────────

_TRANSFORM_URL_DOC = """\
Transform an image **by URL** without uploading a file.

`payload` is a **base64url-encoded** (no padding) JSON object:

```json
{
  "url":      "https://example.com/photo.jpg",
  "pipeline": [
    {"op": "resize", "width": 800},
    {"op": "convert", "format": "webp", "quality": 85}
  ]
}
```

Only `url` is required; `pipeline` defaults to `[]` (passthrough).

**How to build the payload (JavaScript):**
```js
const payload = btoa(JSON.stringify({
  url: "https://…",
  pipeline: [{ op: "resize", width: 800 }, { op: "convert", format: "webp" }]
})).replace(/\\+/g, '-').replace(/\\//g, '_').replace(/=+$/, '');
// → <img src="/transform/${payload}" />
```
"""


@app.get(
    "/transform/{payload}",
    summary="Transform an image by URL",
    tags=["transform"],
    response_class=FileResponse,
    responses={
        200: {
            "description": "Transformed image. `X-Timing` carries per-step durations.",
            "headers": {"X-Timing": {"schema": {"type": "string"}}},
            "content": {"image/jpeg": {}, "image/png": {}, "image/webp": {}},
        },
        400: {"description": "Invalid payload, pipeline, or download error", "model": ErrorResponse},
        415: {"description": "Unsupported image format",                      "model": ErrorResponse},
        500: {"description": "Transform error",                               "model": ErrorResponse},
    },
)
async def transform_url(payload: str) -> FileResponse:
    t_start = time.perf_counter()

    # 1. Decode payload
    try:
        params = decode_payload(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    src_url: str = params.get("url", "")
    if not src_url:
        raise HTTPException(status_code=400, detail="payload must contain 'url'")

    ops_raw = params.get("pipeline", [])
    if not isinstance(ops_raw, list):
        raise HTTPException(status_code=400, detail="'pipeline' must be a JSON array")

    # 2. Validate pipeline
    try:
        ops = pipeline_validator.validate_python(ops_raw)
    except ValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid pipeline operation: {exc.errors(include_url=False)}",
        ) from exc

    # 3. Determine output format from last ConvertOp
    fmt_out_hint = None
    for op_raw in ops_raw:
        if isinstance(op_raw, dict) and op_raw.get("op") == "convert":
            fmt_out_hint = op_raw.get("format")

    # 4. Download source image
    t_dl0 = time.perf_counter()
    try:
        in_path, in_size = await download_to_temp(src_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Failed to download image: {exc}"
        ) from exc
    download_ms = round((time.perf_counter() - t_dl0) * 1000, 1)

    # 5. Detect input format
    try:
        with open(in_path, "rb") as fh:
            header = fh.read(12)
        fmt_in = detect_format(header)
    except ValueError as exc:
        os.unlink(in_path)
        raise HTTPException(status_code=415, detail=str(exc)) from exc

    fmt_out = fmt_out_hint or fmt_in
    out_fd, out_path = tempfile.mkstemp(suffix=EXTENSIONS[fmt_out])
    os.close(out_fd)

    # 6. Run pipeline
    try:
        tf = await asyncio.to_thread(run_pipeline, in_path, out_path, ops, fmt_in)
    except Exception as exc:
        os.unlink(out_path)
        _log.error("transform_url_error",
                   extra={"event": "transform_url_error", "source_url": src_url,
                          "pipeline": ops_raw, "error": str(exc)},
                   exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transform failed: {exc}") from exc
    finally:
        try:
            os.unlink(in_path)
        except OSError:
            pass

    out_size = os.path.getsize(out_path)
    total_ms = round((time.perf_counter() - t_start) * 1000, 1)

    timing = {
        "request_id":  current_request_id(),
        "download_ms": download_ms,
        **tf,
        "total_ms":    total_ms,
    }

    _log.info("transform_url_done", extra={
        "event":       "transform_url_done",
        "source_url":  src_url,
        "output":      {"format": fmt_out, "content_type": CONTENT_TYPES[fmt_out],
                        "size_bytes": out_size,
                        "width": tf["output_width"], "height": tf["output_height"]},
        "pipeline_ops": ops_raw,
        "timing":       timing,
    })

    stem = src_url.rstrip("/").rsplit("/", 1)[-1].rsplit(".", 1)[0] or "image"
    download_name = f"{stem}_transformed{EXTENSIONS[fmt_out]}"

    return FileResponse(
        path=out_path,
        media_type=CONTENT_TYPES[fmt_out],
        filename=download_name,
        headers={"X-Timing": json.dumps(timing)},
        background=BackgroundTask(os.unlink, out_path),
    )
