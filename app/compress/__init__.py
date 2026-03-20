import time
from .jpeg import compress_jpeg
from .png import compress_png
from .webp import compress_webp

_JPEG_MAGIC = b"\xff\xd8\xff"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_WEBP_MAGIC_RIFF = b"RIFF"
_WEBP_MAGIC_WEBP = b"WEBP"


def detect_format(header: bytes) -> str:
    """Detect image format from the first >=12 bytes. Returns 'jpeg', 'png', or 'webp'."""
    if header[:3] == _JPEG_MAGIC:
        return "jpeg"
    if header[:8] == _PNG_MAGIC:
        return "png"
    if header[:4] == _WEBP_MAGIC_RIFF and header[8:12] == _WEBP_MAGIC_WEBP:
        return "webp"
    raise ValueError(f"Unsupported image format (magic: {header[:12].hex()})")


CONTENT_TYPES: dict[str, str] = {
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
}

EXTENSIONS: dict[str, str] = {
    "jpeg": ".jpg",
    "png": ".png",
    "webp": ".webp",
}


def compress(
    in_path: str,
    out_path: str,
    quality: int,
    lossless: bool = False,
    palette: bool = True,
    colors: int = 256,
    dither: float = 1.0,
    effort: int = 10,
) -> tuple[str, str, dict]:
    """
    Compress the image at *in_path*, writing the result to *out_path*.

    Returns (content_type, file_extension, timings).
    *timings* is a flat dict of millisecond durations for each sub-phase
    plus a ``compress_ms`` total for the compression step.
    The format is auto-detected from magic bytes.
    """
    t_detect0 = time.perf_counter()
    with open(in_path, "rb") as f:
        header = f.read(12)
    fmt = detect_format(header)
    detect_ms = (time.perf_counter() - t_detect0) * 1000

    t_compress0 = time.perf_counter()
    if fmt == "jpeg":
        detail = compress_jpeg(in_path, out_path, quality)
    elif fmt == "png":
        detail = compress_png(
            in_path, out_path, quality,
            palette=palette, colors=colors, dither=dither, effort=effort,
        )
    elif fmt == "webp":
        detail = compress_webp(in_path, out_path, quality, lossless=lossless)
    compress_ms = (time.perf_counter() - t_compress0) * 1000

    timings = {
        "detect_ms": round(detect_ms, 1),
        "compress_ms": round(compress_ms, 1),
        **{f"compress_{k}": v for k, v in detail.items()},
    }
    return CONTENT_TYPES[fmt], EXTENSIONS[fmt], timings
