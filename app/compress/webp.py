import time
import pyvips


def compress_webp(in_path: str, out_path: str, quality: int, lossless: bool = False) -> dict:
    """Compress WebP via pyvips. Input/output stay off the Python heap.

    Returns a dict of sub-phase timings in milliseconds.
    """
    t0 = time.perf_counter()
    image = pyvips.Image.new_from_file(in_path, access="sequential")
    read_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    image.webpsave(out_path, Q=quality, lossless=lossless)
    encode_ms = (time.perf_counter() - t1) * 1000

    return {"read_ms": round(read_ms, 1), "encode_ms": round(encode_ms, 1)}

