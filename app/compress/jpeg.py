import time
import pyvips


def compress_jpeg(in_path: str, out_path: str, quality: int) -> dict:
    """Compress JPEG via pyvips using trellis quantization and jpegli.

    Returns a dict of sub-phase timings in milliseconds.
    """
    t0 = time.perf_counter()
    image = pyvips.Image.new_from_file(in_path)
    read_ms = (time.perf_counter() - t0) * 1000

    kwargs = dict(Q=quality, trellis_quant=True, optimize_coding=True)
    t1 = time.perf_counter()
    try:
        image.jpegli_save(out_path, **kwargs)
    except Exception:
        image.jpegsave(out_path, **kwargs)
    encode_ms = (time.perf_counter() - t1) * 1000

    return {"read_ms": round(read_ms, 1), "encode_ms": round(encode_ms, 1)}

