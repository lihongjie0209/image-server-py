import os
import time

import pyvips


def compress_png(
    in_path: str,
    out_path: str,
    quality: int,
    palette: bool = True,
    colors: int = 256,
    dither: float = 1.0,
    effort: int = 10,
) -> dict:
    """
    Compress PNG via pyvips.

    When *palette* is True (default), pyvips quantises the image to at most
    *colors* palette entries using libimagequant internally — no Python heap
    bytes are needed for intermediate buffers.

    Parameters
    ----------
    quality : int
        Minimum quality for palette quantisation (0-100).  Ignored when
        palette=False.
    palette : bool
        Quantise to 8-bit indexed palette (lossy but much smaller).
    colors : int
        Maximum number of palette colours (2-256).
    dither : float
        Amount of Floyd-Steinberg dithering (0.0 = none, 1.0 = full).
    effort : int
        Quantisation CPU effort (1 = fastest, 10 = best quality).
        Maps to libimagequant speed setting internally.

    Returns a dict of sub-phase timings in milliseconds plus oxipng-style
    before/after byte counts so the frontend can display them.
    """
    t0 = time.perf_counter()
    image = pyvips.Image.new_from_file(in_path)

    # pngsave palette mode requires 8-bit sRGB(A).  Cast unconditionally so
    # 16-bit PNGs and other formats all work reliably.
    if image.format != "uchar":
        image = image.cast("uchar")
    # palette mode requires an alpha channel (RGBA)
    if palette and image.bands == 3:
        image = image.bandjoin(255)

    read_ms = (time.perf_counter() - t0) * 1000

    pre_bytes = os.path.getsize(in_path)

    # PNG palette bit-depths must be exactly 1/2/4/8 bits → valid colour counts
    # are 2, 4, 16, 256.  Snap to nearest higher valid value so we never pass
    # an invalid count that makes libvips choose a non-standard bit depth.
    _VALID_COLOURS = (2, 4, 16, 256)

    def _snap_colours(n: int) -> int:
        for v in _VALID_COLOURS:
            if n <= v:
                return v
        return 256

    t1 = time.perf_counter()
    if palette:
        image.pngsave(
            out_path,
            palette=True,
            colours=_snap_colours(min(max(2, colors), 256)),
            Q=quality,
            dither=float(dither),
            effort=min(max(1, effort), 10),
            compression=9,
        )
    else:
        image.pngsave(out_path, compression=9)
    encode_ms = (time.perf_counter() - t1) * 1000

    post_bytes = os.path.getsize(out_path)

    return {
        "read_ms": round(read_ms, 1),
        "encode_ms": round(encode_ms, 1),
        "oxipng_pre_bytes": pre_bytes,    # kept same keys for frontend compat
        "oxipng_post_bytes": post_bytes,
    }
