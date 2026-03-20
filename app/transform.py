"""Image transformation pipeline.

Each pipeline step is a Pydantic model that validates its own parameters.
``run_pipeline`` loads the image, executes every step in the given order,
and saves the result, returning per-step timing data.

Available operations
────────────────────
  resize  — scale to fit/cover/fill/scale-down via libvips thumbnail_image
  crop    — extract a rectangular region by pixel coordinates
  flip    — mirror horizontally, vertically, or both
  rotate  — clockwise rotation; 90°/180°/270° use the lossless rot() path
  convert — set the output format, quality and lossless flag
            (the *last* convert op wins; if absent the input format is kept)
"""
from __future__ import annotations

import time
from typing import Annotated, Literal, Optional, Union

import pyvips
from pydantic import BaseModel, Field, TypeAdapter


# ── operation models ──────────────────────────────────────────────────────────

class ResizeOp(BaseModel):
    """Scale the image to fit within (width × height) using the chosen strategy."""
    op: Literal["resize"]
    width: Optional[int] = Field(
        None, ge=1, le=10000, description="Target width (px)"
    )
    height: Optional[int] = Field(
        None, ge=1, le=10000, description="Target height (px)"
    )
    fit: Literal["contain", "cover", "fill", "scale-down"] = Field(
        "contain",
        description=(
            "contain: preserve aspect ratio (default); "
            "cover: crop to exact WxH; "
            "fill: stretch; "
            "scale-down: only shrink, never enlarge"
        ),
    )
    smart_crop: Literal["none", "centre", "entropy", "attention"] = Field(
        "none",
        description="Focus-point algorithm — effective only when fit=cover",
    )


class CropOp(BaseModel):
    """Extract a rectangular region by pixel coordinates."""
    op: Literal["crop"]
    left: int = Field(0, ge=0, description="Left offset (px)")
    top: int = Field(0, ge=0, description="Top offset (px)")
    width: int = Field(..., ge=1, description="Output width (px)")
    height: int = Field(..., ge=1, description="Output height (px)")


class FlipOp(BaseModel):
    """Mirror the image along one or both axes."""
    op: Literal["flip"]
    direction: Literal["h", "v", "both"] = Field(
        "h",
        description="h = horizontal, v = vertical, both = h then v",
    )


class RotateOp(BaseModel):
    """Rotate the image clockwise. Multiples of 90° use the lossless rot() path."""
    op: Literal["rotate"]
    angle: float = Field(..., ge=-360, le=360, description="Clockwise degrees")


class ConvertOp(BaseModel):
    """Set the output encoding. The last ConvertOp in the pipeline wins."""
    op: Literal["convert"]
    format: Literal["jpeg", "png", "webp"] = Field(..., description="Output format")
    quality: int = Field(80, ge=1, le=100, description="Encoding quality (1–100)")
    lossless: bool = Field(False, description="WebP: force lossless mode")


# Discriminated union — Pydantic uses the 'op' field to pick the right model
PipelineOp = Annotated[
    Union[ResizeOp, CropOp, FlipOp, RotateOp, ConvertOp],
    Field(discriminator="op"),
]

# Module-level validator (built once, reused on every request)
pipeline_validator: TypeAdapter = TypeAdapter(list[PipelineOp])


# ── pyvips enum maps ──────────────────────────────────────────────────────────

_INTERESTING = {
    "none":      pyvips.enums.Interesting.NONE,
    "centre":    pyvips.enums.Interesting.CENTRE,
    "entropy":   pyvips.enums.Interesting.ENTROPY,
    "attention": pyvips.enums.Interesting.ATTENTION,
}

_SIZE = {
    "contain":    pyvips.enums.Size.BOTH,
    "cover":      pyvips.enums.Size.BOTH,
    "fill":       pyvips.enums.Size.FORCE,
    "scale-down": pyvips.enums.Size.DOWN,
}


# ── per-op executor ───────────────────────────────────────────────────────────

def _exec_op(image: pyvips.Image, op: PipelineOp) -> tuple[pyvips.Image, float]:
    """Execute *op* on *image*. Returns ``(new_image, elapsed_ms)``."""
    t = time.perf_counter()

    if op.op == "resize":
        if op.width or op.height:
            w = op.width  if op.width  else 10_000_000
            h = op.height if op.height else 10_000_000
            size_enum = _SIZE.get(op.fit, pyvips.enums.Size.BOTH)
            crop_enum = (
                _INTERESTING.get(op.smart_crop, pyvips.enums.Interesting.NONE)
                if op.fit == "cover"
                else pyvips.enums.Interesting.NONE
            )
            image = image.thumbnail_image(
                w, height=h, size=size_enum, crop=crop_enum, no_rotate=True,
            )
        # Neither width nor height set → deliberate no-op (still records timing)

    elif op.op == "crop":
        cl = max(0, min(op.left,   image.width  - 1))
        ct = max(0, min(op.top,    image.height - 1))
        cw = min(op.width,  image.width  - cl)
        ch = min(op.height, image.height - ct)
        if cw > 0 and ch > 0:
            image = image.crop(cl, ct, cw, ch)

    elif op.op == "flip":
        if op.direction in ("h", "both"):
            image = image.fliphor()
        if op.direction in ("v", "both"):
            image = image.flipver()

    elif op.op == "rotate":
        angle = op.angle % 360
        if angle == 90:
            image = image.rot(pyvips.enums.Angle.D90)
        elif angle == 180:
            image = image.rot(pyvips.enums.Angle.D180)
        elif angle == 270:
            image = image.rot(pyvips.enums.Angle.D270)
        elif angle:
            image = image.rotate(angle)
        # angle == 0 → deliberate no-op

    elif op.op == "convert":
        pass  # Format/quality applied at save time

    return image, round((time.perf_counter() - t) * 1000, 1)


# ── pipeline runner ───────────────────────────────────────────────────────────

def run_pipeline(
    in_path: str,
    out_path: str,
    ops: list,
    default_fmt: str = "jpeg",
) -> dict:
    """
    Execute *ops* sequentially on the image at *in_path*,
    writing the result to *out_path*.

    Output format and quality are taken from the **last** ``ConvertOp`` in the
    pipeline; if none is present the input format (*default_fmt*) is preserved
    with a default quality of 80.

    Returns a timing + dimension dict for the ``X-Timing`` response header.
    """
    # Resolve output settings from the last ConvertOp (if any)
    fmt_out  = default_fmt
    quality  = 80
    lossless = False
    for op in ops:
        if op.op == "convert":
            fmt_out  = op.format
            quality  = op.quality
            lossless = op.lossless

    # Load — JPEG supports EXIF auto-rotate; PNG/WebP loaders may raise
    t0 = time.perf_counter()
    try:
        image = pyvips.Image.new_from_file(in_path, autorotate=True)
    except Exception:
        image = pyvips.Image.new_from_file(in_path)
    load_ms = round((time.perf_counter() - t0) * 1000, 1)

    orig_w, orig_h = image.width, image.height

    # Execute each step, recording per-step timing
    op_timings: list[dict] = []
    for op in ops:
        image, ms = _exec_op(image, op)
        op_timings.append({"op": op.op, "ms": ms})

    total_ops_ms = round(sum(s["ms"] for s in op_timings), 1)

    # Save
    t_save = time.perf_counter()
    if fmt_out == "jpeg":
        if image.hasalpha():
            image = image.flatten(background=[255, 255, 255])
        kwargs = dict(Q=quality, trellis_quant=True, optimize_coding=True)
        try:
            image.jpegli_save(out_path, **kwargs)
        except Exception:
            image.jpegsave(out_path, **kwargs)
    elif fmt_out == "png":
        image.pngsave(out_path, compression=9)
    elif fmt_out == "webp":
        image.webpsave(out_path, Q=quality, lossless=lossless)
    save_ms = round((time.perf_counter() - t_save) * 1000, 1)

    return {
        "load_ms":       load_ms,
        "pipeline":      op_timings,
        "total_ops_ms":  total_ops_ms,
        "save_ms":       save_ms,
        "input_width":   orig_w,
        "input_height":  orig_h,
        "output_width":  image.width,
        "output_height": image.height,
        "output_format": fmt_out,
    }

