"""
Comprehensive integration tests for the image compression service.

Test categories
───────────────
  1. Health endpoint
  2. JPEG compression — quality levels, output validity, size reduction
  3. PNG compression  — quality levels, output validity, size reduction, alpha
  4. WebP compression — lossy, lossless, output validity
  5. Quality parameter validation — boundary values, bad inputs
  6. Error handling — empty file, unsupported format, corrupt data
  7. Response headers — Content-Type, Content-Disposition, filename derivation
  8. Image integrity — compressed output can be re-opened; dimensions preserved
  9. Edge cases — 1×1 pixel, filename without extension, MIME mismatch (format
                   detected from bytes, not Content-Type header)
 10. Concurrent requests — temp-file isolation under parallel load
"""
import concurrent.futures
import io
import json
import struct

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

# ---------------------------------------------------------------------------
# Custom test client: transparently moves URL query-params into the multipart
# form-data body so legacy tests don't need rewriting after the API migrated
# all parameters to Form(...) fields.
# ---------------------------------------------------------------------------
class _FormClient(TestClient):
    """TestClient that converts URL query params into form-data fields."""

    def post(self, url: str, **kwargs):
        from urllib.parse import urlsplit, parse_qs, urlunsplit
        parts = urlsplit(url)
        qp = {k: v[0] for k, v in parse_qs(parts.query).items()}
        if qp:
            clean_parts = parts._replace(query="")
            url = urlunsplit(clean_parts)
            data = dict(kwargs.pop("data", {}) or {})
            data.update(qp)
            kwargs["data"] = data
        return super().post(url, **kwargs)


client = _FormClient(app)

# ────────────────────────────────────────────────────────────────────────────
# Image factories
# ────────────────────────────────────────────────────────────────────────────

def make_jpeg(width: int = 200, height: int = 200, quality: int = 95) -> bytes:
    img = Image.new("RGB", (width, height))
    # add some gradient so the compressor has real work to do
    pixels = img.load()
    for x in range(width):
        for y in range(height):
            pixels[x, y] = (x % 256, y % 256, (x + y) % 256)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def make_png(width: int = 200, height: int = 200, mode: str = "RGBA") -> bytes:
    img = Image.new(mode, (width, height))
    pixels = img.load()
    for x in range(width):
        for y in range(height):
            if mode == "RGBA":
                pixels[x, y] = (x % 256, y % 256, (x + y) % 256, 200)
            else:
                pixels[x, y] = (x % 256, y % 256, (x + y) % 256)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_webp(width: int = 200, height: int = 200, quality: int = 95) -> bytes:
    img = Image.new("RGB", (width, height))
    pixels = img.load()
    for x in range(width):
        for y in range(height):
            pixels[x, y] = (x % 256, y % 256, (x + y) % 256)
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=quality)
    return buf.getvalue()


def is_valid_jpeg(data: bytes) -> bool:
    return data[:3] == b"\xff\xd8\xff"


def is_valid_png(data: bytes) -> bool:
    return data[:8] == b"\x89PNG\r\n\x1a\n"


def is_valid_webp(data: bytes) -> bool:
    return data[:4] == b"RIFF" and data[8:12] == b"WEBP"


def image_size(data: bytes) -> tuple[int, int]:
    """Return (width, height) of an image by decoding it with Pillow."""
    return Image.open(io.BytesIO(data)).size


# ────────────────────────────────────────────────────────────────────────────
# 1. Health endpoint
# ────────────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_returns_ok_status(self):
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "pyvips_version" in data
        assert "libvips_version" in data
        assert "uptime_seconds" in data

    def test_wrong_method_returns_405(self):
        resp = client.post("/health")
        assert resp.status_code == 405


# ────────────────────────────────────────────────────────────────────────────
# 2. JPEG compression
# ────────────────────────────────────────────────────────────────────────────

class TestJpegCompression:
    def test_status_200(self):
        resp = client.post("/compress?quality=70",
                           files={"file": ("img.jpg", make_jpeg(), "image/jpeg")})
        assert resp.status_code == 200

    def test_output_content_type(self):
        resp = client.post("/compress?quality=70",
                           files={"file": ("img.jpg", make_jpeg(), "image/jpeg")})
        assert resp.headers["content-type"].startswith("image/jpeg")

    def test_output_is_valid_jpeg(self):
        resp = client.post("/compress?quality=70",
                           files={"file": ("img.jpg", make_jpeg(), "image/jpeg")})
        assert is_valid_jpeg(resp.content)

    def test_low_quality_reduces_size(self):
        original = make_jpeg(400, 400, quality=95)
        resp = client.post("/compress?quality=20",
                           files={"file": ("img.jpg", original, "image/jpeg")})
        assert resp.status_code == 200
        assert len(resp.content) < len(original)

    def test_quality_1(self):
        resp = client.post("/compress?quality=1",
                           files={"file": ("img.jpg", make_jpeg(), "image/jpeg")})
        assert resp.status_code == 200
        assert is_valid_jpeg(resp.content)

    def test_quality_99(self):
        resp = client.post("/compress?quality=99",
                           files={"file": ("img.jpg", make_jpeg(), "image/jpeg")})
        assert resp.status_code == 200
        assert is_valid_jpeg(resp.content)

    def test_dimensions_preserved(self):
        original = make_jpeg(320, 240)
        resp = client.post("/compress?quality=70",
                           files={"file": ("img.jpg", original, "image/jpeg")})
        w, h = image_size(resp.content)
        assert (w, h) == (320, 240)

    def test_default_quality_when_omitted(self):
        resp = client.post("/compress",
                           files={"file": ("img.jpg", make_jpeg(), "image/jpeg")})
        assert resp.status_code == 200


# ────────────────────────────────────────────────────────────────────────────
# 3. PNG compression
# ────────────────────────────────────────────────────────────────────────────

class TestPngCompression:
    def test_status_200(self):
        resp = client.post("/compress?quality=75",
                           files={"file": ("img.png", make_png(), "image/png")})
        assert resp.status_code == 200

    def test_output_content_type(self):
        resp = client.post("/compress?quality=75",
                           files={"file": ("img.png", make_png(), "image/png")})
        assert resp.headers["content-type"].startswith("image/png")

    def test_output_is_valid_png(self):
        resp = client.post("/compress?quality=75",
                           files={"file": ("img.png", make_png(), "image/png")})
        assert is_valid_png(resp.content)

    def test_low_quality_reduces_size(self):
        # Use a large uncompressed PNG with simple banding so imagequant+oxipng
        # clearly wins: compress_level=0 ≈ 360 KB; palette+oxipng → ~20 KB.
        # Pixel values are banded (few unique colors) so imagequant succeeds.
        img = Image.new("RGBA", (300, 300))
        px = img.load()
        for x in range(300):
            for y in range(300):
                px[x, y] = ((x // 10) * 8 % 256, (y // 10) * 8 % 256, 128, 255)
        buf = io.BytesIO()
        img.save(buf, format="PNG", compress_level=0)   # uncompressed source
        original = buf.getvalue()

        resp = client.post("/compress?quality=50",
                           files={"file": ("img.png", original, "image/png")})
        assert resp.status_code == 200
        assert len(resp.content) < len(original)

    def test_rgba_png_accepted(self):
        original = make_png(200, 200, mode="RGBA")
        resp = client.post("/compress?quality=75",
                           files={"file": ("img.png", original, "image/png")})
        assert resp.status_code == 200
        assert is_valid_png(resp.content)

    def test_rgb_png_accepted(self):
        original = make_png(200, 200, mode="RGB")
        resp = client.post("/compress?quality=75",
                           files={"file": ("img.png", original, "image/png")})
        assert resp.status_code == 200
        assert is_valid_png(resp.content)

    def test_dimensions_preserved(self):
        original = make_png(128, 64)
        resp = client.post("/compress?quality=75",
                           files={"file": ("img.png", original, "image/png")})
        w, h = image_size(resp.content)
        assert (w, h) == (128, 64)

    def test_quality_1(self):
        resp = client.post("/compress?quality=1",
                           files={"file": ("img.png", make_png(), "image/png")})
        assert resp.status_code == 200
        assert is_valid_png(resp.content)


# ────────────────────────────────────────────────────────────────────────────
# 4. WebP compression
# ────────────────────────────────────────────────────────────────────────────

class TestWebpCompression:
    def test_status_200(self):
        resp = client.post("/compress?quality=70",
                           files={"file": ("img.webp", make_webp(), "image/webp")})
        assert resp.status_code == 200

    def test_output_content_type(self):
        resp = client.post("/compress?quality=70",
                           files={"file": ("img.webp", make_webp(), "image/webp")})
        assert resp.headers["content-type"].startswith("image/webp")

    def test_output_is_valid_webp(self):
        resp = client.post("/compress?quality=70",
                           files={"file": ("img.webp", make_webp(), "image/webp")})
        assert is_valid_webp(resp.content)

    def test_low_quality_reduces_size(self):
        original = make_webp(400, 400, quality=95)
        resp = client.post("/compress?quality=20",
                           files={"file": ("img.webp", original, "image/webp")})
        assert resp.status_code == 200
        assert len(resp.content) < len(original)

    def test_lossless_flag(self):
        resp = client.post("/compress?quality=90&lossless=true",
                           files={"file": ("img.webp", make_webp(), "image/webp")})
        assert resp.status_code == 200
        assert is_valid_webp(resp.content)

    def test_dimensions_preserved(self):
        original = make_webp(160, 90)
        resp = client.post("/compress?quality=70",
                           files={"file": ("img.webp", original, "image/webp")})
        w, h = image_size(resp.content)
        assert (w, h) == (160, 90)

    def test_quality_1(self):
        resp = client.post("/compress?quality=1",
                           files={"file": ("img.webp", make_webp(), "image/webp")})
        assert resp.status_code == 200
        assert is_valid_webp(resp.content)


# ────────────────────────────────────────────────────────────────────────────
# 5. Quality parameter validation
# ────────────────────────────────────────────────────────────────────────────

class TestQualityValidation:
    def test_quality_zero_rejected(self):
        resp = client.post("/compress?quality=0",
                           files={"file": ("img.jpg", make_jpeg(), "image/jpeg")})
        assert resp.status_code == 422

    def test_quality_101_rejected(self):
        resp = client.post("/compress?quality=101",
                           files={"file": ("img.jpg", make_jpeg(), "image/jpeg")})
        assert resp.status_code == 422

    def test_quality_string_rejected(self):
        resp = client.post("/compress?quality=high",
                           files={"file": ("img.jpg", make_jpeg(), "image/jpeg")})
        assert resp.status_code == 422

    def test_quality_float_rejected(self):
        resp = client.post("/compress?quality=75.5",
                           files={"file": ("img.jpg", make_jpeg(), "image/jpeg")})
        assert resp.status_code == 422

    def test_quality_boundary_1_accepted(self):
        resp = client.post("/compress?quality=1",
                           files={"file": ("img.jpg", make_jpeg(), "image/jpeg")})
        assert resp.status_code == 200

    def test_quality_boundary_100_accepted(self):
        resp = client.post("/compress?quality=100",
                           files={"file": ("img.jpg", make_jpeg(), "image/jpeg")})
        assert resp.status_code == 200


# ────────────────────────────────────────────────────────────────────────────
# 6. Error handling
# ────────────────────────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_empty_file_returns_400(self):
        resp = client.post("/compress",
                           files={"file": ("empty.jpg", b"", "image/jpeg")})
        assert resp.status_code == 400

    def test_gif_returns_415(self):
        gif_header = b"GIF89a\x01\x00\x01\x00\x00\xff\x00"
        resp = client.post("/compress",
                           files={"file": ("anim.gif", gif_header, "image/gif")})
        assert resp.status_code == 415

    def test_random_bytes_returns_415(self):
        resp = client.post("/compress",
                           files={"file": ("junk.bin", b"\x00" * 64, "application/octet-stream")})
        assert resp.status_code == 415

    def test_truncated_jpeg_returns_5xx(self):
        # Valid JPEG magic, but the rest is garbage
        bad_jpeg = b"\xff\xd8\xff\xe0" + b"\xde\xad\xbe\xef" * 32
        resp = client.post("/compress",
                           files={"file": ("bad.jpg", bad_jpeg, "image/jpeg")})
        assert resp.status_code in (415, 500)

    def test_missing_file_field_returns_422(self):
        resp = client.post("/compress?quality=80")
        assert resp.status_code == 422


# ────────────────────────────────────────────────────────────────────────────
# 7. Response headers
# ────────────────────────────────────────────────────────────────────────────

class TestResponseHeaders:
    def test_jpeg_content_disposition_filename(self):
        resp = client.post("/compress?quality=70",
                           files={"file": ("photo.jpg", make_jpeg(), "image/jpeg")})
        assert "photo_compressed.jpg" in resp.headers.get("content-disposition", "")

    def test_png_content_disposition_filename(self):
        resp = client.post("/compress?quality=70",
                           files={"file": ("banner.png", make_png(), "image/png")})
        assert "banner_compressed.png" in resp.headers.get("content-disposition", "")

    def test_webp_content_disposition_filename(self):
        resp = client.post("/compress?quality=70",
                           files={"file": ("hero.webp", make_webp(), "image/webp")})
        assert "hero_compressed.webp" in resp.headers.get("content-disposition", "")

    def test_content_disposition_is_attachment(self):
        resp = client.post("/compress?quality=70",
                           files={"file": ("img.jpg", make_jpeg(), "image/jpeg")})
        cd = resp.headers.get("content-disposition", "")
        assert "attachment" in cd

    def test_jpeg_mime_is_image_jpeg(self):
        resp = client.post("/compress",
                           files={"file": ("img.jpg", make_jpeg(), "image/jpeg")})
        assert "image/jpeg" in resp.headers["content-type"]

    def test_png_mime_is_image_png(self):
        resp = client.post("/compress",
                           files={"file": ("img.png", make_png(), "image/png")})
        assert "image/png" in resp.headers["content-type"]

    def test_webp_mime_is_image_webp(self):
        resp = client.post("/compress",
                           files={"file": ("img.webp", make_webp(), "image/webp")})
        assert "image/webp" in resp.headers["content-type"]


# ────────────────────────────────────────────────────────────────────────────
# 8. Image integrity — output is decodable, dimensions match
# ────────────────────────────────────────────────────────────────────────────

class TestImageIntegrity:
    def test_jpeg_reopen_with_pillow(self):
        resp = client.post("/compress?quality=70",
                           files={"file": ("img.jpg", make_jpeg(300, 300), "image/jpeg")})
        img = Image.open(io.BytesIO(resp.content))
        assert img.format == "JPEG"

    def test_png_reopen_with_pillow(self):
        resp = client.post("/compress?quality=70",
                           files={"file": ("img.png", make_png(300, 300), "image/png")})
        img = Image.open(io.BytesIO(resp.content))
        assert img.format == "PNG"

    def test_webp_reopen_with_pillow(self):
        resp = client.post("/compress?quality=70",
                           files={"file": ("img.webp", make_webp(300, 300), "image/webp")})
        img = Image.open(io.BytesIO(resp.content))
        assert img.format == "WEBP"

    def test_jpeg_dimensions_exact(self):
        resp = client.post("/compress?quality=70",
                           files={"file": ("img.jpg", make_jpeg(512, 384), "image/jpeg")})
        assert image_size(resp.content) == (512, 384)

    def test_png_dimensions_exact(self):
        resp = client.post("/compress?quality=70",
                           files={"file": ("img.png", make_png(256, 512), "image/png")})
        assert image_size(resp.content) == (256, 512)

    def test_webp_dimensions_exact(self):
        resp = client.post("/compress?quality=70",
                           files={"file": ("img.webp", make_webp(640, 480), "image/webp")})
        assert image_size(resp.content) == (640, 480)


# ────────────────────────────────────────────────────────────────────────────
# 9. Edge cases
# ────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_1x1_jpeg(self):
        img = Image.new("RGB", (1, 1), color=(128, 64, 32))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        resp = client.post("/compress?quality=70",
                           files={"file": ("tiny.jpg", buf.getvalue(), "image/jpeg")})
        assert resp.status_code == 200
        assert is_valid_jpeg(resp.content)

    def test_1x1_png(self):
        img = Image.new("RGBA", (1, 1), color=(0, 255, 0, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        resp = client.post("/compress?quality=70",
                           files={"file": ("tiny.png", buf.getvalue(), "image/png")})
        assert resp.status_code == 200
        assert is_valid_png(resp.content)

    def test_filename_without_extension(self):
        """Service should not crash when the uploaded filename has no dot."""
        resp = client.post("/compress?quality=70",
                           files={"file": ("myimage", make_jpeg(), "image/jpeg")})
        assert resp.status_code == 200
        cd = resp.headers.get("content-disposition", "")
        assert "myimage_compressed.jpg" in cd

    def test_format_detected_from_bytes_not_mime(self):
        """A JPEG sent with wrong MIME type must still compress as JPEG."""
        resp = client.post(
            "/compress?quality=70",
            files={"file": ("img.jpg", make_jpeg(), "application/octet-stream")},
        )
        assert resp.status_code == 200
        assert is_valid_jpeg(resp.content)

    def test_png_sent_with_wrong_mime(self):
        """A PNG sent as image/jpeg must still be detected and compressed as PNG."""
        resp = client.post(
            "/compress?quality=70",
            files={"file": ("img.png", make_png(), "image/jpeg")},
        )
        assert resp.status_code == 200
        assert is_valid_png(resp.content)

    def test_large_jpeg(self):
        """1000×1000 JPEG should compress without error."""
        resp = client.post("/compress?quality=60",
                           files={"file": ("big.jpg", make_jpeg(1000, 1000, quality=95), "image/jpeg")})
        assert resp.status_code == 200
        assert is_valid_jpeg(resp.content)

    def test_large_png(self):
        """800×800 PNG should compress without error."""
        resp = client.post("/compress?quality=60",
                           files={"file": ("big.png", make_png(800, 800), "image/png")})
        assert resp.status_code == 200
        assert is_valid_png(resp.content)


# ────────────────────────────────────────────────────────────────────────────
# 10. Concurrent requests — verify temp-file isolation
# ────────────────────────────────────────────────────────────────────────────

class TestConcurrency:
    def _compress_jpeg(self, quality: int) -> tuple[int, bytes]:
        data = make_jpeg(300, 300)
        resp = client.post(
            f"/compress?quality={quality}",
            files={"file": ("img.jpg", data, "image/jpeg")},
        )
        return resp.status_code, resp.content

    def test_five_concurrent_jpeg_requests(self):
        qualities = [20, 40, 60, 80, 95]
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(self._compress_jpeg, q) for q in qualities]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 5
        for status, content in results:
            assert status == 200
            assert is_valid_jpeg(content)

    def test_mixed_format_concurrent_requests(self):
        def _req(fmt: str, data: bytes) -> tuple[int, str, bytes]:
            resp = client.post(
                "/compress?quality=70",
                files={"file": (f"img.{fmt}", data, f"image/{fmt}")},
            )
            return resp.status_code, resp.headers.get("content-type", ""), resp.content

        jobs = [
            ("jpg",  make_jpeg()),
            ("png",  make_png()),
            ("webp", make_webp()),
            ("jpg",  make_jpeg(300, 300)),
            ("png",  make_png(300, 300)),
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(_req, fmt, data) for fmt, data in jobs]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert all(status == 200 for status, _, _ in results)


# ────────────────────────────────────────────────────────────────────────────
# 11. Transform endpoint — resize, crop, flip, rotate
# ────────────────────────────────────────────────────────────────────────────

def _jpeg(w=300, h=200, quality=90):
    img = Image.new("RGB", (w, h), (100, 150, 200))
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

def _png(w=300, h=200, mode="RGB"):
    img = Image.new(mode, (w, h), (100, 150, 200))
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return buf.getvalue()


# ────────────────────────────────────────────────────────────────────────────
# 11. Transform pipeline — comprehensive tests
# ────────────────────────────────────────────────────────────────────────────

def _jpeg(w: int = 300, h: int = 200, quality: int = 90) -> bytes:
    img = Image.new("RGB", (w, h), (100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _png(w: int = 300, h: int = 200, mode: str = "RGB") -> bytes:
    img = Image.new(mode, (w, h), (100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _webp(w: int = 300, h: int = 200) -> bytes:
    img = Image.new("RGB", (w, h), (100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=80)
    return buf.getvalue()


def _transform(ops: list, file_tuple: tuple) -> object:
    """POST /transform with a pipeline JSON string."""
    return client.post(
        "/transform",
        data={"pipeline": json.dumps(ops)},
        files={"file": file_tuple},
    )


class TestTransformPipeline:

    # ── A. Empty pipeline — passthrough ─────────────────────────────────────

    def test_empty_pipeline_jpeg_200(self):
        r = _transform([], ("t.jpg", _jpeg(), "image/jpeg"))
        assert r.status_code == 200
        assert is_valid_jpeg(r.content)

    def test_empty_pipeline_png_200(self):
        r = _transform([], ("t.png", _png(), "image/png"))
        assert r.status_code == 200
        assert is_valid_png(r.content)

    def test_empty_pipeline_webp_200(self):
        r = _transform([], ("t.webp", _webp(), "image/webp"))
        assert r.status_code == 200
        assert is_valid_webp(r.content)

    def test_empty_pipeline_preserves_jpeg_format(self):
        r = _transform([], ("t.jpg", _jpeg(), "image/jpeg"))
        assert is_valid_jpeg(r.content)

    def test_empty_pipeline_preserves_png_format(self):
        r = _transform([], ("t.png", _png(), "image/png"))
        assert is_valid_png(r.content)

    # ── B. Resize op ─────────────────────────────────────────────────────────

    def test_resize_width_only(self):
        r = _transform([{"op": "resize", "width": 100}],
                        ("t.jpg", _jpeg(400, 300), "image/jpeg"))
        assert r.status_code == 200
        assert Image.open(io.BytesIO(r.content)).width <= 100

    def test_resize_height_only(self):
        r = _transform([{"op": "resize", "height": 80}],
                        ("t.jpg", _jpeg(400, 300), "image/jpeg"))
        assert r.status_code == 200
        assert Image.open(io.BytesIO(r.content)).height <= 80

    def test_resize_contain_preserves_aspect(self):
        # 400×200 (2:1) → contain 200×200 → expected 200×100
        r = _transform([{"op": "resize", "width": 200, "height": 200, "fit": "contain"}],
                        ("t.jpg", _jpeg(400, 200), "image/jpeg"))
        assert r.status_code == 200
        img = Image.open(io.BytesIO(r.content))
        assert img.width == 200 and img.height == 100

    def test_resize_cover_exact_dimensions(self):
        r = _transform(
            [{"op": "resize", "width": 100, "height": 100,
              "fit": "cover", "smart_crop": "centre"}],
            ("t.jpg", _jpeg(400, 200), "image/jpeg"))
        assert r.status_code == 200
        img = Image.open(io.BytesIO(r.content))
        assert img.width == 100 and img.height == 100

    def test_resize_fill_stretches(self):
        r = _transform(
            [{"op": "resize", "width": 150, "height": 80, "fit": "fill"}],
            ("t.jpg", _jpeg(300, 200), "image/jpeg"))
        assert r.status_code == 200
        img = Image.open(io.BytesIO(r.content))
        assert img.width == 150 and img.height == 80

    def test_resize_scale_down_never_enlarges(self):
        # 100×50 source, asking for width=9999 → should stay 100×50
        r = _transform([{"op": "resize", "width": 9999, "fit": "scale-down"}],
                        ("t.jpg", _jpeg(100, 50), "image/jpeg"))
        assert r.status_code == 200
        assert Image.open(io.BytesIO(r.content)).width <= 100

    def test_resize_no_dims_is_noop(self):
        # resize with neither width nor height → image unchanged
        r = _transform([{"op": "resize"}], ("t.jpg", _jpeg(200, 100), "image/jpeg"))
        assert r.status_code == 200
        img = Image.open(io.BytesIO(r.content))
        assert img.width == 200 and img.height == 100

    def test_resize_smart_crop_entropy(self):
        r = _transform(
            [{"op": "resize", "width": 80, "height": 80,
              "fit": "cover", "smart_crop": "entropy"}],
            ("t.jpg", _jpeg(400, 300), "image/jpeg"))
        assert r.status_code == 200

    def test_resize_smart_crop_attention(self):
        r = _transform(
            [{"op": "resize", "width": 80, "height": 80,
              "fit": "cover", "smart_crop": "attention"}],
            ("t.jpg", _jpeg(400, 300), "image/jpeg"))
        assert r.status_code == 200

    # ── C. Rotate op ─────────────────────────────────────────────────────────

    def test_rotate_90_swaps_dims(self):
        r = _transform([{"op": "rotate", "angle": 90}],
                        ("t.jpg", _jpeg(300, 200), "image/jpeg"))
        assert r.status_code == 200
        img = Image.open(io.BytesIO(r.content))
        assert img.width == 200 and img.height == 300

    def test_rotate_180_keeps_dims(self):
        r = _transform([{"op": "rotate", "angle": 180}],
                        ("t.jpg", _jpeg(300, 200), "image/jpeg"))
        assert r.status_code == 200
        img = Image.open(io.BytesIO(r.content))
        assert img.width == 300 and img.height == 200

    def test_rotate_270_swaps_dims(self):
        r = _transform([{"op": "rotate", "angle": 270}],
                        ("t.jpg", _jpeg(300, 200), "image/jpeg"))
        assert r.status_code == 200
        img = Image.open(io.BytesIO(r.content))
        assert img.width == 200 and img.height == 300

    def test_rotate_negative_90_swaps_dims(self):
        r = _transform([{"op": "rotate", "angle": -90}],
                        ("t.jpg", _jpeg(300, 200), "image/jpeg"))
        assert r.status_code == 200
        img = Image.open(io.BytesIO(r.content))
        assert img.width == 200 and img.height == 300

    def test_rotate_0_is_noop(self):
        r = _transform([{"op": "rotate", "angle": 0}],
                        ("t.jpg", _jpeg(300, 200), "image/jpeg"))
        assert r.status_code == 200
        img = Image.open(io.BytesIO(r.content))
        assert img.width == 300 and img.height == 200

    def test_rotate_45_returns_valid_image(self):
        # Arbitrary angle — dimensions will change due to padding
        r = _transform([{"op": "rotate", "angle": 45}],
                        ("t.jpg", _jpeg(200, 200), "image/jpeg"))
        assert r.status_code == 200
        assert is_valid_jpeg(r.content)

    # ── D. Flip op ───────────────────────────────────────────────────────────

    def test_flip_h(self):
        r = _transform([{"op": "flip", "direction": "h"}],
                        ("t.jpg", _jpeg(), "image/jpeg"))
        assert r.status_code == 200
        assert is_valid_jpeg(r.content)

    def test_flip_v(self):
        r = _transform([{"op": "flip", "direction": "v"}],
                        ("t.jpg", _jpeg(), "image/jpeg"))
        assert r.status_code == 200
        assert is_valid_jpeg(r.content)

    def test_flip_both(self):
        r = _transform([{"op": "flip", "direction": "both"}],
                        ("t.jpg", _jpeg(), "image/jpeg"))
        assert r.status_code == 200
        assert is_valid_jpeg(r.content)

    def test_flip_default_direction(self):
        # direction defaults to 'h'
        r = _transform([{"op": "flip"}], ("t.jpg", _jpeg(), "image/jpeg"))
        assert r.status_code == 200

    # ── E. Crop op ───────────────────────────────────────────────────────────

    def test_crop_exact(self):
        # 300×200 source, crop 80×60 from (10, 10)
        r = _transform([{"op": "crop", "left": 10, "top": 10, "width": 80, "height": 60}],
                        ("t.jpg", _jpeg(300, 200), "image/jpeg"))
        assert r.status_code == 200
        img = Image.open(io.BytesIO(r.content))
        assert img.width == 80 and img.height == 60

    def test_crop_at_origin(self):
        r = _transform([{"op": "crop", "left": 0, "top": 0, "width": 50, "height": 50}],
                        ("t.jpg", _jpeg(200, 200), "image/jpeg"))
        assert r.status_code == 200
        img = Image.open(io.BytesIO(r.content))
        assert img.width == 50 and img.height == 50

    def test_crop_clamped_to_image_bounds(self):
        # width=999 but image is only 100px wide → should clamp gracefully
        r = _transform([{"op": "crop", "left": 50, "top": 0, "width": 999, "height": 50}],
                        ("t.jpg", _jpeg(100, 100), "image/jpeg"))
        assert r.status_code == 200
        img = Image.open(io.BytesIO(r.content))
        assert img.width <= 50  # 100 - 50 = 50

    # ── F. Convert op ────────────────────────────────────────────────────────

    def test_convert_jpeg_to_png(self):
        r = _transform([{"op": "convert", "format": "png"}],
                        ("t.jpg", _jpeg(), "image/jpeg"))
        assert r.status_code == 200
        assert is_valid_png(r.content)

    def test_convert_jpeg_to_webp(self):
        r = _transform([{"op": "convert", "format": "webp"}],
                        ("t.jpg", _jpeg(), "image/jpeg"))
        assert r.status_code == 200
        assert is_valid_webp(r.content)

    def test_convert_png_to_jpeg(self):
        r = _transform([{"op": "convert", "format": "jpeg"}],
                        ("t.png", _png(), "image/png"))
        assert r.status_code == 200
        assert is_valid_jpeg(r.content)

    def test_convert_png_to_webp(self):
        r = _transform([{"op": "convert", "format": "webp"}],
                        ("t.png", _png(), "image/png"))
        assert r.status_code == 200
        assert is_valid_webp(r.content)

    def test_convert_quality_10_smaller_than_90(self):
        ops_low  = [{"op": "convert", "format": "jpeg", "quality": 10}]
        ops_high = [{"op": "convert", "format": "jpeg", "quality": 90}]
        r_low  = _transform(ops_low,  ("t.jpg", _jpeg(400, 400), "image/jpeg"))
        r_high = _transform(ops_high, ("t.jpg", _jpeg(400, 400), "image/jpeg"))
        assert r_low.status_code == 200 and r_high.status_code == 200
        assert len(r_low.content) < len(r_high.content)

    def test_convert_webp_lossless(self):
        r = _transform([{"op": "convert", "format": "webp", "lossless": True}],
                        ("t.jpg", _jpeg(), "image/jpeg"))
        assert r.status_code == 200
        assert is_valid_webp(r.content)

    def test_no_convert_preserves_format_jpeg(self):
        r = _transform([{"op": "resize", "width": 100}],
                        ("t.jpg", _jpeg(), "image/jpeg"))
        assert is_valid_jpeg(r.content)

    def test_no_convert_preserves_format_png(self):
        r = _transform([{"op": "resize", "width": 100}],
                        ("t.png", _png(), "image/png"))
        assert is_valid_png(r.content)

    def test_last_convert_wins(self):
        # First convert → png, second convert → webp; output should be webp
        ops = [
            {"op": "convert", "format": "png"},
            {"op": "convert", "format": "webp", "quality": 80},
        ]
        r = _transform(ops, ("t.jpg", _jpeg(), "image/jpeg"))
        assert r.status_code == 200
        assert is_valid_webp(r.content)

    # ── G. Pipeline ordering ─────────────────────────────────────────────────

    def test_resize_then_crop(self):
        ops = [
            {"op": "resize", "width": 200, "height": 100, "fit": "fill"},
            {"op": "crop",   "left": 0, "top": 0, "width": 50, "height": 50},
        ]
        r = _transform(ops, ("t.jpg", _jpeg(400, 300), "image/jpeg"))
        assert r.status_code == 200
        img = Image.open(io.BytesIO(r.content))
        assert img.width == 50 and img.height == 50

    def test_rotate_then_resize(self):
        # 300×200 → rotate 90 → 200×300 → resize width=100 → ≤100 wide
        ops = [
            {"op": "rotate", "angle": 90},
            {"op": "resize", "width": 100},
        ]
        r = _transform(ops, ("t.jpg", _jpeg(300, 200), "image/jpeg"))
        assert r.status_code == 200
        assert Image.open(io.BytesIO(r.content)).width <= 100

    def test_flip_then_rotate(self):
        ops = [{"op": "flip", "direction": "h"}, {"op": "rotate", "angle": 90}]
        r = _transform(ops, ("t.jpg", _jpeg(), "image/jpeg"))
        assert r.status_code == 200
        assert is_valid_jpeg(r.content)

    def test_multiple_resizes(self):
        ops = [{"op": "resize", "width": 400}, {"op": "resize", "width": 100}]
        r = _transform(ops, ("t.jpg", _jpeg(600, 400), "image/jpeg"))
        assert r.status_code == 200
        assert Image.open(io.BytesIO(r.content)).width <= 100

    def test_resize_convert_chain(self):
        ops = [
            {"op": "resize", "width": 150},
            {"op": "convert", "format": "webp", "quality": 75},
        ]
        r = _transform(ops, ("t.jpg", _jpeg(400, 300), "image/jpeg"))
        assert r.status_code == 200
        assert is_valid_webp(r.content)
        assert Image.open(io.BytesIO(r.content)).width <= 150

    # ── H. X-Timing header ───────────────────────────────────────────────────

    def test_x_timing_present(self):
        r = _transform([{"op": "resize", "width": 100}],
                        ("t.jpg", _jpeg(), "image/jpeg"))
        assert "x-timing" in r.headers

    def test_x_timing_has_request_id(self):
        r = _transform([], ("t.jpg", _jpeg(), "image/jpeg"))
        t = json.loads(r.headers["x-timing"])
        assert "request_id" in t
        assert isinstance(t["request_id"], str) and len(t["request_id"]) > 0

    def test_x_timing_has_pipeline_array(self):
        ops = [{"op": "resize", "width": 100}, {"op": "flip", "direction": "h"}]
        r = _transform(ops, ("t.jpg", _jpeg(), "image/jpeg"))
        t = json.loads(r.headers["x-timing"])
        assert "pipeline" in t
        assert isinstance(t["pipeline"], list)
        assert len(t["pipeline"]) == 2

    def test_x_timing_pipeline_op_names(self):
        ops = [{"op": "resize", "width": 100}, {"op": "convert", "format": "webp"}]
        r = _transform(ops, ("t.jpg", _jpeg(), "image/jpeg"))
        t = json.loads(r.headers["x-timing"])
        names = [step["op"] for step in t["pipeline"]]
        assert names == ["resize", "convert"]

    def test_x_timing_pipeline_has_ms(self):
        r = _transform([{"op": "rotate", "angle": 90}], ("t.jpg", _jpeg(), "image/jpeg"))
        t = json.loads(r.headers["x-timing"])
        for step in t["pipeline"]:
            assert "ms" in step
            assert isinstance(step["ms"], (int, float))

    def test_x_timing_dimensions(self):
        r = _transform(
            [{"op": "resize", "width": 100, "height": 100, "fit": "fill"}],
            ("t.jpg", _jpeg(300, 200), "image/jpeg"))
        t = json.loads(r.headers["x-timing"])
        assert t["input_width"] == 300
        assert t["input_height"] == 200
        assert t["output_width"] == 100
        assert t["output_height"] == 100

    def test_x_timing_output_format(self):
        ops = [{"op": "convert", "format": "webp"}]
        r = _transform(ops, ("t.jpg", _jpeg(), "image/jpeg"))
        t = json.loads(r.headers["x-timing"])
        assert t["output_format"] == "webp"

    def test_x_timing_total_ms_present(self):
        r = _transform([], ("t.jpg", _jpeg(), "image/jpeg"))
        t = json.loads(r.headers["x-timing"])
        assert "total_ms" in t
        assert t["total_ms"] >= 0

    # ── I. Compress X-Timing also has request_id ─────────────────────────────

    def test_compress_x_timing_has_request_id(self):
        r = client.post("/compress", files={"file": ("t.jpg", make_jpeg(), "image/jpeg")})
        assert r.status_code == 200
        t = json.loads(r.headers["x-timing"])
        assert "request_id" in t
        assert len(t["request_id"]) > 0

    def test_compress_x_timing_request_ids_are_unique(self):
        """Each request gets a distinct request_id."""
        ids = set()
        for _ in range(5):
            r = client.post("/compress",
                            files={"file": ("t.jpg", make_jpeg(), "image/jpeg")})
            t = json.loads(r.headers["x-timing"])
            ids.add(t["request_id"])
        assert len(ids) == 5

    # ── J. Content-Disposition for transform ─────────────────────────────────

    def test_transform_content_disposition(self):
        r = _transform([], ("photo.jpg", _jpeg(), "image/jpeg"))
        cd = r.headers.get("content-disposition", "")
        assert "photo_transformed.jpg" in cd

    def test_transform_png_content_disposition(self):
        r = _transform([], ("banner.png", _png(), "image/png"))
        cd = r.headers.get("content-disposition", "")
        assert "banner_transformed.png" in cd

    # ── K. Error cases ────────────────────────────────────────────────────────

    def test_invalid_json_returns_400(self):
        r = client.post("/transform",
                        data={"pipeline": "not valid json"},
                        files={"file": ("t.jpg", _jpeg(), "image/jpeg")})
        assert r.status_code == 400
        assert "JSON" in r.json().get("detail", "") or "json" in r.json().get("detail", "").lower()

    def test_pipeline_not_a_list_returns_400(self):
        r = client.post("/transform",
                        data={"pipeline": '{"op":"resize"}'},
                        files={"file": ("t.jpg", _jpeg(), "image/jpeg")})
        assert r.status_code == 400

    def test_unknown_op_returns_400(self):
        r = client.post("/transform",
                        data={"pipeline": '[{"op":"blur","radius":5}]'},
                        files={"file": ("t.jpg", _jpeg(), "image/jpeg")})
        assert r.status_code == 400

    def test_crop_missing_width_returns_400(self):
        r = client.post("/transform",
                        data={"pipeline": '[{"op":"crop","left":0,"top":0,"height":50}]'},
                        files={"file": ("t.jpg", _jpeg(), "image/jpeg")})
        assert r.status_code == 400

    def test_crop_missing_height_returns_400(self):
        r = client.post("/transform",
                        data={"pipeline": '[{"op":"crop","left":0,"top":0,"width":50}]'},
                        files={"file": ("t.jpg", _jpeg(), "image/jpeg")})
        assert r.status_code == 400

    def test_rotate_out_of_range_returns_400(self):
        r = client.post("/transform",
                        data={"pipeline": '[{"op":"rotate","angle":999}]'},
                        files={"file": ("t.jpg", _jpeg(), "image/jpeg")})
        assert r.status_code == 400

    def test_resize_width_too_large_returns_400(self):
        r = client.post("/transform",
                        data={"pipeline": '[{"op":"resize","width":99999}]'},
                        files={"file": ("t.jpg", _jpeg(), "image/jpeg")})
        assert r.status_code == 400

    def test_convert_invalid_format_returns_400(self):
        r = client.post("/transform",
                        data={"pipeline": '[{"op":"convert","format":"bmp"}]'},
                        files={"file": ("t.jpg", _jpeg(), "image/jpeg")})
        assert r.status_code == 400

    def test_convert_quality_out_of_range_returns_400(self):
        r = client.post("/transform",
                        data={"pipeline": '[{"op":"convert","format":"jpeg","quality":200}]'},
                        files={"file": ("t.jpg", _jpeg(), "image/jpeg")})
        assert r.status_code == 400

    def test_empty_file_returns_400(self):
        r = client.post("/transform",
                        data={"pipeline": "[]"},
                        files={"file": ("t.jpg", b"", "image/jpeg")})
        assert r.status_code == 400

    def test_unsupported_format_returns_415(self):
        gif_header = b"GIF89a\x01\x00\x01\x00\x00\xff\x00"
        r = client.post("/transform",
                        data={"pipeline": "[]"},
                        files={"file": ("t.gif", gif_header, "image/gif")})
        assert r.status_code == 415

    def test_flip_invalid_direction_returns_400(self):
        r = client.post("/transform",
                        data={"pipeline": '[{"op":"flip","direction":"diagonal"}]'},
                        files={"file": ("t.jpg", _jpeg(), "image/jpeg")})
        assert r.status_code == 400

    def test_missing_file_returns_422(self):
        r = client.post("/transform", data={"pipeline": "[]"})
        assert r.status_code == 422

    def test_rotate_missing_angle_returns_400(self):
        r = client.post("/transform",
                        data={"pipeline": '[{"op":"rotate"}]'},
                        files={"file": ("t.jpg", _jpeg(), "image/jpeg")})
        assert r.status_code == 400

    def test_convert_missing_format_returns_400(self):
        r = client.post("/transform",
                        data={"pipeline": '[{"op":"convert","quality":80}]'},
                        files={"file": ("t.jpg", _jpeg(), "image/jpeg")})
        assert r.status_code == 400

    # ── L. Audit-friendly: each request gets unique request_id ───────────────

    def test_transform_request_ids_are_unique(self):
        ids = set()
        for _ in range(4):
            r = _transform([], ("t.jpg", _jpeg(), "image/jpeg"))
            t = json.loads(r.headers["x-timing"])
            ids.add(t["request_id"])
        assert len(ids) == 4
