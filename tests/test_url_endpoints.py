"""
Tests for the URL-based GET /compress/{payload} and GET /transform/{payload}
endpoints.

Strategy
────────
• ``app.main.download_to_temp`` is patched via pytest monkeypatch so we
  never make real HTTP calls.  Each test provides a fake async function that
  writes a test image to a temp file and returns ``(path, size)``.
• ``app.urlproc.decode_payload`` / ``encode_payload`` are tested directly
  (pure functions — no patching needed).
• Error paths are tested by making the fake download raise the expected
  exception type.
"""
import base64
import io
import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.urlproc import decode_payload, encode_payload

client = TestClient(app)


# ────────────────────────────────────────────────────────────────────────────
# Image factories (reused from test_compress.py)
# ────────────────────────────────────────────────────────────────────────────

def make_jpeg(w: int = 200, h: int = 150) -> bytes:
    img = Image.new("RGB", (w, h))
    pixels = img.load()
    for x in range(w):
        for y in range(h):
            pixels[x, y] = (x % 256, y % 256, (x + y) % 256)
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=90); return buf.getvalue()


def make_png(w: int = 200, h: int = 150) -> bytes:
    img = Image.new("RGB", (w, h))
    pixels = img.load()
    for x in range(w):
        for y in range(h):
            pixels[x, y] = (x % 256, y % 256, (x + y) % 256)
    buf = io.BytesIO(); img.save(buf, format="PNG"); return buf.getvalue()


def make_webp(w: int = 200, h: int = 150) -> bytes:
    img = Image.new("RGB", (w, h))
    buf = io.BytesIO(); img.save(buf, format="WEBP", quality=80); return buf.getvalue()


def is_valid_jpeg(data: bytes) -> bool:
    return data[:3] == b"\xff\xd8\xff"


def is_valid_png(data: bytes) -> bool:
    return data[:8] == b"\x89PNG\r\n\x1a\n"


def is_valid_webp(data: bytes) -> bool:
    return data[:4] == b"RIFF" and data[8:12] == b"WEBP"


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def b64(obj: dict) -> str:
    """Encode dict → base64url string (no padding)."""
    return encode_payload(obj)


def _make_fake_download(image_bytes: bytes, suffix: str = ".jpg"):
    """Return an async function that writes *image_bytes* to a temp file."""
    async def _fake(url: str):
        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(image_bytes)
        return path, len(image_bytes)
    return _fake


# ────────────────────────────────────────────────────────────────────────────
# 1. Payload codec
# ────────────────────────────────────────────────────────────────────────────

class TestPayloadCodec:
    def test_roundtrip_simple(self):
        obj = {"url": "https://example.com/img.jpg", "quality": 75}
        assert decode_payload(encode_payload(obj)) == obj

    def test_roundtrip_pipeline(self):
        obj = {"url": "https://x.com/a.png",
               "pipeline": [{"op": "resize", "width": 400}, {"op": "convert", "format": "webp"}]}
        assert decode_payload(encode_payload(obj)) == obj

    def test_no_padding_accepted(self):
        raw = base64.urlsafe_b64encode(b'{"url":"https://x.com"}').rstrip(b"=").decode()
        assert decode_payload(raw)["url"] == "https://x.com"

    def test_with_padding_accepted(self):
        raw = base64.urlsafe_b64encode(b'{"url":"https://x.com"}').decode()
        assert decode_payload(raw)["url"] == "https://x.com"

    def test_invalid_base64_raises(self):
        with pytest.raises(ValueError, match="base64url"):
            decode_payload("not!!valid!!base64!!")

    def test_invalid_json_raises(self):
        bad = base64.urlsafe_b64encode(b"not json").rstrip(b"=").decode()
        with pytest.raises(ValueError, match="JSON"):
            decode_payload(bad)

    def test_non_dict_raises(self):
        raw = base64.urlsafe_b64encode(b"[1,2,3]").rstrip(b"=").decode()
        with pytest.raises(ValueError, match="dict"):
            decode_payload(raw)

    def test_unicode_url_survives_roundtrip(self):
        obj = {"url": "https://例子.com/图片.jpg"}
        assert decode_payload(encode_payload(obj)) == obj


# ────────────────────────────────────────────────────────────────────────────
# 2. GET /compress/{payload}
# ────────────────────────────────────────────────────────────────────────────

class TestCompressUrl:

    def test_jpeg_200(self, monkeypatch):
        monkeypatch.setattr("app.main.download_to_temp",
                            _make_fake_download(make_jpeg(), ".jpg"))
        r = client.get(f"/compress/{b64({'url':'https://x.com/a.jpg'})}")
        assert r.status_code == 200
        assert is_valid_jpeg(r.content)

    def test_png_200(self, monkeypatch):
        monkeypatch.setattr("app.main.download_to_temp",
                            _make_fake_download(make_png(), ".png"))
        r = client.get(f"/compress/{b64({'url':'https://x.com/a.png'})}")
        assert r.status_code == 200
        assert is_valid_png(r.content)

    def test_webp_200(self, monkeypatch):
        monkeypatch.setattr("app.main.download_to_temp",
                            _make_fake_download(make_webp(), ".webp"))
        r = client.get(f"/compress/{b64({'url':'https://x.com/a.webp'})}")
        assert r.status_code == 200
        assert is_valid_webp(r.content)

    def test_quality_param_used(self, monkeypatch):
        monkeypatch.setattr("app.main.download_to_temp",
                            _make_fake_download(make_jpeg(400, 400), ".jpg"))
        r_lo = client.get(f"/compress/{b64({'url':'https://x.com/a.jpg','quality':5})}")
        r_hi = client.get(f"/compress/{b64({'url':'https://x.com/a.jpg','quality':99})}")
        assert r_lo.status_code == 200 and r_hi.status_code == 200
        assert len(r_lo.content) < len(r_hi.content)

    def test_default_quality_applied(self, monkeypatch):
        monkeypatch.setattr("app.main.download_to_temp",
                            _make_fake_download(make_jpeg(), ".jpg"))
        r = client.get(f"/compress/{b64({'url':'https://x.com/a.jpg'})}")
        assert r.status_code == 200

    def test_content_type_jpeg(self, monkeypatch):
        monkeypatch.setattr("app.main.download_to_temp",
                            _make_fake_download(make_jpeg(), ".jpg"))
        r = client.get(f"/compress/{b64({'url':'https://x.com/a.jpg'})}")
        assert "image/jpeg" in r.headers["content-type"]

    def test_x_timing_present(self, monkeypatch):
        monkeypatch.setattr("app.main.download_to_temp",
                            _make_fake_download(make_jpeg(), ".jpg"))
        r = client.get(f"/compress/{b64({'url':'https://x.com/a.jpg'})}")
        assert "x-timing" in r.headers

    def test_x_timing_has_request_id(self, monkeypatch):
        monkeypatch.setattr("app.main.download_to_temp",
                            _make_fake_download(make_jpeg(), ".jpg"))
        r = client.get(f"/compress/{b64({'url':'https://x.com/a.jpg'})}")
        t = json.loads(r.headers["x-timing"])
        assert "request_id" in t and t["request_id"]

    def test_x_timing_has_download_ms(self, monkeypatch):
        monkeypatch.setattr("app.main.download_to_temp",
                            _make_fake_download(make_jpeg(), ".jpg"))
        r = client.get(f"/compress/{b64({'url':'https://x.com/a.jpg'})}")
        t = json.loads(r.headers["x-timing"])
        assert "download_ms" in t

    def test_content_disposition_filename(self, monkeypatch):
        monkeypatch.setattr("app.main.download_to_temp",
                            _make_fake_download(make_jpeg(), ".jpg"))
        r = client.get(f"/compress/{b64({'url':'https://x.com/photo.jpg'})}")
        assert "photo_compressed" in r.headers.get("content-disposition", "")

    # ── error paths ──────────────────────────────────────────────────────────

    def test_invalid_base64_returns_400(self):
        r = client.get("/compress/!!not-valid!!")
        assert r.status_code == 400

    def test_invalid_json_returns_400(self):
        bad = base64.urlsafe_b64encode(b"not json").rstrip(b"=").decode()
        r = client.get(f"/compress/{bad}")
        assert r.status_code == 400

    def test_missing_url_returns_400(self):
        r = client.get(f"/compress/{b64({'quality':80})}")
        assert r.status_code == 400

    def test_invalid_url_scheme_returns_400(self, monkeypatch):
        async def _bad_dl(url):
            raise ValueError("URL scheme must be http or https, got 'file'")
        monkeypatch.setattr("app.main.download_to_temp", _bad_dl)
        r = client.get(f"/compress/{b64({'url':'file:///etc/passwd'})}")
        assert r.status_code == 400

    def test_download_error_returns_400(self, monkeypatch):
        import httpx as _httpx
        async def _fail(url):
            raise _httpx.HTTPStatusError("404", request=None, response=None)
        monkeypatch.setattr("app.main.download_to_temp", _fail)
        r = client.get(f"/compress/{b64({'url':'https://x.com/missing.jpg'})}")
        assert r.status_code == 400

    def test_unsupported_format_returns_415(self, monkeypatch):
        # Valid base64/JSON but downloaded bytes are not a supported image
        monkeypatch.setattr("app.main.download_to_temp",
                            _make_fake_download(b"GIF89a\x01\x00\x01\x00\x00", ".gif"))
        r = client.get(f"/compress/{b64({'url':'https://x.com/anim.gif'})}")
        assert r.status_code == 415


# ────────────────────────────────────────────────────────────────────────────
# 3. GET /transform/{payload}
# ────────────────────────────────────────────────────────────────────────────

class TestTransformUrl:

    def test_empty_pipeline_jpeg_200(self, monkeypatch):
        monkeypatch.setattr("app.main.download_to_temp",
                            _make_fake_download(make_jpeg(), ".jpg"))
        r = client.get(f"/transform/{b64({'url':'https://x.com/a.jpg'})}")
        assert r.status_code == 200
        assert is_valid_jpeg(r.content)

    def test_empty_pipeline_png_200(self, monkeypatch):
        monkeypatch.setattr("app.main.download_to_temp",
                            _make_fake_download(make_png(), ".png"))
        r = client.get(f"/transform/{b64({'url':'https://x.com/a.png','pipeline':[]})}")
        assert r.status_code == 200
        assert is_valid_png(r.content)

    def test_resize_op(self, monkeypatch):
        monkeypatch.setattr("app.main.download_to_temp",
                            _make_fake_download(make_jpeg(400, 300), ".jpg"))
        ops = [{"op": "resize", "width": 100}]
        r = client.get(f"/transform/{b64({'url':'https://x.com/a.jpg','pipeline':ops})}")
        assert r.status_code == 200
        assert Image.open(io.BytesIO(r.content)).width <= 100

    def test_convert_to_webp(self, monkeypatch):
        monkeypatch.setattr("app.main.download_to_temp",
                            _make_fake_download(make_jpeg(), ".jpg"))
        ops = [{"op": "convert", "format": "webp", "quality": 80}]
        r = client.get(f"/transform/{b64({'url':'https://x.com/a.jpg','pipeline':ops})}")
        assert r.status_code == 200
        assert is_valid_webp(r.content)

    def test_convert_to_png(self, monkeypatch):
        monkeypatch.setattr("app.main.download_to_temp",
                            _make_fake_download(make_jpeg(), ".jpg"))
        ops = [{"op": "convert", "format": "png"}]
        r = client.get(f"/transform/{b64({'url':'https://x.com/a.jpg','pipeline':ops})}")
        assert r.status_code == 200
        assert is_valid_png(r.content)

    def test_pipeline_chain(self, monkeypatch):
        monkeypatch.setattr("app.main.download_to_temp",
                            _make_fake_download(make_jpeg(400, 300), ".jpg"))
        ops = [{"op": "resize", "width": 200}, {"op": "convert", "format": "webp"}]
        r = client.get(f"/transform/{b64({'url':'https://x.com/a.jpg','pipeline':ops})}")
        assert r.status_code == 200
        assert is_valid_webp(r.content)
        assert Image.open(io.BytesIO(r.content)).width <= 200

    def test_x_timing_has_pipeline_array(self, monkeypatch):
        monkeypatch.setattr("app.main.download_to_temp",
                            _make_fake_download(make_jpeg(), ".jpg"))
        ops = [{"op": "flip", "direction": "h"}, {"op": "rotate", "angle": 90}]
        r = client.get(f"/transform/{b64({'url':'https://x.com/a.jpg','pipeline':ops})}")
        t = json.loads(r.headers["x-timing"])
        assert isinstance(t.get("pipeline"), list) and len(t["pipeline"]) == 2

    def test_x_timing_has_download_ms(self, monkeypatch):
        monkeypatch.setattr("app.main.download_to_temp",
                            _make_fake_download(make_jpeg(), ".jpg"))
        r = client.get(f"/transform/{b64({'url':'https://x.com/a.jpg'})}")
        t = json.loads(r.headers["x-timing"])
        assert "download_ms" in t

    def test_content_disposition_filename(self, monkeypatch):
        monkeypatch.setattr("app.main.download_to_temp",
                            _make_fake_download(make_jpeg(), ".jpg"))
        r = client.get(f"/transform/{b64({'url':'https://x.com/banner.jpg'})}")
        assert "banner_transformed" in r.headers.get("content-disposition", "")

    # ── error paths ──────────────────────────────────────────────────────────

    def test_invalid_base64_returns_400(self):
        r = client.get("/transform/!!not-valid!!")
        assert r.status_code == 400

    def test_missing_url_returns_400(self):
        r = client.get(f"/transform/{b64({'pipeline':[]})}")
        assert r.status_code == 400

    def test_pipeline_not_list_returns_400(self):
        r = client.get(f"/transform/{b64({'url':'https://x.com/a.jpg','pipeline':'resize'})}")
        assert r.status_code == 400

    def test_unknown_op_returns_400(self, monkeypatch):
        monkeypatch.setattr("app.main.download_to_temp",
                            _make_fake_download(make_jpeg(), ".jpg"))
        ops = [{"op": "blur", "radius": 5}]
        r = client.get(f"/transform/{b64({'url':'https://x.com/a.jpg','pipeline':ops})}")
        assert r.status_code == 400

    def test_rotate_out_of_range_returns_400(self, monkeypatch):
        monkeypatch.setattr("app.main.download_to_temp",
                            _make_fake_download(make_jpeg(), ".jpg"))
        ops = [{"op": "rotate", "angle": 999}]
        r = client.get(f"/transform/{b64({'url':'https://x.com/a.jpg','pipeline':ops})}")
        assert r.status_code == 400

    def test_convert_invalid_format_returns_400(self, monkeypatch):
        monkeypatch.setattr("app.main.download_to_temp",
                            _make_fake_download(make_jpeg(), ".jpg"))
        ops = [{"op": "convert", "format": "gif"}]
        r = client.get(f"/transform/{b64({'url':'https://x.com/a.jpg','pipeline':ops})}")
        assert r.status_code == 400

    def test_download_error_returns_400(self, monkeypatch):
        import httpx as _httpx
        async def _fail(url):
            raise _httpx.RequestError("connection refused")
        monkeypatch.setattr("app.main.download_to_temp", _fail)
        r = client.get(f"/transform/{b64({'url':'https://x.com/a.jpg'})}")
        assert r.status_code == 400

    def test_unsupported_format_returns_415(self, monkeypatch):
        monkeypatch.setattr("app.main.download_to_temp",
                            _make_fake_download(b"\x00" * 64, ".bin"))
        r = client.get(f"/transform/{b64({'url':'https://x.com/a.bin'})}")
        assert r.status_code == 415


# ────────────────────────────────────────────────────────────────────────────
# 4. urlproc unit tests (validate_source_url + decode edge cases)
# ────────────────────────────────────────────────────────────────────────────

class TestValidateSourceUrl:
    from app.urlproc import validate_source_url

    def test_http_ok(self):
        from app.urlproc import validate_source_url
        validate_source_url("http://example.com/img.jpg")  # must not raise

    def test_https_ok(self):
        from app.urlproc import validate_source_url
        validate_source_url("https://cdn.example.com/img.png")  # must not raise

    def test_file_scheme_raises(self):
        from app.urlproc import validate_source_url
        with pytest.raises(ValueError):
            validate_source_url("file:///etc/passwd")

    def test_ftp_scheme_raises(self):
        from app.urlproc import validate_source_url
        with pytest.raises(ValueError):
            validate_source_url("ftp://example.com/img.jpg")

    def test_empty_url_raises(self):
        from app.urlproc import validate_source_url
        with pytest.raises(ValueError):
            validate_source_url("https://")

    def test_no_scheme_raises(self):
        from app.urlproc import validate_source_url
        with pytest.raises(ValueError):
            validate_source_url("example.com/img.jpg")

