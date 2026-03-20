"""
Functional smoke tests for the production image.

Run against a *live* server — validates that the production image boots correctly,
all runtime libraries (pyvips, libvips, libjxl, …) are present, and every
endpoint returns sensible responses.

Usage:
    SMOKE_URL=http://localhost:8000 pytest tests/smoke/ -v

In CI this is wired to the actual `app` Docker image started as a container.
"""

import base64
import io
import json
import os

import httpx
import pytest
from PIL import Image

BASE_URL = os.environ.get("SMOKE_URL", "http://localhost:8000").rstrip("/")


# ─── helpers ──────────────────────────────────────────────────────────────────

def _jpeg(w: int = 120, h: int = 80) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color=(200, 100, 50)).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _png(w: int = 120, h: int = 80) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color=(50, 150, 200)).save(buf, format="PNG")
    return buf.getvalue()


def _webp(w: int = 120, h: int = 80) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color=(100, 200, 100)).save(buf, format="WEBP", quality=80)
    return buf.getvalue()


def _b64(obj: dict) -> str:
    raw = json.dumps(obj).encode()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _timing(resp: httpx.Response) -> dict:
    return json.loads(resp.headers.get("x-timing", "{}"))


# ─── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=BASE_URL, timeout=30) as c:
        yield c


@pytest.fixture(scope="module")
def jpeg(): return _jpeg()

@pytest.fixture(scope="module")
def png(): return _png()

@pytest.fixture(scope="module")
def webp(): return _webp()


# ─── health / docs ────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_openapi_accessible(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        assert "paths" in r.json()

    def test_docs_page_accessible(self, client):
        r = client.get("/docs")
        assert r.status_code == 200


# ─── /compress ────────────────────────────────────────────────────────────────

class TestCompress:
    def test_jpeg_200(self, client, jpeg):
        r = client.post("/compress",
                        files={"file": ("photo.jpg", jpeg, "image/jpeg")},
                        data={"quality": "75"})
        assert r.status_code == 200
        assert "image/jpeg" in r.headers["content-type"]
        assert len(r.content) > 0

    def test_png_200(self, client, png):
        r = client.post("/compress",
                        files={"file": ("img.png", png, "image/png")})
        assert r.status_code == 200
        assert "image/png" in r.headers["content-type"]

    def test_webp_200(self, client, webp):
        r = client.post("/compress",
                        files={"file": ("img.webp", webp, "image/webp")})
        assert r.status_code == 200
        assert "image/webp" in r.headers["content-type"]

    def test_png_palette_mode(self, client, png):
        r = client.post("/compress",
                        files={"file": ("img.png", png, "image/png")},
                        data={"palette": "true", "colors": "64", "effort": "3"})
        assert r.status_code == 200

    def test_x_timing_structure(self, client, jpeg):
        r = client.post("/compress",
                        files={"file": ("t.jpg", jpeg, "image/jpeg")})
        assert r.status_code == 200
        t = _timing(r)
        assert "total_ms" in t
        assert "compress_ms" in t
        assert "request_id" in t
        assert isinstance(t["total_ms"], (int, float))

    def test_content_disposition_filename(self, client, jpeg):
        r = client.post("/compress",
                        files={"file": ("myphoto.jpg", jpeg, "image/jpeg")})
        assert r.status_code == 200
        cd = r.headers.get("content-disposition", "")
        assert "myphoto_compressed" in cd

    def test_unsupported_format_415(self, client):
        r = client.post("/compress",
                        files={"file": ("doc.txt", b"hello world", "text/plain")})
        assert r.status_code == 415

    def test_missing_file_422(self, client):
        r = client.post("/compress", data={"quality": "80"})
        assert r.status_code == 422

    def test_output_is_valid_image(self, client, jpeg):
        """Parse the response bytes back into a PIL Image — proves pyvips actually ran."""
        r = client.post("/compress",
                        files={"file": ("t.jpg", jpeg, "image/jpeg")},
                        data={"quality": "60"})
        assert r.status_code == 200
        img = Image.open(io.BytesIO(r.content))
        assert img.width > 0 and img.height > 0


# ─── /transform ───────────────────────────────────────────────────────────────

class TestTransform:
    def test_empty_pipeline_passthrough(self, client, jpeg):
        r = client.post("/transform",
                        files={"file": ("t.jpg", jpeg, "image/jpeg")},
                        data={"pipeline": "[]"})
        assert r.status_code == 200
        assert "image/jpeg" in r.headers["content-type"]

    def test_resize_reduces_dimensions(self, client, jpeg):
        pipeline = json.dumps([{"op": "resize", "width": 60, "height": 40}])
        r = client.post("/transform",
                        files={"file": ("t.jpg", jpeg, "image/jpeg")},
                        data={"pipeline": pipeline})
        assert r.status_code == 200
        img = Image.open(io.BytesIO(r.content))
        assert img.width <= 60 and img.height <= 40

    def test_convert_to_webp(self, client, jpeg):
        pipeline = json.dumps([{"op": "convert", "format": "webp", "quality": 75}])
        r = client.post("/transform",
                        files={"file": ("t.jpg", jpeg, "image/jpeg")},
                        data={"pipeline": pipeline})
        assert r.status_code == 200
        assert "image/webp" in r.headers["content-type"]

    def test_convert_to_png(self, client, jpeg):
        pipeline = json.dumps([{"op": "convert", "format": "png"}])
        r = client.post("/transform",
                        files={"file": ("t.jpg", jpeg, "image/jpeg")},
                        data={"pipeline": pipeline})
        assert r.status_code == 200
        assert "image/png" in r.headers["content-type"]

    def test_flip_horizontal(self, client, jpeg):
        pipeline = json.dumps([{"op": "flip", "direction": "h"}])
        r = client.post("/transform",
                        files={"file": ("t.jpg", jpeg, "image/jpeg")},
                        data={"pipeline": pipeline})
        assert r.status_code == 200

    def test_rotate_90(self, client, jpeg):
        pipeline = json.dumps([{"op": "rotate", "angle": 90}])
        r = client.post("/transform",
                        files={"file": ("t.jpg", jpeg, "image/jpeg")},
                        data={"pipeline": pipeline})
        assert r.status_code == 200
        # 90° rotation: width and height swap
        img = Image.open(io.BytesIO(r.content))
        assert img.width == 80 and img.height == 120

    def test_crop_op(self, client, jpeg):
        pipeline = json.dumps([{"op": "crop", "left": 10, "top": 10, "width": 50, "height": 40}])
        r = client.post("/transform",
                        files={"file": ("t.jpg", jpeg, "image/jpeg")},
                        data={"pipeline": pipeline})
        assert r.status_code == 200
        img = Image.open(io.BytesIO(r.content))
        assert img.width == 50 and img.height == 40

    def test_pipeline_chain(self, client, jpeg):
        pipeline = json.dumps([
            {"op": "resize", "width": 80},
            {"op": "flip", "direction": "v"},
            {"op": "convert", "format": "webp", "quality": 70},
        ])
        r = client.post("/transform",
                        files={"file": ("t.jpg", jpeg, "image/jpeg")},
                        data={"pipeline": pipeline})
        assert r.status_code == 200
        assert "image/webp" in r.headers["content-type"]

    def test_x_timing_pipeline_steps(self, client, jpeg):
        pipeline = json.dumps([
            {"op": "resize", "width": 80},
            {"op": "rotate", "angle": 180},
        ])
        r = client.post("/transform",
                        files={"file": ("t.jpg", jpeg, "image/jpeg")},
                        data={"pipeline": pipeline})
        assert r.status_code == 200
        t = _timing(r)
        steps = t.get("pipeline", [])
        assert len(steps) == 2
        assert steps[0]["op"] == "resize"
        assert steps[1]["op"] == "rotate"

    def test_invalid_op_400(self, client, jpeg):
        r = client.post("/transform",
                        files={"file": ("t.jpg", jpeg, "image/jpeg")},
                        data={"pipeline": '[{"op": "unknown_op"}]'})
        assert r.status_code == 400

    def test_unsupported_format_415(self, client):
        r = client.post("/transform",
                        files={"file": ("doc.gif", b"GIF89a", "image/gif")},
                        data={"pipeline": "[]"})
        assert r.status_code == 415


# ─── GET /compress/{b64} ──────────────────────────────────────────────────────

class TestGetCompress:
    def test_invalid_base64_400(self, client):
        r = client.get("/compress/!!!invalid!!!")
        assert r.status_code == 400

    def test_missing_url_field_400(self, client):
        r = client.get(f"/compress/{_b64({'quality': 80})}")
        assert r.status_code == 400

    def test_file_scheme_rejected_400(self, client):
        r = client.get(f"/compress/{_b64({'url': 'file:///etc/passwd'})}")
        assert r.status_code == 400

    def test_ftp_scheme_rejected_400(self, client):
        r = client.get(f"/compress/{_b64({'url': 'ftp://example.com/img.jpg'})}")
        assert r.status_code == 400

    def test_unreachable_url_400(self, client):
        # .invalid TLD never resolves (RFC 2606) — DNS fails immediately
        r = client.get(f"/compress/{_b64({'url': 'http://no-such-host.invalid/img.jpg'})}")
        assert r.status_code == 400


# ─── GET /transform/{b64} ─────────────────────────────────────────────────────

class TestGetTransform:
    def test_invalid_base64_400(self, client):
        r = client.get("/transform/!!!invalid!!!")
        assert r.status_code == 400

    def test_missing_url_field_400(self, client):
        r = client.get(f"/transform/{_b64({'pipeline': []})}")
        assert r.status_code == 400

    def test_bad_pipeline_op_400(self, client):
        r = client.get(f"/transform/{_b64({'url': 'http://example.com/img.jpg', 'pipeline': [{'op': 'bad'}]})}")
        assert r.status_code == 400

    def test_unreachable_url_400(self, client):
        r = client.get(f"/transform/{_b64({'url': 'http://no-such-host.invalid/img.jpg', 'pipeline': []})}")
        assert r.status_code == 400
