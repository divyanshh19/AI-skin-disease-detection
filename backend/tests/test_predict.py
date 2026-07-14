import io
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def _dummy_image_bytes():
    img = Image.new("RGB", (224, 224), color=(180, 140, 120))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_predict_rejects_bad_content_type():
    files = {"file": ("test.txt", b"not an image", "text/plain")}
    response = client.post("/api/v1/predict", files=files)
    assert response.status_code == 415


def test_predict_with_valid_image_returns_200_or_503():
    """503 is acceptable if models aren't trained/available in this test env."""
    buf = _dummy_image_bytes()
    files = {"file": ("test.jpg", buf, "image/jpeg")}
    response = client.post("/api/v1/predict", files=files)
    assert response.status_code in (200, 503)
