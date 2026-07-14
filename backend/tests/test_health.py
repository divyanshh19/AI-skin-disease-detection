from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "cnn_model_loaded" in body
    assert "vit_model_loaded" in body


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
