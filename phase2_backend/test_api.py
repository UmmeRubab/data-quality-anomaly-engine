"""API smoke tests; requires httpx for FastAPI TestClient."""
from fastapi.testclient import TestClient
from phase2_backend.main import app

client = TestClient(app)


def test_health():
    response = client.get('/api/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_rejects_invalid_noise_rate():
    response = client.post('/api/experiments/label-noise', json={'noise_rate': 1.5, 'seed': 42})
    assert response.status_code == 422


def test_rejects_path_traversal():
    response = client.get('/api/results/preview', params={'filename': '../secrets.csv'})
    assert response.status_code == 400
