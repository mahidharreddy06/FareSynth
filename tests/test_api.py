import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "FareSynth AI" in response.text

def test_api_info():
    response = client.get("/api/info")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "FareSynth AI"
    assert "endpoints" in data

def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "architecture" in data

def test_api_routes():
    response = client.get("/routes")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] > 0
    assert len(data["routes"]) > 0

def test_api_generate_endpoint():
    payload = {
        "route": "JFK-LHR",
        "season": "Summer",
        "demand_scenario": "Normal",
        "num_sequences": 2,
        "seq_len": 30
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_flights_generated"] == 2
    assert len(data["flights"]) == 2
    assert len(data["flights"][0]["timeline"]) == 30

def test_api_pricing_simulation_endpoint():
    payload = {
        "route": "LAX-ORD",
        "demand_scenario": "Surge",
        "base_fare": 280.0,
        "capacity": 180
    }
    response = client.post("/pricing/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "LAX-ORD"
    assert len(data["results"]) == 4
    assert data["recommended_strategy"] is not None

