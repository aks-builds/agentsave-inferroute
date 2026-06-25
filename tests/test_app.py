"""Tests for FastAPI proxy app."""
import json
import os
import pytest
import httpx
import respx
from fastapi.testclient import TestClient


@pytest.fixture
def app_env(monkeypatch):
    """Set minimum required env vars for the app."""
    monkeypatch.setenv("BACKEND_URL", "http://vllm-backend:8000")
    monkeypatch.setenv("BACKEND_TYPE", "vllm")
    monkeypatch.delenv("AGENTSAVE_METRICS_URL", raising=False)


@pytest.fixture
def test_client(app_env):
    """Import app after env vars are set."""
    # Must import AFTER monkeypatch to pick up env vars at module level.
    import importlib
    import inferroute.app as app_module
    importlib.reload(app_module)
    from inferroute.app import app
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, test_client):
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok(self, test_client):
        data = test_client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_includes_backend_type(self, test_client):
        data = test_client.get("/health").json()
        assert data["backend_type"] == "vllm"

    def test_health_includes_version(self, test_client):
        data = test_client.get("/health").json()
        assert data["version"] == "0.1.0"


class TestChatCompletionsProxy:
    @respx.mock
    def test_turn1_proxied_to_backend(self, test_client):
        respx.post("http://vllm-backend:8000/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={"id": "chatcmpl-1", "object": "chat.completion", "choices": []},
            )
        )
        body = {
            "model": "meta-llama/Llama-3-8b-instruct",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        response = test_client.post("/v1/chat/completions", json=body)
        assert response.status_code == 200
        assert response.json()["id"] == "chatcmpl-1"

    @respx.mock
    def test_turn2_decode_header_added_for_vllm(self, test_client):
        captured = {}

        def capture_and_respond(req: httpx.Request):
            captured["x-route-type"] = req.headers.get("x-route-type")
            return httpx.Response(200, json={"id": "chatcmpl-2", "choices": []})

        respx.post("http://vllm-backend:8000/v1/chat/completions").mock(
            side_effect=capture_and_respond
        )
        body = {
            "model": "meta-llama/Llama-3-8b-instruct",
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "Tell me more"},
            ],
        }
        test_client.post("/v1/chat/completions", json=body)
        assert captured["x-route-type"] == "decode"

    @respx.mock
    def test_backend_error_propagated(self, test_client):
        respx.post("http://vllm-backend:8000/v1/chat/completions").mock(
            return_value=httpx.Response(503, text="Service Unavailable")
        )
        body = {"model": "llama", "messages": [{"role": "user", "content": "Hi"}]}
        response = test_client.post("/v1/chat/completions", json=body)
        assert response.status_code == 503
