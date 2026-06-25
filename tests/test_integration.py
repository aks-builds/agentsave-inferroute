"""Integration tests: end-to-end routing paths.

These tests wire the real classifier + scorer + vLLM adapter together
and mock only the HTTP backend.  They verify the complete routing
decision chain without any mock substitutions except at the network edge.
"""
import json
import os
import importlib
import pytest
import httpx
import respx
from fastapi.testclient import TestClient


TURN1_BODY = {
    "model": "meta-llama/Llama-3-8b-instruct",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "stream": False,
}

TURN2_BODY = {
    "model": "meta-llama/Llama-3-8b-instruct",
    "messages": [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "What is its population?"},
    ],
    "stream": False,
}

BACKEND = "http://vllm-integration-test:8000"
BACKEND_CHAT = f"{BACKEND}/v1/chat/completions"


@pytest.fixture(scope="module")
def client(monkeypatch_module=None):
    """Build TestClient with vLLM backend configured."""
    os.environ["BACKEND_URL"] = BACKEND
    os.environ["BACKEND_TYPE"] = "vllm"
    os.environ.pop("AGENTSAVE_METRICS_URL", None)
    import inferroute.app as app_module
    importlib.reload(app_module)
    from inferroute.app import app
    return TestClient(app)


class TestTurn1FullPipeline:
    @respx.mock
    def test_turn1_no_decode_header(self, client):
        captured_headers = {}

        def capture(req: httpx.Request):
            captured_headers.update(dict(req.headers))
            return httpx.Response(200, json={"id": "t1", "choices": []})

        respx.post(BACKEND_CHAT).mock(side_effect=capture)
        client.post("/v1/chat/completions", json=TURN1_BODY)
        assert captured_headers.get("x-route-type") is None

    @respx.mock
    def test_turn1_returns_upstream_body(self, client):
        respx.post(BACKEND_CHAT).mock(
            return_value=httpx.Response(
                200,
                json={"id": "t1-body", "object": "chat.completion", "choices": [
                    {"message": {"role": "assistant", "content": "Paris"}}
                ]},
            )
        )
        resp = client.post("/v1/chat/completions", json=TURN1_BODY)
        assert resp.status_code == 200
        assert resp.json()["id"] == "t1-body"


class TestTurn2PlusPipeline:
    @respx.mock
    def test_turn2_has_decode_header(self, client):
        """With default weights (w_ttft=0.7, tpot=0.3) and static estimates
        (ttft_imp=68, tpot_deg=5), route_score = 0.7*68 - 0.3*5 = 46.1 > 0."""
        captured_headers = {}

        def capture(req: httpx.Request):
            captured_headers.update(dict(req.headers))
            return httpx.Response(200, json={"id": "t2", "choices": []})

        respx.post(BACKEND_CHAT).mock(side_effect=capture)
        client.post("/v1/chat/completions", json=TURN2_BODY)
        assert captured_headers.get("x-route-type") == "decode"

    @respx.mock
    def test_turn2_returns_upstream_body(self, client):
        respx.post(BACKEND_CHAT).mock(
            return_value=httpx.Response(
                200,
                json={"id": "t2-body", "choices": [
                    {"message": {"role": "assistant", "content": "2.2 million"}}
                ]},
            )
        )
        resp = client.post("/v1/chat/completions", json=TURN2_BODY)
        assert resp.status_code == 200
        assert resp.json()["id"] == "t2-body"


class TestHealthIntegration:
    def test_health_endpoint_live(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["backend_type"] == "vllm"
        assert "version" in body


class TestSGLangIntegration:
    @respx.mock
    def test_sglang_turn2_router_prefix(self, monkeypatch):
        monkeypatch.setenv("BACKEND_URL", "http://sglang-backend:30000")
        monkeypatch.setenv("BACKEND_TYPE", "sglang")
        monkeypatch.delenv("AGENTSAVE_METRICS_URL", raising=False)

        import inferroute.app as app_module
        importlib.reload(app_module)
        from inferroute.app import app
        from fastapi.testclient import TestClient as TC
        c = TC(app)

        captured_url = {}

        def capture(req: httpx.Request):
            captured_url["url"] = str(req.url)
            return httpx.Response(200, json={"id": "sg2", "choices": []})

        respx.post("http://sglang-backend:30000/v1/chat/completions").mock(
            side_effect=capture
        )
        c.post("/v1/chat/completions", json=TURN2_BODY)
        assert "router_prefix=decode" in captured_url["url"]
