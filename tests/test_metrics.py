"""Tests for metrics emitter."""
import pytest
import httpx
import respx
from inferroute.metrics import MetricsEvent, MetricsEmitter


class TestMetricsEvent:
    def test_fields(self):
        event = MetricsEvent(
            request_type="turn2",
            routed_to="decode",
            ttft_ms=42.5,
            backend_type="vllm",
        )
        assert event.request_type == "turn2"
        assert event.routed_to == "decode"
        assert event.ttft_ms == 42.5
        assert event.backend_type == "vllm"


class TestMetricsEmitter:
    @pytest.mark.asyncio
    @respx.mock
    async def test_emit_posts_to_dashboard(self):
        mock_route = respx.post("https://dashboard.agentsave.io/api/events").mock(
            return_value=httpx.Response(200)
        )
        emitter = MetricsEmitter(
            dashboard_url="https://dashboard.agentsave.io/api/events",
            token="test-token",
        )
        event = MetricsEvent(
            request_type="turn1",
            routed_to="prefill",
            ttft_ms=120.0,
            backend_type="vllm",
        )
        await emitter.emit(event)
        assert mock_route.called

    @pytest.mark.asyncio
    @respx.mock
    async def test_emit_sends_auth_header(self):
        captured = {}

        def capture(req: httpx.Request):
            captured["auth"] = req.headers.get("authorization")
            return httpx.Response(200)

        respx.post("https://dashboard.agentsave.io/api/events").mock(side_effect=capture)
        emitter = MetricsEmitter(
            dashboard_url="https://dashboard.agentsave.io/api/events",
            token="my-secret-token",
        )
        await emitter.emit(
            MetricsEvent(request_type="turn2", routed_to="decode", ttft_ms=30.0, backend_type="sglang")
        )
        assert captured["auth"] == "Bearer my-secret-token"

    @pytest.mark.asyncio
    @respx.mock
    async def test_emit_does_not_raise_on_http_error(self):
        """404 / 500 from dashboard must not propagate to caller."""
        respx.post("https://dashboard.agentsave.io/api/events").mock(
            return_value=httpx.Response(500)
        )
        emitter = MetricsEmitter(
            dashboard_url="https://dashboard.agentsave.io/api/events",
            token="tok",
        )
        # Should complete without raising
        await emitter.emit(
            MetricsEvent(request_type="turn1", routed_to="prefill", ttft_ms=100.0, backend_type="vllm")
        )

    @pytest.mark.asyncio
    @respx.mock
    async def test_emit_does_not_raise_on_connection_error(self):
        """Network failure must be swallowed silently."""
        respx.post("https://dashboard.agentsave.io/api/events").mock(
            side_effect=httpx.ConnectError("unreachable")
        )
        emitter = MetricsEmitter(
            dashboard_url="https://dashboard.agentsave.io/api/events",
            token="tok",
        )
        await emitter.emit(
            MetricsEvent(request_type="turn2", routed_to="decode", ttft_ms=35.0, backend_type="sglang")
        )

    def test_from_env_returns_none_without_url(self, monkeypatch):
        monkeypatch.delenv("AGENTSAVE_METRICS_URL", raising=False)
        monkeypatch.delenv("AGENTSAVE_TOKEN", raising=False)
        assert MetricsEmitter.from_env() is None

    def test_from_env_returns_emitter_with_url(self, monkeypatch):
        monkeypatch.setenv("AGENTSAVE_METRICS_URL", "https://dashboard.agentsave.io/api/events")
        monkeypatch.setenv("AGENTSAVE_TOKEN", "env-token")
        emitter = MetricsEmitter.from_env()
        assert emitter is not None
        assert emitter.token == "env-token"
