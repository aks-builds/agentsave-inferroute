"""Tests for async dispatcher."""
import json
import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock
from inferroute.adapters.base import AdaptedRequest, RouteTarget
from inferroute.classifier import TurnType
from inferroute.dispatcher import Dispatcher


def _make_fake_request(body: dict, headers: dict | None = None) -> MagicMock:
    """Build a mock FastAPI Request with the given JSON body."""
    req = MagicMock()
    req.headers = httpx.Headers(headers or {"content-type": "application/json"})
    req.method = "POST"
    req.url = MagicMock()
    req.url.path = "/v1/chat/completions"
    req.url.query = ""
    req.body = AsyncMock(return_value=json.dumps(body).encode())
    return req


def _make_adapter(route_target: RouteTarget, url: str = "http://backend/v1/chat/completions") -> MagicMock:
    adapter = MagicMock()
    adapted = AdaptedRequest(url=url, headers={}, route_target=route_target)
    adapter.decode_request.return_value = adapted
    adapter.prefill_request.return_value = AdaptedRequest(
        url=url, headers={}, route_target=RouteTarget.PREFILL
    )
    return adapter


def _make_scorer(should_decode: bool) -> MagicMock:
    scorer = MagicMock()
    scorer.should_route_to_decode.return_value = should_decode
    return scorer


class TestDispatcher:
    @pytest.mark.asyncio
    async def test_turn1_calls_prefill_request(self, respx_mock):
        adapter = _make_adapter(RouteTarget.PREFILL)
        scorer = _make_scorer(should_decode=False)
        dispatcher = Dispatcher(adapter=adapter, scorer=scorer)

        body = {"model": "llama", "messages": [{"role": "user", "content": "Hi"}]}
        req = _make_fake_request(body)

        respx_mock.post("http://backend/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"id": "1", "choices": []})
        )

        await dispatcher.dispatch(req)
        adapter.prefill_request.assert_called_once()
        adapter.decode_request.assert_not_called()

    @pytest.mark.asyncio
    async def test_turn2_positive_score_calls_decode_request(self, respx_mock):
        adapter = _make_adapter(RouteTarget.DECODE)
        scorer = _make_scorer(should_decode=True)
        dispatcher = Dispatcher(adapter=adapter, scorer=scorer)

        body = {
            "model": "llama",
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "More"},
            ],
        }
        req = _make_fake_request(body)

        respx_mock.post("http://backend/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"id": "2", "choices": []})
        )

        await dispatcher.dispatch(req)
        adapter.decode_request.assert_called_once()
        adapter.prefill_request.assert_not_called()

    @pytest.mark.asyncio
    async def test_turn2_negative_score_calls_prefill_request(self, respx_mock):
        adapter = _make_adapter(RouteTarget.PREFILL)
        scorer = _make_scorer(should_decode=False)
        dispatcher = Dispatcher(adapter=adapter, scorer=scorer)

        body = {
            "model": "llama",
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "More"},
            ],
        }
        req = _make_fake_request(body)

        respx_mock.post("http://backend/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"id": "3", "choices": []})
        )

        await dispatcher.dispatch(req)
        adapter.prefill_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatch_returns_httpx_response(self, respx_mock):
        adapter = _make_adapter(RouteTarget.PREFILL)
        scorer = _make_scorer(should_decode=False)
        dispatcher = Dispatcher(adapter=adapter, scorer=scorer)

        body = {"model": "llama", "messages": [{"role": "user", "content": "Hi"}]}
        req = _make_fake_request(body)

        respx_mock.post("http://backend/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"id": "4", "choices": [{"message": {"content": "Hi"}}]})
        )

        response = await dispatcher.dispatch(req)
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "4"
