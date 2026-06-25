"""Tests for vLLM backend adapter."""
import pytest
from inferroute.adapters.base import RouteTarget, AdaptedRequest
from inferroute.adapters.vllm import VLLMAdapter


class TestVLLMAdapter:
    def test_decode_request_adds_header(self):
        adapter = VLLMAdapter(backend_url="http://vllm-service:8000")
        original_headers = {"content-type": "application/json", "authorization": "Bearer tok"}
        result = adapter.decode_request(
            original_url="http://vllm-service:8000/v1/chat/completions",
            original_headers=original_headers,
        )
        assert result.headers["x-route-type"] == "decode"

    def test_decode_request_preserves_existing_headers(self):
        adapter = VLLMAdapter(backend_url="http://vllm-service:8000")
        result = adapter.decode_request(
            original_url="http://vllm-service:8000/v1/chat/completions",
            original_headers={"authorization": "Bearer tok"},
        )
        assert result.headers["authorization"] == "Bearer tok"
        assert result.headers["x-route-type"] == "decode"

    def test_decode_request_url_unchanged(self):
        adapter = VLLMAdapter(backend_url="http://vllm-service:8000")
        url = "http://vllm-service:8000/v1/chat/completions"
        result = adapter.decode_request(original_url=url, original_headers={})
        assert result.url == url

    def test_decode_request_route_target_is_decode(self):
        adapter = VLLMAdapter(backend_url="http://vllm-service:8000")
        result = adapter.decode_request(
            original_url="http://vllm-service:8000/v1/chat/completions",
            original_headers={},
        )
        assert result.route_target == RouteTarget.DECODE

    def test_prefill_request_no_decode_header(self):
        adapter = VLLMAdapter(backend_url="http://vllm-service:8000")
        result = adapter.prefill_request(
            original_url="http://vllm-service:8000/v1/chat/completions",
            original_headers={"content-type": "application/json"},
        )
        assert "x-route-type" not in result.headers

    def test_prefill_request_route_target_is_prefill(self):
        adapter = VLLMAdapter(backend_url="http://vllm-service:8000")
        result = adapter.prefill_request(
            original_url="http://vllm-service:8000/v1/chat/completions",
            original_headers={},
        )
        assert result.route_target == RouteTarget.PREFILL

    def test_does_not_mutate_input_headers(self):
        """Adapter must not modify the caller's header dict in-place."""
        adapter = VLLMAdapter(backend_url="http://vllm-service:8000")
        original = {"content-type": "application/json"}
        adapter.decode_request(
            original_url="http://vllm-service:8000/v1/chat/completions",
            original_headers=original,
        )
        assert "x-route-type" not in original
