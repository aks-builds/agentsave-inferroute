"""Tests for SGLang backend adapter."""
import pytest
from urllib.parse import urlparse, parse_qs
from inferroute.adapters.base import RouteTarget
from inferroute.adapters.sglang import SGLangAdapter


class TestSGLangAdapter:
    def test_decode_request_appends_router_prefix(self):
        adapter = SGLangAdapter(backend_url="http://sglang-service:30000")
        result = adapter.decode_request(
            original_url="http://sglang-service:30000/v1/chat/completions",
            original_headers={},
        )
        parsed = urlparse(result.url)
        params = parse_qs(parsed.query)
        assert params["router_prefix"] == ["decode"]

    def test_decode_request_preserves_path(self):
        adapter = SGLangAdapter(backend_url="http://sglang-service:30000")
        result = adapter.decode_request(
            original_url="http://sglang-service:30000/v1/chat/completions",
            original_headers={},
        )
        assert urlparse(result.url).path == "/v1/chat/completions"

    def test_decode_request_preserves_existing_query_params(self):
        adapter = SGLangAdapter(backend_url="http://sglang-service:30000")
        result = adapter.decode_request(
            original_url="http://sglang-service:30000/v1/chat/completions?timeout=30",
            original_headers={},
        )
        parsed = urlparse(result.url)
        params = parse_qs(parsed.query)
        assert params["timeout"] == ["30"]
        assert params["router_prefix"] == ["decode"]

    def test_decode_request_route_target_is_decode(self):
        adapter = SGLangAdapter(backend_url="http://sglang-service:30000")
        result = adapter.decode_request(
            original_url="http://sglang-service:30000/v1/chat/completions",
            original_headers={},
        )
        assert result.route_target == RouteTarget.DECODE

    def test_decode_request_preserves_headers(self):
        adapter = SGLangAdapter(backend_url="http://sglang-service:30000")
        result = adapter.decode_request(
            original_url="http://sglang-service:30000/v1/chat/completions",
            original_headers={"authorization": "Bearer tok"},
        )
        assert result.headers["authorization"] == "Bearer tok"

    def test_prefill_request_no_router_prefix(self):
        adapter = SGLangAdapter(backend_url="http://sglang-service:30000")
        result = adapter.prefill_request(
            original_url="http://sglang-service:30000/v1/chat/completions",
            original_headers={},
        )
        assert "router_prefix" not in result.url

    def test_prefill_request_url_unchanged(self):
        adapter = SGLangAdapter(backend_url="http://sglang-service:30000")
        url = "http://sglang-service:30000/v1/chat/completions"
        result = adapter.prefill_request(original_url=url, original_headers={})
        assert result.url == url

    def test_prefill_request_route_target_is_prefill(self):
        adapter = SGLangAdapter(backend_url="http://sglang-service:30000")
        result = adapter.prefill_request(
            original_url="http://sglang-service:30000/v1/chat/completions",
            original_headers={},
        )
        assert result.route_target == RouteTarget.PREFILL

    def test_does_not_mutate_input_headers(self):
        adapter = SGLangAdapter(backend_url="http://sglang-service:30000")
        original = {"content-type": "application/json"}
        adapter.decode_request(
            original_url="http://sglang-service:30000/v1/chat/completions",
            original_headers=original,
        )
        assert original == {"content-type": "application/json"}
