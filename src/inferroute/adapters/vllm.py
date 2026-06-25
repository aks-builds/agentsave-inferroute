"""vLLM backend adapter.

Decode routing strategy: add ``X-Route-Type: decode`` header.
vLLM's routing layer reads this header and directs the request to a
worker that already holds the prior-turn KV cache.
"""
from __future__ import annotations

from inferroute.adapters.base import AdaptedRequest, RouteTarget


class VLLMAdapter:
    """Adapter for a vLLM backend cluster."""

    def __init__(self, backend_url: str) -> None:
        self.backend_url = backend_url.rstrip("/")

    def decode_request(
        self,
        original_url: str,
        original_headers: dict[str, str],
    ) -> AdaptedRequest:
        """Route to decode node by injecting ``X-Route-Type: decode``."""
        headers = dict(original_headers)
        headers["x-route-type"] = "decode"
        return AdaptedRequest(
            url=original_url,
            headers=headers,
            route_target=RouteTarget.DECODE,
        )

    def prefill_request(
        self,
        original_url: str,
        original_headers: dict[str, str],
    ) -> AdaptedRequest:
        """Standard prefill path — pass through unchanged."""
        return AdaptedRequest(
            url=original_url,
            headers=dict(original_headers),
            route_target=RouteTarget.PREFILL,
        )
