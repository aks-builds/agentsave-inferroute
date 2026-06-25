"""SGLang backend adapter.

Decode routing strategy: append ``router_prefix=decode`` query parameter.
SGLang's built-in router reads this parameter and selects a worker whose
KV cache matches the prior-turn prefix.
"""
from __future__ import annotations

from urllib.parse import urlencode, urlparse, parse_qs, urlunparse

from inferroute.adapters.base import AdaptedRequest, RouteTarget


class SGLangAdapter:
    """Adapter for an SGLang backend cluster."""

    def __init__(self, backend_url: str) -> None:
        self.backend_url = backend_url.rstrip("/")

    def decode_request(
        self,
        original_url: str,
        original_headers: dict[str, str],
    ) -> AdaptedRequest:
        """Route to decode node by appending ``router_prefix=decode``."""
        parsed = urlparse(original_url)
        existing_params = parse_qs(parsed.query, keep_blank_values=True)
        existing_params["router_prefix"] = ["decode"]
        new_query = urlencode(
            {k: v[0] for k, v in existing_params.items()}, doseq=False
        )
        new_url = urlunparse(parsed._replace(query=new_query))
        return AdaptedRequest(
            url=new_url,
            headers=dict(original_headers),
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
