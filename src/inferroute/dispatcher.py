"""Async HTTP dispatcher — the core proxy engine.

Reads the incoming FastAPI request body, classifies the turn,
consults the PPD scorer, selects prefill or decode routing via the
adapter, then forwards the request with httpx.

Static TTFT/TPOT estimates for v0.1.0 (real instrumentation is Phase 2):
  ttft_improvement = 68.0   (ms, conservative from arXiv:2603.13358)
  tpot_degradation = 5.0    (ms)
"""
from __future__ import annotations

import json
from typing import AsyncGenerator

import httpx
from fastapi import Request

from inferroute.adapters.base import AdaptedRequest, BackendAdapter
from inferroute.classifier import TurnType, classify_turn
from inferroute.scoring import PPDScorer

# Conservative static estimates from ICML 2026 paper.
_DEFAULT_TTFT_IMPROVEMENT = 68.0
_DEFAULT_TPOT_DEGRADATION = 5.0

# Headers the proxy must not forward upstream (hop-by-hop).
_HOP_BY_HOP = frozenset(
    {
        "host",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "content-length",  # httpx recalculates this
    }
)


class Dispatcher:
    """Routes a FastAPI Request to the correct backend endpoint."""

    def __init__(self, adapter: BackendAdapter, scorer: PPDScorer) -> None:
        self._adapter = adapter
        self._scorer = scorer
        self._client = httpx.AsyncClient(timeout=120.0)

    async def _build_adapted(self, request: Request, body_bytes: bytes) -> AdaptedRequest:
        """Decide turn type and return the adapter's routing decision."""
        try:
            body = json.loads(body_bytes)
            messages = body.get("messages", [])
        except (json.JSONDecodeError, AttributeError):
            messages = []

        turn_type = classify_turn(messages)

        # Build clean outgoing headers (strip hop-by-hop, lowercase keys).
        clean_headers: dict[str, str] = {
            k.lower(): v
            for k, v in request.headers.items()
            if k.lower() not in _HOP_BY_HOP
        }

        # Reconstruct the downstream URL from the adapter's base URL.
        base = getattr(self._adapter, "backend_url", "")
        path = request.url.path
        query = request.url.query
        adapted_url = f"{base}{path}"
        if query:
            adapted_url = f"{adapted_url}?{query}"

        if turn_type == TurnType.TURN2_PLUS and self._scorer.should_route_to_decode(
            ttft_improvement=_DEFAULT_TTFT_IMPROVEMENT,
            tpot_degradation=_DEFAULT_TPOT_DEGRADATION,
        ):
            return self._adapter.decode_request(adapted_url, clean_headers)
        else:
            return self._adapter.prefill_request(adapted_url, clean_headers)

    async def dispatch(self, request: Request) -> httpx.Response:
        """Proxy the request and return the full (buffered) upstream response."""
        body_bytes = await request.body()
        adapted = await self._build_adapted(request, body_bytes)

        response = await self._client.request(
            method=request.method,
            url=adapted.url,
            headers=adapted.headers,
            content=body_bytes,
        )
        return response

    async def dispatch_stream(
        self, request: Request
    ) -> AsyncGenerator[bytes, None]:
        """Stream the upstream response chunk-by-chunk without buffering."""
        body_bytes = await request.body()
        adapted = await self._build_adapted(request, body_bytes)

        async with self._client.stream(
            method=request.method,
            url=adapted.url,
            headers=adapted.headers,
            content=body_bytes,
        ) as response:
            async for chunk in response.aiter_bytes():
                yield chunk

    async def close(self) -> None:
        """Release the httpx client. Call on app shutdown."""
        await self._client.aclose()
