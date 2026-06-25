"""FastAPI proxy application.

Reads configuration from environment variables at import time:
  BACKEND_URL          (required) upstream vLLM / SGLang base URL
  BACKEND_TYPE         "vllm" (default) or "sglang"
  AGENTSAVE_TOKEN      bearer token for dashboard metrics
  AGENTSAVE_METRICS_URL  dashboard POST endpoint; metrics disabled if absent
  PPD_W_TTFT           float, default 0.7
  PPD_W_TPOT           float, default 0.3
"""
from __future__ import annotations

import asyncio
import json as _json
import logging
import os
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from inferroute import __version__
from inferroute.adapters.base import RouteTarget
from inferroute.adapters.sglang import SGLangAdapter
from inferroute.adapters.vllm import VLLMAdapter
from inferroute.classifier import classify_turn
from inferroute.dispatcher import Dispatcher
from inferroute.metrics import MetricsEmitter, MetricsEvent
from inferroute.scoring import PPDScorer, PPDWeights

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from env
# ---------------------------------------------------------------------------
_BACKEND_URL: str = os.environ.get("BACKEND_URL", "http://localhost:8000")
_BACKEND_TYPE: str = os.environ.get("BACKEND_TYPE", "vllm").lower()

# ---------------------------------------------------------------------------
# Build adapter, scorer, dispatcher, emitter
# ---------------------------------------------------------------------------
_weights = PPDWeights.from_env()
_scorer = PPDScorer(weights=_weights)

if _BACKEND_TYPE == "sglang":
    _adapter = SGLangAdapter(backend_url=_BACKEND_URL)
else:
    _adapter = VLLMAdapter(backend_url=_BACKEND_URL)

_dispatcher = Dispatcher(adapter=_adapter, scorer=_scorer)
_emitter = MetricsEmitter.from_env()


# ---------------------------------------------------------------------------
# Lifespan: clean up httpx clients on shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(application: FastAPI):
    yield
    await _dispatcher.close()
    if _emitter is not None:
        await _emitter.close()


app = FastAPI(title="InferRoute", version=__version__, lifespan=lifespan)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "backend_type": _BACKEND_TYPE, "version": __version__}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    """Proxy POST /v1/chat/completions to the configured backend."""
    t0 = time.perf_counter()

    body_bytes = await request.body()

    # Determine turn type for metrics (classifier is pure, cheap to re-run).
    try:
        parsed = _json.loads(body_bytes)
        messages = parsed.get("messages", [])
    except Exception:
        parsed = {}
        messages = []

    turn_type = classify_turn(messages)

    # Check whether the request has stream=true.
    is_stream = bool(parsed.get("stream", False)) if isinstance(parsed, dict) else False

    if is_stream:
        async def _gen():
            async for chunk in _dispatcher.dispatch_stream(request):
                yield chunk

        return StreamingResponse(_gen(), media_type="text/event-stream")

    upstream: httpx.Response = await _dispatcher.dispatch(request)
    ttft_ms = (time.perf_counter() - t0) * 1000.0

    # Fire-and-forget metrics
    if _emitter is not None:
        adapted_url_has_decode_prefix = "router_prefix=decode" in str(upstream.request.url)
        has_decode_header = upstream.request.headers.get("x-route-type") == "decode"
        routed_to = (
            RouteTarget.DECODE.value
            if (adapted_url_has_decode_prefix or has_decode_header)
            else RouteTarget.PREFILL.value
        )
        event = MetricsEvent(
            request_type=turn_type.value,
            routed_to=routed_to,
            ttft_ms=ttft_ms,
            backend_type=_BACKEND_TYPE,
        )
        asyncio.create_task(_emitter.emit(event))

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=dict(upstream.headers),
        media_type=upstream.headers.get("content-type", "application/json"),
    )


def main() -> None:
    """CLI entry point: inferroute"""
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host=host, port=port)
