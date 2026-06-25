"""Fire-and-forget async metrics emitter.

Posts telemetry events to the AgentSave dashboard.  All exceptions are
swallowed — a metrics failure must never block or degrade the proxy.
"""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass, asdict

import httpx

logger = logging.getLogger(__name__)


@dataclass
class MetricsEvent:
    """One proxied-request telemetry record."""

    request_type: str   # "turn1" | "turn2"
    routed_to: str      # "prefill" | "decode"
    ttft_ms: float
    backend_type: str   # "vllm" | "sglang"


class MetricsEmitter:
    """Sends ``MetricsEvent`` records to the AgentSave dashboard."""

    def __init__(self, dashboard_url: str, token: str) -> None:
        self.dashboard_url = dashboard_url
        self.token = token
        self._client = httpx.AsyncClient(timeout=5.0)

    @classmethod
    def from_env(cls) -> "MetricsEmitter | None":
        """Build from env vars, or return ``None`` if URL is not configured."""
        url = os.environ.get("AGENTSAVE_METRICS_URL")
        token = os.environ.get("AGENTSAVE_TOKEN", "")
        if not url:
            return None
        return cls(dashboard_url=url, token=token)

    async def emit(self, event: MetricsEvent) -> None:
        """POST the event to the dashboard.  Swallows all errors silently."""
        try:
            await self._client.post(
                self.dashboard_url,
                json=asdict(event),
                headers={"authorization": f"Bearer {self.token}"},
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Metrics emission failed (suppressed): %s", exc)

    async def close(self) -> None:
        """Release the underlying httpx client."""
        await self._client.aclose()
