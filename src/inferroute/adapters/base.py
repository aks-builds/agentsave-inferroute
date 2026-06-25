"""Shared types and Protocol for backend adapters."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class RouteTarget(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass
class AdaptedRequest:
    """The downstream URL and headers after adapter transformation."""

    url: str
    headers: dict[str, str]
    route_target: RouteTarget


class BackendAdapter(Protocol):
    """Protocol every backend adapter must satisfy."""

    def decode_request(
        self,
        original_url: str,
        original_headers: dict[str, str],
    ) -> AdaptedRequest:
        """Return the request adapted for direct decode-node routing."""
        ...

    def prefill_request(
        self,
        original_url: str,
        original_headers: dict[str, str],
    ) -> AdaptedRequest:
        """Return the request unchanged for standard prefill-node routing."""
        ...
