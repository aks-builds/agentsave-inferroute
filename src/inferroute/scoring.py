"""PPD (Per-turn Prefill Disaggregation) scoring function.

Computes route_score = w_ttft * TTFT_improvement - w_tpot * TPOT_degradation.

When route_score > 0, routing to the decode node is worthwhile.
Default weights from the ICML 2026 paper (arXiv:2603.13358):
  w_ttft = 0.7, w_tpot = 0.3
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class PPDWeights:
    """Weights for the PPD routing score."""

    w_ttft: float = 0.7
    w_tpot: float = 0.3

    @classmethod
    def from_env(cls) -> "PPDWeights":
        """Build weights from ``PPD_W_TTFT`` and ``PPD_W_TPOT`` env vars.

        Falls back to defaults when the variables are absent.
        """
        w_ttft = float(os.environ.get("PPD_W_TTFT", 0.7))
        w_tpot = float(os.environ.get("PPD_W_TPOT", 0.3))
        return cls(w_ttft=w_ttft, w_tpot=w_tpot)


class PPDScorer:
    """Evaluates whether a Turn 2+ request should be routed to a decode node."""

    def __init__(self, weights: PPDWeights | None = None) -> None:
        self.weights = weights if weights is not None else PPDWeights()

    def score(self, ttft_improvement: float, tpot_degradation: float) -> float:
        """Compute the PPD routing score.

        Args:
            ttft_improvement: Expected TTFT reduction (in ms or %) from
                routing to decode rather than prefill.
            tpot_degradation: Expected TPOT increase (same units) from
                skipping the prefill node.

        Returns:
            Signed float.  Positive means decode routing is net beneficial.
        """
        return (
            self.weights.w_ttft * ttft_improvement
            - self.weights.w_tpot * tpot_degradation
        )

    def should_route_to_decode(
        self, ttft_improvement: float, tpot_degradation: float
    ) -> bool:
        """Return True only when route_score is strictly positive."""
        return self.score(ttft_improvement, tpot_degradation) > 0.0
