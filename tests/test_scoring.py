"""Tests for PPD scoring function."""
import os
import pytest
from inferroute.scoring import PPDWeights, PPDScorer


class TestPPDWeights:
    def test_default_weights(self):
        weights = PPDWeights()
        assert weights.w_ttft == 0.7
        assert weights.w_tpot == 0.3

    def test_custom_weights(self):
        weights = PPDWeights(w_ttft=0.5, w_tpot=0.5)
        assert weights.w_ttft == 0.5
        assert weights.w_tpot == 0.5

    def test_from_env_uses_defaults_when_vars_absent(self, monkeypatch):
        monkeypatch.delenv("PPD_W_TTFT", raising=False)
        monkeypatch.delenv("PPD_W_TPOT", raising=False)
        weights = PPDWeights.from_env()
        assert weights.w_ttft == 0.7
        assert weights.w_tpot == 0.3

    def test_from_env_reads_env_vars(self, monkeypatch):
        monkeypatch.setenv("PPD_W_TTFT", "0.6")
        monkeypatch.setenv("PPD_W_TPOT", "0.4")
        weights = PPDWeights.from_env()
        assert weights.w_ttft == pytest.approx(0.6)
        assert weights.w_tpot == pytest.approx(0.4)


class TestPPDScorer:
    def test_score_formula(self):
        """route_score = w_ttft * ttft_improvement - w_tpot * tpot_degradation"""
        scorer = PPDScorer(PPDWeights(w_ttft=0.7, w_tpot=0.3))
        # 0.7 * 68 - 0.3 * 10 = 47.6 - 3.0 = 44.6
        assert scorer.score(ttft_improvement=68.0, tpot_degradation=10.0) == pytest.approx(44.6)

    def test_score_zero_degradation(self):
        scorer = PPDScorer(PPDWeights(w_ttft=0.7, w_tpot=0.3))
        # 0.7 * 50 - 0.3 * 0 = 35.0
        assert scorer.score(ttft_improvement=50.0, tpot_degradation=0.0) == pytest.approx(35.0)

    def test_score_negative_when_tpot_dominates(self):
        scorer = PPDScorer(PPDWeights(w_ttft=0.1, w_tpot=0.9))
        # 0.1 * 5 - 0.9 * 100 = 0.5 - 90.0 = -89.5
        assert scorer.score(ttft_improvement=5.0, tpot_degradation=100.0) == pytest.approx(-89.5)

    def test_should_route_to_decode_positive_score(self):
        scorer = PPDScorer()
        # High TTFT improvement, low TPOT cost → route to decode
        assert scorer.should_route_to_decode(ttft_improvement=68.0, tpot_degradation=5.0) is True

    def test_should_route_to_decode_zero_score(self):
        """Score of exactly zero — do NOT route (must be strictly positive)."""
        scorer = PPDScorer(PPDWeights(w_ttft=0.5, w_tpot=0.5))
        # 0.5 * 10 - 0.5 * 10 = 0
        assert scorer.should_route_to_decode(ttft_improvement=10.0, tpot_degradation=10.0) is False

    def test_should_route_to_decode_negative_score(self):
        scorer = PPDScorer(PPDWeights(w_ttft=0.1, w_tpot=0.9))
        assert scorer.should_route_to_decode(ttft_improvement=5.0, tpot_degradation=100.0) is False

    def test_default_weights_used_when_none(self):
        scorer = PPDScorer()
        assert scorer.weights.w_ttft == 0.7
        assert scorer.weights.w_tpot == 0.3
