"""Tests for turn classifier."""
import pytest
from inferroute.classifier import TurnType, classify_turn


class TestClassifyTurn:
    def test_single_user_message_is_turn1(self):
        messages = [{"role": "user", "content": "Hello"}]
        assert classify_turn(messages) == TurnType.TURN1

    def test_system_plus_user_is_turn1(self):
        """System prompt + one user message is still Turn 1."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        assert classify_turn(messages) == TurnType.TURN1

    def test_user_assistant_user_is_turn2(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "Tell me more."},
        ]
        assert classify_turn(messages) == TurnType.TURN2_PLUS

    def test_long_conversation_is_turn2(self):
        messages = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
            {"role": "user", "content": "Q3"},
        ]
        assert classify_turn(messages) == TurnType.TURN2_PLUS

    def test_empty_messages_is_turn1(self):
        """Degenerate input: treat as Turn 1 (full prefill needed)."""
        assert classify_turn([]) == TurnType.TURN1

    def test_system_only_is_turn1(self):
        messages = [{"role": "system", "content": "You are helpful."}]
        assert classify_turn(messages) == TurnType.TURN1

    def test_assistant_turn_anywhere_triggers_turn2(self):
        """If any assistant message exists, KV cache may be present."""
        messages = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Response"},
        ]
        assert classify_turn(messages) == TurnType.TURN2_PLUS

    def test_messages_with_none_content_handled(self):
        """Tool call messages may have None content — should not crash."""
        messages = [
            {"role": "user", "content": "Call a tool"},
            {"role": "assistant", "content": None},
            {"role": "user", "content": "Follow up"},
        ]
        assert classify_turn(messages) == TurnType.TURN2_PLUS
