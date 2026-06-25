"""Turn classifier: detects whether a request is Turn 1 or Turn 2+.

Turn 1  — only user/system messages; no prior assistant response.
Turn 2+ — at least one assistant message exists in the history,
           meaning KV cache from a prior turn may be present on a
           decode node.
"""
from enum import Enum


class TurnType(Enum):
    TURN1 = "turn1"
    TURN2_PLUS = "turn2"


def classify_turn(messages: list[dict]) -> TurnType:
    """Classify an incoming request as Turn 1 or Turn 2+.

    Args:
        messages: The ``messages`` list from a chat completions request body.

    Returns:
        ``TurnType.TURN2_PLUS`` if any message with role ``"assistant"``
        is present in the history (indicating a prior exchange whose KV
        cache may be resident on a decode node).  ``TurnType.TURN1``
        otherwise.
    """
    for message in messages:
        if message.get("role") == "assistant":
            return TurnType.TURN2_PLUS
    return TurnType.TURN1
