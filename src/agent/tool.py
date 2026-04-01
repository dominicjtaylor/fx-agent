"""
tool.py
Shared types and utilities for agent tool functions.
"""

import time
from dataclasses import dataclass
from typing import Any, Optional

import anthropic


@dataclass
class ToolResult:
    ok: bool
    value: Any
    error: Optional[str] = None


def retry_api_call(fn, max_retries: int = 3, base_delay: float = 1.0):
    """
    Retry a callable on transient Anthropic API errors with exponential backoff.

    Retries on: rate limit (429), overload (529), connection errors.
    Raises immediately on all other exceptions.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn()
        except anthropic.APIStatusError as e:
            if e.status_code in (429, 529):
                last_exc = e
                time.sleep(base_delay * (2 ** attempt))
            else:
                raise
        except anthropic.APIConnectionError as e:
            last_exc = e
            time.sleep(base_delay * (2 ** attempt))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("retry_api_call failed without capturing exception")