"""MCP Resources package - exports all resource registration functions."""

from __future__ import annotations

from .library import register_library_resources
from .players import register_player_resources

__all__ = [
    "register_library_resources",
    "register_player_resources",
]
