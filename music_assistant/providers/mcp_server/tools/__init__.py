"""MCP Tools package - exports all tool registration functions."""

from __future__ import annotations

from .library import (
    register_library_delete_tools,
    register_library_edit_tools,
    register_library_query_tools,
)
from .media import register_audiobook_tools, register_podcast_tools, register_radio_tools
from .metadata import register_metadata_tools
from .playback import register_playback_control_tools, register_playback_query_tools
from .players import register_player_control_tools, register_player_query_tools
from .playlists import (
    register_playlist_delete_tools,
    register_playlist_edit_tools,
    register_playlist_query_tools,
)
from .queue import (
    register_queue_control_tools,
    register_queue_delete_tools,
    register_queue_edit_tools,
    register_queue_query_tools,
)
from .volume import register_volume_control_tools

__all__ = [
    "register_audiobook_tools",
    "register_library_delete_tools",
    "register_library_edit_tools",
    "register_library_query_tools",
    "register_metadata_tools",
    "register_playback_control_tools",
    "register_playback_query_tools",
    "register_player_control_tools",
    "register_player_query_tools",
    "register_playlist_delete_tools",
    "register_playlist_edit_tools",
    "register_playlist_query_tools",
    "register_podcast_tools",
    "register_queue_control_tools",
    "register_queue_delete_tools",
    "register_queue_edit_tools",
    "register_queue_query_tools",
    "register_radio_tools",
    "register_volume_control_tools",
]
