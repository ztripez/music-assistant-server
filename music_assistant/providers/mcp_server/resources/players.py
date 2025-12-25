"""Player-related MCP resources."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from music_assistant.mass import MusicAssistant


def register_player_resources(mcp: FastMCP, mass: MusicAssistant) -> None:
    """Register player-related MCP resources.

    :param mcp: FastMCP server instance.
    :param mass: MusicAssistant instance.
    """

    @mcp.resource("players://")
    async def list_players() -> str:
        """List all available players/speakers with their capabilities."""
        players = []
        for player in mass.players.all():
            # Convert supported features to list of capability names
            capabilities = [f.name.lower() for f in player.supported_features]
            players.append(
                {
                    "id": player.player_id,
                    "name": player.display_name,
                    "available": player.available,
                    "state": player.playback_state.value,
                    "volume": player.volume_level,
                    "muted": player.volume_muted,
                    "type": player.type.value if player.type else "unknown",
                    "powered": player.powered,
                    "capabilities": capabilities,
                }
            )

        return json.dumps({"players": players}, indent=2)

    @mcp.resource("player://{player_id}")
    async def get_player(player_id: str) -> str:
        """Get detailed information about a specific player."""
        player = mass.players.get(player_id)
        if not player:
            return json.dumps({"error": f"Player {player_id} not found"})

        # Get queue info if available
        queue = mass.player_queues.get(player_id)
        queue_info = None
        if queue:
            current_item = queue.current_item
            queue_info = {
                "state": queue.state.value if queue.state else "unknown",
                "shuffle": queue.shuffle_enabled,
                "repeat": queue.repeat_mode.value if queue.repeat_mode else "off",
                "current_index": queue.current_index,
                "current_track": (
                    {
                        "name": current_item.name if current_item else None,
                        "artist": (
                            getattr(current_item, "artist_str", None) if current_item else None
                        ),
                        "duration": current_item.duration if current_item else None,
                    }
                    if current_item
                    else None
                ),
                "elapsed_time": queue.elapsed_time,
            }

        # Convert supported features to list of capability names
        capabilities = [f.name.lower() for f in player.supported_features]

        return json.dumps(
            {
                "player": {
                    "id": player.player_id,
                    "name": player.display_name,
                    "available": player.available,
                    "state": player.playback_state.value,
                    "volume": player.volume_level,
                    "muted": player.volume_muted,
                    "type": player.type.value if player.type else "unknown",
                    "powered": player.powered,
                    "group_members": player.group_members,
                    "capabilities": capabilities,
                },
                "queue": queue_info,
            },
            indent=2,
        )

    @mcp.resource("nowplaying://{player_id}")
    async def get_now_playing(player_id: str) -> str:
        """Get the currently playing track on a player."""
        queue = mass.player_queues.get(player_id)
        if not queue:
            return json.dumps({"error": f"Queue for {player_id} not found"})

        current_item = queue.current_item
        if not current_item:
            return json.dumps({"now_playing": None, "message": "Nothing currently playing"})

        return json.dumps(
            {
                "now_playing": {
                    "name": current_item.name,
                    "uri": current_item.uri if hasattr(current_item, "uri") else None,
                    "duration": current_item.duration,
                    "elapsed": queue.elapsed_time,
                },
                "state": queue.state.value if queue.state else "unknown",
            },
            indent=2,
        )

    @mcp.resource("queue://{player_id}")
    async def get_queue_contents(player_id: str) -> str:
        """Get the full queue contents for a player."""
        queue = mass.player_queues.get(player_id)
        if not queue:
            return json.dumps({"error": f"Queue for {player_id} not found"})

        items = mass.player_queues.items(player_id, limit=100)
        queue_items = []
        for item in items:
            queue_items.append(
                {
                    "item_id": item.queue_item_id,
                    "name": item.name,
                    "uri": item.uri if hasattr(item, "uri") else None,
                    "duration": item.duration,
                }
            )

        return json.dumps(
            {
                "queue": {
                    "player_id": player_id,
                    "current_index": queue.current_index,
                    "shuffle": queue.shuffle_enabled,
                    "repeat": queue.repeat_mode.value if queue.repeat_mode else "off",
                    "items": queue_items,
                    "total_items": len(queue_items),
                }
            },
            indent=2,
        )
