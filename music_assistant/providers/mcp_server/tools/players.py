"""Player management tools for MCP server."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from music_assistant.mass import MusicAssistant


def register_player_query_tools(mcp: FastMCP, mass: MusicAssistant) -> None:
    """Register player query tools.

    :param mcp: FastMCP server instance.
    :param mass: MusicAssistant instance.
    """

    @mcp.tool()
    async def get_player_by_name(name: str) -> str:
        """Find a player by name, including its capabilities.

        :param name: Full or partial player name.
        """
        try:
            name_lower = name.lower()
            matches = []
            for player in mass.players.all():
                if name_lower in player.display_name.lower():
                    capabilities = [f.name.lower() for f in player.supported_features]
                    matches.append(
                        {
                            "id": player.player_id,
                            "name": player.display_name,
                            "available": player.available,
                            "state": player.playback_state.value,
                            "capabilities": capabilities,
                        }
                    )

            if not matches:
                return f"No players found matching '{name}'"
            return json.dumps({"matches": matches}, indent=2)
        except Exception as e:
            return f"Error: {e}"


def register_player_control_tools(mcp: FastMCP, mass: MusicAssistant) -> None:
    """Register player control tools.

    :param mcp: FastMCP server instance.
    :param mass: MusicAssistant instance.
    """

    @mcp.tool()
    async def power_player(player_id: str, powered: bool) -> str:
        """Power on or off a player.

        :param player_id: Player ID.
        :param powered: Power on.
        """
        try:
            await mass.players.cmd_power(player_id, powered)
            state = "on" if powered else "off"
            return f"Player {player_id} powered {state}"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def group_players(
        target_player_id: str,
        child_player_ids: str,
    ) -> str:
        """Group players for synchronized playback.

        :param target_player_id: Group leader player ID.
        :param child_player_ids: Comma-separated player IDs to add.
        """
        try:
            child_ids = [p.strip() for p in child_player_ids.split(",")]
            await mass.players.cmd_group_many(target_player_id, child_ids)
            return f"Grouped players with {target_player_id}"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def ungroup_player(player_id: str) -> str:
        """Remove a player from its group.

        :param player_id: Player ID.
        """
        try:
            await mass.players.cmd_ungroup(player_id)
            return f"Player {player_id} ungrouped"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def play_announcement(
        player_id: str,
        url: str,
        volume: int | None = None,
    ) -> str:
        """Play an announcement on a player (TTS or audio URL).

        :param player_id: Player ID.
        :param url: URL of the audio to play.
        :param volume: Optional volume override (0-100).
        """
        try:
            await mass.players.play_announcement(player_id, url, volume_level=volume)
            return f"Playing announcement on {player_id}"
        except Exception as e:
            return f"Error: {e}"
