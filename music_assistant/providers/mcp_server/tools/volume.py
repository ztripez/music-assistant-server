"""Volume control tools for MCP server."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from music_assistant.mass import MusicAssistant


def register_volume_control_tools(mcp: FastMCP, mass: MusicAssistant) -> None:
    """Register volume control tools.

    :param mcp: FastMCP server instance.
    :param mass: MusicAssistant instance.
    """

    @mcp.tool()
    async def set_volume(player_id: str, volume: int) -> str:
        """Set the volume level of a player.

        :param player_id: Player ID.
        :param volume: Volume level from 0 to 100.
        """
        try:
            volume = max(0, min(100, volume))
            await mass.players.cmd_volume_set(player_id, volume)
            return f"Volume set to {volume}% on {player_id}"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def volume_up(player_id: str) -> str:
        """Increase the volume of a player by one step.

        :param player_id: Player ID.
        """
        try:
            await mass.players.cmd_volume_up(player_id)
            return f"Volume increased on {player_id}"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def volume_down(player_id: str) -> str:
        """Decrease the volume of a player by one step.

        :param player_id: Player ID.
        """
        try:
            await mass.players.cmd_volume_down(player_id)
            return f"Volume decreased on {player_id}"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def mute(player_id: str, muted: bool) -> str:
        """Mute or unmute a player.

        :param player_id: Player ID.
        :param muted: Mute player.
        """
        try:
            await mass.players.cmd_volume_mute(player_id, muted)
            state = "muted" if muted else "unmuted"
            return f"Player {player_id} {state}"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def set_group_volume(player_id: str, volume: int) -> str:
        """Set the volume for all players in a group.

        :param player_id: Group player ID.
        :param volume: Volume level from 0 to 100.
        """
        try:
            volume = max(0, min(100, volume))
            await mass.players.cmd_group_volume(player_id, volume)
            return f"Group volume set to {volume}% on {player_id}"
        except Exception as e:
            return f"Error: {e}"
