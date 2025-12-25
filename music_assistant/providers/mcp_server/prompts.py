"""MCP prompts for Music Assistant."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from music_assistant.mass import MusicAssistant


def register_prompts(mcp: FastMCP, mass: MusicAssistant) -> None:
    """Register MCP prompts as user-invokable templates.

    :param mcp: FastMCP server instance.
    :param mass: MusicAssistant instance.
    """

    @mcp.prompt()
    async def play_music(query: str = "", player: str = "") -> str:
        """Request to play music."""
        players_info = ""
        players = [p.display_name for p in mass.players.all() if p.available]
        players_info = f"\n\nAvailable players: {', '.join(players)}" if players else ""

        query_part = f'"{query}"' if query else "some music"
        player_part = f" on {player}" if player else ""

        return f"I want to play {query_part}{player_part}.{players_info}"

    @mcp.prompt()
    async def whats_playing(player: str = "") -> str:
        """Check current playback status."""
        if player:
            # Try to find the player and get current track
            for p in mass.players.all():
                if player.lower() in p.display_name.lower() or player == p.player_id:
                    queue = mass.player_queues.get(p.player_id)
                    if queue and queue.current_item:
                        return (
                            f"What's playing on {p.display_name}? "
                            f"(Currently: {queue.current_item.name})"
                        )
                    return f"What's playing on {p.display_name}?"

        # List all players with their current state
        playing_info = []
        for p in mass.players.all():
            if p.available:
                queue = mass.player_queues.get(p.player_id)
                track = queue.current_item.name if queue and queue.current_item else "Nothing"
                playing_info.append(f"{p.display_name}: {track}")

        return "What's currently playing?\n\n" + "\n".join(playing_info)

    @mcp.prompt()
    async def control_playback(player: str = "", action: str = "") -> str:
        """Playback control request."""
        actions = "play, pause, stop, next, previous, volume up, volume down"
        player_part = f" on {player}" if player else ""
        action_part = action if action else f"[{actions}]"
        return f"I want to {action_part}{player_part}."

    @mcp.prompt()
    async def discover_music(mood: str = "", genre: str = "") -> str:
        """Music discovery and recommendations."""
        parts = []
        if mood:
            parts.append(mood)
        if genre:
            parts.append(genre)

        if parts:
            return f"Suggest some {' '.join(parts)} music for me to listen to."
        return "Suggest some music based on my listening history and preferences."

    @mcp.prompt()
    async def manage_queue(player: str = "") -> str:
        """Queue management request."""
        player_part = f" on {player}" if player else ""
        return f"Help me manage the music queue{player_part}. Show me what's queued up."

    @mcp.prompt()
    async def setup_multiroom(rooms: str = "") -> str:
        """Multi-room audio setup."""
        players_info = ""
        players = [p.display_name for p in mass.players.all() if p.available]
        players_info = f"\n\nAvailable speakers: {', '.join(players)}" if players else ""

        if rooms:
            return f"Help me sync music across these rooms: {rooms}.{players_info}"
        return f"Help me set up multi-room audio to play music in sync.{players_info}"

    @mcp.prompt()
    async def transfer_playback(from_player: str = "", to_player: str = "") -> str:
        """Move playback between players."""
        players_info = ""
        players = [p.display_name for p in mass.players.all() if p.available]
        players_info = f"\n\nAvailable players: {', '.join(players)}" if players else ""

        if from_player and to_player:
            return f"Transfer what's playing from {from_player} to {to_player}.{players_info}"
        if to_player:
            return f"I want to continue listening on {to_player}.{players_info}"
        if from_player:
            return f"Move what's playing on {from_player} to another player.{players_info}"
        return f"Help me transfer music from one player to another.{players_info}"
