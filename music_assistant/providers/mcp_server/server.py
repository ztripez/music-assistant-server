"""MCP Server implementation for Music Assistant."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from music_assistant.mass import MusicAssistant


# Module-level state container
_state: dict[str, Any] = {
    "mass": None,
    "logger": None,
    "intro_prompt": "",
    "player_context_prompt": "",
}


def create_mcp_server(
    mass: MusicAssistant,
    require_auth: bool = True,
) -> FastMCP:
    """Create and configure the MCP server instance.

    :param mass: MusicAssistant instance.
    :param require_auth: Whether to require authentication.
    :return: Configured FastMCP server instance.
    """
    _state["mass"] = mass

    # Build server kwargs
    server_kwargs: dict[str, Any] = {
        "name": "Music Assistant",
        "instructions": (
            "Music Assistant MCP server for controlling music playback "
            "and managing your music library."
        ),
        "stateless_http": True,
        "json_response": True,
    }

    # Add authentication if required
    if require_auth:
        from .auth import MusicAssistantTokenVerifier  # noqa: PLC0415

        server_kwargs["token_verifier"] = MusicAssistantTokenVerifier(mass)

    mcp = FastMCP(**server_kwargs)

    # Register all tools, resources, and prompts
    _register_playback_tools(mcp)
    _register_media_tools(mcp)
    _register_resources(mcp)
    _register_prompts(mcp)

    return mcp


def _get_mass() -> MusicAssistant | None:
    """Get the MusicAssistant instance from module state."""
    mass = _state["mass"]
    if mass is None:
        return None
    return mass  # type: ignore[no-any-return]


def _register_playback_tools(mcp: FastMCP) -> None:
    """Register playback control tools."""

    @mcp.tool()
    async def play(player_id: str) -> str:
        """Start or resume playback on a player.

        :param player_id: The ID of the player/queue to control.
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            await mass.player_queues.play(player_id)
            return f"Playback started on {player_id}"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def pause(player_id: str) -> str:
        """Pause playback on a player.

        :param player_id: The ID of the player/queue to control.
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            await mass.player_queues.pause(player_id)
            return f"Playback paused on {player_id}"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def stop(player_id: str) -> str:
        """Stop playback on a player.

        :param player_id: The ID of the player/queue to control.
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            await mass.player_queues.stop(player_id)
            return f"Playback stopped on {player_id}"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def next_track(player_id: str) -> str:
        """Skip to the next track on a player.

        :param player_id: The ID of the player/queue to control.
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            await mass.player_queues.next(player_id)
            return f"Skipped to next track on {player_id}"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def previous_track(player_id: str) -> str:
        """Go to the previous track on a player.

        :param player_id: The ID of the player/queue to control.
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            await mass.player_queues.previous(player_id)
            return f"Went to previous track on {player_id}"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def set_volume(player_id: str, volume: int) -> str:
        """Set the volume level of a player.

        :param player_id: The ID of the player to control.
        :param volume: Volume level from 0 to 100.
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            volume = max(0, min(100, volume))
            await mass.players.cmd_volume_set(player_id, volume)
            return f"Volume set to {volume}% on {player_id}"
        except Exception as e:
            return f"Error: {e}"


def _register_media_tools(mcp: FastMCP) -> None:
    """Register media search and playback tools."""

    @mcp.tool()
    async def search_music(
        query: str,
        media_types: str = "track,artist,album,playlist",
        limit: int = 10,
    ) -> str:
        """Search the music library.

        :param query: Search query string.
        :param media_types: Comma-separated types (track, artist, album, playlist, radio).
        :param limit: Maximum number of results per media type.
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            from music_assistant_models.enums import MediaType  # noqa: PLC0415

            types_map = {
                "track": MediaType.TRACK,
                "artist": MediaType.ARTIST,
                "album": MediaType.ALBUM,
                "playlist": MediaType.PLAYLIST,
                "radio": MediaType.RADIO,
            }
            search_types = [
                types_map[t.strip().lower()]
                for t in media_types.split(",")
                if t.strip().lower() in types_map
            ]

            results = await mass.music.search(
                search_query=query,
                media_types=search_types,
                limit=limit,
            )

            output: dict[str, Any] = {"query": query, "results": {}}
            for media_type, items in [
                ("tracks", results.tracks),
                ("artists", results.artists),
                ("albums", results.albums),
                ("playlists", results.playlists),
                ("radio", results.radio),
            ]:
                if items:
                    output["results"][media_type] = [
                        {"name": item.name, "uri": item.uri} for item in items[:limit]
                    ]

            return json.dumps(output, indent=2)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def play_media(player_id: str, uri: str) -> str:
        """Play a specific media item by URI.

        :param player_id: The ID of the player/queue to play on.
        :param uri: The URI of the media item to play (from search results or library).
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            await mass.player_queues.play_media(queue_id=player_id, media=uri)
            return f"Playing {uri} on {player_id}"
        except Exception as e:
            return f"Error: {e}"


def _register_resources(mcp: FastMCP) -> None:
    """Register all MCP resources."""

    @mcp.resource("players://")
    async def list_players() -> str:
        """List all available players/speakers."""
        mass = _get_mass()
        if mass is None:
            return json.dumps({"error": "Music Assistant not initialized"})

        players = []
        for player in mass.players.all():
            players.append(
                {
                    "id": player.player_id,
                    "name": player.display_name,
                    "available": player.available,
                    "state": player.playback_state.value,
                    "volume": player.volume_level,
                    "muted": player.volume_muted,
                    "type": player.type.value if player.type else "unknown",
                }
            )

        return json.dumps({"players": players}, indent=2)

    @mcp.resource("player://{player_id}")
    async def get_player(player_id: str) -> str:
        """Get detailed information about a specific player."""
        mass = _get_mass()
        if mass is None:
            return json.dumps({"error": "Music Assistant not initialized"})

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
                },
                "queue": queue_info,
            },
            indent=2,
        )

    @mcp.resource("nowplaying://{player_id}")
    async def get_now_playing(player_id: str) -> str:
        """Get the currently playing track on a player."""
        mass = _get_mass()
        if mass is None:
            return json.dumps({"error": "Music Assistant not initialized"})

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


def _register_prompts(mcp: FastMCP) -> None:
    """Register all MCP prompts."""

    @mcp.prompt()
    async def music_assistant_intro() -> str:
        """Introduction prompt to prime an AI assistant with MA capabilities."""
        mass = _get_mass()
        if mass is None:
            return "Music Assistant is not currently available."

        players = mass.players.all()
        player_list = (
            ", ".join(p.display_name for p in players if p.available) or "No players available"
        )

        # Use configured prompt template
        intro_template = str(_state.get("intro_prompt", ""))
        if intro_template:
            return intro_template.format(player_list=player_list)

        # Fallback to default if no template configured
        return f"Music Assistant is available. Players: {player_list}"

    @mcp.prompt()
    async def playback_control_context(player_id: str) -> str:
        """Context prompt for controlling a specific player."""
        mass = _get_mass()
        if mass is None:
            return "Music Assistant is not currently available."

        player = mass.players.get(player_id)
        if not player:
            return f"Player {player_id} not found."

        queue = mass.player_queues.get(player_id)
        current_track = "Nothing playing"
        if queue and queue.current_item:
            current_track = queue.current_item.name

        state = player.playback_state.value
        volume = player.volume_level or 0

        # Use configured prompt template
        context_template = str(_state.get("player_context_prompt", ""))
        if context_template:
            return context_template.format(
                player_name=player.display_name,
                state=state,
                volume=volume,
                current_track=current_track,
            )

        # Fallback to default if no template configured
        return f"Controlling {player.display_name}, state: {state}, volume: {volume}%"


async def start_mcp_server(
    mass: MusicAssistant,
    port: int,
    require_auth: bool,
    intro_prompt: str,
    player_context_prompt: str,
    logger: logging.Logger,
) -> tuple[asyncio.Task[Any], asyncio.Event]:
    """Start the MCP server.

    :param mass: MusicAssistant instance.
    :param port: Port to run the server on.
    :param require_auth: Whether to require authentication.
    :param intro_prompt: Template for the introduction prompt.
    :param player_context_prompt: Template for the player context prompt.
    :param logger: Logger instance.
    :return: Tuple of (server task, shutdown event).
    """
    _state["logger"] = logger
    _state["intro_prompt"] = intro_prompt
    _state["player_context_prompt"] = player_context_prompt

    # Create the MCP server with authentication if required
    mcp = create_mcp_server(mass, require_auth)

    shutdown_event = asyncio.Event()

    async def run_server() -> None:
        """Run the uvicorn server."""
        import uvicorn  # noqa: PLC0415

        # Configure the MCP server path
        mcp.settings.streamable_http_path = "/"

        config = uvicorn.Config(
            app=mcp.streamable_http_app(),
            host="0.0.0.0",
            port=port,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)

        # Run server until shutdown is requested
        server_task = asyncio.create_task(server.serve())

        # Wait for shutdown signal
        await shutdown_event.wait()

        # Graceful shutdown
        server.should_exit = True
        try:
            await asyncio.wait_for(server_task, timeout=5.0)
        except TimeoutError:
            server_task.cancel()

    task = asyncio.create_task(run_server())
    return task, shutdown_event
