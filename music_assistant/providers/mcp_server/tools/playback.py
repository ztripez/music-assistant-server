"""Playback control tools for MCP server."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from music_assistant.mass import MusicAssistant


def register_playback_query_tools(mcp: FastMCP, mass: MusicAssistant) -> None:
    """Register playback query tools (search).

    :param mcp: FastMCP server instance.
    :param mass: MusicAssistant instance.
    """

    @mcp.tool()
    async def search_music(
        query: str,
        media_types: str = "track,artist,album,playlist",
        limit: int = 10,
        library_only: bool = False,
    ) -> str:
        """Search for music. Returns items with URIs for play_media.

        :param query: Search query.
        :param media_types: Comma-separated: track, artist, album, playlist, radio.
        :param limit: Max results per type.
        :param library_only: Only search library, not streaming providers.
        """
        try:
            from music_assistant_models.enums import MediaType  # noqa: PLC0415

            types_map = {
                "track": MediaType.TRACK,
                "artist": MediaType.ARTIST,
                "album": MediaType.ALBUM,
                "playlist": MediaType.PLAYLIST,
                "radio": MediaType.RADIO,
                "podcast": MediaType.PODCAST,
                "audiobook": MediaType.AUDIOBOOK,
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
                library_only=library_only,
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


def register_playback_control_tools(mcp: FastMCP, mass: MusicAssistant) -> None:
    """Register playback control tools.

    :param mcp: FastMCP server instance.
    :param mass: MusicAssistant instance.
    """

    @mcp.tool()
    async def play(player_id: str) -> str:
        """Start or resume playback on a player.

        :param player_id: Player ID from players:// resource.
        """
        try:
            await mass.player_queues.play(player_id)
            return f"Playback started on {player_id}"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def pause(player_id: str) -> str:
        """Pause playback on a player.

        :param player_id: Player ID.
        """
        try:
            await mass.player_queues.pause(player_id)
            return f"Playback paused on {player_id}"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def stop(player_id: str) -> str:
        """Stop playback on a player and clear the queue.

        :param player_id: Player ID.
        """
        try:
            await mass.player_queues.stop(player_id)
            return f"Playback stopped on {player_id}"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def next_track(player_id: str) -> str:
        """Skip to the next track on a player.

        :param player_id: Player ID.
        """
        try:
            await mass.player_queues.next(player_id)
            return f"Skipped to next track on {player_id}"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def previous_track(player_id: str) -> str:
        """Go to the previous track on a player.

        :param player_id: Player ID.
        """
        try:
            await mass.player_queues.previous(player_id)
            return f"Went to previous track on {player_id}"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def seek(player_id: str, position: int) -> str:
        """Seek to a specific position in the current track.

        :param player_id: Player ID.
        :param position: Position in seconds to seek to.
        """
        try:
            await mass.player_queues.seek(player_id, position)
            return f"Seeked to {position}s on {player_id}"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def skip_forward(player_id: str, seconds: int = 30) -> str:
        """Skip forward by a number of seconds.

        :param player_id: Player ID.
        :param seconds: Number of seconds to skip forward.
        """
        try:
            await mass.player_queues.skip(player_id, seconds)
            return f"Skipped forward {seconds}s on {player_id}"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def skip_backward(player_id: str, seconds: int = 30) -> str:
        """Skip backward by a number of seconds.

        :param player_id: Player ID.
        :param seconds: Number of seconds to skip backward.
        """
        try:
            await mass.player_queues.skip(player_id, -seconds)
            return f"Skipped backward {seconds}s on {player_id}"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def play_media(
        player_id: str,
        uri: str,
        enqueue_mode: str = "play",
        radio_mode: bool = False,
    ) -> str:
        """Play a media item by URI on a player.

        :param player_id: Player ID.
        :param uri: Media URI (e.g., spotify://track/abc).
        :param enqueue_mode: play, next, add, or replace.
        :param radio_mode: Create endless radio based on this item.
        """
        try:
            from music_assistant_models.enums import QueueOption  # noqa: PLC0415

            option_map = {
                "play": QueueOption.PLAY,
                "next": QueueOption.NEXT,
                "add": QueueOption.ADD,
                "replace": QueueOption.REPLACE,
            }
            option = option_map.get(enqueue_mode.lower(), QueueOption.PLAY)
            await mass.player_queues.play_media(
                queue_id=player_id,
                media=uri,
                option=option,
                radio_mode=radio_mode,
            )
            mode_str = " (radio mode)" if radio_mode else ""
            return f"Playing {uri} on {player_id}{mode_str}"
        except Exception as e:
            return f"Error: {e}"
