"""MCP Server implementation for Music Assistant."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from music_assistant.mass import MusicAssistant


# Valid sort options for library_items queries
# Maps user-friendly names to internal sort keys
VALID_SORT_OPTIONS = (
    "name",
    "name_desc",
    "sort_name",
    "sort_name_desc",
    "timestamp_added",
    "timestamp_added_desc",
    "last_played",
    "last_played_desc",
    "play_count",
    "play_count_desc",
    "random",
)

# Additional sort options for tracks/albums that have duration/year
EXTENDED_SORT_OPTIONS = (
    *VALID_SORT_OPTIONS,
    "duration",
    "duration_desc",
    "year",
    "year_desc",
    "artist_name",
    "artist_name_desc",
)

# Module-level state container
_state: dict[str, Any] = {
    "mass": None,
    "logger": None,
    "enabled_features": {},
}


def create_mcp_server(
    mass: MusicAssistant,
    require_auth: bool = True,
    enabled_features: dict[str, bool] | None = None,
) -> FastMCP:
    """Create and configure the MCP server instance.

    :param mass: MusicAssistant instance.
    :param require_auth: Whether to require authentication.
    :param enabled_features: Dictionary of feature flags to enable/disable tool categories.
    :return: Configured FastMCP server instance.
    """
    _state["mass"] = mass
    if enabled_features is not None:
        _state["enabled_features"] = enabled_features

    from mcp.server.transport_security import TransportSecuritySettings  # noqa: PLC0415

    server_kwargs: dict[str, Any] = {
        "name": "Music Assistant",
        "instructions": (
            "Music Assistant MCP server for controlling music playback "
            "and managing your music library."
        ),
        "stateless_http": True,
        "json_response": True,
        # Disable DNS rebinding protection to allow connections from any host
        "transport_security": TransportSecuritySettings(
            enable_dns_rebinding_protection=False,
        ),
    }

    if require_auth:
        from mcp.server.auth.settings import AuthSettings  # noqa: PLC0415
        from pydantic import AnyHttpUrl  # noqa: PLC0415

        from .auth import MusicAssistantTokenVerifier  # noqa: PLC0415

        # Use MA webserver as issuer for token validation
        # Set resource_server_url=None to avoid OAuth protected resource discovery
        base_url = mass.webserver.base_url
        server_kwargs["auth"] = AuthSettings(
            issuer_url=AnyHttpUrl(base_url),
            resource_server_url=None,
        )
        server_kwargs["token_verifier"] = MusicAssistantTokenVerifier(mass)

    mcp = FastMCP(**server_kwargs)
    features = enabled_features or {}

    if features.get("playback_tools", True):
        _register_playback_tools(mcp)
    if features.get("queue_tools", True):
        _register_queue_tools(mcp)
    if features.get("volume_tools", True):
        _register_volume_tools(mcp)
    if features.get("library_tools", True):
        _register_library_tools(mcp)
        _register_podcast_tools(mcp)
        _register_radio_tools(mcp)
        _register_audiobook_tools(mcp)
        _register_metadata_tools(mcp)
    if features.get("playlist_tools", True):
        _register_playlist_tools(mcp)
    if features.get("player_tools", True):
        _register_player_tools(mcp)
    if features.get("player_resources", True):
        _register_player_resources(mcp)
    if features.get("library_resources", True):
        _register_library_resources(mcp)
    if features.get("prompts", True):
        _register_prompts(mcp)

    return mcp


def _get_mass() -> MusicAssistant | None:
    """Get the MusicAssistant instance from module state."""
    mass = _state["mass"]
    if mass is None:
        return None
    return mass  # type: ignore[no-any-return]


# =============================================================================
# PLAYBACK CONTROL TOOLS
# =============================================================================


def _register_playback_tools(mcp: FastMCP) -> None:  # noqa: PLR0915
    """Register playback control tools."""

    @mcp.tool()
    async def play(player_id: str) -> str:
        """Start or resume playback on a player.

        :param player_id: Player ID from players:// resource.
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

        :param player_id: Player ID.
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
        """Stop playback on a player and clear the queue.

        :param player_id: Player ID.
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

        :param player_id: Player ID.
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

        :param player_id: Player ID.
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
    async def seek(player_id: str, position: int) -> str:
        """Seek to a specific position in the current track.

        :param player_id: Player ID.
        :param position: Position in seconds to seek to.
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
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
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
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
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
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
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
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


# =============================================================================
# QUEUE MANAGEMENT TOOLS
# =============================================================================


def _register_queue_tools(mcp: FastMCP) -> None:  # noqa: PLR0915
    """Register queue management tools."""

    @mcp.tool()
    async def get_queue(player_id: str, limit: int = 50) -> str:
        """Get items in a player's queue.

        :param player_id: Player ID.
        :param limit: Maximum number of items to return.
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            items = mass.player_queues.items(player_id, limit=limit)
            queue = mass.player_queues.get(player_id)
            current_index = queue.current_index if queue else 0

            output = {
                "queue_id": player_id,
                "current_index": current_index,
                "total_items": len(items),
                "items": [
                    {
                        "index": i,
                        "queue_item_id": item.queue_item_id,
                        "name": item.name,
                        "uri": item.uri if hasattr(item, "uri") else None,
                        "duration": item.duration,
                        "is_current": i == current_index,
                    }
                    for i, item in enumerate(items)
                ],
            }
            return json.dumps(output, indent=2)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def clear_queue(player_id: str) -> str:
        """Clear all items from a player's queue.

        :param player_id: Player ID.
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            mass.player_queues.clear(player_id)
            return f"Queue cleared on {player_id}"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def shuffle_queue(player_id: str, enabled: bool) -> str:
        """Set shuffle mode.

        :param player_id: Player ID.
        :param enabled: Enable shuffle.
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            await mass.player_queues.set_shuffle(player_id, enabled)
            state = "enabled" if enabled else "disabled"
            return f"Shuffle {state} on {player_id}"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def repeat_queue(player_id: str, mode: str) -> str:
        """Set repeat mode.

        :param player_id: Player ID.
        :param mode: off, one, or all.
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            from music_assistant_models.enums import RepeatMode  # noqa: PLC0415

            mode_map = {
                "off": RepeatMode.OFF,
                "one": RepeatMode.ONE,
                "all": RepeatMode.ALL,
            }
            repeat_mode = mode_map.get(mode.lower(), RepeatMode.OFF)
            mass.player_queues.set_repeat(player_id, repeat_mode)
            return f"Repeat mode set to '{mode}' on {player_id}"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def move_queue_item(
        player_id: str,
        position_shift: int,
        queue_item_id: str | None = None,
        index: int | None = None,
    ) -> str:
        """Move an item in the queue by a relative position.

        :param player_id: Player ID.
        :param position_shift: Number of positions to move (+/- for up/down).
        :param queue_item_id: The queue_item_id from get_queue output.
        :param index: Alternatively, the index of the item to move (0-based).
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        if queue_item_id is None and index is None:
            return "Error: Must provide either queue_item_id or index"
        try:
            # If index provided, look up the queue_item_id
            if queue_item_id is None and index is not None:
                items = mass.player_queues.items(player_id)
                if index < 0 or index >= len(items):
                    return f"Error: Index {index} out of range"
                queue_item_id = items[index].queue_item_id
            mass.player_queues.move_item(player_id, queue_item_id, position_shift)  # type: ignore[arg-type]
            direction = "up" if position_shift < 0 else "down"
            return f"Moved item {abs(position_shift)} position(s) {direction}"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def remove_queue_item(
        player_id: str, queue_item_id: str | None = None, index: int | None = None
    ) -> str:
        """Remove an item from the queue.

        :param player_id: Player ID.
        :param queue_item_id: The queue_item_id from get_queue output.
        :param index: Alternatively, the index of the item to remove (0-based).
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        if queue_item_id is None and index is None:
            return "Error: Must provide either queue_item_id or index"
        try:
            item_id_or_index: str | int = queue_item_id if queue_item_id else index  # type: ignore[assignment]
            mass.player_queues.delete_item(player_id, item_id_or_index)
            return "Removed item from queue"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def play_queue_index(player_id: str, index: int) -> str:
        """Play a specific item in the queue by index.

        :param player_id: Player ID.
        :param index: The index of the item to play (0-based).
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            await mass.player_queues.play_index(player_id, index)
            return f"Playing queue item at index {index}"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def transfer_queue(
        source_player_id: str,
        target_player_id: str,
    ) -> str:
        """Transfer a queue from one player to another.

        :param source_player_id: The player to transfer from.
        :param target_player_id: The player to transfer to.
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            await mass.player_queues.transfer_queue(source_player_id, target_player_id)
            return f"Queue transferred from {source_player_id} to {target_player_id}"
        except Exception as e:
            return f"Error: {e}"


# =============================================================================
# VOLUME CONTROL TOOLS
# =============================================================================


def _register_volume_tools(mcp: FastMCP) -> None:
    """Register volume control tools."""

    @mcp.tool()
    async def set_volume(player_id: str, volume: int) -> str:
        """Set the volume level of a player.

        :param player_id: Player ID.
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

    @mcp.tool()
    async def volume_up(player_id: str) -> str:
        """Increase the volume of a player by one step.

        :param player_id: Player ID.
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
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
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
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
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
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
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            volume = max(0, min(100, volume))
            await mass.players.cmd_group_volume(player_id, volume)
            return f"Group volume set to {volume}% on {player_id}"
        except Exception as e:
            return f"Error: {e}"


# =============================================================================
# LIBRARY & DISCOVERY TOOLS
# =============================================================================


def _register_library_tools(mcp: FastMCP) -> None:  # noqa: PLR0915
    """Register library and discovery tools."""

    @mcp.tool()
    async def get_recommendations() -> str:
        """Get personalized music recommendations."""
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            recommendations = await mass.music.recommendations()
            output = []
            for folder in recommendations:
                items = []
                for item in folder.items[:10]:  # Limit items per folder
                    items.append({"name": item.name, "uri": item.uri})
                output.append({"category": folder.name, "items": items})
            return json.dumps({"recommendations": output}, indent=2)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def get_recently_played(limit: int = 20) -> str:
        """Get recently played items.

        :param limit: Maximum number of items to return.
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            items = await mass.music.recently_played(limit=limit)
            output = [
                {"name": item.name, "uri": item.uri, "type": item.media_type.value}
                for item in items
            ]
            return json.dumps({"recently_played": output}, indent=2)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def get_recently_added(limit: int = 20) -> str:
        """Get recently added tracks to the library.

        :param limit: Maximum number of items to return.
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            items = await mass.music.recently_added_tracks(limit=limit)
            output = [{"name": item.name, "uri": item.uri} for item in items]
            return json.dumps({"recently_added": output}, indent=2)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def get_similar_tracks(track_uri: str, limit: int = 25) -> str:
        """Get tracks similar to a given track.

        :param track_uri: The URI of the track to find similar tracks for.
        :param limit: Maximum number of similar tracks.
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            from music_assistant.helpers.uri import parse_uri  # noqa: PLC0415

            # Parse the URI to get provider and item_id
            _, provider, item_id = await parse_uri(track_uri)

            similar = await mass.music.tracks.similar_tracks(item_id, provider, limit=limit)
            output = [{"name": t.name, "uri": t.uri} for t in similar]
            return json.dumps({"similar_tracks": output}, indent=2)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def browse_library(path: str = "") -> str:
        """Browse the music library by path.

        :param path: Path to browse (empty for root). Examples: 'library://artists'.
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            items = await mass.music.browse(path or None)
            output = []
            for item in items[:50]:  # Limit to 50 items
                entry = {"name": item.name, "uri": item.uri}
                if hasattr(item, "path"):
                    entry["path"] = item.path
                output.append(entry)
            return json.dumps({"items": output, "path": path}, indent=2)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def get_in_progress_items(limit: int = 20) -> str:
        """Get audiobooks and podcast episodes that are in progress.

        Returns items that have been partially played but not finished.

        :param limit: Maximum number of items to return.
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            items = await mass.music.in_progress_items(limit=limit)
            output = [
                {
                    "name": item.name,
                    "uri": item.uri,
                    "type": item.media_type.value if hasattr(item, "media_type") else None,
                }
                for item in items
            ]
            return json.dumps({"in_progress": output}, indent=2)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def get_artist_tracks(artist_uri: str, limit: int = 50) -> str:
        """Get all tracks by an artist.

        :param artist_uri: The URI of the artist.
        :param limit: Maximum number of tracks.
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            from music_assistant.helpers.uri import parse_uri  # noqa: PLC0415

            _, provider, item_id = await parse_uri(artist_uri)
            artist = await mass.music.get_item_by_uri(artist_uri)
            if not artist:
                return f"Error: Artist not found: {artist_uri}"

            all_tracks = await mass.music.artists.tracks(item_id, provider)
            tracks = [{"name": t.name, "uri": t.uri} for t in all_tracks[:limit]]

            return json.dumps({"artist": artist.name, "tracks": tracks}, indent=2)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def get_artist_albums(artist_uri: str, limit: int = 50) -> str:
        """Get all albums by an artist.

        :param artist_uri: The URI of the artist.
        :param limit: Maximum number of albums.
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            from music_assistant.helpers.uri import parse_uri  # noqa: PLC0415

            _, provider, item_id = await parse_uri(artist_uri)
            artist = await mass.music.get_item_by_uri(artist_uri)
            if not artist:
                return f"Error: Artist not found: {artist_uri}"

            all_albums = await mass.music.artists.albums(item_id, provider)
            albums = [{"name": a.name, "uri": a.uri} for a in all_albums[:limit]]

            return json.dumps({"artist": artist.name, "albums": albums}, indent=2)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def get_album_tracks(album_uri: str) -> str:
        """Get all tracks on an album.

        :param album_uri: The URI of the album.
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            from music_assistant.helpers.uri import parse_uri  # noqa: PLC0415

            _, provider, item_id = await parse_uri(album_uri)
            album = await mass.music.get_item_by_uri(album_uri)
            if not album:
                return f"Error: Album not found: {album_uri}"

            all_tracks = await mass.music.albums.tracks(item_id, provider)
            tracks = [
                {"name": t.name, "uri": t.uri, "track_number": t.track_number} for t in all_tracks
            ]

            return json.dumps({"album": album.name, "tracks": tracks}, indent=2)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def add_to_library(uri: str) -> str:
        """Add an item to the user's library.

        :param uri: The URI of the item to add to library.
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            await mass.music.add_item_to_library(uri)
            return f"Added {uri} to library"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def remove_from_library(uri: str) -> str:
        """Remove an item from the user's library.

        :param uri: The URI of the library item to remove (must be a library:// URI).
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            from music_assistant.helpers.uri import parse_uri  # noqa: PLC0415

            media_type, provider, item_id = await parse_uri(uri)
            if provider != "library":
                return "Error: Can only remove library items (use library:// URI)"

            await mass.music.remove_item_from_library(media_type, item_id)
            return f"Removed {uri} from library"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def add_to_favorites(uri: str) -> str:
        """Mark an item as favorite.

        :param uri: The URI of the item to favorite.
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            await mass.music.add_item_to_favorites(uri)
            return f"Added {uri} to favorites"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def remove_from_favorites(uri: str) -> str:
        """Remove an item from favorites.

        :param uri: The URI of the library item to unfavorite (must be a library:// URI).
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            from music_assistant.helpers.uri import parse_uri  # noqa: PLC0415

            media_type, provider, item_id = await parse_uri(uri)
            if provider != "library":
                return "Error: Can only unfavorite library items (use library:// URI)"

            await mass.music.remove_item_from_favorites(media_type, item_id)
            return f"Removed {uri} from favorites"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def get_library_artists(
        search: str = "",
        limit: int = 50,
        favorites_only: bool = False,
        order_by: str = "sort_name",
        provider: str = "",
    ) -> str:
        """Get artists from the library.

        :param search: Optional search filter.
        :param limit: Maximum number of artists.
        :param favorites_only: Only return favorited artists.
        :param order_by: Sort order. Options: name, name_desc, sort_name, sort_name_desc,
            timestamp_added, timestamp_added_desc, last_played, last_played_desc,
            play_count, play_count_desc, random.
        :param provider: Filter by provider instance ID (e.g., 'spotify_1').
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            if order_by and order_by not in VALID_SORT_OPTIONS:
                return f"Error: Invalid order_by. Valid options: {', '.join(VALID_SORT_OPTIONS)}"
            artists = await mass.music.artists.library_items(
                search=search or None,
                limit=limit,
                favorite=favorites_only if favorites_only else None,
                order_by=order_by or "sort_name",
                provider=provider or None,
            )
            output = [{"name": a.name, "uri": a.uri} for a in artists]
            return json.dumps({"artists": output}, indent=2)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def get_library_albums(
        search: str = "",
        limit: int = 50,
        favorites_only: bool = False,
        order_by: str = "sort_name",
        provider: str = "",
    ) -> str:
        """Get albums from the library.

        :param search: Optional search filter.
        :param limit: Maximum number of albums.
        :param favorites_only: Only return favorited albums.
        :param order_by: Sort order. Options: name, name_desc, sort_name, sort_name_desc,
            timestamp_added, timestamp_added_desc, last_played, last_played_desc,
            play_count, play_count_desc, random, year, year_desc, artist_name, artist_name_desc.
        :param provider: Filter by provider instance ID (e.g., 'spotify_1').
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            if order_by and order_by not in EXTENDED_SORT_OPTIONS:
                return f"Error: Invalid order_by. Valid options: {', '.join(EXTENDED_SORT_OPTIONS)}"
            albums = await mass.music.albums.library_items(
                search=search or None,
                limit=limit,
                favorite=favorites_only if favorites_only else None,
                order_by=order_by or "sort_name",
                provider=provider or None,
            )
            output = [{"name": a.name, "uri": a.uri} for a in albums]
            return json.dumps({"albums": output}, indent=2)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def get_library_tracks(
        search: str = "",
        limit: int = 50,
        favorites_only: bool = False,
        order_by: str = "sort_name",
        provider: str = "",
    ) -> str:
        """Get tracks from the library.

        :param search: Optional search filter.
        :param limit: Maximum number of tracks.
        :param favorites_only: Only return favorited tracks.
        :param order_by: Sort order. Options: name, name_desc, sort_name, sort_name_desc,
            timestamp_added, timestamp_added_desc, last_played, last_played_desc,
            play_count, play_count_desc, random, duration, duration_desc, year, year_desc,
            artist_name, artist_name_desc.
        :param provider: Filter by provider instance ID (e.g., 'spotify_1').
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            if order_by and order_by not in EXTENDED_SORT_OPTIONS:
                return f"Error: Invalid order_by. Valid options: {', '.join(EXTENDED_SORT_OPTIONS)}"
            tracks = await mass.music.tracks.library_items(
                search=search or None,
                limit=limit,
                favorite=favorites_only if favorites_only else None,
                order_by=order_by or "sort_name",
                provider=provider or None,
            )
            output = [{"name": t.name, "uri": t.uri} for t in tracks]
            return json.dumps({"tracks": output}, indent=2)
        except Exception as e:
            return f"Error: {e}"


# =============================================================================
# PLAYLIST MANAGEMENT TOOLS
# =============================================================================


def _register_playlist_tools(mcp: FastMCP) -> None:  # noqa: PLR0915
    """Register playlist management tools."""

    @mcp.tool()
    async def get_playlists(
        search: str = "",
        limit: int = 50,
        order_by: str = "sort_name",
        provider: str = "",
    ) -> str:
        """Get playlists from the library.

        :param search: Optional search filter.
        :param limit: Maximum number of playlists.
        :param order_by: Sort order. Options: name, name_desc, sort_name, sort_name_desc,
            timestamp_added, timestamp_added_desc, random.
        :param provider: Filter by provider instance ID (e.g., 'spotify_1').
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            if order_by and order_by not in VALID_SORT_OPTIONS:
                return f"Error: Invalid order_by. Valid options: {', '.join(VALID_SORT_OPTIONS)}"
            playlists = await mass.music.playlists.library_items(
                search=search or None,
                limit=limit,
                order_by=order_by or "sort_name",
                provider=provider or None,
            )
            output = [{"name": p.name, "uri": p.uri} for p in playlists]
            return json.dumps({"playlists": output}, indent=2)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def get_playlist_tracks(playlist_uri: str, limit: int = 100) -> str:
        """Get tracks in a playlist.

        :param playlist_uri: The URI of the playlist.
        :param limit: Maximum number of tracks.
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            playlist = await mass.music.get_item_by_uri(playlist_uri)
            if not playlist:
                return f"Error: Playlist not found: {playlist_uri}"

            tracks = []
            async for track in mass.music.playlists.tracks(playlist.item_id, playlist.provider):
                tracks.append({"name": track.name, "uri": track.uri})
                if len(tracks) >= limit:
                    break

            return json.dumps({"playlist": playlist.name, "tracks": tracks}, indent=2)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def create_playlist(name: str) -> str:
        """Create a new playlist.

        :param name: The name for the new playlist.
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            playlist = await mass.music.playlists.create_playlist(name)
            return json.dumps(
                {"created": True, "name": playlist.name, "uri": playlist.uri},
                indent=2,
            )
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def add_to_playlist(playlist_uri: str, track_uri: str) -> str:
        """Add a track to a playlist.

        :param playlist_uri: The URI of the playlist.
        :param track_uri: The URI of the track to add.
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            playlist = await mass.music.get_item_by_uri(playlist_uri)
            if not playlist:
                return f"Error: Playlist not found: {playlist_uri}"

            await mass.music.playlists.add_playlist_track(playlist.item_id, track_uri)
            return f"Added track to playlist {playlist.name}"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def remove_from_playlist(playlist_uri: str, position: int) -> str:
        """Remove a track from a playlist by position.

        :param playlist_uri: The URI of the playlist.
        :param position: The position (0-based index) of the track to remove.
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            from music_assistant.helpers.uri import parse_uri  # noqa: PLC0415

            _, _, item_id = await parse_uri(playlist_uri)
            playlist = await mass.music.get_item_by_uri(playlist_uri)
            if not playlist:
                return f"Error: Playlist not found: {playlist_uri}"

            await mass.music.playlists.remove_playlist_tracks(item_id, (position,))
            return f"Removed track at position {position} from playlist {playlist.name}"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def delete_playlist(playlist_uri: str) -> str:
        """Delete a playlist from the library.

        :param playlist_uri: The URI of the playlist to delete.
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            playlist = await mass.music.get_item_by_uri(playlist_uri)
            if not playlist:
                return f"Error: Playlist not found: {playlist_uri}"

            await mass.music.playlists.remove_item_from_library(playlist.item_id)
            return json.dumps(
                {"deleted": True, "name": playlist.name, "uri": playlist_uri},
                indent=2,
            )
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def clear_playlist(playlist_uri: str) -> str:
        """Remove all tracks from a playlist.

        :param playlist_uri: The URI of the playlist to clear.
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            from music_assistant.helpers.uri import parse_uri  # noqa: PLC0415

            _, _, item_id = await parse_uri(playlist_uri)
            playlist = await mass.music.get_item_by_uri(playlist_uri)
            if not playlist:
                return f"Error: Playlist not found: {playlist_uri}"

            # Get all track positions
            positions = []
            idx = 0
            async for _track in mass.music.playlists.tracks(item_id, playlist.provider):
                positions.append(idx)
                idx += 1

            if not positions:
                return f"Playlist {playlist.name} is already empty"

            await mass.music.playlists.remove_playlist_tracks(item_id, tuple(positions))
            return f"Cleared all {len(positions)} tracks from playlist {playlist.name}"
        except Exception as e:
            return f"Error: {e}"


# =============================================================================
# PODCAST TOOLS
# =============================================================================


def _register_podcast_tools(mcp: FastMCP) -> None:
    """Register podcast management tools."""

    @mcp.tool()
    async def get_library_podcasts(
        search: str = "",
        limit: int = 50,
        order_by: str = "sort_name",
        provider: str = "",
    ) -> str:
        """Get podcasts from the library.

        :param search: Optional search filter.
        :param limit: Maximum number of podcasts.
        :param order_by: Sort order. Options: name, name_desc, sort_name, sort_name_desc,
            timestamp_added, timestamp_added_desc, last_played, last_played_desc, random.
        :param provider: Filter by provider instance ID (e.g., 'spotify_1').
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            if order_by and order_by not in VALID_SORT_OPTIONS:
                return f"Error: Invalid order_by. Valid options: {', '.join(VALID_SORT_OPTIONS)}"
            podcasts = await mass.music.podcasts.library_items(
                search=search or None,
                limit=limit,
                order_by=order_by or "sort_name",
                provider=provider or None,
            )
            output = [
                {
                    "name": p.name,
                    "uri": p.uri,
                    "publisher": getattr(p, "publisher", None),
                    "total_episodes": getattr(p, "total_episodes", None),
                }
                for p in podcasts
            ]
            return json.dumps({"podcasts": output}, indent=2)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def get_podcast_episodes(podcast_uri: str, limit: int = 50) -> str:
        """Get episodes for a podcast.

        :param podcast_uri: The URI of the podcast.
        :param limit: Maximum number of episodes.
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            from music_assistant.helpers.uri import parse_uri  # noqa: PLC0415

            _, provider, item_id = await parse_uri(podcast_uri)
            podcast = await mass.music.get_item_by_uri(podcast_uri)
            if not podcast:
                return f"Error: Podcast not found: {podcast_uri}"

            episodes = []
            async for episode in mass.music.podcasts.episodes(item_id, provider):
                episodes.append(
                    {
                        "name": episode.name,
                        "uri": episode.uri,
                        "duration": episode.duration,
                        "position": getattr(episode, "position", None),
                        "resume_position_ms": getattr(episode, "resume_position_ms", None),
                        "fully_played": getattr(episode, "fully_played", None),
                    }
                )
                if len(episodes) >= limit:
                    break

            return json.dumps(
                {"podcast": podcast.name, "episodes": episodes},
                indent=2,
            )
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def play_podcast_episode(
        player_id: str,
        episode_uri: str,
        resume: bool = True,
    ) -> str:
        """Play a podcast episode on a player.

        :param player_id: Player ID from players:// resource.
        :param episode_uri: The URI of the podcast episode to play.
        :param resume: Resume from last position if available.
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            from music_assistant_models.enums import QueueOption  # noqa: PLC0415

            await mass.player_queues.play_media(
                queue_id=player_id,
                media=episode_uri,
                option=QueueOption.PLAY,
                radio_mode=False,
            )
            resume_note = " (resuming from last position)" if resume else ""
            return f"Playing podcast episode on {player_id}{resume_note}"
        except Exception as e:
            return f"Error: {e}"


# =============================================================================
# RADIO TOOLS
# =============================================================================


def _register_radio_tools(mcp: FastMCP) -> None:
    """Register radio station tools."""

    @mcp.tool()
    async def get_library_radios(
        search: str = "",
        limit: int = 50,
        order_by: str = "sort_name",
        provider: str = "",
    ) -> str:
        """Get radio stations from the library.

        :param search: Optional search filter.
        :param limit: Maximum number of radio stations.
        :param order_by: Sort order. Options: name, name_desc, sort_name, sort_name_desc,
            timestamp_added, timestamp_added_desc, random.
        :param provider: Filter by provider instance ID (e.g., 'tunein').
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            if order_by and order_by not in VALID_SORT_OPTIONS:
                return f"Error: Invalid order_by. Valid options: {', '.join(VALID_SORT_OPTIONS)}"
            radios = await mass.music.radio.library_items(
                search=search or None,
                limit=limit,
                order_by=order_by or "sort_name",
                provider=provider or None,
            )
            output = [
                {
                    "name": r.name,
                    "uri": r.uri,
                    "favorite": r.favorite,
                }
                for r in radios
            ]
            return json.dumps({"radios": output}, indent=2)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def play_radio_station(player_id: str, radio_uri: str) -> str:
        """Play a radio station on a player.

        :param player_id: Player ID from players:// resource.
        :param radio_uri: The URI of the radio station to play.
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            from music_assistant_models.enums import QueueOption  # noqa: PLC0415

            radio = await mass.music.get_item_by_uri(radio_uri)
            if not radio:
                return f"Error: Radio station not found: {radio_uri}"

            await mass.player_queues.play_media(
                queue_id=player_id,
                media=radio_uri,
                option=QueueOption.PLAY,
            )
            return f"Playing radio station '{radio.name}' on {player_id}"
        except Exception as e:
            return f"Error: {e}"


# =============================================================================
# AUDIOBOOK TOOLS
# =============================================================================


def _register_audiobook_tools(mcp: FastMCP) -> None:
    """Register audiobook management tools."""

    @mcp.tool()
    async def get_library_audiobooks(
        search: str = "",
        limit: int = 50,
        order_by: str = "sort_name",
        provider: str = "",
    ) -> str:
        """Get audiobooks from the library.

        :param search: Optional search filter.
        :param limit: Maximum number of audiobooks.
        :param order_by: Sort order. Options: name, name_desc, sort_name, sort_name_desc,
            timestamp_added, timestamp_added_desc, last_played, last_played_desc, random,
            duration, duration_desc.
        :param provider: Filter by provider instance ID (e.g., 'audible').
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            if order_by and order_by not in EXTENDED_SORT_OPTIONS:
                return f"Error: Invalid order_by. Valid options: {', '.join(EXTENDED_SORT_OPTIONS)}"
            audiobooks = await mass.music.audiobooks.library_items(
                search=search or None,
                limit=limit,
                order_by=order_by or "sort_name",
                provider=provider or None,
            )
            output = [
                {
                    "name": ab.name,
                    "uri": ab.uri,
                    "authors": getattr(ab, "authors", []),
                    "narrators": getattr(ab, "narrators", []),
                    "duration": getattr(ab, "duration", None),
                    "resume_position_ms": getattr(ab, "resume_position_ms", None),
                    "fully_played": getattr(ab, "fully_played", None),
                }
                for ab in audiobooks
            ]
            return json.dumps({"audiobooks": output}, indent=2)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def get_audiobook_chapters(audiobook_uri: str) -> str:
        """Get chapters for an audiobook.

        :param audiobook_uri: The URI of the audiobook.
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            audiobook = await mass.music.get_item_by_uri(audiobook_uri)
            if not audiobook:
                return f"Error: Audiobook not found: {audiobook_uri}"

            chapters = []
            if hasattr(audiobook, "metadata") and hasattr(audiobook.metadata, "chapters"):
                for chapter in audiobook.metadata.chapters or []:
                    chapters.append(
                        {
                            "position": chapter.position,
                            "name": getattr(chapter, "name", f"Chapter {chapter.position}"),
                            "start_seconds": chapter.start,
                        }
                    )

            return json.dumps(
                {
                    "audiobook": audiobook.name,
                    "chapters": chapters,
                    "resume_position_ms": getattr(audiobook, "resume_position_ms", None),
                    "fully_played": getattr(audiobook, "fully_played", None),
                },
                indent=2,
            )
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def play_audiobook(
        player_id: str,
        audiobook_uri: str,
        chapter: int | None = None,
    ) -> str:
        """Play an audiobook on a player.

        :param player_id: Player ID from players:// resource.
        :param audiobook_uri: The URI of the audiobook to play.
        :param chapter: Optional chapter number to start from (1-based).
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            from music_assistant_models.enums import QueueOption  # noqa: PLC0415

            # Convert chapter to string for play_media (it will be parsed internally)
            start_item = str(chapter) if chapter is not None else None
            await mass.player_queues.play_media(
                queue_id=player_id,
                media=audiobook_uri,
                option=QueueOption.PLAY,
                radio_mode=False,
                start_item=start_item,
            )
            chapter_note = f" from chapter {chapter}" if chapter else " (resuming)"
            return f"Playing audiobook on {player_id}{chapter_note}"
        except Exception as e:
            return f"Error: {e}"


# =============================================================================
# METADATA TOOLS
# =============================================================================


def _register_metadata_tools(mcp: FastMCP) -> None:
    """Register metadata and lyrics tools."""

    @mcp.tool()
    async def get_track_lyrics(track_uri: str) -> str:
        """Get lyrics for a track.

        :param track_uri: The URI of the track.
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            from music_assistant_models.enums import MediaType  # noqa: PLC0415
            from music_assistant_models.media_items import BrowseFolder  # noqa: PLC0415

            item = await mass.music.get_item_by_uri(track_uri)
            if not item or isinstance(item, BrowseFolder):
                return f"Error: Track not found: {track_uri}"
            if item.media_type != MediaType.TRACK:
                return f"Error: URI is not a track: {track_uri}"

            lyrics, lrc_lyrics = await mass.metadata.get_track_lyrics(item)  # type: ignore[arg-type]
            if not lyrics and not lrc_lyrics:
                return json.dumps(
                    {"track": item.name, "lyrics": None, "message": "No lyrics found"},
                    indent=2,
                )

            return json.dumps(
                {
                    "track": item.name,
                    "artist": getattr(item, "artist_str", None),
                    "lyrics": lyrics,
                    "synced_lyrics": lrc_lyrics is not None,
                },
                indent=2,
            )
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def get_item_artwork(uri: str) -> str:
        """Get artwork URL for a media item.

        :param uri: The URI of the media item (track, album, artist, playlist, etc.).
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            from music_assistant_models.enums import ImageType  # noqa: PLC0415
            from music_assistant_models.media_items import BrowseFolder  # noqa: PLC0415

            item = await mass.music.get_item_by_uri(uri)
            if not item or isinstance(item, BrowseFolder):
                return f"Error: Item not found: {uri}"

            # Get different image types
            thumb_url = await mass.metadata.get_image_url_for_item(item, img_type=ImageType.THUMB)
            fanart_url = await mass.metadata.get_image_url_for_item(item, img_type=ImageType.FANART)

            return json.dumps(
                {
                    "name": item.name,
                    "uri": uri,
                    "thumbnail": thumb_url,
                    "fanart": fanart_url,
                },
                indent=2,
            )
        except Exception as e:
            return f"Error: {e}"


# =============================================================================
# PLAYER MANAGEMENT TOOLS
# =============================================================================


def _register_player_tools(mcp: FastMCP) -> None:
    """Register player management tools."""

    @mcp.tool()
    async def power_player(player_id: str, powered: bool) -> str:
        """Power on or off a player.

        :param player_id: Player ID.
        :param powered: Power on.
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
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
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
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
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
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
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
        try:
            await mass.players.play_announcement(player_id, url, volume_level=volume)
            return f"Playing announcement on {player_id}"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def get_player_by_name(name: str) -> str:
        """Find a player by name, including its capabilities.

        :param name: Full or partial player name.
        """
        mass = _get_mass()
        if mass is None:
            return "Error: Music Assistant not initialized"
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


# =============================================================================
# PLAYER RESOURCES
# =============================================================================


def _register_player_resources(mcp: FastMCP) -> None:
    """Register player-related MCP resources."""

    @mcp.resource("players://")
    async def list_players() -> str:
        """List all available players/speakers with their capabilities."""
        mass = _get_mass()
        if mass is None:
            return json.dumps({"error": "Music Assistant not initialized"})

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

    @mcp.resource("queue://{player_id}")
    async def get_queue_contents(player_id: str) -> str:
        """Get the full queue contents for a player."""
        mass = _get_mass()
        if mass is None:
            return json.dumps({"error": "Music Assistant not initialized"})

        queue = mass.player_queues.get(player_id)
        if not queue:
            return json.dumps({"error": f"Queue for {player_id} not found"})

        items = mass.player_queues.items(player_id, limit=100)
        queue_items = []
        for item in items:
            queue_items.append(
                {
                    "index": item.queue_item_id,
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
                    "tota   l_items": len(queue_items),
                }
            },
            indent=2,
        )


# =============================================================================
# LIBRARY RESOURCES
# =============================================================================


def _register_library_resources(mcp: FastMCP) -> None:  # noqa: PLR0915
    """Register library-related MCP resources."""

    @mcp.resource("library://stats")
    async def get_library_stats() -> str:
        """Get statistics about the music library."""
        mass = _get_mass()
        if mass is None:
            return json.dumps({"error": "Music Assistant not initialized"})

        try:
            stats = {
                "artists": await mass.music.artists.library_count(),
                "albums": await mass.music.albums.library_count(),
                "tracks": await mass.music.tracks.library_count(),
                "playlists": await mass.music.playlists.library_count(),
                "podcasts": await mass.music.podcasts.library_count(),
                "audiobooks": await mass.music.audiobooks.library_count(),
                "radios": await mass.music.radio.library_count(),
            }
            return json.dumps({"library_stats": stats}, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource("library://favorites")
    async def get_favorites() -> str:
        """Get the user's favorite items from the library."""
        mass = _get_mass()
        if mass is None:
            return json.dumps({"error": "Music Assistant not initialized"})

        try:
            favorites: dict[str, list[dict[str, str | None]]] = {
                "artists": [],
                "albums": [],
                "tracks": [],
            }

            # Get favorite artists (limit 20)
            artists = await mass.music.artists.library_items(favorite=True, limit=20)
            for artist in artists:
                favorites["artists"].append({"name": artist.name, "uri": artist.uri})

            # Get favorite albums (limit 20)
            albums = await mass.music.albums.library_items(favorite=True, limit=20)
            for album in albums:
                favorites["albums"].append({"name": album.name, "uri": album.uri})

            # Get favorite tracks (limit 30)
            tracks = await mass.music.tracks.library_items(favorite=True, limit=30)
            for track in tracks:
                favorites["tracks"].append({"name": track.name, "uri": track.uri})

            return json.dumps({"favorites": favorites}, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource("library://recently_played")
    async def get_recently_played_resource() -> str:
        """Get recently played items from the library."""
        mass = _get_mass()
        if mass is None:
            return json.dumps({"error": "Music Assistant not initialized"})

        try:
            recently_played = await mass.music.recently_played(limit=30)
            items = [{"name": item.name, "uri": item.uri} for item in recently_played]
            return json.dumps({"recently_played": items}, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource("providers://")
    async def list_providers() -> str:
        """List all configured music providers."""
        mass = _get_mass()
        if mass is None:
            return json.dumps({"error": "Music Assistant not initialized"})

        providers = []
        for provider in mass.music.providers:
            providers.append(
                {
                    "id": provider.instance_id,
                    "name": provider.name,
                    "domain": provider.domain,
                    "available": provider.available,
                }
            )

        return json.dumps({"providers": providers}, indent=2)

    @mcp.resource("library://podcasts")
    async def get_library_podcasts_resource() -> str:
        """List all podcasts in the library."""
        mass = _get_mass()
        if mass is None:
            return json.dumps({"error": "Music Assistant not initialized"})

        try:
            podcasts = await mass.music.podcasts.library_items(limit=100)
            items = [
                {
                    "name": p.name,
                    "uri": p.uri,
                    "publisher": getattr(p, "publisher", None),
                    "total_episodes": getattr(p, "total_episodes", None),
                }
                for p in podcasts
            ]
            return json.dumps({"podcasts": items}, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource("library://audiobooks")
    async def get_library_audiobooks_resource() -> str:
        """List all audiobooks in the library."""
        mass = _get_mass()
        if mass is None:
            return json.dumps({"error": "Music Assistant not initialized"})

        try:
            audiobooks = await mass.music.audiobooks.library_items(limit=100)
            items = [
                {
                    "name": ab.name,
                    "uri": ab.uri,
                    "authors": getattr(ab, "authors", []),
                    "narrators": getattr(ab, "narrators", []),
                }
                for ab in audiobooks
            ]
            return json.dumps({"audiobooks": items}, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource("library://radios")
    async def get_library_radios_resource() -> str:
        """List all radio stations in the library."""
        mass = _get_mass()
        if mass is None:
            return json.dumps({"error": "Music Assistant not initialized"})

        try:
            radios = await mass.music.radio.library_items(limit=100)
            items = [
                {
                    "name": r.name,
                    "uri": r.uri,
                    "favorite": r.favorite,
                }
                for r in radios
            ]
            return json.dumps({"radios": items}, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})


# =============================================================================
# PROMPTS (User-invokable templates per MCP spec)
# =============================================================================


def _register_prompts(mcp: FastMCP) -> None:  # noqa: PLR0915
    """Register MCP prompts as user-invokable templates."""

    @mcp.prompt()
    async def play_music(query: str = "", player: str = "") -> str:
        """Request to play music."""
        mass = _get_mass()
        players_info = ""
        if mass:
            players = [p.display_name for p in mass.players.all() if p.available]
            players_info = f"\n\nAvailable players: {', '.join(players)}" if players else ""

        query_part = f'"{query}"' if query else "some music"
        player_part = f" on {player}" if player else ""

        return f"I want to play {query_part}{player_part}.{players_info}"

    @mcp.prompt()
    async def whats_playing(player: str = "") -> str:
        """Check current playback status."""
        mass = _get_mass()
        if not mass:
            return "What's currently playing?"

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
        mass = _get_mass()
        players_info = ""
        if mass:
            players = [p.display_name for p in mass.players.all() if p.available]
            players_info = f"\n\nAvailable speakers: {', '.join(players)}" if players else ""

        if rooms:
            return f"Help me sync music across these rooms: {rooms}.{players_info}"
        return f"Help me set up multi-room audio to play music in sync.{players_info}"

    @mcp.prompt()
    async def transfer_playback(from_player: str = "", to_player: str = "") -> str:
        """Move playback between players."""
        mass = _get_mass()
        players_info = ""
        if mass:
            players = [p.display_name for p in mass.players.all() if p.available]
            players_info = f"\n\nAvailable players: {', '.join(players)}" if players else ""

        if from_player and to_player:
            return f"Transfer what's playing from {from_player} to {to_player}.{players_info}"
        if to_player:
            return f"I want to continue listening on {to_player}.{players_info}"
        if from_player:
            return f"Move what's playing on {from_player} to another player.{players_info}"
        return f"Help me transfer music from one player to another.{players_info}"


# =============================================================================
# SERVER STARTUP
# =============================================================================
#
# Config Changes and Client Notifications
# ----------------------------------------
# The MCP server uses stateless HTTP mode for simplicity.
# When config toggles change, the provider reloads and restarts the server.
# Clients detect disconnection, reconnect, and receive updated capabilities.
#
# Note: Stateful connections (for real-time notifications like now-playing
# updates) would make sense for a music app, but are out of scope for this
# iteration. The current stateless approach prioritizes reliability.
# =============================================================================


async def start_mcp_server(
    mass: MusicAssistant,
    port: int,
    require_auth: bool,
    enabled_features: dict[str, bool],
    logger: logging.Logger,
) -> tuple[asyncio.Task[Any], asyncio.Event]:
    """Start the MCP server.

    :param mass: MusicAssistant instance.
    :param port: Port to run the server on.
    :param require_auth: Whether to require authentication.
    :param enabled_features: Dictionary of feature flags to enable/disable tool categories.
    :param logger: Logger instance.
    :return: Tuple of (server task, shutdown event).
    """
    _state["logger"] = logger
    _state["enabled_features"] = enabled_features

    # Create the MCP server with authentication and feature flags
    mcp = create_mcp_server(mass, require_auth, enabled_features)

    shutdown_event = asyncio.Event()

    async def run_server() -> None:
        """Run the uvicorn server."""
        import uvicorn  # noqa: PLC0415
        from starlette.middleware.cors import CORSMiddleware  # noqa: PLC0415

        # Get the base app
        app = mcp.streamable_http_app()

        # Add CORS middleware to handle preflight OPTIONS requests
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"],
        )

        # Map MA log level to uvicorn log level
        ma_log_level = logger.getEffectiveLevel()
        uvicorn_log_level = logging.getLevelName(ma_log_level).lower()

        config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=port,
            log_level=uvicorn_log_level,
            access_log=ma_log_level <= logging.DEBUG,  # Only show access log in debug
            log_config=None,  # Disable uvicorn's default logging config
        )
        server = uvicorn.Server(config)

        # Replace uvicorn's loggers with MA logger
        logging.getLogger("uvicorn").handlers = logger.handlers
        logging.getLogger("uvicorn").setLevel(ma_log_level)
        logging.getLogger("uvicorn.error").handlers = logger.handlers
        logging.getLogger("uvicorn.error").setLevel(ma_log_level)
        logging.getLogger("uvicorn.access").handlers = logger.handlers
        logging.getLogger("uvicorn.access").setLevel(ma_log_level)

        try:
            # Run server until shutdown is requested
            server_task = asyncio.create_task(server.serve())

            # Wait for shutdown signal
            await shutdown_event.wait()

            # Graceful shutdown - give connections 2 seconds to close
            server.should_exit = True
            try:
                await asyncio.wait_for(server_task, timeout=2.0)
            except TimeoutError:
                # Force exit if connections don't close in time
                server.force_exit = True
                try:
                    await asyncio.wait_for(server_task, timeout=1.0)
                except TimeoutError:
                    server_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await server_task
        except asyncio.CancelledError:
            # Handle task cancellation during shutdown
            server.force_exit = True
            server_task.cancel()
        except RuntimeError:
            # Event loop closing, force immediate exit
            server.force_exit = True
            server_task.cancel()

    task = asyncio.create_task(run_server())
    return task, shutdown_event
