"""Playlist management tools for MCP server."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from .library import VALID_SORT_OPTIONS

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from music_assistant.mass import MusicAssistant


def register_playlist_tools(mcp: FastMCP, mass: MusicAssistant) -> None:  # noqa: PLR0915
    """Register playlist management tools.

    :param mcp: FastMCP server instance.
    :param mass: MusicAssistant instance.
    """

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
        try:
            from music_assistant.helpers.uri import parse_uri  # noqa: PLC0415

            _, _, item_id = await parse_uri(playlist_uri)
            playlist = await mass.music.get_item_by_uri(playlist_uri)
            if not playlist:
                return f"Error: Playlist not found: {playlist_uri}"

            positions = []
            idx = 0
            async for _track in mass.music.playlists.tracks(item_id, playlist.provider):
                positions.append(idx)
                idx += 1

            if not positions:
                return f"Playlist {playlist.name} is already empty"

            await mass.music.playlists.remove_playlist_tracks(item_id, tuple(positions))
            return f"Cleared {len(positions)} tracks from playlist {playlist.name}"
        except Exception as e:
            return f"Error: {e}"
