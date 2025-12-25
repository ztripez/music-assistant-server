"""Library and discovery tools for MCP server."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from music_assistant.controllers.media.base import SORT_KEYS

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from music_assistant.mass import MusicAssistant

# Valid sort options for library_items queries
VALID_SORT_OPTIONS = tuple(SORT_KEYS.keys())

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


def register_library_query_tools(mcp: FastMCP, mass: MusicAssistant) -> None:  # noqa: PLR0915
    """Register library query tools.

    :param mcp: FastMCP server instance.
    :param mass: MusicAssistant instance.
    """

    @mcp.tool()
    async def get_recommendations() -> str:
        """Get personalized music recommendations."""
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
        try:
            from music_assistant.helpers.uri import parse_uri  # noqa: PLC0415

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


def register_library_edit_tools(mcp: FastMCP, mass: MusicAssistant) -> None:
    """Register library edit tools.

    :param mcp: FastMCP server instance.
    :param mass: MusicAssistant instance.
    """

    @mcp.tool()
    async def add_to_library(uri: str) -> str:
        """Add an item to the user's library.

        :param uri: The URI of the item to add to library.
        """
        try:
            await mass.music.add_item_to_library(uri)
            return f"Added {uri} to library"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def add_to_favorites(uri: str) -> str:
        """Mark an item as favorite.

        :param uri: The URI of the item to favorite.
        """
        try:
            await mass.music.add_item_to_favorites(uri)
            return f"Added {uri} to favorites"
        except Exception as e:
            return f"Error: {e}"


def register_library_delete_tools(mcp: FastMCP, mass: MusicAssistant) -> None:
    """Register library delete tools.

    :param mcp: FastMCP server instance.
    :param mass: MusicAssistant instance.
    """

    @mcp.tool()
    async def remove_from_library(uri: str) -> str:
        """Remove an item from the user's library.

        :param uri: The URI of the library item to remove (must be a library:// URI).
        """
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
    async def remove_from_favorites(uri: str) -> str:
        """Remove an item from favorites.

        :param uri: The URI of the library item to unfavorite (must be a library:// URI).
        """
        try:
            from music_assistant.helpers.uri import parse_uri  # noqa: PLC0415

            media_type, provider, item_id = await parse_uri(uri)
            if provider != "library":
                return "Error: Can only unfavorite library items (use library:// URI)"

            await mass.music.remove_item_from_favorites(media_type, item_id)
            return f"Removed {uri} from favorites"
        except Exception as e:
            return f"Error: {e}"
