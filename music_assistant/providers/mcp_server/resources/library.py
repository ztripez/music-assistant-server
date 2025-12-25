"""Library-related MCP resources."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from music_assistant.mass import MusicAssistant


def register_library_resources(mcp: FastMCP, mass: MusicAssistant) -> None:
    """Register library-related MCP resources.

    :param mcp: FastMCP server instance.
    :param mass: MusicAssistant instance.
    """

    @mcp.resource("library://stats")
    async def get_library_stats() -> str:
        """Get statistics about the music library."""
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
        try:
            recently_played = await mass.music.recently_played(limit=30)
            items = [{"name": item.name, "uri": item.uri} for item in recently_played]
            return json.dumps({"recently_played": items}, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource("providers://")
    async def list_providers() -> str:
        """List all configured music providers."""
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
