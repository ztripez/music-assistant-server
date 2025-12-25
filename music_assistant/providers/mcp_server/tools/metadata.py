"""Metadata and lyrics tools for MCP server."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from music_assistant.mass import MusicAssistant


def register_metadata_tools(mcp: FastMCP, mass: MusicAssistant) -> None:
    """Register metadata and lyrics tools.

    :param mcp: FastMCP server instance.
    :param mass: MusicAssistant instance.
    """

    @mcp.tool()
    async def get_track_lyrics(track_uri: str) -> str:
        """Get lyrics for a track.

        :param track_uri: The URI of the track.
        """
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
