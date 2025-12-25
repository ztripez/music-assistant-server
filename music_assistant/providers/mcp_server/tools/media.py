"""Podcast, radio, and audiobook tools for MCP server."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from .library import EXTENDED_SORT_OPTIONS, VALID_SORT_OPTIONS

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from music_assistant.mass import MusicAssistant


def register_podcast_tools(mcp: FastMCP, mass: MusicAssistant) -> None:
    """Register podcast management tools.

    :param mcp: FastMCP server instance.
    :param mass: MusicAssistant instance.
    """

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
    async def play_podcast_episode(player_id: str, episode_uri: str) -> str:
        """Play a podcast episode on a player.

        :param player_id: Player ID from players:// resource.
        :param episode_uri: The URI of the podcast episode to play.
        """
        try:
            from music_assistant_models.enums import QueueOption  # noqa: PLC0415

            await mass.player_queues.play_media(
                queue_id=player_id,
                media=episode_uri,
                option=QueueOption.PLAY,
                radio_mode=False,
            )
            return f"Playing podcast episode on {player_id}"
        except Exception as e:
            return f"Error: {e}"


def register_radio_tools(mcp: FastMCP, mass: MusicAssistant) -> None:
    """Register radio station tools.

    :param mcp: FastMCP server instance.
    :param mass: MusicAssistant instance.
    """

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


def register_audiobook_tools(mcp: FastMCP, mass: MusicAssistant) -> None:
    """Register audiobook management tools.

    :param mcp: FastMCP server instance.
    :param mass: MusicAssistant instance.
    """

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
