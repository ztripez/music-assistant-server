"""Recommendation engine for Music Assistant based on user interactions."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING

from music_assistant_models.enums import MediaType

from music_assistant.helpers.scrobbler import ScrobblerHelper
from music_assistant.mass import LOGGER

if TYPE_CHECKING:
    from music_assistant_models.media_items import RecommendationFolder
    from music_assistant_models.playback_progress_report import MediaItemPlaybackProgressReport

    from music_assistant.mass import MusicAssistant

    from .chroma_embeddings import ChromaEmbeddings


class RecommendationEngine:
    """Handles recommendation logic based on user interactions."""

    def __init__(self, mass: MusicAssistant, chroma: ChromaEmbeddings):
        """Initialize RecommendationEngine."""
        self.mass = mass
        self.logger = LOGGER.getChild("music_insights.recommendations")
        self.chroma = chroma
        self._player_current_track: dict[str, str | None] = {}

    async def record_interaction(self, event: MediaItemPlaybackProgressReport) -> None:
        """
        Record a user interaction event (track playback progress) in ChromaDB.

        Stores metadata about the playback event, including timestamp, URI,
        play duration, and whether the track was fully played.

        Args:
            event: The MediaItemPlaybackProgressReport event.
        """
        if event.media_type != MediaType.TRACK:
            return

        if event.mbid is None:
            return

        metadata: Mapping[str, str | int | float | bool] = {
            "timestamp": float(time.time()),
            "uri": event.uri,
            "fully_played": event.fully_played,
            "seconds_played": event.seconds_played,
            "duration": event.duration,
            "score": 1,
        }
        try:
            await asyncio.to_thread(
                self.chroma.user_collection.add,
                ids=[event.mbid],
                metadatas=[metadata],
            )
            self.logger.debug("Recorded interaction to ChromaDB: %s", metadata)
        except Exception as e:
            self.logger.error(
                "Failed to record interaction to ChromaDB: %s - %s", metadata, e, exc_info=e
            )

    async def get_recommendations(self, limit: int = 25) -> list[RecommendationFolder]:
        """
        Generate recommendations based on stored user interactions.

        NOTE: This is currently a placeholder and needs implementation.
              It should query ChromaDB for user interactions, calculate scores,
              and find similar tracks to generate recommendations.

        Args:
            limit: The maximum number of recommendation folders/items to return.

        Returns:
            A list of RecommendationFolder objects (currently empty).
        """
        # TODO: Implement actual recommendation logic using interaction scores
        # and ChromaDB similarity searches.
        return []


class InsightScrobbler(ScrobblerHelper):
    """Handles the event handling."""

    def __init__(self, logger: logging.Logger, rec: RecommendationEngine) -> None:
        """
        Initialize the InsightScrobbler.

        Args:
            logger: The logger instance to use.
            rec: The RecommendationEngine instance.
        """
        super().__init__(logger)
        self.rec = rec

    async def _update_now_playing(self, report: MediaItemPlaybackProgressReport) -> None:
        """
        Handle the 'now playing' update event.

        Records the interaction using the RecommendationEngine.

        Args:
            report: The playback progress report.
        """
        await self.rec.record_interaction(report)

    async def _scrobble(self, report: MediaItemPlaybackProgressReport) -> None:
        """
        Handle the 'scrobble' event (track finished or played significantly).

        Records the interaction using the RecommendationEngine.

        Args:
            report: The playback progress report.
        """
        await self.rec.record_interaction(report)
