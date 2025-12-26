"""Recommendation engine for Music Assistant based on user interactions."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from music_assistant_models.enums import MediaType
from music_assistant_models.media_items import RecommendationFolder
from music_assistant_models.unique_list import UniqueList

from music_assistant.helpers.scrobbler import ScrobblerHelper
from music_assistant.mass import LOGGER

if TYPE_CHECKING:
    from music_assistant_models.playback_progress_report import MediaItemPlaybackProgressReport

    from music_assistant.mass import MusicAssistant

    from .sidecar_embeddings import SidecarEmbeddings

# Cache key for storing user interactions
INTERACTIONS_CACHE_KEY = "music_insights_interactions"


class RecommendationEngine:
    """Handles recommendation logic based on user interactions."""

    def __init__(self, mass: MusicAssistant, embeddings: SidecarEmbeddings) -> None:
        """
        Initialize RecommendationEngine.

        :param mass: The MusicAssistant instance.
        :param embeddings: The SidecarEmbeddings instance for similarity search.
        """
        self.mass = mass
        self.logger = LOGGER.getChild("music_insights.recommendations")
        self.embeddings = embeddings
        self._player_current_track: dict[str, str | None] = {}

    async def record_interaction(self, event: MediaItemPlaybackProgressReport) -> None:
        """
        Record a user interaction event (track playback progress).

        Stores metadata about the playback event in the MA cache,
        including timestamp, URI, play duration, and whether the track was fully played.

        :param event: The MediaItemPlaybackProgressReport event.
        """
        if event.media_type != MediaType.TRACK:
            return

        if event.mbid is None:
            return

        # Get existing interactions from cache
        interactions: dict[str, dict[str, float | str | bool]] = await self.mass.cache.get(
            INTERACTIONS_CACHE_KEY, default={}
        )

        # Create or update interaction record
        interaction: dict[str, float | str | bool] = {
            "timestamp": float(time.time()),
            "uri": event.uri,
            "fully_played": event.fully_played,
            "seconds_played": float(event.seconds_played),
            "duration": float(event.duration),
            "score": 1.0,
        }

        # Merge with existing if present (accumulate score)
        if event.mbid in interactions:
            existing = interactions[event.mbid]
            interaction["score"] = float(existing.get("score", 0)) + 1.0

        interactions[event.mbid] = interaction

        # Store back to cache
        await self.mass.cache.set(INTERACTIONS_CACHE_KEY, interactions)
        self.logger.debug("Recorded interaction: %s", event.mbid)

    async def get_recommendations(
        self,
        limit: int = 25,
        user_id: str | None = None,
    ) -> list[RecommendationFolder]:
        """
        Generate recommendations based on stored user interactions.

        Uses the taste profile API to compute personalized recommendations
        from aggregated user interactions.

        :param limit: The maximum number of recommendation items to return.
        :param user_id: User ID for multi-user setups (defaults to "default").
        :return: A list of RecommendationFolder objects with recommended tracks.
        """
        # Get interactions from cache
        interactions: dict[str, dict[str, float | str | bool]] = await self.mass.cache.get(
            INTERACTIONS_CACHE_KEY, default={}
        )

        if not interactions:
            return []

        # Convert to sidecar format
        interaction_list: list[dict[str, float | str]] = []
        for track_id, meta in interactions.items():
            # Classify signal type based on metadata
            signal_type = self._classify_signal(meta)

            interaction_list.append(
                {
                    "track_id": track_id,
                    "timestamp": int(meta.get("timestamp", time.time())),
                    "signal_type": signal_type,
                    "seconds_played": float(meta.get("seconds_played", 0)),
                    "duration": float(meta.get("duration", 1)),
                }
            )

        # Use user_id from MA user profile or fallback to "default"
        uid = user_id or "default"

        # Compute profile and get recommendations
        try:
            await self.embeddings.compute_user_profile(uid, interaction_list, cutoff_days=21)
            tracks = await self.embeddings.get_user_recommendations(uid, limit=limit)
        except Exception as err:
            self.logger.error("Failed to get taste recommendations: %s", err)
            return []

        if not tracks:
            return []

        folder = RecommendationFolder(
            item_id="music_insights_recommended",
            provider="library",
            name="Recommended for you",
            translation_key="recommended_tracks",
            icon="mdi-brain",
            items=UniqueList(tracks[:limit]),
        )
        return [folder]

    def _classify_signal(self, meta: dict[str, float | str | bool]) -> str:
        """
        Classify interaction as a signal type for the taste profile API.

        :param meta: Interaction metadata dict.
        :return: Signal type string (full_play, partial_play, skip, repeat).
        """
        fully_played = meta.get("fully_played", False)
        seconds_played = float(meta.get("seconds_played", 0))
        duration = float(meta.get("duration", 1))
        score = float(meta.get("score", 0))

        if score >= 2.0:  # Multiple plays
            return "repeat"
        if fully_played:
            return "full_play"
        if duration > 0 and seconds_played / duration > 0.5:
            return "partial_play"
        return "skip"


class InsightScrobbler(ScrobblerHelper):
    """Handles playback event handling for recommendations."""

    def __init__(self, logger: logging.Logger, rec: RecommendationEngine) -> None:
        """
        Initialize the InsightScrobbler.

        :param logger: The logger instance to use.
        :param rec: The RecommendationEngine instance.
        """
        super().__init__(logger)
        self.rec = rec

    async def _update_now_playing(self, report: MediaItemPlaybackProgressReport) -> None:
        """
        Handle the 'now playing' update event.

        :param report: The playback progress report.
        """
        await self.rec.record_interaction(report)

    async def _scrobble(self, report: MediaItemPlaybackProgressReport) -> None:
        """
        Handle the 'scrobble' event (track finished or played significantly).

        :param report: The playback progress report.
        """
        await self.rec.record_interaction(report)
