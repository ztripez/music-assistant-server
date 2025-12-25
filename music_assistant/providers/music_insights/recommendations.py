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
    from music_assistant_models.media_items import Track
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

    async def get_recommendations(self, limit: int = 25) -> list[RecommendationFolder]:
        """
        Generate recommendations based on stored user interactions.

        Reads stored interactions from cache, determines the most relevant
        tracks, and returns similar tracks based on the user's taste profile.

        :param limit: The maximum number of recommendation items to return.
        :return: A list of RecommendationFolder objects with recommended tracks.
        """
        # Get interactions from cache
        interactions: dict[str, dict[str, float | str | bool]] = await self.mass.cache.get(
            INTERACTIONS_CACHE_KEY, default={}
        )

        if not interactions:
            return []

        # Score and rank interactions
        scored: dict[str, float] = {}
        id_to_uri: dict[str, str] = {}
        now = time.time()

        for track_id, meta in interactions.items():
            uri = str(meta.get("uri", ""))
            id_to_uri[track_id] = uri

            # Calculate base score
            base = float(meta.get("score", 0))
            if meta.get("fully_played"):
                base += 2

            duration = float(meta.get("duration", 0))
            if duration > 0:
                fraction = float(meta.get("seconds_played", 0)) / duration
                base += fraction * 2

            # Apply time decay
            timestamp = float(meta.get("timestamp", now))
            age_days = max(0.0, now - timestamp) / 86_400
            decay = 0.9**age_days
            score = base * decay

            scored[track_id] = scored.get(track_id, 0) + score

        # Sort by score, pick most relevant interactions
        sorted_ids = [
            track_id for track_id, _ in sorted(scored.items(), key=lambda x: x[1], reverse=True)
        ]

        recommended: UniqueList[Track] = UniqueList()

        for track_id in sorted_ids:
            if len(recommended) >= limit:
                break

            uri = id_to_uri.get(track_id) or ""
            if not uri:
                continue

            try:
                track = await self.mass.music.get_item_by_uri(uri)
            except Exception as err:
                self.logger.debug("Could not resolve track for %s: %s", uri, err)
                continue

            if not track:
                continue

            # Get similar tracks from sidecar
            similar = await self.embeddings.get_similar_tracks(track.item_id, limit=5)
            for item in similar:
                if item not in recommended:
                    recommended.append(item)
                if len(recommended) >= limit:
                    break

        if not recommended:
            return []

        folder = RecommendationFolder(
            item_id="music_insights_recommended",
            provider="library",
            name="Recommended for you",
            translation_key="recommended_tracks",
            icon="mdi-brain",
            items=UniqueList(recommended[:limit]),
        )
        return [folder]


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
