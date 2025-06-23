"""Recommendation engine for Music Assistant based on user interactions."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping
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
        except Exception:
            # fall back to update in case the item already exists
            try:
                await asyncio.to_thread(
                    self.chroma.user_collection.update,
                    ids=[event.mbid],
                    metadatas=[metadata],
                )
            except Exception as e:
                self.logger.error(
                    "Failed to record interaction to ChromaDB: %s - %s", metadata, e, exc_info=e
                )
                return
        self.logger.debug("Recorded interaction to ChromaDB: %s", metadata)

    async def get_recommendations(self, limit: int = 25) -> list[RecommendationFolder]:
        """
        Generate recommendations based on stored user interactions.

        This reads the stored interactions from ChromaDB, determines the
        most relevant tracks and returns a folder with similar tracks
        based on the user's taste profile.

        Args:
            limit: The maximum number of recommendation folders/items to return.

        Returns:
            A list of RecommendationFolder objects with recommended tracks.
        """
        try:
            data = await asyncio.to_thread(
                self.chroma.user_collection.get, include=["ids", "metadatas"]
            )
        except Exception as err:
            self.logger.warning("Failed to fetch user interactions: %s", err)
            return []

        ids = data.get("ids", [])
        metas = data.get("metadatas", [])
        interactions = [m for m in metas if isinstance(m, Mapping)]
        if not ids or not interactions:
            return []

        scored: dict[str, float] = {}
        id_to_uri: dict[str, str] = {}
        now = time.time()
        for id_, meta in zip(ids, interactions, strict=False):
            uri = str(meta.get("uri", ""))
            id_to_uri[id_] = uri
            base = float(meta.get("score", 0))
            if meta.get("fully_played"):
                base += 2
            duration = float(meta.get("duration", 0))
            if duration:
                fraction = float(meta.get("seconds_played", 0)) / duration
                base += fraction * 2
            timestamp = float(meta.get("timestamp", now))
            age_days = max(0.0, now - timestamp) / 86_400
            decay = 0.9**age_days
            score = base * decay
            scored[id_] = scored.get(id_, 0) + score

        # sort by score, pick most relevant interactions
        sorted_ids = [id_ for id_, _ in sorted(scored.items(), key=lambda x: x[1], reverse=True)]

        recommended: UniqueList[Track] = UniqueList()
        for track_id in sorted_ids:
            if len(recommended) >= limit:
                break
            uri = id_to_uri.get(track_id) or ""
            if not uri:
                continue
            try:
                track = await self.mass.music.get_item_by_uri(uri)
            except Exception as err:  # item might be missing
                self.logger.debug("Could not resolve track for %s: %s", uri, err)
                continue
            if not track:
                continue
            similar = await self.chroma.get_similar_tracks(track.item_id, limit=5)
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
