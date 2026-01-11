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

# Cache key prefix for storing user interactions (per-user)
INTERACTIONS_CACHE_KEY_PREFIX = "music_insights_interactions_"

# Cache key prefix for storing track favorite states (per-user)
FAVORITES_CACHE_KEY_PREFIX = "music_insights_favorites_"

# Default user ID when no user context is available
DEFAULT_USER_ID = "default"

# Skip threshold in seconds (industry standard from Spotify/Apple/Deezer)
SKIP_THRESHOLD_SECONDS = 30


def _get_cache_key(user_id: str | None) -> str:
    """Get the cache key for a specific user's interactions."""
    uid = user_id or DEFAULT_USER_ID
    return f"{INTERACTIONS_CACHE_KEY_PREFIX}{uid}"


def _get_favorites_cache_key(user_id: str | None) -> str:
    """Get the cache key for a specific user's favorite states."""
    uid = user_id or DEFAULT_USER_ID
    return f"{FAVORITES_CACHE_KEY_PREFIX}{uid}"


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

    async def record_interaction(
        self, event: MediaItemPlaybackProgressReport, user_id: str | None = None
    ) -> None:
        """
        Record a user interaction event (track playback progress).

        Stores metadata about the playback event in the MA cache,
        including timestamp, URI, play duration, and whether the track was fully played.

        :param event: The MediaItemPlaybackProgressReport event.
        :param user_id: User ID for multi-user setups (defaults to event.userid or "default").
        """
        if event.media_type != MediaType.TRACK:
            return

        if event.mbid is None:
            return

        # Use user_id from event if not provided explicitly
        uid = user_id or event.userid or DEFAULT_USER_ID
        cache_key = _get_cache_key(uid)

        # Get existing interactions from cache
        interactions: dict[str, dict[str, float | str | bool]] = await self.mass.cache.get(
            cache_key, default={}
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
        await self.mass.cache.set(cache_key, interactions)
        self.logger.debug("Recorded interaction for user %s: %s", uid, event.mbid)

    async def record_favorite_signal(
        self, track_id: str, is_favorite: bool, user_id: str | None = None
    ) -> None:
        """
        Record a favorite or unfavorite signal for a track.

        Favorite signals have weight 2.0 (strongest positive signal).
        Unfavorite is treated as a dislike signal with weight -1.0.

        :param track_id: The track ID (mbid or item_id).
        :param is_favorite: True if favorited, False if unfavorited.
        :param user_id: User ID for multi-user setups.
        """
        uid = user_id or DEFAULT_USER_ID
        cache_key = _get_cache_key(uid)

        interactions: dict[str, dict[str, float | str | bool]] = await self.mass.cache.get(
            cache_key, default={}
        )

        signal_type = "favorite" if is_favorite else "dislike"

        interaction: dict[str, float | str | bool] = {
            "timestamp": float(time.time()),
            "signal_type": signal_type,
            "is_favorite": is_favorite,
            "seconds_played": 0.0,
            "duration": 0.0,
            "score": 0.0,
        }

        # Merge with existing playback data if present
        if track_id in interactions:
            existing = interactions[track_id]
            interaction["seconds_played"] = existing.get("seconds_played", 0.0)
            interaction["duration"] = existing.get("duration", 0.0)
            interaction["score"] = existing.get("score", 0.0)

        interactions[track_id] = interaction

        await self.mass.cache.set(cache_key, interactions)
        self.logger.info(
            "Recorded %s signal for user %s: track=%s",
            signal_type,
            uid,
            track_id,
        )

    async def check_and_record_favorite_change(
        self, track_id: str, current_favorite: bool, user_id: str | None = None
    ) -> bool:
        """
        Check if favorite status changed and record signal if it did.

        Compares current favorite status with cached previous status.
        Records favorite/dislike signal only when status actually changes.

        :param track_id: The track ID.
        :param current_favorite: Current favorite status.
        :param user_id: User ID for multi-user setups.
        :return: True if a signal was recorded (status changed), False otherwise.
        """
        uid = user_id or DEFAULT_USER_ID
        favorites_cache_key = _get_favorites_cache_key(uid)

        # Get cached favorite states
        favorites: dict[str, bool] = await self.mass.cache.get(favorites_cache_key, default={})

        previous_favorite = favorites.get(track_id)

        # Only record if status actually changed
        if previous_favorite is not None and previous_favorite != current_favorite:
            await self.record_favorite_signal(track_id, current_favorite, uid)
            favorites[track_id] = current_favorite
            await self.mass.cache.set(favorites_cache_key, favorites)
            return True

        # Update cache with current status (for first-time or unchanged)
        if previous_favorite is None:
            favorites[track_id] = current_favorite
            await self.mass.cache.set(favorites_cache_key, favorites)

        return False

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
        # Use provided user_id or default
        uid = user_id or DEFAULT_USER_ID
        cache_key = _get_cache_key(uid)

        # Get interactions from cache for this user
        interactions: dict[str, dict[str, float | str | bool]] = await self.mass.cache.get(
            cache_key, default={}
        )

        if not interactions:
            self.logger.debug("No interactions found for user %s", uid)
            return []

        # Convert to sidecar format
        interaction_list: list[dict[str, str | int | float]] = []
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

        Uses industry-standard thresholds:
        - Skip: < 30 seconds played (Spotify/Apple/Deezer standard)
        - Partial play: >= 30s but < 50% of track
        - Full play: >= 50% or marked as fully_played
        - Repeat: score >= 2.0 (multiple plays)
        - Favorite/Dislike: explicit signals stored in signal_type

        :param meta: Interaction metadata dict.
        :return: Signal type string (full_play, partial_play, skip, repeat, favorite, dislike).
        """
        # Check for explicit signal types first (favorite/dislike)
        explicit_signal = meta.get("signal_type")
        if explicit_signal in ("favorite", "dislike"):
            return str(explicit_signal)

        fully_played = meta.get("fully_played", False)
        seconds_played = float(meta.get("seconds_played", 0))
        duration = float(meta.get("duration", 1))
        score = float(meta.get("score", 0))

        # Multiple plays = repeat (strong positive signal)
        if score >= 2.0:
            return "repeat"

        # Explicit full play flag
        if fully_played:
            return "full_play"

        # 30-second skip threshold (industry standard)
        if seconds_played < SKIP_THRESHOLD_SECONDS:
            return "skip"

        # Partial play if >= 30s but < 50%
        if duration > 0 and seconds_played / duration >= 0.5:
            return "full_play"

        return "partial_play"


class InsightScrobbler(ScrobblerHelper):
    """Handles playback event handling for recommendations.

    Records user interactions from playback events to build per-user taste profiles.
    Uses report.userid to track interactions separately for each user.
    """

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

        Records the interaction for the user specified in report.userid.

        :param report: The playback progress report (includes userid).
        """
        await self.rec.record_interaction(report)

    async def _scrobble(self, report: MediaItemPlaybackProgressReport) -> None:
        """
        Handle the 'scrobble' event (track finished or played significantly).

        Records the interaction for the user specified in report.userid.

        :param report: The playback progress report (includes userid).
        """
        await self.rec.record_interaction(report)
