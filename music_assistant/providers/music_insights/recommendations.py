"""Recommendation engine for Music Assistant based on user interactions."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING

from music_assistant_models.enums import EventType, PlayerState

from music_assistant.mass import LOGGER

if TYPE_CHECKING:
    from music_assistant_models.event import MassEvent
    from music_assistant_models.media_items import RecommendationFolder

    from music_assistant.mass import MusicAssistant

    from .chroma_embeddings import ChromaEmbeddings


# Simple score mapping for POC - focus on explicit actions for now
# We can refine this later based on inferred actions (skip, complete)
EVENT_SCORE_MAP = {
    "track_start": 1,  # User initiated playback
    "track_stop": -1,  # User stopped playback
    "track_pause": 0,  # Pause is neutral for now
    # Add scores for inferred actions later if needed
    # "track_complete": 5,
    # "track_skip": -3,
}


class RecommendationEngine:
    """Handles recommendation logic based on user interactions."""

    def __init__(self, mass: MusicAssistant, chroma: ChromaEmbeddings):
        """Initialize RecommendationEngine."""
        self.mass = mass
        self.logger = LOGGER.getChild("music_insights.recommendations")
        self.chroma = chroma
        # Store minimal state: current track_id per player
        self._player_current_track: dict[str, str | None] = {}

    async def handle_event(self, event: MassEvent) -> None:
        """Handle incoming player/queue events for interaction tracking."""
        player_id = event.object_id
        if player_id is None:
            return
        player = self.mass.players.get(str(player_id))
        if not player:
            return

        track_id: str | None = None
        event_type_str: str | None = None
        score: int | None = None

        # --- Event Handling Logic ---

        # if event.event == EventType.QUEUE_ITEM_PLAYING:
        #     queue_item: QueueItem = event.data
        #     if isinstance(queue_item.media_item, Track):
        #         track: Track = queue_item.media_item
        #         track_id = track.item_id
        #         self._player_current_track[player_id] = track_id
        #         event_type_str = "track_start"
        #         score = EVENT_SCORE_MAP.get(event_type_str)
        #         self.logger.debug(
        #             "Player %s started track %s (%s)", player_id, track_id, track.name
        #         )

        if event.event == EventType.PLAYER_UPDATED:
            # Infer stop/pause based on state change while a track is loaded
            current_track_id = self._player_current_track.get(player_id)
            if current_track_id:
                if player.state == PlayerState.IDLE and player.powered:
                    # Considered a 'stop' if player goes idle while powered and track was playing
                    track_id = current_track_id
                    event_type_str = "track_stop"
                    score = EVENT_SCORE_MAP.get(event_type_str)
                    self._player_current_track[player_id] = None  # Clear current track on stop
                    self.logger.debug("Player %s stopped track %s", player_id, track_id)
                elif player.state == PlayerState.PAUSED:
                    track_id = current_track_id
                    event_type_str = "track_pause"
                    score = EVENT_SCORE_MAP.get(event_type_str)
                    self.logger.debug("Player %s paused track %s", player_id, track_id)
                elif player.state == PlayerState.PLAYING and event_type_str == "track_pause":
                    # Player resumed after pause - could be logged but score is 0
                    pass

        # --- Record Interaction ---
        if track_id and event_type_str and score is not None:
            await self._record_interaction(player_id, track_id, event_type_str, score)

    async def _record_interaction(
        self, player_id: str, track_id: str, event_type: str, score: int
    ) -> None:
        """Record a user interaction event in ChromaDB."""
        interaction_id = f"{player_id}_{track_id}_{event_type}_{time.time()}"
        # Explicitly type metadata for ChromaDB compatibility
        metadata: Mapping[str, str | int | float | bool] = {
            "player_id": str(player_id),
            "track_id": str(track_id),
            "event_type": str(event_type),
            "score": int(score),
            "timestamp": float(time.time()),
        }  # Ensure all values are explicitly cast to the expected types
        try:
            # Use asyncio.to_thread for the blocking chromadb call
            await asyncio.to_thread(
                self.chroma.user_collection.add,
                ids=[interaction_id],
                metadatas=[metadata],
                # No embeddings needed for interactions themselves yet
            )
            self.logger.debug("Recorded interaction to ChromaDB: %s", metadata)
        except Exception as e:
            self.logger.error(
                "Failed to record interaction to ChromaDB: %s - %s", metadata, e, exc_info=e
            )

    # Placeholder for recommendation generation
    async def get_recommendations(self, limit: int = 25) -> list[RecommendationFolder]:
        """Generate recommendations based on stored interactions."""
        # TODO: Implement actual recommendation logic using interaction scores
        # and ChromaDB similarity searches.
        self.logger.warning("Recommendation generation not yet implemented.")
        return []
