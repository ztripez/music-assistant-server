"""Provider for Music Insights based on embeddings and recommendations."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from music_assistant_models.enums import EventType, MediaType, ProviderFeature
from music_assistant_models.media_items import (
    RecommendationFolder,
    SearchResults,
    Track,
)
from music_assistant_models.unique_list import UniqueList

from music_assistant.models.music_provider import MusicProvider

# Import the new ChromaEmbeddings and RecommendationEngine classes
from .chroma_embeddings import ChromaEmbeddings
from .recommendations import RecommendationEngine

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ConfigEntry, ProviderConfig
    from music_assistant_models.event import MassEvent
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return MusicInsightProvider(mass, manifest, config)


async def get_config_entries() -> tuple[ConfigEntry, ...]:
    """
    Return Config entries to setup this provider.

    instance_id: id of an existing provider instance (None if new instance setup).
    action: [optional] action key called from config entries UI.
    values: the (intermediate) raw values for config entries sent with the action.
    """
    # NOTE: Removed unused arguments: mass, instance_id, action, values
    # If they become necessary later for configuration logic, they should be added back.
    return ()


class MusicInsightProvider(MusicProvider):
    """
    Example/demo Music provider.

    Note that this is always subclassed from MusicProvider,
    which in turn is a subclass of the generic Provider model.

    The base implementation already takes care of some convenience methods,
    such as the mass object and the logger. Take a look at the base class
    for more information on what is available.

    Just like with any other subclass, make sure that if you override
    any of the default methods (such as __init__), you call the super() method.
    In most cases its not needed to override any of the builtin methods and you only
    implement the abc methods with your actual implementation.
    """

    def __init__(
        self, mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
    ) -> None:
        """Initialize MusicInsightProvider."""
        super().__init__(mass, manifest, config)
        # Instantiate the ChromaEmbeddings handler
        self.chroma_embeddings = ChromaEmbeddings(mass)
        # Instantiate the RecommendationEngine
        self.recommendation_engine = RecommendationEngine(mass, self.chroma_embeddings)
        self._library_update_listener: Callable[[], None] | None = None
        self._player_event_listener: Callable[[], None] | None = None

    async def loaded_in_mass(self) -> None:
        """Subscribe to relevant events after MA is loaded."""
        # Subscribe to library update events for embeddings
        self._library_update_listener = self.mass.subscribe(
            self.handle_library_update,
            event_filter=(
                EventType.MEDIA_ITEM_UPDATED,
                EventType.MEDIA_ITEM_ADDED,
                EventType.MEDIA_ITEM_DELETED,
            ),
        )
        self.logger.info("Subscribed to library updates for embedding generation.")

        # Subscribe to player/queue events for recommendations
        self._player_event_listener = self.mass.subscribe(
            self.recommendation_engine.handle_event,
            event_filter=(
                EventType.PLAYER_UPDATED,
                # EventType.QUEUE_ITEM_PLAYING,
                # Add other relevant events like QUEUE_ENDED if needed later
            ),
        )
        self.logger.info("Subscribed to player events for recommendations.")

    async def handle_library_update(self, event: MassEvent) -> None:
        """Handle library updates to keep embeddings current."""
        if not isinstance(event.data, Track):
            return
        track: Track = event.data
        if event.event in (EventType.MEDIA_ITEM_ADDED, EventType.MEDIA_ITEM_UPDATED):
            await self.chroma_embeddings.upsert_track(track)
        elif event.event == EventType.MEDIA_ITEM_DELETED:
            await self.chroma_embeddings.remove_track(track.item_id)

    @property
    def supported_features(self) -> set[ProviderFeature]:
        """Return the features supported by this Provider."""
        # MANDATORY
        # you should return a tuple of provider-level features
        # here that your player provider supports or an empty tuple if none.
        # for example 'ProviderFeature.SYNC_PLAYERS' if you can sync players.
        return {
            ProviderFeature.SEARCH,
            ProviderFeature.RECOMMENDATIONS,
            ProviderFeature.SIMILAR_TRACKS,
            # see the ProviderFeature enum for all available features
        }

    async def unload(self, is_removed: bool = False) -> None:
        """
        Handle unload/close of the provider.

        Called when provider is deregistered (e.g. MA exiting or config reloading).
        is_removed will be set to True when the provider is removed from the configuration.
        """
        # unsubscribe from events
        if callable(self._library_update_listener):
            self._library_update_listener()
            self._library_update_listener = None
            self.logger.info("Unsubscribed from library updates.")
        # unsubscribe from player events
        if callable(self._player_event_listener):
            self._player_event_listener()
            self._player_event_listener = None
            self.logger.info("Unsubscribed from player events.")

        # delete chromadb collections if provider is removed
        if is_removed:
            await self.chroma_embeddings.cleanup()

    async def search(
        self, search_query: str, media_types: list[MediaType], limit: int = 5
    ) -> SearchResults:
        """Search for tracks using text embeddings."""
        tracks: UniqueList[Track] = UniqueList()
        if MediaType.TRACK in media_types:
            tracks = await self.chroma_embeddings.search_tracks(search_query, limit=limit)
        return SearchResults(tracks=tracks)

    async def get_similar_tracks(self, prov_track_id: str, limit: int = 25) -> list[Track]:
        """Get tracks similar to the given track ID using embeddings."""
        return await self.chroma_embeddings.get_similar_tracks(prov_track_id, limit=limit)

    async def recommendations(self) -> list[RecommendationFolder]:
        """
        Get this provider's recommendations.

        Returns an actual (and often personalised) list of recommendations
        from this provider for the user/account.
        """
        # Get this provider's recommendations.
        # This is only called if you reported the RECOMMENDATIONS feature in the supported_features.
        # Delegate to the recommendation engine
        return await self.recommendation_engine.get_recommendations()
