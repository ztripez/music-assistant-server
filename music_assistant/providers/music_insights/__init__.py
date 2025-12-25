"""Provider for Music Insights based on embeddings and recommendations."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from music_assistant_models.config_entries import (
    ConfigEntry,
    ConfigValueType,
    ProviderConfig,
)
from music_assistant_models.enums import ConfigEntryType, EventType, MediaType, ProviderFeature
from music_assistant_models.media_items import (
    RecommendationFolder,
    SearchResults,
    Track,
)
from music_assistant_models.unique_list import UniqueList

from music_assistant.models.music_provider import MusicProvider

from .audio_streamer import AudioStreamer
from .recommendations import InsightScrobbler, RecommendationEngine
from .sidecar_embeddings import SidecarEmbeddings

if TYPE_CHECKING:
    from music_assistant_models.event import MassEvent
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return MusicInsightProvider(mass, manifest, config)


async def get_config_entries(
    mass: MusicAssistant,  # noqa: ARG001
    instance_id: str | None = None,  # noqa: ARG001
    action: str | None = None,  # noqa: ARG001
    values: dict[str, ConfigValueType] | None = None,  # noqa: ARG001
) -> tuple[ConfigEntry, ...]:
    """Return Config entries for provider setup."""
    return (
        ConfigEntry(
            key="sidecar_url",
            type=ConfigEntryType.STRING,
            label="Insight Sidecar URL",
            default_value="http://localhost:8096",
            description=(
                "URL of the Music Assistant Insight Sidecar service. "
                "The sidecar handles ML inference and vector storage."
            ),
            required=True,
        ),
        ConfigEntry(
            key="model_id",
            type=ConfigEntryType.STRING,
            label="CLAP Model",
            default_value="Xenova/clap-htsat-unfused",
            description=(
                "HuggingFace model ID for the CLAP model to use for embeddings. "
                "Recommended models: Xenova/clap-htsat-unfused (default, general purpose), "
                "laion/larger_clap_music (music-optimized). "
                "The model will be downloaded if not already cached."
            ),
            required=True,
        ),
        ConfigEntry(
            key="cut_off_taste",
            type=ConfigEntryType.INTEGER,
            range=(1, 356),
            label="Days of interactions to include in taste profile",
            default_value=21,
            description=(
                "How many days of interactions (plays, stops, favourites) "
                "to include when building recommendations. "
                "21 days (three weeks) is usually a sweet-spot; "
                "higher values risk over-fitting and repetition."
            ),
            required=True,
        ),
        ConfigEntry(
            key="enable_audio_streaming",
            type=ConfigEntryType.BOOLEAN,
            label="Enable audio streaming",
            default_value=True,
            description=(
                "Stream audio to the sidecar during playback to generate audio embeddings. "
                "This enables audio-based similarity search (find tracks that sound similar). "
                "Disable if you only want text-based embeddings from metadata."
            ),
            required=False,
        ),
        ConfigEntry(
            key="rebuild_on_start",
            type=ConfigEntryType.BOOLEAN,
            label="Rebuild embeddings on startup",
            default_value=False,
            description=(
                "If enabled, all track embeddings will be rebuilt when the provider starts. "
                "Use this after changing sidecar configuration or to fix sync issues."
            ),
            required=False,
            category="advanced",
        ),
    )


class MusicInsightProvider(MusicProvider):
    """
    Provider for Music Insights based on embeddings and recommendations.

    This provider uses the Insight Sidecar service for ML inference and
    Qdrant for vector storage. It provides features like:
    - Semantic search for tracks based on text queries.
    - Finding similar tracks based on embeddings.
    - Generating recommendations based on user listening history.
    """

    _on_unload: list[Callable[[], None]] = []

    def __init__(
        self, mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
    ) -> None:
        """Initialize MusicInsightProvider."""
        super().__init__(mass, manifest, config)

    async def handle_async_init(self) -> None:
        """
        Handle asynchronous initialization of the provider.

        Sets up the sidecar connection, ensures the configured model is loaded,
        sets up the recommendation engine, and subscribes to relevant events.
        """
        sidecar_url = cast("str", self.config.get_value("sidecar_url") or "http://localhost:8096")
        model_id = cast("str", self.config.get_value("model_id") or "Xenova/clap-htsat-unfused")
        enable_audio_streaming = cast("bool", self.config.get_value("enable_audio_streaming") or True)
        rebuild_on_start = cast("bool", self.config.get_value("rebuild_on_start") or False)

        self.embeddings = SidecarEmbeddings(
            self.mass,
            self.logger,
            sidecar_url=sidecar_url,
        )

        try:
            await self.embeddings.async_init()
        except Exception as e:
            self.logger.error(
                "Failed to connect to insight sidecar at %s: %s. Make sure the sidecar is running.",
                sidecar_url,
                e,
            )
            raise

        # Ensure the configured model is loaded
        try:
            status = await self.embeddings.get_status()
            current_model = status.model
            current_model_id = current_model.model_id if current_model else None

            if current_model_id != model_id:
                self.logger.info(
                    "Configured model '%s' differs from loaded model '%s'. Switching...",
                    model_id,
                    current_model_id or "none",
                )
                if await self.embeddings.ensure_model_loaded(model_id):
                    self.logger.info("Model '%s' loaded successfully.", model_id)
                else:
                    self.logger.warning(
                        "Failed to load configured model '%s'. Using current model '%s' instead.",
                        model_id,
                        current_model_id,
                    )
            else:
                self.logger.info("Configured model '%s' is already loaded.", model_id)
        except Exception as e:
            self.logger.warning(
                "Could not verify/load model '%s': %s. Continuing with current model.",
                model_id,
                e,
            )

        self.recommendation_engine = RecommendationEngine(self.mass, self.embeddings)

        # Initialize audio streamer for real-time audio embeddings
        self.audio_streamer: AudioStreamer | None = None
        if enable_audio_streaming:
            self.audio_streamer = AudioStreamer(
                self.mass,
                sidecar_url=sidecar_url,
                logger=self.logger,
            )
            try:
                await self.audio_streamer.start()
                self.logger.info("Audio streaming enabled for real-time audio embeddings")
            except Exception as e:
                self.logger.warning(
                    "Failed to start audio streamer: %s. Audio embeddings will not be generated.",
                    e,
                )
                self.audio_streamer = None

        # Subscribe to library events
        self._on_unload.append(
            self.mass.subscribe(
                self.handle_library_update,
                event_filter=(
                    EventType.MEDIA_ITEM_UPDATED,
                    EventType.MEDIA_ITEM_ADDED,
                    EventType.MEDIA_ITEM_DELETED,
                ),
            )
        )

        # Subscribe to playback events for recommendations
        handler = InsightScrobbler(self.logger, self.recommendation_engine)
        self._on_unload.append(
            self.mass.subscribe(handler._on_mass_media_item_played, EventType.MEDIA_ITEM_PLAYED)
        )

        self.logger.info("Subscribed to library and player events.")

        # Optionally rebuild embeddings on start
        if rebuild_on_start:
            self.logger.info("Scheduling full embedding rebuild (rebuild_on_start=True)")
            self.mass.create_task(self._rebuild_embeddings())

    async def handle_library_update(self, event: MassEvent) -> None:
        """
        Handle library update events (add, update, delete) for tracks.

        Upserts or removes track embeddings via the sidecar.

        :param event: The MassEvent containing track data and event type.
        """
        if not isinstance(event.data, Track):
            return

        track: Track = event.data

        if event.event in (EventType.MEDIA_ITEM_ADDED, EventType.MEDIA_ITEM_UPDATED):
            await self.embeddings.upsert_track(track)
        elif event.event == EventType.MEDIA_ITEM_DELETED:
            await self.embeddings.remove_track(track.item_id)

    @property
    def supported_features(self) -> set[ProviderFeature]:
        """Return the features supported by this Provider."""
        return {
            ProviderFeature.SEARCH,
            ProviderFeature.RECOMMENDATIONS,
            ProviderFeature.SIMILAR_TRACKS,
        }

    async def unload(self, is_removed: bool = False) -> None:
        """
        Handle unload/close of the provider.

        Called when provider is deregistered (e.g. MA exiting or config reloading).
        is_removed will be set to True when the provider is removed from the configuration.
        """
        for unload_cb in self._on_unload:
            unload_cb()

        # Stop audio streamer
        if self.audio_streamer:
            await self.audio_streamer.stop()

        await self.embeddings.cleanup()

        if is_removed:
            self.logger.info("Provider removed.")

    async def search(
        self, search_query: str, media_types: list[MediaType], limit: int = 5
    ) -> SearchResults:
        """
        Perform a search for tracks based on a text query using embeddings.

        :param search_query: The text query to search for.
        :param media_types: A list of media types to include (only TRACK is supported).
        :param limit: The maximum number of results to return.
        :return: SearchResults containing the found tracks.
        """
        tracks: UniqueList[Track] = UniqueList()
        if MediaType.TRACK in media_types:
            tracks = await self.embeddings.search_tracks(search_query, limit=limit)
        return SearchResults(tracks=tracks)

    async def get_similar_tracks(self, prov_track_id: str, limit: int = 25) -> list[Track]:
        """
        Get tracks similar to a given track ID using embeddings.

        :param prov_track_id: The provider-specific ID of the track.
        :param limit: The maximum number of similar tracks to return.
        :return: A list of similar Track objects.
        """
        return await self.embeddings.get_similar_tracks(prov_track_id, limit=limit)

    async def recommendations(self) -> list[RecommendationFolder]:
        """
        Get this provider's recommendations.

        Returns an actual (and often personalised) list of recommendations.
        """
        return await self.recommendation_engine.get_recommendations()

    async def _rebuild_embeddings(self) -> None:
        """
        Perform a full rebuild of all track embeddings in the library.

        Uses batch operations for efficiency.
        """
        self.logger.info("Starting full embedding rebuild...")
        count = 0

        try:
            tracks = await self.mass.music.tracks.library_items()
            track_list = list(tracks)
            total = len(track_list)

            self.logger.info("Found %d tracks to embed", total)

            # Process in batches
            batch_size = 50
            for i in range(0, total, batch_size):
                batch = track_list[i : i + batch_size]
                succeeded = await self.embeddings.upsert_tracks_batch(batch)
                count += succeeded

                if (i + batch_size) % 200 == 0 or i + batch_size >= total:
                    self.logger.info(
                        "Embedding rebuild progress: %d/%d tracks processed",
                        min(i + batch_size, total),
                        total,
                    )

            self.logger.info("Completed full embedding rebuild. %d tracks processed.", count)

        except Exception as e:
            self.logger.error("Failed during embedding rebuild: %s", e, exc_info=e)
