"""Provider for Music Insights based on embeddings and recommendations."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import aiofiles
import torch
from music_assistant_models.config_entries import (
    ConfigEntry,
    ConfigValueOption,
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

from music_assistant.helpers.scrobbler import ScrobblerHelper
from music_assistant.models.music_provider import MusicProvider

# Import the new ChromaEmbeddings and RecommendationEngine classes
from .chroma_embeddings import ChromaEmbeddings
from .recommendations import RecommendationEngine

if TYPE_CHECKING:
    from music_assistant_models.event import MassEvent
    from music_assistant_models.playback_progress_report import MediaItemPlaybackProgressReport
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return MusicInsightProvider(mass, manifest, config)


DEFAULT_PRESET = "everyday_laptop"


async def _load_preset() -> Any:
    """Load presets asynchronously from the JSON file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    presets_path = os.path.join(current_dir, "_config_presets.json")
    async with aiofiles.open(presets_path, encoding="utf-8") as f:
        content = await f.read()
        return json.loads(content)


async def get_config_entries(
    mass: MusicAssistant,  # noqa: ARG001
    instance_id: str | None = None,  # noqa: ARG001
    action: str | None = None,
    values: dict[str, ConfigValueType] | None = None,
) -> tuple[ConfigEntry, ...]:
    """Return Config entries for provider setup, with optional preset autofill."""
    presets = await _load_preset()
    preset_defaults = presets[DEFAULT_PRESET]

    if action == "select_preset" and values and (preset_key := values.get("preset")):
        preset_defaults = presets[preset_key]

    entries = [
        ConfigEntry(
            key="preset",
            type=ConfigEntryType.STRING,
            label="Preset",
            default_value=DEFAULT_PRESET,
            description=(
                "Choose a ready-made hardware preset. All presents can be run on most "
                "machines but the inference time will be heavliy affected."
            ),
            options=[
                ConfigValueOption(title=f"{v['title']} - {v['description']}", value=k)
                for k, v in presets.items()
            ],
            required=True,
            action="select_preset",
        ),
        ConfigEntry(
            key="model_name",
            category="advanced",
            type=ConfigEntryType.STRING,
            label="CLAP embedding model",
            default_value=preset_defaults.get("model_name", "laion/clap-htsat-fused"),
            description=(
                "The CLAP embedding model to use. **Changing this rebuilds all vectors and retrains any taste profiles.**\n"
                "‣ `laion/clap-htsat-fused` (≈155 M) - good middle-ground; ~3 GB VRAM or CPU-only ok."
                "‣ `hf-internal-testing/tiny-clap-htsat-unfused` (30 M) - recommended for Raspberry Pi and laptops without CUDA.\n"
                "‣ `laion/larger_clap_music` (≈200 M) - music-tuned, needs a beefier GPU or fast CPU."
            ),
            required=True,
        ),
        ConfigEntry(
            key="window_size",
            type=ConfigEntryType.INTEGER,
            range=(1, 20),
            label="Audio window length (seconds) (Place holder)",
            default_value=preset_defaults.get("window_size", 10),
            description="Sliding-window length when embedding audio. Most models want 10 s;",
            required=True,
            category="advanced",
        ),
        ConfigEntry(
            key="enable_audio_features",
            type=ConfigEntryType.BOOLEAN,
            label="Enable audio-feature training (Place holder)",
            default_value=False,
            description=(
                "Produces the most accurate recommendations and similar-track matching, "
                "but **currently works only with local providers**. "
                "Training can take considerable time; a CUDA-enabled device is strongly recommended. "
                "(Note: full audio-feature training is a placeholder and not yet implemented.)"
            ),
            required=False,
        ),
        ConfigEntry(
            key="cut_off_taste",
            type=ConfigEntryType.INTEGER,
            range=(1, 356),
            label="Days of interactions to include in taste profile",
            default_value=21,
            description=(
                "How many days of interactions (plays, stops, favourites) "
                "to include when training the taste profile. "
                "21 days (three weeks) is usually a sweet-spot;"
                "higher values risk over-fitting and repetition."
            ),
            required=True,
        ),
    ]

    if torch.cuda.is_available():
        entries.append(
            ConfigEntry(
                key="enable_cuda",
                type=ConfigEntryType.BOOLEAN,
                label="Enable GPU",
                default_value=True,
                description="Enable CUDA acceleration",
                required=False,
            )
        )

    return tuple(entries)


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

    _on_unload: list[Callable[[], None]]

    def __init__(
        self, mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
    ) -> None:
        """Initialize MusicInsightProvider."""
        super().__init__(mass, manifest, config)

    async def handle_async_init(self) -> None:
        """Async init of the provider."""
        current_model_name = cast("str", self.config.get_value("model_name") or "")
        current_window_size = cast("int", self.config.get_value("window_size") or 0)
        enable_cuda = cast("bool", self.config.get_value("enable_cuda") or False)

        self.chroma_embeddings = ChromaEmbeddings(
            self.mass,
            model_name=current_model_name,
            audio_window_s=current_window_size,
            enable_cuda=enable_cuda,
        )
        self.recommendation_engine = RecommendationEngine(self.mass, self.chroma_embeddings)
        self._library_update_listener: Callable[[], None] | None = None
        self._previous_config_values: dict[str, Any] = {}
        config_changed = False
        if str(await self.mass.cache.get("model_name")) != current_model_name:
            config_changed = True
            await self.mass.cache.set("model_name", current_model_name)
        if int(await self.mass.cache.get("window_size", default=10)) != current_window_size:
            config_changed = True
            await self.mass.cache.set("window_size", current_window_size)

        if config_changed:
            self.logger.info("Scheduling full embedding rebuild due to configuration change.")
            self.mass.create_task(self._rebuild_embeddings())

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
        handler = InsightScrobbler(self.logger, self.recommendation_engine)
        self._on_unload.append(
            self.mass.subscribe(handler._on_mass_media_item_played, EventType.MEDIA_ITEM_PLAYED)
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

        if is_removed:
            self.logger.info("Provider removed, cleaning up embeddings and stored config state.")
            await self.mass.cache.delete("model_name")
            await self.mass.cache.delete("window_size")
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
        return await self.recommendation_engine.get_recommendations()

    async def _rebuild_embeddings(self) -> None:
        """Perform a full rebuild of track embeddings."""
        self.logger.info("Starting full embedding rebuild...")
        count = 0
        try:
            self.logger.info("Cleaning up existing embeddings...")
            await self.chroma_embeddings.cleanup()

            self.logger.info("Starting re-embedding process...")
            for track in await self.mass.music.tracks.library_items():
                try:
                    await self.chroma_embeddings.upsert_track(track)
                    count += 1
                    if count % 100 == 0:
                        self.logger.info("Embedding rebuild progress: %d tracks processed.", count)
                except Exception as e:
                    self.logger.warning(
                        "Error embedding track %s (%s): %s",
                        track.item_id,
                        track.name,
                        str(e),
                        exc_info=e,
                    )

            self.logger.info("Completed full embedding rebuild. %d tracks processed.", count)
        except Exception as e:
            self.logger.error("Failed during embedding rebuild process: %s", str(e), exc_info=e)


class InsightScrobbler(ScrobblerHelper):
    """Handles the event handling."""

    def __init__(self, logger: logging.Logger, rec: RecommendationEngine) -> None:
        """Initialize."""
        super().__init__(logger)

    async def _update_now_playing(self, report: MediaItemPlaybackProgressReport) -> None:
        pass

    async def _scrobble(self, report: MediaItemPlaybackProgressReport) -> None:
        pass
