"""Sidecar-based embeddings handler for Music Assistant."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from music_assistant_models.unique_list import UniqueList

from .sidecar_client import (
    DeleteModelResult,
    DownloadModelResult,
    DownloadProgress,
    DownloadStatus,
    HealthStatus,
    LoadModelResult,
    ModelDetail,
    ModelStatus,
    SidecarClient,
    StorageStats,
    SystemStatus,
)

if TYPE_CHECKING:
    from music_assistant_models.media_items import Track

    from music_assistant.mass import MusicAssistant


# Re-export for convenience
__all__ = [
    "DeleteModelResult",
    "DownloadModelResult",
    "DownloadProgress",
    "DownloadStatus",
    "HealthStatus",
    "LoadModelResult",
    "ModelDetail",
    "ModelStatus",
    "SidecarEmbeddings",
    "StorageStats",
    "SystemStatus",
]


class SidecarEmbeddings:
    """
    Handles track embeddings using the Rust sidecar service.

    This replaces the ChromaDB + PyTorch implementation with HTTP calls
    to the insight sidecar, which handles ONNX inference and Qdrant storage.
    """

    def __init__(
        self,
        mass: MusicAssistant,
        logger: logging.Logger,
        sidecar_url: str = "http://localhost:8096",
    ) -> None:
        """
        Initialize SidecarEmbeddings.

        :param mass: The MusicAssistant instance.
        :param logger: Logger instance.
        :param sidecar_url: URL of the insight sidecar service.
        """
        self.mass = mass
        self.logger = logger.getChild("sidecar")
        self.client = SidecarClient(base_url=sidecar_url)
        self._connected = False

    async def async_init(self) -> None:
        """
        Asynchronously initialize the connection to the sidecar.

        Performs a health check to verify the sidecar is available.
        """
        try:
            health = await self.client.health_check()
            self._connected = True
            self.logger.info(
                "Connected to sidecar: model_loaded=%s, storage_ready=%s",
                health.get("model_loaded", False),
                health.get("storage_ready", False),
            )
        except Exception as e:
            self._connected = False
            self.logger.error("Failed to connect to sidecar: %s", e)
            raise

    @property
    def is_connected(self) -> bool:
        """Return whether the sidecar connection is established."""
        return self._connected

    async def upsert_track(self, track: Track) -> bool:
        """
        Generate and store text embeddings for a track.

        :param track: The Track object to process.
        :return: True if track was upserted, False if failed.
        """
        if not self._connected:
            self.logger.warning("Sidecar not connected, skipping upsert for %s", track.item_id)
            return False

        metadata = SidecarClient.track_to_metadata(track)
        # Run hash computation in thread pool to avoid blocking event loop
        loop = asyncio.get_running_loop()
        track_hash = await loop.run_in_executor(None, SidecarClient.compute_track_hash, track)

        try:
            result = await self.client.embed_text_and_store(
                track.item_id, metadata, metadata_hash=track_hash
            )
            self.logger.debug(
                "Upserted track %s: %s",
                track.item_id,
                result.get("text", "")[:50],
            )
            return True
        except Exception as e:
            self.logger.warning("Failed to upsert track %s: %s", track.item_id, e)
            return False

    async def upsert_tracks_batch(self, tracks: list[Track], batch_size: int = 50) -> int:
        """
        Batch upsert multiple tracks.

        :param tracks: List of tracks to upsert.
        :param batch_size: Number of tracks per batch request.
        :return: Number of successfully processed tracks.
        """
        if not self._connected:
            self.logger.warning("Sidecar not connected, skipping batch upsert")
            return 0

        success_count = 0

        for i in range(0, len(tracks), batch_size):
            batch = tracks[i : i + batch_size]
            batch_data = [
                (track.item_id, SidecarClient.track_to_metadata(track)) for track in batch
            ]

            try:
                result = await self.client.batch_embed_text_and_store(batch_data)
                success_count += result.get("succeeded", 0)
                if result.get("failed", 0) > 0:
                    self.logger.warning(
                        "Batch had %d failures out of %d",
                        result.get("failed", 0),
                        len(batch),
                    )
            except Exception as e:
                self.logger.error("Batch upsert failed: %s", e)

        return success_count

    async def remove_track(self, track_id: str) -> None:
        """
        Remove a track's embeddings from storage.

        :param track_id: The item_id of the track to remove.
        """
        if not self._connected:
            return

        try:
            await self.client.delete_track(track_id)
            self.logger.debug("Removed track %s", track_id)
        except Exception as e:
            self.logger.warning("Failed to remove track %s: %s", track_id, e)

    async def get_similar_tracks(
        self,
        prov_track_id: str,
        limit: int = 25,
        cutoff: float = 0.5,
    ) -> list[Track]:
        """
        Find tracks similar to a given track ID.

        :param prov_track_id: The provider-specific ID of the track.
        :param limit: The maximum number of similar tracks to return.
        :param cutoff: Minimum similarity score (0-1, higher is more similar).
        :return: A list of similar Track objects.
        """
        if not self._connected:
            return []

        try:
            results = await self.client.get_similar_tracks(prov_track_id, limit=limit)
        except Exception as e:
            self.logger.warning("Similar tracks search failed: %s", e)
            return []

        # Filter by cutoff (sidecar returns cosine similarity scores)
        filtered_results = [r for r in results if r.score >= cutoff]

        # Fetch full track objects from Music Assistant
        tracks: list[Track] = []
        for result in filtered_results[:limit]:
            try:
                # Extract base track ID (remove any collection suffix)
                base_id = result.track_id.split("#")[0]
                track = await self.mass.music.tracks.get(
                    base_id, provider_instance_id_or_domain="library"
                )
                if track:
                    tracks.append(track)
            except Exception as e:
                self.logger.debug("Could not fetch track %s: %s", result.track_id, e)

        return tracks

    async def search_tracks(
        self,
        search_query: str,
        limit: int = 50,
        cutoff: float = 0.4,
        filter_: dict[str, Any] | None = None,
    ) -> UniqueList[Track]:
        """
        Search for tracks based on a text query.

        :param search_query: The text query string.
        :param limit: The maximum number of results to return.
        :param cutoff: Minimum similarity score (0-1, higher is more similar).
        :param filter_: Optional filter dict with keys like:
            - moods: list[str] - include tracks with any of these moods
            - exclude_moods: list[str] - exclude tracks with these moods
            - min_valence/max_valence: float - valence range (-1 to 1)
            - min_arousal/max_arousal: float - arousal range (-1 to 1)
            - artists: list[str] - filter by artists
            - genres: list[str] - filter by genres
            - exclude_ids: list[str] - exclude specific track IDs
        :return: A UniqueList of matching Track objects.
        """
        if not self._connected:
            return UniqueList()

        try:
            results = await self.client.search(search_query, limit=limit, filter_=filter_)
        except Exception as e:
            self.logger.warning("Search failed: %s", e)
            return UniqueList()

        # Filter by cutoff
        filtered_results = [r for r in results if r.score >= cutoff]

        # Fetch full track objects
        tracks: UniqueList[Track] = UniqueList()
        for result in filtered_results[:limit]:
            try:
                base_id = result.track_id.split("#")[0]
                track = await self.mass.music.tracks.get(
                    base_id, provider_instance_id_or_domain="library"
                )
                if track:
                    tracks.append(track)
            except Exception as e:
                self.logger.debug("Could not fetch track %s: %s", result.track_id, e)

        return tracks

    async def cleanup(self) -> None:
        """Clean up resources and close connections."""
        await self.client.close()
        self._connected = False
        self.logger.info("Sidecar client closed")

    # ========================================================================
    # Management API Methods
    # ========================================================================

    async def get_status(self) -> SystemStatus:
        """
        Get comprehensive system status from the sidecar.

        :return: SystemStatus with version, health, model info, storage stats.
        """
        return await self.client.get_status()

    async def list_models(self) -> tuple[list[ModelDetail], str | None]:
        """
        List all available models (known + cached).

        :return: Tuple of (list of models, current model ID or None).
        """
        return await self.client.list_models()

    async def get_current_model(self) -> ModelDetail | None:
        """
        Get the currently loaded model details.

        :return: ModelDetail if a model is loaded, None otherwise.
        """
        models, current_id = await self.list_models()
        if current_id is None:
            return None
        for model in models:
            if model.model_id == current_id:
                return model
        return None

    async def start_model_download(self, model_id: str) -> DownloadModelResult:
        """
        Start downloading a model.

        :param model_id: HuggingFace model ID (e.g., "Xenova/clap-htsat-unfused").
        :return: DownloadModelResult with download_id if started.
        """
        return await self.client.start_download(model_id)

    async def get_download_progress(self, download_id: str) -> DownloadProgress | None:
        """
        Get progress for a specific download.

        :param download_id: The download ID returned from start_model_download.
        :return: DownloadProgress or None if not found.
        """
        return await self.client.get_download_progress(download_id)

    async def list_downloads(self) -> list[DownloadProgress]:
        """
        Get status of all downloads (active and recent).

        :return: List of download progress objects.
        """
        return await self.client.list_downloads()

    async def load_model(self, model_id: str) -> LoadModelResult:
        """
        Load a downloaded model (hot-swap).

        :param model_id: Model ID to load (must be downloaded first).
        :return: LoadModelResult with success status.
        """
        result = await self.client.load_model(model_id)
        if result.loaded:
            self.logger.info("Model loaded: %s (device: %s)", model_id, result.device)
        return result

    async def delete_cached_model(self, model_id: str) -> DeleteModelResult:
        """
        Delete a cached model.

        Cannot delete the currently loaded model.

        :param model_id: Model ID to delete.
        :return: DeleteModelResult with success status.
        """
        result = await self.client.delete_model(model_id)
        if result.deleted:
            self.logger.info("Model deleted: %s", model_id)
        return result

    async def get_storage_stats(self) -> StorageStats:
        """
        Get detailed storage statistics.

        :return: StorageStats with track counts and connection info.
        """
        return await self.client.get_storage_stats()

    async def ensure_model_loaded(self, model_id: str) -> bool:
        """
        Ensure a specific model is downloaded and loaded.

        Downloads if needed, then loads if not already current.

        :param model_id: Model ID to ensure is loaded.
        :return: True if model is loaded, False if failed.
        """
        try:
            models, current = await self.list_models()

            # Already loaded?
            if current == model_id:
                self.logger.debug("Model %s already loaded", model_id)
                return True

            # Check if downloaded
            model_info = None
            for m in models:
                if m.model_id == model_id:
                    model_info = m
                    break

            # Download if needed
            if model_info is None or model_info.status == ModelStatus.NOT_DOWNLOADED:
                self.logger.info("Downloading model %s...", model_id)
                download_result = await self.start_model_download(model_id)
                if download_result.download_id:
                    # Wait for download to complete
                    progress = await self.client.wait_for_download(
                        download_result.download_id,
                        poll_interval=2.0,
                        timeout=1800.0,  # 30 minutes
                    )
                    if progress.status != DownloadStatus.COMPLETED:
                        self.logger.error("Download failed: %s", progress.error)
                        return False

            # Load the model
            load_result = await self.load_model(model_id)
            return load_result.loaded

        except Exception as e:
            self.logger.error("Failed to ensure model %s is loaded: %s", model_id, e)
            return False

    # ========================================================================
    # Taste Profile API Methods
    # ========================================================================

    async def compute_user_profile(
        self,
        user_id: str,
        interactions: list[dict[str, Any]],
        cutoff_days: int = 21,
    ) -> dict[str, Any]:
        """
        Compute taste profile from user interactions.

        :param user_id: User ID to compute profile for.
        :param interactions: List of interaction dicts with track_id, timestamp, signal_type, etc.
        :param cutoff_days: Number of days of history to consider (default: 21).
        :return: Response dict with user_id and profile metadata.
        """
        if not self._connected:
            msg = "Sidecar not connected"
            raise RuntimeError(msg)

        return await self.client.compute_taste_profile(
            user_id=user_id,
            interactions=interactions,
            cutoff_days=cutoff_days,
        )

    async def get_user_recommendations(
        self,
        user_id: str,
        limit: int = 25,
        exclude_ids: list[str] | None = None,
        filter_: dict[str, Any] | None = None,
    ) -> list[Track]:
        """
        Get personalized recommendations based on user's taste profile.

        :param user_id: User ID to get recommendations for.
        :param limit: Maximum number of recommendations (default: 25).
        :param exclude_ids: Track IDs to exclude from results.
        :param filter_: Optional filter dict with keys like:
            - moods: list[str] - include tracks with any of these moods
            - exclude_moods: list[str] - exclude tracks with these moods
            - min_valence/max_valence: float - valence range (-1 to 1)
            - min_arousal/max_arousal: float - arousal range (-1 to 1)
            - artists: list[str] - filter by artists
            - genres: list[str] - filter by genres
        :return: List of recommended Track objects.
        """
        if not self._connected:
            return []

        try:
            data = await self.client.get_taste_recommendations(
                user_id=user_id,
                limit=limit,
                exclude_ids=exclude_ids,
                filter_=filter_,
            )
        except Exception as e:
            self.logger.warning("Failed to get recommendations for %s: %s", user_id, e)
            return []

        # Convert response to Track objects
        tracks: list[Track] = []
        for item in data.get("tracks", []):
            try:
                base_id = item["track_id"].split("#")[0]
                track = await self.mass.music.tracks.get(
                    base_id, provider_instance_id_or_domain="library"
                )
                if track:
                    tracks.append(track)
            except Exception as e:
                self.logger.debug("Could not fetch track %s: %s", item["track_id"], e)

        return tracks
