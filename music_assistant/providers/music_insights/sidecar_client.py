"""HTTP client for the Music Assistant Insight Sidecar."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import aiohttp
import msgpack

if TYPE_CHECKING:
    from music_assistant_models.media_items import Track

LOGGER = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================


class HealthStatus(Enum):
    """Health status of the sidecar."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ModelStatus(Enum):
    """Status of a model."""

    NOT_DOWNLOADED = "not_downloaded"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    LOADED = "loaded"
    FAILED = "failed"


class DownloadStatus(Enum):
    """Status of a download."""

    PENDING = "pending"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ============================================================================
# Core Data Classes
# ============================================================================


@dataclass
class SearchResult:
    """A search result from the sidecar."""

    track_id: str
    score: float
    metadata: dict[str, Any]


@dataclass
class TrackMetadata:
    """Metadata for a track to be embedded."""

    name: str
    artists: list[str]
    album: str | None = None
    genres: list[str] | None = None


# ============================================================================
# Management API Data Classes
# ============================================================================


@dataclass
class StorageStats:
    """Storage statistics from the sidecar."""

    mode: str
    connected: bool
    text_collection_count: int
    audio_collection_count: int
    total_tracks: int


@dataclass
class ModelDetail:
    """Detailed information about a model."""

    model_id: str
    name: str
    status: ModelStatus
    description: str | None = None
    estimated_size_bytes: int | None = None
    actual_size_bytes: int | None = None
    cache_path: str | None = None
    recommended: bool = False
    is_current: bool = False
    device: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelDetail:
        """Create from API response dict."""
        status_str = data.get("status", "not_downloaded")
        status = ModelStatus(status_str)
        return cls(
            model_id=data["model_id"],
            name=data["name"],
            status=status,
            description=data.get("description"),
            estimated_size_bytes=data.get("estimated_size_bytes"),
            actual_size_bytes=data.get("actual_size_bytes"),
            cache_path=data.get("cache_path"),
            recommended=data.get("recommended", False),
            is_current=data.get("is_current", False),
            device=data.get("device"),
        )


@dataclass
class DownloadProgress:
    """Progress information for a model download."""

    download_id: str
    model_id: str
    status: DownloadStatus
    bytes_downloaded: int
    progress_percent: float
    started_at: int
    bytes_total: int | None = None
    completed_at: int | None = None
    error: str | None = None
    current_file: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DownloadProgress:
        """Create from API response dict."""
        status_str = data.get("status", "pending")
        status = DownloadStatus(status_str)
        return cls(
            download_id=data["download_id"],
            model_id=data["model_id"],
            status=status,
            bytes_downloaded=data.get("bytes_downloaded", 0),
            progress_percent=data.get("progress_percent", 0.0),
            started_at=data.get("started_at", 0),
            bytes_total=data.get("bytes_total"),
            completed_at=data.get("completed_at"),
            error=data.get("error"),
            current_file=data.get("current_file"),
        )


@dataclass
class SystemStatus:
    """Comprehensive system status from the sidecar."""

    version: str
    health: HealthStatus
    uptime_seconds: int
    storage: StorageStats
    features: list[str] = field(default_factory=list)
    model: ModelDetail | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SystemStatus:
        """Create from API response dict."""
        health_str = data.get("health", "unhealthy")
        health = HealthStatus(health_str)

        storage_data = data.get("storage", {})
        storage = StorageStats(
            mode=storage_data.get("mode", "unknown"),
            connected=storage_data.get("connected", False),
            text_collection_count=storage_data.get("text_collection_count", 0),
            audio_collection_count=storage_data.get("audio_collection_count", 0),
            total_tracks=storage_data.get("total_tracks", 0),
        )

        model = None
        if model_data := data.get("model"):
            model = ModelDetail.from_dict(model_data)

        return cls(
            version=data.get("version", "unknown"),
            health=health,
            uptime_seconds=data.get("uptime_seconds", 0),
            storage=storage,
            features=data.get("features", []),
            model=model,
        )


@dataclass
class DownloadModelResult:
    """Result from starting a model download."""

    model_id: str
    message: str
    download_id: str | None = None
    already_exists: bool = False


@dataclass
class LoadModelResult:
    """Result from loading a model."""

    model_id: str
    loaded: bool
    message: str
    device: str | None = None


@dataclass
class DeleteModelResult:
    """Result from deleting a model."""

    model_id: str
    deleted: bool
    message: str


class SidecarClient:
    """
    Async HTTP client for the Music Assistant Insight Sidecar.

    Communicates with the Rust sidecar using MessagePack serialization
    for efficient binary transfer of embeddings.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8096",
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize the sidecar client.

        :param base_url: Base URL of the sidecar API.
        :param timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: aiohttp.ClientSession | None = None
        self.logger = LOGGER.getChild("sidecar_client")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
                headers={"Content-Type": "application/msgpack"},
            )
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    @staticmethod
    async def _get_error_body(resp: aiohttp.ClientResponse) -> str:
        """Safely extract error body from response (msgpack or text)."""
        try:
            data = await resp.read()
            # Try to decode as msgpack first
            try:
                result = msgpack.unpackb(data, raw=False)
                if isinstance(result, dict):
                    return str(result.get("error", result))
                return str(result)
            except Exception:
                # Fall back to text decoding
                return data.decode("utf-8", errors="replace")
        except Exception:
            return "<failed to read response body>"

    async def health_check(self) -> dict[str, Any]:
        """
        Check if the sidecar is healthy.

        :return: Health status dict with 'status', 'model_loaded', 'storage_ready'.
        """
        session = await self._get_session()
        url = f"{self.base_url}/api/v1/health"
        async with session.get(url) as resp:
            if resp.status != 200:
                raise ConnectionError(f"Sidecar health check failed: {resp.status}")
            data = await resp.read()
            result: dict[str, Any] = msgpack.unpackb(data, raw=False)
            return result

    async def embed_text_and_store(
        self,
        track_id: str,
        metadata: TrackMetadata,
        metadata_hash: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate text embedding from metadata and store it in one operation.

        :param track_id: Unique track identifier.
        :param metadata: Track metadata for embedding.
        :param metadata_hash: Optional hash for change detection.
        :return: Response with track_id, stored status, and embedded text.
        """
        session = await self._get_session()
        url = f"{self.base_url}/api/v1/tracks/embed-text"

        payload = {
            "track_id": track_id,
            "metadata": {
                "name": metadata.name,
                "artists": metadata.artists,
                "album": metadata.album,
                "genres": metadata.genres or [],
                "metadata_hash": metadata_hash,
            },
        }

        data = msgpack.packb(payload)
        async with session.post(url, data=data) as resp:
            if resp.status != 200:
                body = await self._get_error_body(resp)
                raise RuntimeError(f"Failed to embed and store track: {resp.status} - {body}")
            result: dict[str, Any] = msgpack.unpackb(await resp.read(), raw=False)
            return result

    async def batch_embed_text_and_store(
        self,
        tracks: list[tuple[str, TrackMetadata]],
    ) -> dict[str, Any]:
        """
        Batch generate text embeddings and store them.

        :param tracks: List of (track_id, metadata) tuples.
        :return: Response with results for each track.
        """
        session = await self._get_session()
        url = f"{self.base_url}/api/v1/tracks/batch-embed-text"

        payload = {
            "tracks": [
                {
                    "track_id": track_id,
                    "metadata": {
                        "name": meta.name,
                        "artists": meta.artists,
                        "album": meta.album,
                        "genres": meta.genres or [],
                    },
                }
                for track_id, meta in tracks
            ]
        }

        data = msgpack.packb(payload)
        async with session.post(url, data=data) as resp:
            if resp.status != 200:
                body = await self._get_error_body(resp)
                raise RuntimeError(f"Batch embed failed: {resp.status} - {body}")
            result: dict[str, Any] = msgpack.unpackb(await resp.read(), raw=False)
            return result

    async def search(
        self,
        query: str,
        limit: int = 25,
    ) -> list[SearchResult]:
        """
        Search for tracks using a text query.

        Generates an embedding for the query and searches the vector store.

        :param query: Text search query.
        :param limit: Maximum number of results.
        :return: List of search results with scores.
        """
        session = await self._get_session()

        # First, generate embedding for the query
        embed_url = f"{self.base_url}/api/v1/embed/text"
        embed_payload = {"text": query}
        embed_data = msgpack.packb(embed_payload)

        async with session.post(embed_url, data=embed_data) as resp:
            if resp.status != 200:
                body = await self._get_error_body(resp)
                raise RuntimeError(f"Failed to embed query: {resp.status} - {body}")
            embed_result = msgpack.unpackb(await resp.read(), raw=False)
            embedding = embed_result["embedding"]

        # Then search with the embedding
        search_url = f"{self.base_url}/api/v1/tracks/search"
        search_payload = {
            "embedding": embedding,
            "collection": "text",
            "limit": limit,
        }
        search_data = msgpack.packb(search_payload)

        async with session.post(search_url, data=search_data) as resp:
            if resp.status != 200:
                body = await self._get_error_body(resp)
                raise RuntimeError(f"Search failed: {resp.status} - {body}")
            result = msgpack.unpackb(await resp.read(), raw=False)

        return [
            SearchResult(
                track_id=r["track_id"],
                score=r["score"],
                metadata=r.get("metadata", {}),
            )
            for r in result.get("results", [])
        ]

    async def get_similar_tracks(
        self,
        track_id: str,
        limit: int = 25,
    ) -> list[SearchResult]:
        """
        Find tracks similar to a given track.

        :param track_id: The track ID to find similar tracks for.
        :param limit: Maximum number of results.
        :return: List of similar tracks with scores.
        """
        session = await self._get_session()

        # First, get the track's embedding
        get_url = f"{self.base_url}/api/v1/tracks/{track_id}?include_text=true"
        async with session.get(get_url) as resp:
            if resp.status == 404:
                return []
            if resp.status != 200:
                body = await self._get_error_body(resp)
                raise RuntimeError(f"Failed to get track: {resp.status} - {body}")
            track_data = msgpack.unpackb(await resp.read(), raw=False)

        embedding = track_data.get("text_embedding")
        if not embedding:
            return []

        # Search for similar tracks
        search_url = f"{self.base_url}/api/v1/tracks/search"
        search_payload = {
            "embedding": embedding,
            "collection": "text",
            "limit": limit + 1,  # +1 to exclude self
            "filter": {"exclude_ids": [track_id]},
        }
        search_data = msgpack.packb(search_payload)

        async with session.post(search_url, data=search_data) as resp:
            if resp.status != 200:
                body = await self._get_error_body(resp)
                raise RuntimeError(f"Similar search failed: {resp.status} - {body}")
            result = msgpack.unpackb(await resp.read(), raw=False)

        return [
            SearchResult(
                track_id=r["track_id"],
                score=r["score"],
                metadata=r.get("metadata", {}),
            )
            for r in result.get("results", [])
            if r["track_id"] != track_id
        ][:limit]

    async def track_exists(self, track_id: str) -> bool:
        """
        Check if a track has embeddings in storage.

        :param track_id: The track ID to check.
        :return: True if the track exists in storage.
        """
        session = await self._get_session()
        url = f"{self.base_url}/api/v1/tracks/{track_id}"

        async with session.get(url) as resp:
            return resp.status == 200

    async def get_track_hash(self, track_id: str) -> str | None:
        """
        Get the stored metadata hash for a track.

        :param track_id: The track ID to check.
        :return: The stored hash or None if track not found.
        """
        session = await self._get_session()
        url = f"{self.base_url}/api/v1/tracks/{track_id}"

        async with session.get(url) as resp:
            if resp.status == 404:
                return None
            if resp.status != 200:
                return None
            data: dict[str, Any] = msgpack.unpackb(await resp.read(), raw=False)
            metadata: dict[str, Any] = data.get("metadata", {})
            hash_val: str | None = metadata.get("metadata_hash")
            return hash_val

    async def delete_track(self, track_id: str) -> bool:
        """
        Delete a track's embeddings from storage.

        :param track_id: The track ID to delete.
        :return: True if deletion was successful.
        """
        session = await self._get_session()
        url = f"{self.base_url}/api/v1/tracks/{track_id}"

        # Request body specifies which collections to delete from
        payload = {"text": True, "audio": True}
        data = msgpack.packb(payload)

        async with session.delete(url, data=data) as resp:
            if resp.status == 404:
                return False
            if resp.status != 200:
                body = await self._get_error_body(resp)
                raise RuntimeError(f"Delete failed: {resp.status} - {body}")
            return True

    @staticmethod
    def compute_track_hash(track: Track) -> str:
        """
        Compute a hash of Track for change detection.

        :param track: The Track object.
        :return: SHA-256 hex digest.
        """
        data = track.to_dict()
        json_str = json.dumps(data, sort_keys=True, ensure_ascii=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    @staticmethod
    def track_to_metadata(track: Track) -> TrackMetadata:
        """
        Convert a Music Assistant Track to TrackMetadata.

        :param track: The Track object.
        :return: TrackMetadata for the sidecar.
        """
        album_name = track.album.name if track.album else None
        artist_names = [a.name for a in track.artists] if track.artists else []
        genres = list(track.metadata.genres) if track.metadata.genres else []

        return TrackMetadata(
            name=track.name,
            artists=artist_names,
            album=album_name,
            genres=genres,
        )

    # ========================================================================
    # Management API Methods
    # ========================================================================

    async def get_status(self) -> SystemStatus:
        """
        Get comprehensive system status from the sidecar.

        :return: SystemStatus with version, health, model info, storage stats.
        """
        session = await self._get_session()
        url = f"{self.base_url}/api/v1/status"
        async with session.get(url) as resp:
            if resp.status != 200:
                body = await self._get_error_body(resp)
                raise RuntimeError(f"Failed to get status: {resp.status} - {body}")
            data = msgpack.unpackb(await resp.read(), raw=False)
            return SystemStatus.from_dict(data)

    async def list_models(self) -> tuple[list[ModelDetail], str | None]:
        """
        List all available models (known + cached).

        :return: Tuple of (list of models, current model ID or None).
        """
        session = await self._get_session()
        url = f"{self.base_url}/api/v1/models"
        async with session.get(url) as resp:
            if resp.status != 200:
                body = await self._get_error_body(resp)
                raise RuntimeError(f"Failed to list models: {resp.status} - {body}")
            data = msgpack.unpackb(await resp.read(), raw=False)

            models = [ModelDetail.from_dict(m) for m in data.get("models", [])]
            current_model = data.get("current_model")
            return models, current_model

    async def start_download(self, model_id: str) -> DownloadModelResult:
        """
        Start downloading a model.

        :param model_id: HuggingFace model ID (e.g., "Xenova/clap-htsat-unfused").
        :return: DownloadModelResult with download_id if started.
        """
        session = await self._get_session()
        url = f"{self.base_url}/api/v1/models/download"

        payload = {"model_id": model_id}
        data = msgpack.packb(payload)

        async with session.post(url, data=data) as resp:
            if resp.status != 200:
                body = await self._get_error_body(resp)
                raise RuntimeError(f"Failed to start download: {resp.status} - {body}")
            result = msgpack.unpackb(await resp.read(), raw=False)
            return DownloadModelResult(
                model_id=result["model_id"],
                message=result["message"],
                download_id=result.get("download_id"),
                already_exists=result.get("already_exists", False),
            )

    async def list_downloads(self) -> list[DownloadProgress]:
        """
        Get status of all downloads (active and recent).

        :return: List of download progress objects.
        """
        session = await self._get_session()
        url = f"{self.base_url}/api/v1/models/downloads"
        async with session.get(url) as resp:
            if resp.status != 200:
                body = await self._get_error_body(resp)
                raise RuntimeError(f"Failed to list downloads: {resp.status} - {body}")
            data = msgpack.unpackb(await resp.read(), raw=False)
            return [DownloadProgress.from_dict(d) for d in data.get("downloads", [])]

    async def get_download_progress(self, download_id: str) -> DownloadProgress | None:
        """
        Get progress for a specific download.

        :param download_id: The download ID returned from start_download.
        :return: DownloadProgress or None if not found.
        """
        downloads = await self.list_downloads()
        for d in downloads:
            if d.download_id == download_id:
                return d
        return None

    async def load_model(self, model_id: str) -> LoadModelResult:
        """
        Load a downloaded model (hot-swap).

        :param model_id: Model ID to load (must be downloaded first).
        :return: LoadModelResult with success status.
        """
        session = await self._get_session()
        url = f"{self.base_url}/api/v1/models/{model_id}/load"

        # The request body can be empty, but we send the model_id for consistency
        payload = {"model_id": model_id}
        data = msgpack.packb(payload)

        async with session.post(url, data=data) as resp:
            if resp.status != 200:
                body = await self._get_error_body(resp)
                raise RuntimeError(f"Failed to load model: {resp.status} - {body}")
            result = msgpack.unpackb(await resp.read(), raw=False)
            return LoadModelResult(
                model_id=result["model_id"],
                loaded=result["loaded"],
                message=result["message"],
                device=result.get("device"),
            )

    async def delete_model(self, model_id: str) -> DeleteModelResult:
        """
        Delete a cached model.

        Cannot delete the currently loaded model.

        :param model_id: Model ID to delete.
        :return: DeleteModelResult with success status.
        """
        session = await self._get_session()
        url = f"{self.base_url}/api/v1/models/{model_id}"
        async with session.delete(url) as resp:
            if resp.status != 200:
                body = await self._get_error_body(resp)
                raise RuntimeError(f"Failed to delete model: {resp.status} - {body}")
            result = msgpack.unpackb(await resp.read(), raw=False)
            return DeleteModelResult(
                model_id=result["model_id"],
                deleted=result["deleted"],
                message=result["message"],
            )

    async def get_storage_stats(self) -> StorageStats:
        """
        Get detailed storage statistics.

        :return: StorageStats with track counts and connection info.
        """
        session = await self._get_session()
        url = f"{self.base_url}/api/v1/storage/stats"
        async with session.get(url) as resp:
            if resp.status != 200:
                body = await self._get_error_body(resp)
                raise RuntimeError(f"Failed to get storage stats: {resp.status} - {body}")
            data = msgpack.unpackb(await resp.read(), raw=False)
            stats = data.get("stats", data)  # Handle wrapped and unwrapped format
            return StorageStats(
                mode=stats.get("mode", "unknown"),
                connected=stats.get("connected", False),
                text_collection_count=stats.get("text_collection_count", 0),
                audio_collection_count=stats.get("audio_collection_count", 0),
                total_tracks=stats.get("total_tracks", 0),
            )

    async def wait_for_download(
        self,
        download_id: str,
        poll_interval: float = 1.0,
        timeout: float = 600.0,
    ) -> DownloadProgress:
        """
        Wait for a download to complete, polling for progress.

        :param download_id: The download ID to wait for.
        :param poll_interval: Seconds between progress checks.
        :param timeout: Maximum seconds to wait.
        :return: Final DownloadProgress when complete.
        :raises TimeoutError: If download doesn't complete in time.
        :raises RuntimeError: If download fails.
        """
        elapsed = 0.0
        while elapsed < timeout:
            progress = await self.get_download_progress(download_id)
            if progress is None:
                raise RuntimeError(f"Download {download_id} not found")

            if progress.status == DownloadStatus.COMPLETED:
                return progress
            if progress.status == DownloadStatus.FAILED:
                raise RuntimeError(f"Download failed: {progress.error}")
            if progress.status == DownloadStatus.CANCELLED:
                raise RuntimeError("Download was cancelled")

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        raise TimeoutError(f"Download {download_id} timed out after {timeout}s")

    # ========================================================================
    # Taste Profile API
    # ========================================================================

    async def compute_taste_profile(
        self,
        user_id: str,
        interactions: list[dict[str, Any]],
        cutoff_days: int = 21,
    ) -> dict[str, Any]:
        """
        Compute taste profile from user interactions.

        :param user_id: User ID to compute profile for.
        :param interactions: List of interaction dicts.
        :param cutoff_days: Number of days of history to consider.
        :return: Response with user_id and profile metadata.
        """
        session = await self._get_session()
        url = f"{self.base_url}/api/v1/users/{user_id}/profile/compute"

        payload = {
            "interactions": interactions,
            "cutoff_days": cutoff_days,
            "profile_type": "global",
        }

        data = msgpack.packb(payload)
        async with session.post(url, data=data) as resp:
            if resp.status != 200:
                body = await self._get_error_body(resp)
                raise RuntimeError(f"Failed to compute taste profile: {resp.status} - {body}")
            result: dict[str, Any] = msgpack.unpackb(await resp.read(), raw=False)
            return result

    async def get_taste_recommendations(
        self,
        user_id: str,
        limit: int = 25,
        exclude_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Get personalized recommendations based on taste profile.

        :param user_id: User ID to get recommendations for.
        :param limit: Maximum number of recommendations.
        :param exclude_ids: Track IDs to exclude from results.
        :return: Response with tracks list and profile_confidence.
        """
        session = await self._get_session()
        url = f"{self.base_url}/api/v1/users/{user_id}/recommend"

        payload = {
            "limit": limit,
            "profile_type": {"type": "Global"},  # Adjacently tagged enum format
            "exclude_ids": exclude_ids or [],
            "filter": {},
        }

        data = msgpack.packb(payload)
        async with session.post(url, data=data) as resp:
            if resp.status != 200:
                body = await self._get_error_body(resp)
                raise RuntimeError(f"Failed to get recommendations: {resp.status} - {body}")
            result: dict[str, Any] = msgpack.unpackb(await resp.read(), raw=False)
            return result
